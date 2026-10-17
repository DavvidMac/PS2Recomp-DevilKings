#include "ps2_runtime.h"
#include "ps2_log.h"
#include "ps2_stubs.h"
#include "runtime/ps2_profiler.h"
#include "ps2_syscalls.h"
#include "game_overrides.h"
#include "ps2_runtime_macros.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_hud_layout.h"
#include "runtime/gs/gs_pgs.h"
#include "runtime/gs/gs_gles.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/ps2_touch_controls.h"
#include "runtime/ee_scheduler.h"
#include "ThreadNaming.h"
#include "Kernel/Stubs/Audio.h"
#include "Kernel/Stubs/GS.h"
#include "Kernel/Stubs/MPEG.h"
#include "ps2_host_backend.h"
#include "ps2_iop_host.h"
#include "ps2x/iop/iop_subsystem.h"

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <chrono>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <sstream>
#include "runtime/ps2_settings.h"

namespace
{
    struct WatchRange
    {
        uint32_t start = 0u;
        uint32_t length = 0u;
        uint64_t fromVsync = 0u;
    };

    WatchRange parseWatchRange()
    {
        WatchRange range;
        const char *value = std::getenv("PS2X_WATCH");
        if (!value || !*value)
            return range;
        char *end = nullptr;
        range.start = static_cast<uint32_t>(std::strtoul(value, &end, 0)) & 0x1FFFFFFFu;
        range.length = 4u;
        if (end && *end == ':')
            range.length = static_cast<uint32_t>(std::strtoul(end + 1, &end, 0));
        if (end && *end == ':')
            range.fromVsync = std::strtoull(end + 1, nullptr, 0);
        return range;
    }

    const WatchRange kWatchRange = parseWatchRange();
    const uint8_t *s_watchRdram = nullptr;
    const GSRegisters *s_watchGs = nullptr;

    bool looksLikeReturnAddress(const uint8_t *rdram, uint32_t value)
    {
        if ((value & 3u) != 0u || value < 0x00100008u || value >= PS2_RAM_SIZE)
            return false;
        uint32_t call = 0u;
        std::memcpy(&call, rdram + value - 8u, sizeof(call));
        const bool isJal = (call >> 26) == 3u;
        const bool isJalr = (call >> 26) == 0u && (call & 0x3Fu) == 9u;
        return isJal || isJalr;
    }
}

uint32_t g_ps2WatchStart = kWatchRange.start;
uint32_t g_ps2WatchLength = kWatchRange.length;

void ps2ReportWatchedWrite(uint32_t guestAddr, uint32_t size, uint64_t valueLo, uint64_t valueHi,
                           const char *op, const R5900Context *ctx)
{
    if (kWatchRange.fromVsync != 0u &&
        (!s_watchGs || s_watchGs->vsyncTick.load(std::memory_order_relaxed) < kWatchRange.fromVsync))
        return;
    static std::atomic<uint32_t> s_reports{0u};
    if (s_reports.fetch_add(1u, std::memory_order_relaxed) >= 4000u)
        return;
    std::fprintf(stderr, "[watch] %s addr=0x%08x size=%u value=0x%016llx%016llx pc=0x%08x ra=0x%08x\n",
                 op, guestAddr, size, static_cast<unsigned long long>(valueHi),
                 static_cast<unsigned long long>(valueLo), ctx ? ctx->pc : 0u,
                 ctx ? getRegU32(ctx, 31) : 0u);
    if (!ctx || !s_watchRdram)
        return;
    // Saved return addresses found in the stack frame give a rough backtrace.
    const uint32_t sp = getRegU32(ctx, 29) & 0x1FFFFFFFu;
    std::fprintf(stderr, "[watch]   a0-a3=%08x %08x %08x %08x s0-s8=%08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
                 getRegU32(ctx, 4), getRegU32(ctx, 5), getRegU32(ctx, 6), getRegU32(ctx, 7),
                 getRegU32(ctx, 16), getRegU32(ctx, 17), getRegU32(ctx, 18), getRegU32(ctx, 19),
                 getRegU32(ctx, 20), getRegU32(ctx, 21), getRegU32(ctx, 22), getRegU32(ctx, 23),
                 getRegU32(ctx, 30));
    std::fprintf(stderr, "[watch]   sp=0x%08x stack:", sp);
    for (uint32_t offset = 0u; offset < 0x400u && sp + offset + 4u <= PS2_RAM_SIZE; offset += 4u)
    {
        uint32_t value = 0u;
        std::memcpy(&value, s_watchRdram + sp + offset, sizeof(value));
        if (looksLikeReturnAddress(s_watchRdram, value))
            std::fprintf(stderr, " +0x%x=0x%08x", offset, value);
    }
    std::fputc('\n', stderr);
}

namespace ps2_stubs
{
    void resetSifState();
}

#define ELF_MAGIC 0x464C457F // "\x7FELF" in little endian
#define ET_EXEC 2            // Executable file
#define EM_MIPS 8            // MIPS architecture
#define PT_LOAD 1            // Loadable segment

static constexpr int FB_WIDTH = 640;
static constexpr int FB_HEIGHT = 512;
static constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
static constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;
static constexpr uint32_t DEFAULT_FB_ADDR = (PS2_RAM_SIZE - DEFAULT_FB_SIZE - 0x10000u);
#if defined(PLATFORM_VITA)
static constexpr int HOST_WINDOW_WIDTH = 960;
static constexpr int HOST_WINDOW_HEIGHT = 544;
#elif defined(__ANDROID__)
// raylib takes 0 x 0 as the phone's own size: the picture and the touch controls get the whole
// screen and follow rotation.
static constexpr int HOST_WINDOW_WIDTH = 0;
static constexpr int HOST_WINDOW_HEIGHT = 0;
#else
static constexpr int HOST_WINDOW_WIDTH = FB_WIDTH;
static constexpr int HOST_WINDOW_HEIGHT = DEFAULT_DISPLAY_HEIGHT;
#endif
struct ElfHeader
{
    uint32_t magic;
    uint8_t elf_class;
    uint8_t endianness;
    uint8_t version;
    uint8_t os_abi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
};

struct ProgramHeader
{
    uint32_t type;
    uint32_t offset;
    uint32_t vaddr;
    uint32_t paddr;
    uint32_t filesz;
    uint32_t memsz;
    uint32_t flags;
    uint32_t align;
};

namespace
{
    constexpr uint32_t kGuestHeapDefaultBase = 0x00100000u;
    constexpr uint32_t kGuestHeapDefaultAlignment = 16u;
    constexpr uint32_t kGuestHeapSafetyPad = 0x1000u;
    constexpr uint32_t kGuestHeapHardLimit = 0x01F00000u;

    // Interrupt/alarm/callback handlers run on the EE kernel stack on hardware,
    // which lives below the user area (0x00080000). Keep their stacks there:
    // games freely place thread stacks at the top of user RAM.
    constexpr uint32_t kAsyncCallbackStackFloor = 0x00020000u;
    constexpr uint32_t kAsyncCallbackStackTop = 0x00080000u;

    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_GENERAL = 0x80000080u;
    constexpr uint32_t EXCEPTION_VECTOR_TLB_REFILL = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT = 0xBFC00200u;

    struct DispatchHistory
    {
        std::array<uint32_t, 64> pcs{};
        uint32_t next = 0u;
        bool wrapped = false;
    };

    thread_local DispatchHistory g_dispatchHistory;

    bool computeFileCrc32(const std::string &path, uint32_t &crcOut)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        static const std::array<uint32_t, 256> table = []
        {
            std::array<uint32_t, 256> values{};
            for (uint32_t i = 0; i < values.size(); ++i)
            {
                uint32_t value = i;
                for (uint32_t bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1u) ? (0xEDB88320u ^ (value >> 1u)) : (value >> 1u);
                }
                values[i] = value;
            }
            return values;
        }();

        uint32_t crc = 0xFFFFFFFFu;
        std::array<uint8_t, 16 * 1024> buffer{};
        while (file.good())
        {
            file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = file.gcount();
            for (std::streamsize i = 0; i < count; ++i)
            {
                crc = table[(crc ^ buffer[static_cast<size_t>(i)]) & 0xFFu] ^ (crc >> 8u);
            }
        }
        if (file.bad())
        {
            return false;
        }
        crcOut = ~crc;
        return true;
    }

    void pushDispatchPc(uint32_t pc)
    {
        DispatchHistory &h = g_dispatchHistory;
        h.pcs[h.next] = pc;
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.pcs.size());
        if (h.next == 0u)
        {
            h.wrapped = true;
        }
    }

    std::string formatDispatchHistory()
    {
        const DispatchHistory &h = g_dispatchHistory;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return "(empty)";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t idx = (h.next + h.pcs.size() - count + i) % static_cast<uint32_t>(h.pcs.size());
            if (!first)
            {
                oss << " -> ";
            }
            first = false;
            oss << "0x" << std::hex << h.pcs[idx];
        }
        return oss.str();
    }

    uint32_t selectExceptionVector(const R5900Context *ctx, bool tlbRefill)
    {
        if (ctx->cop0_status & COP0_STATUS_BEV)
        {
            return EXCEPTION_VECTOR_BOOT;
        }
        return tlbRefill ? EXCEPTION_VECTOR_TLB_REFILL : EXCEPTION_VECTOR_GENERAL;
    }

    void seedVu0IdleSuccess(R5900Context *ctx)
    {
        if (!ctx)
        {
            return;
        }

        ctx->vu0_clip_flags = 0;
        ctx->vu0_clip_flags2 = 0;
        ctx->vu0_mac_flags = 0;
        ctx->vu0_status = 0;
        ctx->vu0_q = 1.0f;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;
    }

    void copyVu0ContextToState(const R5900Context *ctx, VU1State &state)
    {
        std::memset(&state, 0, sizeof(state));

        for (uint32_t i = 0; i < 32u; ++i)
        {
            _mm_storeu_ps(state.vf[i], ctx->vu0_vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            state.vi[i] = static_cast<int16_t>(ctx->vi[i]);
        }

        _mm_storeu_ps(state.acc, ctx->vu0_acc);
        state.q = ctx->vu0_q;
        state.p = ctx->vu0_p;
        state.i = ctx->vu0_i;
        alignas(16) uint32_t rWords[4]{};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(rWords), _mm_castps_si128(ctx->vu0_r));
        state.r = 0x3F800000u | (rWords[0] & 0x007FFFFFu);
        state.pc = ctx->vu0_pc;
        state.mac = ctx->vu0_mac_flags;
        state.clip = ctx->vu0_clip_flags;
        state.status = ctx->vu0_status;
        state.itop = ctx->vu0_itop;
        state.dBitEnabled = (ctx->vu0_fbrst & (1u << 2)) != 0u;
        state.tBitEnabled = (ctx->vu0_fbrst & (1u << 3)) != 0u;

        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;
    }

    void copyVu0StateToContext(const VU1State &state, R5900Context *ctx)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            ctx->vu0_vf[i] = _mm_loadu_ps(state.vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            ctx->vi[i] = static_cast<uint16_t>(state.vi[i]);
        }

        ctx->vu0_acc = _mm_loadu_ps(state.acc);
        ctx->vu0_q = state.q;
        ctx->vu0_p = state.p;
        ctx->vu0_i = state.i;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int32_t>(state.r)));
        ctx->vu0_mac_flags = state.mac;
        ctx->vu0_clip_flags = state.clip;
        ctx->vu0_clip_flags2 = state.clip;
        ctx->vu0_status = static_cast<uint16_t>(state.status);
        ctx->vu0_itop = state.itop;
        ctx->vu0_pc = state.pc;
        ctx->vu0_tpc = state.pc;
        ctx->vu0_vpu_stat = (ctx->vu0_vpu_stat & 0xFF00u) | (state.stoppedByD ? (1u << 1) : 0u) | (state.stoppedByT ? (1u << 2) : 0u);
        ctx->vu0_vpu_stat2 = 0;

        ctx->vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
        ctx->vi[0] = 0;
    }

    void raiseCop0Exception(R5900Context *ctx, uint32_t exceptionCode, bool tlbRefill = false)
    {
        if (ctx->in_delay_slot)
        {
            ctx->cop0_epc = ctx->branch_pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~COP0_CAUSE_EXCCODE_MASK) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK) |
                              COP0_CAUSE_BD;
        }
        else
        {
            ctx->cop0_epc = ctx->pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~(COP0_CAUSE_EXCCODE_MASK | COP0_CAUSE_BD)) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK);
        }

        ctx->cop0_status |= COP0_STATUS_EXL;
        ctx->pc = selectExceptionVector(ctx, tlbRefill);
        ctx->in_delay_slot = false;
    }

    std::filesystem::path normalizeAbsolutePath(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return {};
        }

#if defined(PLATFORM_VITA)
        const std::string generic = path.generic_string();
        const std::size_t colon = generic.find(':');
        if (colon != std::string::npos && colon != 0u)
        {
            const std::size_t slash = generic.find_first_of("/\\");
            if (slash == std::string::npos || colon < slash)
            {
                return path.lexically_normal();
            }
        }
#endif

        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        if (ec)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    PS2Runtime::IoPaths &runtimeIoPaths()
    {
        static PS2Runtime::IoPaths paths = []()
        {
            PS2Runtime::IoPaths defaults;
            std::error_code ec;
            const std::filesystem::path cwd = std::filesystem::current_path(ec);
            defaults.elfDirectory = ec ? std::filesystem::path(".") : cwd.lexically_normal();
            defaults.hostRoot = defaults.elfDirectory;
            defaults.cdRoot = defaults.elfDirectory;
            defaults.mcRoot = defaults.elfDirectory / "mc0";
            return defaults;
        }();

        return paths;
    }

    std::string readGuestPrintableString(const uint8_t *rdram, uint32_t addr, size_t maxLen)
    {
        std::string out;
        if (!rdram || maxLen == 0)
        {
            return out;
        }

        out.reserve(std::min<size_t>(maxLen, 64));
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char ch = static_cast<char>(rdram[(addr + static_cast<uint32_t>(i)) & PS2_RAM_MASK]);
            if (ch == '\0')
            {
                break;
            }
            if (ch >= 0x20 && ch < 0x7F)
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('.');
            }
        }
        return out;
    }
}

namespace
{
    // SPU2 output from the IOP, pulled by raylib's audio thread.
    std::atomic<ps2x::iop::IopSubsystem *> g_spu2AudioSource{nullptr};
#if !defined(PLATFORM_VITA)
    AudioStream g_spu2Stream{};

    void spu2AudioCallback(void *buffer, unsigned int frames)
    {
        auto *out = static_cast<int16_t *>(buffer);
        size_t produced = 0u;
        if (ps2x::iop::IopSubsystem *iop = g_spu2AudioSource.load())
            produced = iop->readAudio(out, frames);
        std::fill(out + produced * 2u, out + static_cast<size_t>(frames) * 2u, int16_t{0});

        // PS2X_AUDIO_DUMP=<file>: append the played 48 kHz stereo s16 PCM.
        static FILE *s_dump = []() -> FILE *
        {
            const char *path = std::getenv("PS2X_AUDIO_DUMP");
            return (path && *path) ? std::fopen(path, "wb") : nullptr;
        }();
        if (s_dump)
        {
            std::fwrite(out, sizeof(int16_t) * 2u, frames, s_dump);
            std::fflush(s_dump);
        }
    }
#endif
}

// Interlaced field rendering: games draw each field into a half-height buffer (e.g. 512x224 for
// a 448-line display) and alternate buffers per VSync, offsetting the odd field by half a line.
// Showing each field as a full frame makes the picture bob up and down; weave the latest two
// fields instead, like a TV does. PS2X_DEINTERLACE=0 disables this.
static bool DeinterlaceEnabled()
{
    static const bool s_deinterlace = []()
    {
        const char *value = std::getenv("PS2X_DEINTERLACE");
        return !(value && value[0] == '0');
    }();
    return s_deinterlace;
}

class FieldWeaver
{
public:
    // Writes `field` (width x height, packed RGBA) woven with the previous field into `out`:
    // 2 * height rows, outStrideBytes apart.
    void weave(const uint8_t *field, size_t fieldBytes, uint32_t width, uint32_t height, uint64_t tick, uint8_t *out,
               size_t outStrideBytes)
    {
        const size_t rowBytes = static_cast<size_t>(width) * 4u;
        const bool havePrevious = m_width == width && m_height == height && m_previous.size() >= rowBytes * height;
        // Which output rows the current field owns. The field shown can lag the VSync it is
        // shown on (asynchronous rasterisation, dropped frames, a GPU readback), so pick the
        // order that combs least against the previous field instead of trusting VSync parity,
        // and only switch on a clear win.
        uint32_t parity = static_cast<uint32_t>(tick & 1u);
        if (havePrevious && height > 4u)
        {
            uint64_t combEven = 0u; // current on even rows: missing row above is previous[y-1]
            uint64_t combOdd = 0u;  // current on odd rows: missing row below is previous[y+1]
            for (uint32_t y = 1u; y + 1u < height; y += 2u)
            {
                const uint8_t *current = field + static_cast<size_t>(y) * rowBytes;
                const uint8_t *above = m_previous.data() + static_cast<size_t>(y - 1u) * rowBytes;
                const uint8_t *below = m_previous.data() + static_cast<size_t>(y + 1u) * rowBytes;
                for (uint32_t x = 0; x < width; x += 4u)
                {
                    const uint8_t *c = current + x * 4u;
                    const uint8_t *a = above + x * 4u;
                    const uint8_t *b = below + x * 4u;
                    for (uint32_t ch = 0; ch < 3u; ++ch)
                    {
                        combEven += static_cast<uint32_t>(std::abs(static_cast<int>(c[ch]) - static_cast<int>(a[ch])));
                        combOdd += static_cast<uint32_t>(std::abs(static_cast<int>(c[ch]) - static_cast<int>(b[ch])));
                    }
                }
            }
            if (combEven * 5u < combOdd * 4u)
                m_parity = 0u;
            else if (combOdd * 5u < combEven * 4u)
                m_parity = 1u;
            parity = m_parity;
        }
        const uint32_t fieldRows = std::min<uint32_t>(height, static_cast<uint32_t>(fieldBytes / rowBytes));
        for (uint32_t y = 0; y < fieldRows; ++y)
        {
            const uint8_t *current = field + static_cast<size_t>(y) * rowBytes;
            const uint8_t *previous = havePrevious ? m_previous.data() + static_cast<size_t>(y) * rowBytes : current;
            // The missing line sits between current lines y and y+1 (even field) or y-1 and y
            // (odd field).
            const uint32_t neighbourY = parity == 0u ? std::min(y + 1u, fieldRows - 1u) : (y == 0u ? 0u : y - 1u);
            const uint8_t *neighbour = field + static_cast<size_t>(neighbourY) * rowBytes;
            std::memcpy(out + static_cast<size_t>(2u * y + parity) * outStrideBytes, current, rowBytes);
            uint8_t *missing = out + static_cast<size_t>(2u * y + (1u - parity)) * outStrideBytes;
            // Motion adaptive: keep the previous field where the picture is static (full
            // vertical detail), interpolate where it moved so consecutive frames do not comb.
            for (uint32_t x = 0; x < width; ++x)
            {
                const uint8_t *cur = current + x * 4u;
                const uint8_t *nb = neighbour + x * 4u;
                const uint8_t *prev = previous + x * 4u;
                uint8_t *dst = missing + x * 4u;
                int difference = 0;
                uint8_t interpolated[4];
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    interpolated[c] = static_cast<uint8_t>((static_cast<uint32_t>(cur[c]) + nb[c] + 1u) >> 1u);
                    if (c < 3u)
                        difference += std::abs(static_cast<int>(prev[c]) - static_cast<int>(interpolated[c]));
                }
                if (difference > 48)
                    std::memcpy(dst, interpolated, 4u);
                else
                    std::memcpy(dst, prev, 4u);
            }
        }
        m_previous.assign(field, field + std::min(fieldBytes, rowBytes * height));
        m_width = width;
        m_height = height;
    }

private:
    std::vector<uint8_t> m_previous;
    uint32_t m_width = 0u;
    uint32_t m_height = 0u;
    uint32_t m_parity = 0u;
};

// Diagnostics: PS2X_FRAME_DUMP_DIR=<dir> saves the presented frame as PNG every
// PS2X_FRAME_DUMP_INTERVAL ticks (120 by default) from PS2X_FRAME_DUMP_FROM on, so a run can be
// inspected without watching the window.
static void DumpPresentedFrame(uint64_t currentTick, const uint8_t *rgba, uint32_t width, uint32_t height,
                               uint32_t strideBytes, uint32_t displayFbp)
{
    static const char *s_frameDumpDir = std::getenv("PS2X_FRAME_DUMP_DIR");
    static uint64_t s_lastFrameDumpTick = 0u;
    static uint32_t s_frameDumpIndex = 0u;
    static const uint64_t s_frameDumpInterval = []() -> uint64_t
    {
        const char *value = std::getenv("PS2X_FRAME_DUMP_INTERVAL");
        return value ? std::max<uint64_t>(1u, std::strtoull(value, nullptr, 10)) : 120u;
    }();
    static const uint64_t s_frameDumpFromTick = []() -> uint64_t
    {
        const char *value = std::getenv("PS2X_FRAME_DUMP_FROM");
        return value ? std::strtoull(value, nullptr, 10) : 0u;
    }();
    if (!s_frameDumpDir || !*s_frameDumpDir || currentTick < s_frameDumpFromTick ||
        currentTick < s_lastFrameDumpTick + s_frameDumpInterval)
        return;
    s_lastFrameDumpTick = currentTick;
    const uint32_t dumpWidth = std::max<uint32_t>(1u, width);
    const uint32_t dumpHeight = std::max<uint32_t>(1u, height);
    std::vector<uint8_t> pixels(static_cast<size_t>(dumpWidth) * dumpHeight * 4u);
    for (uint32_t y = 0; y < dumpHeight; ++y)
    {
        std::memcpy(pixels.data() + static_cast<size_t>(y) * dumpWidth * 4u,
                    rgba + static_cast<size_t>(y) * strideBytes,
                    static_cast<size_t>(dumpWidth) * 4u);
    }
    const std::string name = std::string(s_frameDumpDir) + "/frame_" + std::to_string(s_frameDumpIndex++) + "_t" +
                             std::to_string(currentTick) + "_fbp" + std::to_string(displayFbp);
#if defined(__ANDROID__)
    // raylib on Android prefixes every file it writes with the app's internal data folder, so
    // write a PPM here instead of going through ExportImage.
    if (FILE *out = std::fopen((name + ".ppm").c_str(), "wb"))
    {
        std::fprintf(out, "P6\n%u %u\n255\n", dumpWidth, dumpHeight);
        for (size_t i = 0; i < pixels.size(); i += 4u)
            std::fwrite(&pixels[i], 1u, 3u, out);
        std::fclose(out);
    }
#else
    for (size_t i = 3; i < pixels.size(); i += 4u)
    {
        pixels[i] = 0xFFu; // PS2 alpha is 0..0x80; make the dump opaque
    }
    Image frame{};
    frame.data = pixels.data();
    frame.width = static_cast<int>(dumpWidth);
    frame.height = static_cast<int>(dumpHeight);
    frame.mipmaps = 1;
    frame.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    ExportImage(frame, (name + ".png").c_str());
#endif
}

// PS2X_FRAME_DUMP_DIR with the OpenGL renderer: the picture is read back from its texture,
// only when a dump is due.
static void DumpGlesFrame(const Texture2D &picture, uint64_t currentTick)
{
    static const char *s_frameDumpDir = std::getenv("PS2X_FRAME_DUMP_DIR");
    if (!s_frameDumpDir || !*s_frameDumpDir)
        return;
    static uint64_t s_lastTick = 0u;
    static const uint64_t s_interval = []() -> uint64_t
    {
        const char *value = std::getenv("PS2X_FRAME_DUMP_INTERVAL");
        return value ? std::max<uint64_t>(1u, std::strtoull(value, nullptr, 10)) : 120u;
    }();
    static const uint64_t s_fromTick = []() -> uint64_t
    {
        const char *value = std::getenv("PS2X_FRAME_DUMP_FROM");
        return value ? std::strtoull(value, nullptr, 10) : 0u;
    }();
    if (currentTick < s_fromTick || currentTick < s_lastTick + s_interval)
        return;
    s_lastTick = currentTick;
    Image image = LoadImageFromTexture(picture);
    if (image.data)
        DumpPresentedFrame(currentTick, static_cast<const uint8_t *>(image.data), static_cast<uint32_t>(image.width),
                           static_cast<uint32_t>(image.height), static_cast<uint32_t>(image.width) * 4u, 0u);
    UnloadImage(image);
}

// The Vulkan renderer's newest scanout, in a texture resized to it: 2x and 4x internal
// resolution scan out larger frames than the 640x512 one the software renderer fills.
static void UploadVulkanFrame(Texture2D &tex, uint64_t currentTick, uint32_t &outWidth, uint32_t &outHeight)
{
    static std::vector<uint8_t> s_frame;
    static uint32_t s_width = 0u;
    static uint32_t s_height = 0u;
    static std::vector<uint8_t> s_woven;
    static FieldWeaver s_weaver;
    uint32_t width = 0u;
    uint32_t height = 0u;
    bool field = false;
    if (ps2x::pgs::takeFrame(s_frame, width, height, field) && width != 0u && height != 0u)
    {
        const uint8_t *pixels = s_frame.data();
        if (field && DeinterlaceEnabled())
        {
            s_woven.resize(static_cast<size_t>(width) * height * 8u);
            s_weaver.weave(s_frame.data(), s_frame.size(), width, height, currentTick, s_woven.data(),
                           static_cast<size_t>(width) * 4u);
            pixels = s_woven.data();
            height *= 2u;
        }
        if (tex.width != static_cast<int>(width) || tex.height != static_cast<int>(height))
        {
            UnloadTexture(tex);
            Image blank = GenImageColor(static_cast<int>(width), static_cast<int>(height), BLANK);
            tex = LoadTextureFromImage(blank);
            UnloadImage(blank);
        }
        UpdateTexture(tex, pixels);
        s_width = width;
        s_height = height;
        DumpPresentedFrame(currentTick, pixels, width, height, width * 4u, 0u);
    }
    outWidth = s_width != 0u ? s_width : FB_WIDTH;
    outHeight = s_height != 0u ? s_height : DEFAULT_DISPLAY_HEIGHT;
}

static void UploadFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    static uint64_t s_lastPresentationTick = std::numeric_limits<uint64_t>::max();
    static bool s_hasLatchedInitialFrame = false;
    static uint32_t s_lastDisplayFbp = std::numeric_limits<uint32_t>::max();
    static uint32_t s_lastSourceFbp = std::numeric_limits<uint32_t>::max();
    static bool s_lastPreferred = false;
    static uint32_t s_lastWidth = 0u;
    static uint32_t s_lastHeight = 0u;
    static bool s_hasUploadedFrame = false;
    static std::vector<uint8_t> s_scratch;
    static std::vector<uint8_t> s_uploadBuffer(DEFAULT_FB_SIZE, 0u);

    const uint64_t currentTick = rt->eeScheduler().currentVSyncTick();
    const bool needsLatch = !s_hasLatchedInitialFrame || currentTick != s_lastPresentationTick;
    if (needsLatch)
    {
        rt->gs().latchPresentationOnDisplaySwap();
        if (!rt->gs().isPresentationSwapDriven(currentTick))
            rt->gs().latchHostPresentationFrame();
        s_lastPresentationTick = currentTick;
        s_hasLatchedInitialFrame = true;
    }
    if (ps2x::pgs::active())
    {
        UploadVulkanFrame(tex, currentTick, outWidth, outHeight);
        return;
    }
    // The OpenGL renderer's picture is a texture already (ps2x::gles::displayTexture).
    if (ps2x::gles::active())
        return;
    if (!needsLatch && s_hasUploadedFrame)
    {
        outWidth = (s_lastWidth != 0u) ? s_lastWidth : FB_WIDTH;
        outHeight = (s_lastHeight != 0u) ? s_lastHeight : DEFAULT_DISPLAY_HEIGHT;
        return;
    }

    s_scratch.clear();
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferredDisplaySource = false;
    if (!rt->gs().copyLatchedHostPresentationFrame(s_scratch,
                                                   width,
                                                   height,
                                                   &displayFbp,
                                                   &sourceFbp,
                                                   &usedPreferredDisplaySource))
    {
        Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, MAGENTA);
        UpdateTexture(tex, blank.data);
        UnloadImage(blank);
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        s_lastWidth = outWidth;
        s_lastHeight = outHeight;
        s_hasUploadedFrame = true;
        return;
    }

    PS2_IF_AGRESSIVE_LOGS({
        static uint32_t s_uploadDebugCount = 0u;
        if (s_uploadDebugCount < 128u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred ||
            width != s_lastWidth ||
            height != s_lastHeight)
        {
            std::cout << "[frame:upload] idx=" << s_uploadDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        ++s_uploadDebugCount;
    });
    s_lastDisplayFbp = displayFbp;
    s_lastSourceFbp = sourceFbp;
    s_lastPreferred = usedPreferredDisplaySource;
    s_lastWidth = width;
    s_lastHeight = height;

    static FieldWeaver s_weaver;
    const bool weaveFields = DeinterlaceEnabled() && !s_scratch.empty() && width != 0u && height != 0u &&
                             height <= 256u && height * 2u <= static_cast<uint32_t>(FB_HEIGHT) &&
                             width <= static_cast<uint32_t>(FB_WIDTH);

    std::fill(s_uploadBuffer.begin(), s_uploadBuffer.end(), 0u);
    uint32_t presentedHeight = height;
    if (weaveFields)
    {
        s_weaver.weave(s_scratch.data(), s_scratch.size(), width, height, currentTick, s_uploadBuffer.data(),
                       static_cast<size_t>(FB_WIDTH) * 4u);
        presentedHeight = height * 2u;
    }
    else if (!s_scratch.empty() && width != 0u && height != 0u)
    {
        const uint32_t copyWidth = std::min<uint32_t>(width, FB_WIDTH);
        const uint32_t copyHeight = std::min<uint32_t>(height, FB_HEIGHT);
        const size_t srcRowBytes = static_cast<size_t>(width) * 4u;
        const size_t dstRowBytes = static_cast<size_t>(FB_WIDTH) * 4u;
        const size_t copyRowBytes = static_cast<size_t>(copyWidth) * 4u;
        for (uint32_t y = 0; y < copyHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * srcRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * dstRowBytes;
            if (srcOffset + copyRowBytes > s_scratch.size() ||
                dstOffset + copyRowBytes > s_uploadBuffer.size())
            {
                break;
            }
            std::memcpy(s_uploadBuffer.data() + dstOffset, s_scratch.data() + srcOffset, copyRowBytes);
        }
    }

    UpdateTexture(tex, s_uploadBuffer.data());
    outWidth = width;
    outHeight = presentedHeight;
    s_lastHeight = presentedHeight;
    s_hasUploadedFrame = true;

    DumpPresentedFrame(currentTick, s_uploadBuffer.data(), std::min<uint32_t>(width, FB_WIDTH),
                       std::min<uint32_t>(presentedHeight, FB_HEIGHT), FB_WIDTH * 4u, displayFbp);
}

PS2Runtime::PS2Runtime()
{
    m_iopHost = std::make_unique<PS2IopHostAdapter>(*this);
    m_iopSubsystem = std::make_unique<ps2x::iop::IopSubsystem>(*m_iopHost);

    m_eeScheduler = std::make_unique<EeScheduler>(*this);

    // Assign rather than memset: R5900Context's constructor zeroes itself and
    // then applies the COP0 reset values, which a memset here would discard.
    m_cpuContext = R5900Context{};

    // R0 is always zero in MIPS
    m_cpuContext.r[0] = _mm_set1_epi32(0);
    m_cpuContext.vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    m_cpuContext.vu0_q = 1.0f;
    m_cpuContext.vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));

    // Stack pointer (SP) and global pointer (GP) will be set by the loaded ELF

    m_loadedModules.clear();
    m_guestHeapBlocks.clear();
    m_guestHeapBase = kGuestHeapDefaultBase;
    m_guestHeapEnd = kGuestHeapDefaultBase;
    m_guestHeapLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_guestHeapSuggestedBase = kGuestHeapDefaultBase;
    m_guestHeapConfigured = false;
    m_asyncCallbackStackFloor = kAsyncCallbackStackFloor;
    m_asyncCallbackStackTop = kAsyncCallbackStackTop;
}

void PS2Runtime::setDebugUiCallbacks(DebugUiCallback initCallback,
                                     DebugUiCallback drawCallback,
                                     DebugUiCallback shutdownCallback,
                                     void *userData)
{
    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    m_debugUiInitCallback = initCallback;
    m_debugUiDrawCallback = drawCallback;
    m_debugUiShutdownCallback = shutdownCallback;
    m_debugUiUserData = userData;
}

PS2Runtime::~PS2Runtime()
{
    try
    {
        requestStop();
        m_memory.shutdownVif1();
        g_spu2AudioSource.store(nullptr);
#if !defined(PLATFORM_VITA)
        if (IsAudioDeviceReady() && IsAudioStreamValid(g_spu2Stream))
            StopAudioStream(g_spu2Stream);
#endif
        m_iopSubsystem.reset();
        m_iopHost.reset();
#if defined(PLATFORM_VITA)
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
#else
        if (IsAudioDeviceReady())
        {
            g_spu2AudioSource.store(nullptr);
            if (IsAudioStreamValid(g_spu2Stream))
                UnloadAudioStream(g_spu2Stream);
            CloseAudioDevice();
            m_audioBackend.setAudioReady(false);
        }
#endif
        if (m_debugUiInitialized && m_debugUiShutdownCallback)
        {
            m_debugUiShutdownCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = false;
        }

        if (IsWindowReady())
        {
            CloseWindow();
        }

        m_loadedModules.clear();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: unknown" << std::endl;
    }
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModule(std::string_view path, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModule(path, arguments, argumentSize);
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModuleBuffer(guestAddress, arguments, argumentSize);
}

bool PS2Runtime::stopIopModule(int32_t moduleId, int32_t *result)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->stopModule(moduleId, result);
}

ps2x::iop::RpcAbi PS2Runtime::selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const
{
    return m_iopSubsystem->selectRpcAbi(request);
}

bool PS2Runtime::canBindIopRpc(uint32_t sid) const noexcept
{
    return m_iopSubsystem->canBindRpc(sid);
}

ps2x::iop::RpcResult PS2Runtime::handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request)
{
    auto scope = m_iopHost->enterCall(ctx, rdram);
    request.callToken = scope.token();
    return m_iopSubsystem->handleRpc(request);
}

void PS2Runtime::notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
{
    auto scope = m_iopHost->enterCall(nullptr, rdram);
    m_iopSubsystem->onSifTransfer(transfer);
}

void PS2Runtime::advanceIopEeCycles(uint64_t eeCycles) noexcept
{
    m_iopSubsystem->runEeCycles(eeCycles);
}

void PS2Runtime::resetIop()
{
    m_iopSubsystem->reset();
}

ps2x::iop::DebugSnapshot PS2Runtime::iopDebugSnapshot() const
{
    return m_iopSubsystem->debugSnapshot();
}

uint32_t PS2Runtime::allocateIopMemory(uint32_t size, uint32_t alignment)
{
    return m_iopSubsystem ? m_iopSubsystem->allocateMemory(size, alignment) : 0u;
}

bool PS2Runtime::freeIopMemory(uint32_t address)
{
    return m_iopSubsystem && m_iopSubsystem->freeMemory(address);
}

bool PS2Runtime::readIopMemory(uint32_t address, void *destination, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->readMemory(address, destination, size);
}

bool PS2Runtime::writeIopMemory(uint32_t address, const void *source, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->writeMemory(address, source, size);
}

bool PS2Runtime::zeroIopMemory(uint32_t address, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->zeroMemory(address, size);
}

bool PS2Runtime::isIopMemoryRange(uint32_t address, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->isMemoryRange(address, size);
}

bool PS2Runtime::syncCoreSubsystems()
{
    // The GS helpers that call this touch GS state: queued VIF1 transfers come first.
    m_memory.syncVif1();
    uint8_t *const rdram = m_memory.getRDRAM();
    uint8_t *const gsVram = m_memory.getGSVRAM();
    if (!rdram || !gsVram)
    {
        return false;
    }

    if (m_boundRdram == rdram && m_boundGSVram == gsVram)
    {
        return true;
    }

    m_gs.init(gsVram, static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &m_memory.gs());
    // PS2X_RENDERER=opengl (Android's default): the renderer's GL thread starts once the window
    // exists (initialize), which goes back to a CPU backend if it cannot.
    if (ps2x::gles::requested())
        m_gs.setRasterBackend(ps2x::gles::createBackend());
    m_gifArbiter.setProcessPacketFn([this](const uint8_t *data, uint32_t size)
                                    { m_gs.processGIFPacket(data, size); });
    m_memory.setGifArbiter(&m_gifArbiter);
    // VU1 runs read the D/T enables from FBRST and report D/T stops in VPU-STAT of the EE
    // context; with PS2X_VIF1_THREAD they run on the VIF1 thread, so PS2Memory hands them a
    // copy of FBRST taken when the transfer was queued and applies the stops on the EE thread.
    m_memory.setVu1ContextHooks(
        [this]() -> uint32_t
        {
            R5900Context *cpuContext = m_eeScheduler ? m_eeScheduler->currentContext() : nullptr;
            return (cpuContext ? cpuContext : &m_cpuContext)->vu0_fbrst;
        },
        [this](uint32_t stopBits)
        {
            R5900Context *cpuContext = m_eeScheduler ? m_eeScheduler->currentContext() : nullptr;
            if (!cpuContext)
                cpuContext = &m_cpuContext;
            cpuContext->vu0_vpu_stat = (cpuContext->vu0_vpu_stat & ~0x0600u) | stopBits;
        });
    m_memory.setVif1IdleHook([this]()
                             {
                                 if (m_eeScheduler)
                                     m_eeScheduler->postEvent(EeEvent{EeEventType::Dmac, 1u, 0u}); });
    // While guest code waits for the VIF1 thread, pull the EE clock to host time as a
    // checkpoint would, so the IOP runs during the wait instead of in one burst after it.
    m_memory.setGsKickHook([this]() { m_gs.kickBackend(); });
    m_memory.setVif1WaitHook([this]()
                             {
                                 if (m_eeScheduler && m_eeScheduler->isExecutingGuest())
                                     m_eeScheduler->syncCyclesToHostTime(); });
    m_memory.setVu1MscalCallback([this](uint32_t startPC, uint32_t top, uint32_t itop)
                                 {
                                     const uint32_t fbrst = m_memory.vu1RunFbrst();
                                     m_vu1.state().dBitEnabled = (fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled = (fbrst & (1u << 11)) != 0u;
                                     m_vu1.execute(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                   m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                   m_gs, &m_memory, startPC, top, itop, 65536);
                                     m_memory.noteVu1RunStop((m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                                             (m_vu1.state().stoppedByT ? 0x0400u : 0u)); });
    m_memory.setVu1RunQueue(m_vu1.runQueue());
    m_memory.setVu1MscntCallback([this](uint32_t top, uint32_t itop)
                                 {
                                     const uint32_t fbrst = m_memory.vu1RunFbrst();
                                     m_vu1.state().dBitEnabled = (fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled = (fbrst & (1u << 11)) != 0u;
                                     m_vu1.resume(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                  m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                  m_gs, &m_memory, top, itop, 65536);
                                     m_memory.noteVu1RunStop((m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                                             (m_vu1.state().stoppedByT ? 0x0400u : 0u)); });
    resetIop();
    m_vu0.reset();
    m_vu1.reset();

    m_boundRdram = rdram;
    m_boundGSVram = gsVram;
    return true;
}

bool PS2Runtime::initialize(const char *title)
{
    try
    {
        if (!m_memory.initialize())
        {
            std::cerr << "Failed to initialize PS2 memory" << std::endl;
            return false;
        }
        // VIF1 transfers (and the VU1 programs they start) run alongside the EE, as on the
        // console; PS2X_VIF1_THREAD=0 processes them on the EE thread instead.
        m_memory.enableVif1ThreadByDefault();

        if (!syncCoreSubsystems())
        {
            std::cerr << "Failed to bind runtime core subsystems" << std::endl;
            return false;
        }
#if defined(PLATFORM_VITA)
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title); // raylib vita does not support audio
#else
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title);
        InitAudioDevice();
        m_audioBackend.setAudioReady(IsAudioDeviceReady());
        if (IsAudioDeviceReady() && m_iopSubsystem)
        {
            g_spu2AudioSource.store(m_iopSubsystem.get());
            SetAudioStreamBufferSizeDefault(1024);
            g_spu2Stream = LoadAudioStream(48000, 16, 2);
            SetAudioStreamCallback(g_spu2Stream, spu2AudioCallback);
            PlayAudioStream(g_spu2Stream);
        }
#endif
        SetTargetFPS(60);
        if (ps2x::gles::requested() && !ps2x::gles::attachToWindow())
        {
            std::cerr << "[video] the OpenGL renderer is unavailable; using the software renderer" << std::endl;
            m_gs.setRasterBackend(std::make_unique<GSCpuBackend>());
        }
        if (m_debugUiInitCallback)
        {
            m_debugUiInitCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = true;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize PS2 runtime: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to initialize PS2 runtime: unknown exception" << std::endl;
    }

    return false;
}

bool PS2Runtime::loadELF(const std::string &elfPath)
{
    configureIoPathsFromElf(elfPath);

    std::ifstream file(elfPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open ELF file: " << elfPath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(ElfHeader)))
    {
        std::cerr << "ELF file is too small: " << elfPath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    ElfHeader header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Failed to read ELF header from: " << elfPath << std::endl;
        return false;
    }

    if (header.magic != ELF_MAGIC)
    {
        std::cerr << "Invalid ELF magic number" << std::endl;
        return false;
    }

    if (header.elf_class != 1u || header.endianness != 1u)
    {
        std::cerr << "Unsupported ELF format (expected 32-bit little-endian)." << std::endl;
        return false;
    }

    if (header.machine != EM_MIPS || header.type != ET_EXEC)
    {
        std::cerr << "Not a MIPS executable ELF file" << std::endl;
        return false;
    }

    if (header.phnum != 0u && header.phentsize < sizeof(ProgramHeader))
    {
        std::cerr << "Unsupported ELF program-header entry size: " << header.phentsize << std::endl;
        return false;
    }

    const uint64_t programHeaderTableEnd =
        static_cast<uint64_t>(header.phoff) +
        static_cast<uint64_t>(header.phnum) * static_cast<uint64_t>(header.phentsize);
    if (programHeaderTableEnd > static_cast<uint64_t>(fileSize))
    {
        std::cerr << "ELF program-header table is out of range." << std::endl;
        return false;
    }

    m_cpuContext.pc = header.entry;
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);

    uint32_t maxLoadedRdramEnd = kGuestHeapDefaultBase;
    uint32_t moduleBase = std::numeric_limits<uint32_t>::max();
    uint32_t moduleEnd = 0u;
    bool loadedAnySegment = false;

    for (uint16_t i = 0; i < header.phnum; i++)
    {
        const uint64_t phOffset =
            static_cast<uint64_t>(header.phoff) +
            static_cast<uint64_t>(i) * static_cast<uint64_t>(header.phentsize);
        if (phOffset + sizeof(ProgramHeader) > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF program header " << i << " is out of range." << std::endl;
            return false;
        }

        ProgramHeader ph{};
        file.seekg(static_cast<std::streamoff>(phOffset), std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(&ph), sizeof(ph)))
        {
            std::cerr << "Failed to read ELF program header " << i << std::endl;
            return false;
        }

        if (ph.type != PT_LOAD || ph.memsz == 0u)
        {
            continue;
        }

        if (ph.filesz > ph.memsz)
        {
            std::cerr << "ELF segment " << i << " has filesz > memsz." << std::endl;
            return false;
        }

        const uint64_t segmentFileEnd = static_cast<uint64_t>(ph.offset) + static_cast<uint64_t>(ph.filesz);
        if (segmentFileEnd > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF segment " << i << " exceeds file bounds." << std::endl;
            return false;
        }

        const bool scratch =
            ph.vaddr >= PS2_SCRATCHPAD_BASE &&
            ph.vaddr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);

        uint32_t physAddr = 0u;
        try
        {
            physAddr = m_memory.translateAddress(ph.vaddr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to translate ELF segment " << i
                      << " virtual address 0x" << std::hex << ph.vaddr
                      << std::dec << ": " << e.what() << std::endl;
            return false;
        }
        const uint64_t regionSize = scratch ? static_cast<uint64_t>(PS2_SCRATCHPAD_SIZE)
                                            : static_cast<uint64_t>(PS2_RAM_SIZE);
        const uint64_t segmentMemEnd = static_cast<uint64_t>(physAddr) + static_cast<uint64_t>(ph.memsz);
        if (segmentMemEnd > regionSize)
        {
            std::cerr << "ELF segment " << i << " exceeds "
                      << (scratch ? "scratchpad" : "RDRAM")
                      << " bounds (vaddr=0x" << std::hex << ph.vaddr
                      << " memsz=0x" << ph.memsz << std::dec << ")." << std::endl;
            return false;
        }

        uint8_t *destBase = scratch ? m_memory.getScratchpad() : m_memory.getRDRAM();
        if (!destBase)
        {
            std::cerr << "ELF segment " << i << " has no destination memory backing." << std::endl;
            return false;
        }

        uint8_t *dest = destBase + physAddr;
        if (ph.filesz > 0u)
        {
            file.seekg(static_cast<std::streamoff>(ph.offset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(dest), ph.filesz))
            {
                std::cerr << "Failed to read ELF segment " << i << " payload." << std::endl;
                return false;
            }
        }

        if (ph.memsz > ph.filesz)
        {
            std::memset(dest + ph.filesz, 0, ph.memsz - ph.filesz);
        }

        RUNTIME_LOG("Loading segment: 0x" << std::hex << ph.vaddr
                                          << " - 0x" << (static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz))
                                          << " (filesz: 0x" << ph.filesz
                                          << ", memsz: 0x" << ph.memsz << ")"
                                          << std::dec << std::endl);

        if (!scratch)
        {
            maxLoadedRdramEnd = std::max(maxLoadedRdramEnd, static_cast<uint32_t>(segmentMemEnd));
        }

        if (ph.flags & 0x1u) // PF_X
        {
            const uint64_t execEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.filesz);
            if (execEnd <= std::numeric_limits<uint32_t>::max())
            {
                m_memory.registerCodeRegion(ph.vaddr, static_cast<uint32_t>(execEnd));
            }
        }

        loadedAnySegment = true;
        moduleBase = std::min(moduleBase, ph.vaddr);
        const uint64_t segmentVirtualEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
        const uint32_t clampedVirtualEnd =
            (segmentVirtualEnd > std::numeric_limits<uint32_t>::max())
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(segmentVirtualEnd);
        moduleEnd = std::max(moduleEnd, clampedVirtualEnd);
    }

    if (!loadedAnySegment)
    {
        std::cerr << "ELF contains no loadable PT_LOAD segments." << std::endl;
        return false;
    }

    if (maxLoadedRdramEnd > PS2_RAM_SIZE)
    {
        maxLoadedRdramEnd = PS2_RAM_SIZE;
    }

    const uint32_t paddedEnd = (maxLoadedRdramEnd > (PS2_RAM_SIZE - kGuestHeapSafetyPad))
                                   ? PS2_RAM_SIZE
                                   : (maxLoadedRdramEnd + kGuestHeapSafetyPad);
    const uint32_t suggestedHeapBase = alignGuestHeapValue(paddedEnd, kGuestHeapDefaultAlignment);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        if (!m_guestHeapConfigured)
        {
            const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
            m_guestHeapSuggestedBase = std::min(suggestedHeapBase, hardLimit);
            m_guestHeapBase = m_guestHeapSuggestedBase;
            m_guestHeapEnd = m_guestHeapSuggestedBase;
            m_guestHeapLimit = hardLimit;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        m_asyncCallbackStackFloor = kAsyncCallbackStackFloor;
        m_asyncCallbackStackTop = kAsyncCallbackStackTop;
    }

    LoadedModule module;
    module.name = elfPath.substr(elfPath.find_last_of("/\\") + 1);
    module.baseAddress = (moduleBase == std::numeric_limits<uint32_t>::max()) ? 0x00100000u : moduleBase;
    module.size = (moduleEnd > module.baseAddress) ? static_cast<size_t>(moduleEnd - module.baseAddress) : 0u;
    module.active = true;

    m_loadedModules.push_back(module);

    uint32_t elfCrc32 = 0u;
    const bool elfCrc32Valid = computeFileCrc32(elfPath, elfCrc32);
    if (!elfCrc32Valid)
    {
        std::cerr << "[ps2xIOP] failed to compute ELF CRC32 for '" << elfPath << "'" << std::endl;
    }
    ps2x::iop::GameIdentity identity;
    identity.elfName = module.name;
    identity.entryPoint = m_cpuContext.pc;
    identity.crc32 = elfCrc32;
    std::string romError;
    if (!m_romDevice.configure(identity, &romError))
    {
        std::cerr << "[ROM0] failed to configure profile: " << romError << std::endl;
        return false;
    }

    m_iopSubsystem->reset();

    ps2_game_overrides::applyMatching(*this,
                                      elfPath,
                                      m_cpuContext.pc,
                                      elfCrc32,
                                      elfCrc32Valid);

    RUNTIME_LOG("ELF file loaded successfully. Entry point: 0x" << std::hex << m_cpuContext.pc << std::dec);
    return true;
}

const PS2Runtime::IoPaths &PS2Runtime::getIoPaths()
{
    return runtimeIoPaths();
}

void PS2Runtime::setIoPaths(const IoPaths &paths)
{
    IoPaths normalized = paths;
    normalized.elfPath = normalizeAbsolutePath(normalized.elfPath);
    normalized.elfDirectory = normalizeAbsolutePath(normalized.elfDirectory);
    normalized.hostRoot = normalizeAbsolutePath(normalized.hostRoot);
    normalized.cdRoot = normalizeAbsolutePath(normalized.cdRoot);
    normalized.mcRoot = normalizeAbsolutePath(normalized.mcRoot);
    normalized.cdImage = normalizeAbsolutePath(normalized.cdImage);

    if (normalized.elfDirectory.empty() && !normalized.elfPath.empty())
    {
        normalized.elfDirectory = normalized.elfPath.parent_path();
    }

    if (normalized.hostRoot.empty())
    {
        normalized.hostRoot = normalized.elfDirectory;
    }
    if (normalized.cdRoot.empty())
    {
        normalized.cdRoot = normalized.elfDirectory;
    }
    if (normalized.mcRoot.empty())
    {
        normalized.mcRoot = normalized.elfDirectory / "mc0";
    }

    runtimeIoPaths() = normalized;
}

void PS2Runtime::configureIoPathsFromElf(const std::string &elfPath)
{
    IoPaths paths = runtimeIoPaths();
    paths.elfPath = normalizeAbsolutePath(std::filesystem::path(elfPath));
    if (!paths.elfPath.empty())
    {
        paths.elfDirectory = paths.elfPath.parent_path();
    }

    if (!paths.elfDirectory.empty())
    {
        paths.hostRoot = paths.elfDirectory;
        paths.cdRoot = paths.elfDirectory;
        paths.mcRoot = paths.elfDirectory / "mc0";
    }

    setIoPaths(paths);
}

namespace
{
    bool generatedFunctionTableSlot(uint32_t address, uint32_t &slot)
    {
        if ((address & 3u) != 0u || g_ps2RecompiledFunctionTableSlotCount == 0u)
        {
            return false;
        }

        if (address < g_ps2RecompiledFunctionTableBase || address >= g_ps2RecompiledFunctionTableEnd)
        {
            return false;
        }

        const uint32_t offset = address - g_ps2RecompiledFunctionTableBase;
        slot = offset >> 2;
        return slot < g_ps2RecompiledFunctionTableSlotCount;
    }
}

bool PS2Runtime::replaceFunction(uint32_t address, RecompiledFunction func)
{
    uint32_t slot = 0u;
    if (generatedFunctionTableSlot(address, slot))
    {
        g_ps2RecompiledFunctionTable[slot] = func;
        return true;
    }

    if ((address & 3u) != 0u)
    {
        std::cerr << "[function-table] cannot register unaligned guest PC 0x" << std::hex << address
                  << std::dec << std::endl;
        return false;
    }

    // Outside the generated dense table: code loaded at runtime.
    forgetCachedDynamicFunction(address);
    if (func != nullptr)
    {
        m_dynamicFunctions[address] = func;
    }
    else
    {
        m_dynamicFunctions.erase(address);
    }
    return true;
}

void PS2Runtime::setMissingFunctionResolver(MissingFunctionResolver resolver)
{
    m_missingFunctionResolver = std::move(resolver);
}

void PS2Runtime::registerGuestModule(uint32_t base, uint32_t size)
{
    unregisterGuestModule(base);
    m_guestModules.push_back(GuestModuleRange{base, base + size});
}

void PS2Runtime::unregisterGuestModule(uint32_t base)
{
    m_guestModules.erase(std::remove_if(m_guestModules.begin(), m_guestModules.end(),
                                        [base](const GuestModuleRange &module)
                                        { return module.base == base; }),
                         m_guestModules.end());
}

uint32_t PS2Runtime::guestModuleBase(uint32_t address) const
{
    // Loaders may hand a module's tail (relocation tables and the like) back to
    // the heap once it is linked, so a later module can sit inside an earlier
    // module's registered range. The innermost module, the one with the
    // highest base, is the one that owns the address.
    uint32_t best = 0u;
    bool found = false;
    for (const GuestModuleRange &module : m_guestModules)
    {
        if (address >= module.base && address < module.end && (!found || module.base > best))
        {
            best = module.base;
            found = true;
        }
    }
    if (!found)
    {
        std::cerr << "[guest-module] no loaded module contains PC 0x" << std::hex << address << std::dec << std::endl;
    }
    return best;
}

void PS2Runtime::unregisterFunctionRange(uint32_t start, uint32_t end)
{
    m_dynamicFunctionCache.fill({});
    for (auto it = m_dynamicFunctions.begin(); it != m_dynamicFunctions.end();)
    {
        if (it->first >= start && it->first < end)
        {
            it = m_dynamicFunctions.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

PS2Runtime::RecompiledFunction PS2Runtime::findFunction(uint32_t address)
{
    uint32_t slot = 0u;
    if (generatedFunctionTableSlot(address, slot))
    {
        if (RecompiledFunction fn = g_ps2RecompiledFunctionTable[slot])
        {
            return fn;
        }
    }

    if (!m_dynamicFunctions.empty())
    {
        DynamicFunctionCacheEntry &cached = dynamicFunctionCacheSlot(address);
        if (cached.address == address && cached.fn != nullptr)
        {
            return cached.fn;
        }
        const auto it = m_dynamicFunctions.find(address);
        if (it != m_dynamicFunctions.end())
        {
            cached = {address, it->second};
            return it->second;
        }
    }

    if (m_missingFunctionResolver && !m_resolvingMissingFunction)
    {
        m_resolvingMissingFunction = true;
        RecompiledFunction fn = m_missingFunctionResolver(*this, address);
        m_resolvingMissingFunction = false;
        if (fn != nullptr)
        {
            forgetCachedDynamicFunction(address);
            m_dynamicFunctions[address] = fn;
            return fn;
        }
        // The resolver may have registered a whole module instead.
        const auto it = m_dynamicFunctions.find(address);
        if (it != m_dynamicFunctions.end())
        {
            return it->second;
        }
    }
    return nullptr;
}

bool PS2Runtime::registerFunction(uint32_t address, RecompiledFunction func)
{
    return replaceFunction(address, func);
}

bool PS2Runtime::hasFunction(uint32_t address) const
{
    // findFunction() may consult the resolver, which registers loaded code.
    return const_cast<PS2Runtime *>(this)->findFunction(address) != nullptr;
}

const char *describeGuestBranchKind(PS2Runtime::GuestBranchKind kind)
{
    switch (kind)
    {
    case PS2Runtime::GuestBranchKind::DirectJump:
        return "DirectJump";
    case PS2Runtime::GuestBranchKind::DirectCall:
        return "DirectCall";
    case PS2Runtime::GuestBranchKind::IndirectJump:
        return "IndirectJump";
    case PS2Runtime::GuestBranchKind::IndirectCall:
        return "IndirectCall";
    case PS2Runtime::GuestBranchKind::Return:
        return "Return";
    default:
        return "Unknown";
    }
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(uint32_t address)
{
    pushDispatchPc(address);

    if (RecompiledFunction fn = findFunction(address))
    {
        return fn;
    }

    std::cerr << "Error: No exact recompiled function for guest PC 0x" << std::hex << address
              << " tableBase=0x" << g_ps2RecompiledFunctionTableBase
              << " tableEnd=0x" << g_ps2RecompiledFunctionTableEnd
              << " codeRegion=" << (m_memory.isCodeAddress(address) ? "yes" : "no")
              << " trace=" << formatDispatchHistory()
              << std::dec << std::endl;

    static RecompiledFunction missingFunction = [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t badPc = ctx->pc;
        runtime->reportMissingFunction(rdram,
                                       ctx,
                                       badPc,
                                       0u,
                                       PS2Runtime::GuestBranchKind::IndirectJump,
                                       "dispatch");
    };

    return missingFunction;
}

void PS2Runtime::setMissingFunctionPolicy(MissingFunctionPolicy policy)
{
    m_missingFunctionPolicy.store(static_cast<uint32_t>(policy), std::memory_order_release);
}

PS2Runtime::MissingFunctionPolicy PS2Runtime::missingFunctionPolicy() const
{
    return static_cast<MissingFunctionPolicy>(m_missingFunctionPolicy.load(std::memory_order_acquire));
}

void PS2Runtime::resetMissingFunctionReportOnce()
{
    m_missingFunctionReported.store(false, std::memory_order_release);
}

void PS2Runtime::reportMissingFunction(uint8_t *rdram,
                                       R5900Context *ctx,
                                       uint32_t targetPc,
                                       uint32_t sourcePc,
                                       GuestBranchKind kind,
                                       const char *debugName)
{
    const MissingFunctionPolicy policy = missingFunctionPolicy();
    const bool firstReport = !m_missingFunctionReported.exchange(true, std::memory_order_acq_rel);

    const uint32_t pc = ctx->pc;
    const uint32_t ra = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));
    const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
    const uint32_t gp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0));
    const uint32_t a0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[4], 0));
    const uint32_t a1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[5], 0));
    const uint32_t a2 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[6], 0));
    const uint32_t a3 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[7], 0));
    const uint32_t s0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[16], 0));
    const uint32_t s1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[17], 0));
    const uint32_t v0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[2], 0));
    const uint32_t v1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[3], 0));

    auto readGuestU32At = [rdram](uint32_t addr, uint32_t &out) -> bool
    {
        // TODO this !rdram exist only because of test fix those test later
        if (!rdram || addr > PS2_RAM_SIZE - sizeof(uint32_t))
        {
            out = 0u;
            return false;
        }

        std::memcpy(&out, rdram + addr, sizeof(uint32_t));
        return true;
    };

    auto readGuestU32Offset = [&readGuestU32At](uint32_t base, uint32_t offset, uint32_t &out) -> bool
    {
        if (base > PS2_RAM_SIZE - sizeof(uint32_t) || offset > PS2_RAM_SIZE - sizeof(uint32_t) - base)
        {
            out = 0u;
            return false;
        }

        return readGuestU32At(base + offset, out);
    };

    uint32_t a0Word0 = 0u;
    uint32_t a0Word4 = 0u;
    uint32_t a0Word8 = 0u;
    uint32_t a0WordC = 0u;
    const bool a0Readable =
        readGuestU32Offset(a0, 0x00u, a0Word0) &&
        readGuestU32Offset(a0, 0x04u, a0Word4) &&
        readGuestU32Offset(a0, 0x08u, a0Word8) &&
        readGuestU32Offset(a0, 0x0cu, a0WordC);

    uint32_t s0Word0 = 0u;
    uint32_t s0Word4 = 0u;
    uint32_t s0Word8 = 0u;
    uint32_t s0WordC = 0u;
    const bool s0Readable =
        readGuestU32Offset(s0, 0x00u, s0Word0) &&
        readGuestU32Offset(s0, 0x04u, s0Word4) &&
        readGuestU32Offset(s0, 0x08u, s0Word8) &&
        readGuestU32Offset(s0, 0x0cu, s0WordC);

    uint32_t recordWord0 = 0u;
    uint32_t recordWord4 = 0u;
    uint32_t recordWord8 = 0u;
    uint32_t recordWordC = 0u;
    const bool recordReadable =
        s0Readable && s0Word4 != 0u &&
        readGuestU32Offset(s0Word4, 0x00u, recordWord0) &&
        readGuestU32Offset(s0Word4, 0x04u, recordWord4) &&
        readGuestU32Offset(s0Word4, 0x08u, recordWord8) &&
        readGuestU32Offset(s0Word4, 0x0cu, recordWordC);

    uint32_t vtableSlot0 = 0u;
    uint32_t vtableSlot4 = 0u;
    uint32_t vtableSlot8 = 0u;
    uint32_t vtableSlotC = 0u;
    const bool vtableReadable =
        a0Readable && a0Word0 != 0u &&
        readGuestU32Offset(a0Word0, 0x00u, vtableSlot0) &&
        readGuestU32Offset(a0Word0, 0x04u, vtableSlot4) &&
        readGuestU32Offset(a0Word0, 0x08u, vtableSlot8) &&
        readGuestU32Offset(a0Word0, 0x0cu, vtableSlotC);

    if (firstReport)
    {
        std::ostringstream oss;
        oss << "[guest-branch:missing-target] kind=" << describeGuestBranchKind(kind)
            << " op=" << (debugName ? debugName : "<unknown>")
            << " source=0x" << std::hex << sourcePc
            << " target=0x" << targetPc
            << " pc=0x" << pc
            << " ra=0x" << ra
            << " sp=0x" << sp
            << " gp=0x" << gp
            << " a0=0x" << a0
            << " a1=0x" << a1
            << " a2=0x" << a2
            << " a3=0x" << a3
            << " s0=0x" << s0
            << " s1=0x" << s1
            << " v0=0x" << v0
            << " v1=0x" << v1
            << " a0Readable=" << (a0Readable ? "yes" : "no")
            << " a0[0]=0x" << a0Word0
            << " a0[4]=0x" << a0Word4
            << " a0[8]=0x" << a0Word8
            << " a0[c]=0x" << a0WordC
            << " s0Readable=" << (s0Readable ? "yes" : "no")
            << " s0[0]=0x" << s0Word0
            << " s0[4]=0x" << s0Word4
            << " s0[8]=0x" << s0Word8
            << " s0[c]=0x" << s0WordC
            << " recordReadable=" << (recordReadable ? "yes" : "no")
            << " record[0]=0x" << recordWord0
            << " record[4]=0x" << recordWord4
            << " record[8]=0x" << recordWord8
            << " record[c]=0x" << recordWordC
            << " vtableReadable=" << (vtableReadable ? "yes" : "no")
            << " vtbl[0]=0x" << vtableSlot0
            << " vtbl[4]=0x" << vtableSlot4
            << " vtbl[8]=0x" << vtableSlot8
            << " vtbl[c]=0x" << vtableSlotC
            << " codeRegion=" << (m_memory.isCodeAddress(targetPc) ? "yes" : "no")
            << " policy=" << static_cast<uint32_t>(policy)
            << " trace=" << formatDispatchHistory()
            << std::dec;

        static std::mutex s_missingFunctionLogMutex;
        {
            std::lock_guard<std::mutex> lock(s_missingFunctionLogMutex);
            std::cerr << oss.str() << std::endl;
        }
    }

    if (firstReport && policy == MissingFunctionPolicy::BreakOnce)
    {
#if defined(_MSC_VER)
        __debugbreak();
#endif // TODO others breakpoints
    }

    if (ctx)
    {
        ctx->pc = targetPc;
    }

    if (policy == MissingFunctionPolicy::Stop)
    {
        requestStop();
    }
}

bool PS2Runtime::dispatchGuestBranch(uint8_t *rdram,
                                     R5900Context *ctx,
                                     uint32_t targetPc,
                                     uint32_t sourcePc,
                                     uint32_t fallthroughPc,
                                     GuestBranchKind kind,
                                     const char *debugName)
{
    ctx->pc = targetPc;
    const bool isCall = (kind == GuestBranchKind::DirectCall || kind == GuestBranchKind::IndirectCall);

    // Every inter-function transfer is also a deterministic EE safe point.
    // Backward edges inside generated functions use eeCheckpointDue(), while
    // this charge bounds straight-line call chains that have no local loop.
    if (m_eeScheduler && m_eeScheduler->checkpointDue(EeScheduler::kGuestDispatchCycles))
    {
        return false;
    }

    if (!isCall)
    {
        if (!hasFunction(targetPc))
        {
            reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);
        }

        ctx->pc = targetPc;
        return false;
    }

    if (!hasFunction(targetPc))
    {
        reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);

        const MissingFunctionPolicy policy = missingFunctionPolicy();

        if (policy == MissingFunctionPolicy::SkipCallDebug && isCall)
        {
            ctx->pc = fallthroughPc;
            return true;
        }

        if (policy == MissingFunctionPolicy::ContinueToTarget)
        {
            ctx->pc = targetPc;
            // if you need the app to keep open to open debug pannel change this to false
            return false;
        }

        return false;
    }

    RecompiledFunction targetFn = lookupFunction(targetPc);
    const uint32_t entryPc = ctx->pc;
    targetFn(rdram, ctx, this);

    if (isStopRequested() || ctx->pc == 0u)
    {
        return false;
    }

    if (ctx->pc == entryPc)
    {
        ctx->pc = fallthroughPc;
    }

    return ctx->pc == fallthroughPc;
}

void PS2Runtime::SignalException(R5900Context *ctx, PS2Exception exception)
{
    if (exception == EXCEPTION_INTEGER_OVERFLOW)
    {
        HandleIntegerOverflow(ctx);
        return;
    }

    raiseCop0Exception(ctx, static_cast<uint32_t>(exception),
                       exception == EXCEPTION_TLB_REFILL);
}

void PS2Runtime::executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    m_vu0.reset();
    copyVu0ContextToState(ctx, m_vu0.state());
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop, 4096);
    copyVu0StateToContext(m_vu0.state(), ctx);
}

void PS2Runtime::vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    // VCALLMS and VCALLMSR both route here.
    executeVU0Microprogram(rdram, ctx, address);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx)
{
    handleSyscall(rdram, ctx, 0);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId)
{
    if (ctx->in_delay_slot)
    {
        throw std::runtime_error("Attempted to execute a syscall inside a branch delay slot! "
                                 "This breaks the atomic basic block model and is structurally unsupported by the emulator.");
    }

    const uint32_t syscallId = (encodedSyscallId != 0u)
                                   ? encodedSyscallId
                                   : getRegU32(ctx, 3); // $v1 / $3 is the EE kernel syscall number

    if (ps2_syscalls::dispatchNumericSyscall(syscallId, rdram, ctx, this))
    {
        return;
    }

    // God help you
    ps2_syscalls::TODO(rdram, ctx, this, encodedSyscallId);
}

void PS2Runtime::handleBreak(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_BREAKPOINT);
}

void PS2Runtime::drainCompletedDmacHandlers(uint8_t *rdram)
{
    for (uint32_t cause : m_memory.consumeCompletedDmacCauses())
    {
        ps2_syscalls::dispatchDmacHandlersForCause(rdram, this, cause);
    }
}

void PS2Runtime::handleTrap(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_TRAP);
}

void PS2Runtime::handleTLBR(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t vpn = 0;
    uint32_t pfn = 0;
    uint32_t mask = 0;
    bool valid = false;

    const uint32_t index = ctx->cop0_index & 0x3Fu;
    if (!m_memory.tlbRead(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Preserve low ASID bits in EntryHi.
    ctx->cop0_entryhi = (ctx->cop0_entryhi & 0x00000FFFu) | (vpn & 0xFFFFF000u);
    ctx->cop0_entrylo0 = (ctx->cop0_entrylo0 & ~0x03FFFFC2u) |
                         ((pfn & 0x000FFFFFu) << 6) |
                         (valid ? 0x2u : 0u);
    ctx->cop0_pagemask = mask & 0x01FFE000u;
}

void PS2Runtime::handleTLBWI(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t index = ctx->cop0_index & 0x3Fu;
    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
    }
}

void PS2Runtime::handleTLBWR(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t entryCount = static_cast<uint32_t>(m_memory.tlbEntryCount());
    if (entryCount == 0)
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    const uint32_t wired = std::min(ctx->cop0_wired, entryCount - 1);
    uint32_t random = ctx->cop0_random % entryCount;
    if (random < wired)
    {
        random = wired;
    }

    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(random, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Keep COP0 bookkeeping in sync with the selected slot.
    ctx->cop0_index = (ctx->cop0_index & ~0x3Fu) | (random & 0x3Fu);
    ctx->cop0_random = (random <= wired) ? (entryCount - 1) : (random - 1);
}

void PS2Runtime::handleTLBP(uint8_t *rdram, R5900Context *ctx)
{
    const int32_t index = m_memory.tlbProbe(ctx->cop0_entryhi & 0xFFFFF000u);
    if (index >= 0)
    {
        ctx->cop0_index = (ctx->cop0_index & ~0x8000003Fu) |
                          (static_cast<uint32_t>(index) & 0x3Fu);
    }
    else
    {
        // MIPS sets probe failure bit (P) in Index[31].
        ctx->cop0_index |= 0x80000000u;
    }
}

void PS2Runtime::clearLLBit(R5900Context *ctx)
{
    // LL/SC reservation is tracked separately from COP0 Status.
    ctx->llbit = 0;
    ctx->lladdr = 0;
}

uint32_t PS2Runtime::alignGuestHeapValue(uint32_t value, uint32_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    const uint32_t mask = alignment - 1u;
    if (value > (std::numeric_limits<uint32_t>::max() - mask))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return (value + mask) & ~mask;
}

bool PS2Runtime::isGuestHeapAlignmentValid(uint32_t alignment)
{
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

uint32_t PS2Runtime::normalizeGuestHeapAlignment(uint32_t alignment)
{
    if (!isGuestHeapAlignmentValid(alignment))
    {
        return kGuestHeapDefaultAlignment;
    }
    return std::max(alignment, kGuestHeapDefaultAlignment);
}

uint32_t PS2Runtime::clampGuestHeapBase(uint32_t guestBase) const
{
    uint32_t normalized = guestBase;
    if (normalized >= PS2_RAM_SIZE)
    {
        normalized &= PS2_RAM_MASK;
    }
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    return std::min(normalized, hardLimit);
}

uint32_t PS2Runtime::clampGuestHeapLimit(uint32_t guestLimit) const
{
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    if (guestLimit == 0u || guestLimit > hardLimit)
    {
        return hardLimit;
    }
    return guestLimit;
}

void PS2Runtime::resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit)
{
    uint32_t base = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    uint32_t limit = clampGuestHeapLimit(guestLimit);
    if (base == 0u)
    {
        const uint32_t fallbackBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
        base = alignGuestHeapValue(clampGuestHeapBase(fallbackBase), kGuestHeapDefaultAlignment);
    }

    if (limit <= base)
    {
        base = alignGuestHeapValue(clampGuestHeapBase(m_guestHeapSuggestedBase), kGuestHeapDefaultAlignment);
        limit = clampGuestHeapLimit(0u);
    }

    if (limit <= base)
    {
        base = 0u;
        limit = 0u;
    }

    m_guestHeapBlocks.clear();
    if (limit > base)
    {
        m_guestHeapBlocks.push_back({base, limit - base, true});
    }

    m_guestHeapBase = base;
    m_guestHeapEnd = base;
    m_guestHeapLimit = limit;
    m_guestHeapConfigured = true;
}

void PS2Runtime::ensureGuestHeapInitializedLocked()
{
    if (m_guestHeapConfigured)
    {
        return;
    }

    const uint32_t suggested = (m_guestHeapSuggestedBase == 0u) ? kGuestHeapDefaultBase : m_guestHeapSuggestedBase;
    resetGuestHeapLocked(suggested, clampGuestHeapLimit(0u));
}

int32_t PS2Runtime::findGuestHeapBlockIndexLocked(uint32_t guestAddr) const
{
    const uint32_t normalizedAddr = guestAddr & PS2_RAM_MASK;
    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock &block = m_guestHeapBlocks[i];
        if (!block.free && block.addr == normalizedAddr)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

uint32_t PS2Runtime::allocateGuestBlockLocked(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    if (size > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock block = m_guestHeapBlocks[i];
        if (!block.free)
        {
            continue;
        }

        const uint64_t blockStart = block.addr;
        const uint64_t blockEnd = blockStart + static_cast<uint64_t>(block.size);
        const uint32_t alignedAddr = alignGuestHeapValue(block.addr, normalizedAlignment);
        if (alignedAddr < block.addr)
        {
            continue;
        }

        const uint64_t alignedStart = alignedAddr;
        if (alignedStart > blockEnd)
        {
            continue;
        }

        const uint64_t allocEnd = alignedStart + static_cast<uint64_t>(allocSize);
        if (allocEnd > blockEnd)
        {
            continue;
        }

        const uint32_t prefixSize = static_cast<uint32_t>(alignedStart - blockStart);
        const uint32_t suffixSize = static_cast<uint32_t>(blockEnd - allocEnd);

        std::vector<GuestHeapBlock> replacement;
        replacement.reserve(3);
        if (prefixSize > 0u)
        {
            replacement.push_back({block.addr, prefixSize, true});
        }
        replacement.push_back({alignedAddr, allocSize, false});
        if (suffixSize > 0u)
        {
            replacement.push_back({static_cast<uint32_t>(allocEnd), suffixSize, true});
        }

        m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
        m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i),
                                 replacement.begin(),
                                 replacement.end());

        m_guestHeapEnd = std::max(m_guestHeapEnd, static_cast<uint32_t>(allocEnd));
        return alignedAddr;
    }

    return 0u;
}

void PS2Runtime::coalesceGuestHeapLocked()
{
    if (m_guestHeapBlocks.empty())
    {
        return;
    }

    size_t i = 1;
    while (i < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &prev = m_guestHeapBlocks[i - 1];
        GuestHeapBlock &curr = m_guestHeapBlocks[i];
        const uint64_t prevEnd = static_cast<uint64_t>(prev.addr) + static_cast<uint64_t>(prev.size);
        if (prev.free && curr.free && prevEnd == curr.addr)
        {
            prev.size += curr.size;
            m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void PS2Runtime::freeGuestBlockLocked(uint32_t guestAddr)
{
    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return;
    }

    m_guestHeapBlocks[static_cast<size_t>(index)].free = true;
    coalesceGuestHeapLocked();
}

void PS2Runtime::configureGuestHeap(uint32_t guestBase, uint32_t guestLimit)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    uint32_t normalizedBase = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    if (normalizedBase == 0u)
    {
        normalizedBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
    }
    m_guestHeapSuggestedBase = normalizedBase;
    resetGuestHeapLocked(normalizedBase, guestLimit);
}

uint32_t PS2Runtime::guestMalloc(uint32_t size, uint32_t alignment)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    return allocateGuestBlockLocked(size, alignment);
}

uint32_t PS2Runtime::guestCalloc(uint32_t count, uint32_t size, uint32_t alignment)
{
    if (count == 0u || size == 0u)
    {
        return 0u;
    }
    if (count > (std::numeric_limits<uint32_t>::max() / size))
    {
        return 0u;
    }

    const uint32_t totalSize = count * size;
    const uint32_t guestAddr = guestMalloc(totalSize, alignment);
    if (guestAddr != 0u)
    {
        uint8_t *rdram = m_memory.getRDRAM();
        if (rdram)
        {
            uint32_t physAddr = guestAddr & PS2_RAM_MASK;
            if (physAddr + totalSize <= PS2_RAM_SIZE)
                std::memset(rdram + physAddr, 0, totalSize);
        }
    }

    return guestAddr;
}

uint32_t PS2Runtime::guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment)
{
    if (guestAddr == 0u)
    {
        return guestMalloc(newSize, alignment);
    }
    if (newSize == 0u)
    {
        guestFree(guestAddr);
        return 0u;
    }

    if (newSize > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t requestedSize = alignGuestHeapValue(newSize, kGuestHeapDefaultAlignment);

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();

    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return 0u;
    }

    const size_t blockIndex = static_cast<size_t>(index);
    const uint32_t oldAddr = m_guestHeapBlocks[blockIndex].addr;
    const uint32_t oldSize = m_guestHeapBlocks[blockIndex].size;

    if (requestedSize <= oldSize)
    {
        if (requestedSize < oldSize)
        {
            const uint32_t tailAddr = oldAddr + requestedSize;
            const uint32_t tailSize = oldSize - requestedSize;
            m_guestHeapBlocks[blockIndex].size = requestedSize;
            m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u),
                                     GuestHeapBlock{tailAddr, tailSize, true});
            coalesceGuestHeapLocked();
        }
        return oldAddr;
    }

    if (blockIndex + 1u < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &next = m_guestHeapBlocks[blockIndex + 1u];
        const uint64_t blockEnd = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].addr) +
                                  static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size);
        if (next.free && blockEnd == next.addr)
        {
            const uint64_t combined = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size) +
                                      static_cast<uint64_t>(next.size);
            if (combined >= requestedSize)
            {
                const uint32_t extraNeeded = requestedSize - m_guestHeapBlocks[blockIndex].size;
                m_guestHeapBlocks[blockIndex].size = requestedSize;
                if (next.size == extraNeeded)
                {
                    m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u));
                }
                else
                {
                    next.addr += extraNeeded;
                    next.size -= extraNeeded;
                }
                m_guestHeapEnd = std::max(m_guestHeapEnd, oldAddr + requestedSize);
                return oldAddr;
            }
        }
    }

    const uint32_t newAddr = allocateGuestBlockLocked(newSize, normalizedAlignment);
    if (newAddr == 0u)
    {
        return 0u;
    }

    uint8_t *rdram = m_memory.getRDRAM();
    if (rdram)
    {
        const uint32_t copyBytes = std::min(oldSize, newSize);
        uint32_t dstPhys = newAddr & PS2_RAM_MASK;
        uint32_t srcPhys = oldAddr & PS2_RAM_MASK;
        if (dstPhys + copyBytes <= PS2_RAM_SIZE && srcPhys + copyBytes <= PS2_RAM_SIZE)
            std::memmove(rdram + dstPhys, rdram + srcPhys, copyBytes);
    }

    freeGuestBlockLocked(oldAddr);
    return newAddr;
}

void PS2Runtime::guestFree(uint32_t guestAddr)
{
    if (guestAddr == 0u)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    freeGuestBlockLocked(guestAddr);
}

uint32_t PS2Runtime::guestHeapBase() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapBase : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapEnd() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapEnd : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapLimit() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapLimit : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::reserveAsyncCallbackStack(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
    uint32_t top = m_asyncCallbackStackTop;
    if (top > kAsyncCallbackStackTop)
    {
        top = kAsyncCallbackStackTop;
    }
    top &= ~(kGuestHeapDefaultAlignment - 1u);

    if (top <= allocSize)
    {
        return 0u;
    }

    uint32_t base = top - allocSize;
    base &= ~(normalizedAlignment - 1u);
    if (base < m_asyncCallbackStackFloor || base >= top)
    {
        return 0u;
    }

    m_asyncCallbackStackTop = base;
    return top - 0x10u;
}

uint8_t PS2Runtime::Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read8(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint16_t PS2Runtime::Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read16(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint32_t PS2Runtime::Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read32(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint64_t PS2Runtime::Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read64(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

__m128i PS2Runtime::Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read128(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return _mm_setzero_si128();
    }
}

void PS2Runtime::Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 1u, value, 0u, "WRITE8", ctx);
    try
    {
        m_memory.write8(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 2u, value, 0u, "WRITE16", ctx);
    try
    {
        m_memory.write16(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 4u, value, 0u, "WRITE32", ctx);
    try
    {
        m_memory.write32(vaddr, value);
        drainCompletedDmacHandlers(rdram);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 8u, value, 0u, "WRITE64", ctx);
    try
    {
        m_memory.write64(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
{
    alignas(16) uint64_t _parts[2];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(_parts), value);
    ps2TraceGuestWrite(rdram, vaddr, 16u, _parts[0], _parts[1], "WRITE128", ctx);
    try
    {
        m_memory.write128(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::kickGifDmaChainFromMMIO(uint8_t *rdram,
                                         R5900Context *ctx,
                                         uint32_t dPcrValue,
                                         uint32_t dStatValue,
                                         uint32_t tadr,
                                         uint32_t chcr)
{
    constexpr uint32_t D_PCR = 0x1000E020u;
    constexpr uint32_t D_STAT = 0x1000E010u;
    constexpr uint32_t GIF_TADR = 0x1000A030u;
    constexpr uint32_t GIF_CHCR = 0x1000A000u;
    m_memory.syncVif1();

    ps2TraceGuestWrite(rdram, D_PCR, 4u, dPcrValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_PCR, dPcrValue);
    ps2TraceGuestWrite(rdram, D_STAT, 4u, dStatValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_STAT, dStatValue);
    ps2TraceGuestWrite(rdram, GIF_TADR, 4u, tadr, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(GIF_TADR, tadr);
    ps2TraceGuestWrite(rdram, GIF_CHCR, 4u, chcr, 0u, "WRITE32", ctx);
    if (m_memory.tryProcessNativeGifImageUploadChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    if (m_memory.tryProcessNativeGifPackedChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    m_memory.writeIORegister(GIF_CHCR, chcr);
    m_memory.processPendingTransfers();
    drainCompletedDmacHandlers(rdram);
}

void PS2Runtime::requestStop()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
    if (m_eeScheduler)
    {
        m_eeScheduler->requestStop();
    }
}

bool PS2Runtime::isStopRequested() const
{
    return m_stopRequested.load(std::memory_order_relaxed);
}

EeScheduler &PS2Runtime::eeScheduler()
{
    return *m_eeScheduler;
}

const EeScheduler &PS2Runtime::eeScheduler() const
{
    return *m_eeScheduler;
}

void PS2Runtime::postEeEvent(EeEvent event)
{
    m_eeScheduler->postEvent(event);
}

bool PS2Runtime::eeCheckpointDue(uint32_t cycles) noexcept
{
    return m_eeScheduler->checkpointDue(cycles);
}

[[noreturn]] void PS2Runtime::eeWaitVSyncTicks(uint32_t ticks, uint32_t resumePc)
{
    const uint64_t currentTick = m_eeScheduler->currentVSyncTick();
    const uint64_t waitTicks = std::max<uint64_t>(1u, ticks);
    m_eeScheduler->waitVSync(currentTick + waitTicks - 1u,
                             0,
                             [resumePc](R5900Context &context)
                             {
                                 context.pc = resumePc;
                             });
}

void PS2Runtime::addEeExitHandler(int threadId, uint32_t function, uint32_t argument)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers[threadId].push_back({function, argument});
}

std::vector<PS2Runtime::EeExitHandlerRegistration> PS2Runtime::takeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    auto it = m_eeExitHandlers.find(threadId);
    if (it == m_eeExitHandlers.end())
    {
        return {};
    }
    auto handlers = std::move(it->second);
    m_eeExitHandlers.erase(it);
    return handlers;
}

void PS2Runtime::removeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers.erase(threadId);
}

bool PS2Runtime::findEeSyscallOverride(uint32_t syscallNumber, uint32_t &handler) const
{
    std::lock_guard lock(m_eeKernelStateMutex);
    const auto it = m_eeSyscallOverrides.find(syscallNumber);
    if (it == m_eeSyscallOverrides.end())
    {
        return false;
    }
    handler = it->second;
    return true;
}

void PS2Runtime::setEeSyscallOverride(uint8_t *rdram, uint32_t syscallNumber, uint32_t handler)
{
    constexpr uint32_t kTableBase = 0x80011F80u & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
    const int64_t address = static_cast<int64_t>(kTableBase) + offset;

    std::lock_guard lock(m_eeKernelStateMutex);
    if (handler == 0u)
    {
        m_eeSyscallOverrides.erase(syscallNumber);
    }
    else
    {
        m_eeSyscallOverrides[syscallNumber] = handler;
    }
    if (!rdram || address < 0 || address + 4 > kMirrorLimit)
    {
        return;
    }
    const uint32_t guestAddress = static_cast<uint32_t>(address);
    std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
    if (handler == 0u)
    {
        m_eeSyscallMirrorAddresses.erase(guestAddress);
    }
    else
    {
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::initializeEeKernelState(uint8_t *rdram)
{
    if (!rdram)
    {
        return;
    }
    constexpr uint32_t kTableGuestBase = 0x80011F80u;
    constexpr uint32_t kTableBase = kTableGuestBase & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    constexpr uint32_t kProbeBase = 0x000002F0u;

    std::lock_guard lock(m_eeKernelStateMutex);
    for (const uint32_t address : m_eeSyscallMirrorAddresses)
    {
        const uint32_t zero = 0u;
        std::memcpy(rdram + address, &zero, sizeof(zero));
    }
    m_eeSyscallMirrorAddresses.clear();
    const uint32_t high = kTableGuestBase >> 16;
    const uint32_t low = kTableGuestBase & 0xFFFFu;
    std::memcpy(rdram + kProbeBase, &high, sizeof(high));
    std::memcpy(rdram + kProbeBase + 8u, &low, sizeof(low));
    m_eeSyscallMirrorAddresses.insert(kProbeBase);
    m_eeSyscallMirrorAddresses.insert(kProbeBase + 8u);

    for (const auto &[syscallNumber, handler] : m_eeSyscallOverrides)
    {
        const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
        const int64_t address = static_cast<int64_t>(kTableBase) + offset;
        if (address < 0 || address + 4 > kMirrorLimit)
        {
            continue;
        }
        const uint32_t guestAddress = static_cast<uint32_t>(address);
        std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::HandleIntegerOverflow(R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_INTEGER_OVERFLOW);
}

namespace
{
    // Live player settings for the window and the picture (ps2x::settings, the F1 menu),
    // applied on the thread that owns the window.
    // Presentation filters, applied while drawing the frame to the window (GLSL 330; phones get
    // the GLSL ES spelling from filterShaderSource()).
    const char *kFilterShader = R"(#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform vec2 textureSize;  // texels of the frame texture
uniform vec2 pictureUv;    // texture coordinates of the picture's far corner
uniform int mode;          // 0 plain, 1 scanlines, 2 CRT
uniform int fxaa;
out vec4 finalColor;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// FXAA in the style of the console version: blend along the local edge direction.
vec3 antialiased(vec2 uv)
{
    vec2 px = 1.0 / textureSize;
    vec3 nw = texture(texture0, uv + vec2(-1.0, -1.0) * px).rgb;
    vec3 ne = texture(texture0, uv + vec2(1.0, -1.0) * px).rgb;
    vec3 sw = texture(texture0, uv + vec2(-1.0, 1.0) * px).rgb;
    vec3 se = texture(texture0, uv + vec2(1.0, 1.0) * px).rgb;
    vec3 m = texture(texture0, uv).rgb;
    float lnw = luma(nw), lne = luma(ne), lsw = luma(sw), lse = luma(se), lm = luma(m);
    float lmin = min(lm, min(min(lnw, lne), min(lsw, lse)));
    float lmax = max(lm, max(max(lnw, lne), max(lsw, lse)));
    vec2 dir = vec2(-((lnw + lne) - (lsw + lse)), (lnw + lsw) - (lne + lse));
    float reduce = max((lnw + lne + lsw + lse) * 0.03125, 1.0 / 128.0);
    float scale = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * scale, vec2(-8.0), vec2(8.0)) * px;
    vec3 a = 0.5 * (texture(texture0, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                    texture(texture0, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(texture0, uv - dir * 0.5).rgb + texture(texture0, uv + dir * 0.5).rgb);
    float lb = luma(b);
    return (lb < lmin || lb > lmax) ? a : b;
}

void main()
{
    vec2 uv = fragTexCoord;
    float shade = 1.0;
    if (mode == 2)
    {
        // Slightly curved glass and darker corners.
        vec2 n = uv / pictureUv * 2.0 - 1.0;
        n *= 1.0 + 0.035 * dot(n.yx, n.yx);
        if (abs(n.x) > 1.0 || abs(n.y) > 1.0)
        {
            finalColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        uv = (n * 0.5 + 0.5) * pictureUv;
        shade *= clamp(1.0 - 0.25 * dot(n * n, n * n), 0.0, 1.0);
    }
    vec3 color = fxaa == 1 ? antialiased(uv) : texture(texture0, uv).rgb;
    if (mode >= 1)
    {
        // A dark gap between the picture's lines.
        float line = fract(uv.y * textureSize.y);
        shade *= mix(0.62, 1.0, smoothstep(0.0, 0.35, line) * smoothstep(1.0, 0.65, line));
        color *= 1.08;
    }
    if (mode == 2)
    {
        // Phosphor triads across the screen.
        int column = int(mod(gl_FragCoord.x, 3.0));
        vec3 mask = column == 0 ? vec3(1.0, 0.82, 0.82) : column == 1 ? vec3(0.82, 1.0, 0.82) : vec3(0.82, 0.82, 1.0);
        color *= mask * 1.12;
    }
    finalColor = vec4(color * shade, 1.0) * colDiffuse * fragColor;
}
)";

    // The same filters for OpenGL ES 2 (phones): GLSL ES 1.00 spellings of the desktop source.
    std::string filterShaderSource()
    {
        std::string source = kFilterShader;
#if defined(__ANDROID__)
        const auto replace = [&source](const std::string &from, const std::string &to)
        {
            for (size_t at = source.find(from); at != std::string::npos; at = source.find(from, at + to.size()))
                source.replace(at, from.size(), to);
        };
        replace("#version 330", "#version 100\nprecision mediump float;");
        replace("in vec2 fragTexCoord;", "varying vec2 fragTexCoord;");
        replace("in vec4 fragColor;", "varying vec4 fragColor;");
        replace("out vec4 finalColor;", "");
        replace("finalColor", "gl_FragColor");
        replace("texture(", "texture2D(");
#endif
        return source;
    }

    struct PresentationSettings
    {
        Shader shader{};
        bool shaderTried = false;
        int shaderMode = 0;
        bool antialias = false;
        int uniformTextureSize = -1;
        int uniformPictureUv = -1;
        int uniformMode = -1;
        int uniformFxaa = -1;
        std::string window = "windowed";
        std::string aspect = "4:3";
        std::string filter;
        int volume = -1;
        int vsync = -1;
        bool showFps = false;
        int restoreWidth = 0;
        int restoreHeight = 0;
        uint64_t fpsFrames = 0;
        double fpsTime = 0.0;
        float fps = 0.0f;

        void apply(Texture2D &frameTex)
        {
            namespace settings = ps2x::settings;
            const std::string wantedWindow = settings::get("video", "window");
            if (wantedWindow != window)
            {
                if (window == "fullscreen")
                    ToggleFullscreen();
                else if (window == "borderless")
                    ToggleBorderlessWindowed();
                if (window != "windowed" && restoreWidth > 0)
                    SetWindowSize(restoreWidth, restoreHeight);
                if (wantedWindow != "windowed")
                {
                    restoreWidth = GetScreenWidth();
                    restoreHeight = GetScreenHeight();
                }
                if (wantedWindow == "fullscreen")
                {
                    const int monitor = GetCurrentMonitor();
                    SetWindowSize(GetMonitorWidth(monitor), GetMonitorHeight(monitor));
                    ToggleFullscreen();
                }
                else if (wantedWindow == "borderless")
                {
                    ToggleBorderlessWindowed();
                }
                window = wantedWindow;
            }
            const int wantedVsync = settings::getBool("video", "vsync") ? 1 : 0;
            if (wantedVsync != vsync)
            {
                if (wantedVsync)
                    SetWindowState(FLAG_VSYNC_HINT);
                else if (vsync >= 0)
                    ClearWindowState(FLAG_VSYNC_HINT);
                vsync = wantedVsync;
            }
            const std::string wantedFilter = settings::get("video", "filter");
            if (wantedFilter != filter)
            {
                SetTextureFilter(frameTex, wantedFilter == "sharp" ? TEXTURE_FILTER_POINT : TEXTURE_FILTER_BILINEAR);
                filter = wantedFilter;
                shaderMode = wantedFilter == "scanlines" ? 1 : wantedFilter == "crt" ? 2 : 0;
            }
            aspect = settings::get("video", "aspect");
            const int wantedVolume = settings::getNumber("audio", "volume");
            if (wantedVolume != volume && IsAudioDeviceReady())
            {
                SetMasterVolume(static_cast<float>(std::clamp(wantedVolume, 0, 100)) / 100.0f);
                volume = wantedVolume;
            }
            showFps = settings::getBool("performance", "show_fps");
            antialias = settings::getBool("video", "antialias");
        }

        // Draws the frame, through the filter shader when a filter or FXAA is on.
        void drawFrame(const Texture2D &frameTex, const Rectangle &src, const Rectangle &dst)
        {
            const bool filtered = shaderMode != 0 || antialias;
#if !defined(PLATFORM_VITA)
            if (filtered && !shaderTried)
            {
                shaderTried = true;
                shader = LoadShaderFromMemory(nullptr, filterShaderSource().c_str());
                if (!IsShaderValid(shader))
                    std::cerr << "[video] filter shader unavailable; showing the plain picture" << std::endl;
                uniformTextureSize = GetShaderLocation(shader, "textureSize");
                uniformPictureUv = GetShaderLocation(shader, "pictureUv");
                uniformMode = GetShaderLocation(shader, "mode");
                uniformFxaa = GetShaderLocation(shader, "fxaa");
            }
            if (filtered && IsShaderValid(shader))
            {
                const float size[2] = {static_cast<float>(frameTex.width), static_cast<float>(frameTex.height)};
                const float corner[2] = {src.width / size[0], src.height / size[1]};
                const int fxaa = antialias ? 1 : 0;
                SetShaderValue(shader, uniformTextureSize, size, SHADER_UNIFORM_VEC2);
                SetShaderValue(shader, uniformPictureUv, corner, SHADER_UNIFORM_VEC2);
                SetShaderValue(shader, uniformMode, &shaderMode, SHADER_UNIFORM_INT);
                SetShaderValue(shader, uniformFxaa, &fxaa, SHADER_UNIFORM_INT);
                BeginShaderMode(shader);
                DrawTexturePro(frameTex, src, dst, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
                EndShaderMode();
                return;
            }
#endif
            DrawTexturePro(frameTex, src, dst, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        }

        // Game frames per second (display swaps), not the 60 Hz of the window.
        void drawFps(const Rectangle &picture)
        {
            const double now = GetTime();
            const uint64_t frames = ps2x::settings::gameFrames();
            if (now - fpsTime >= 0.5)
            {
                if (fpsTime > 0.0)
                    fps = static_cast<float>((frames - fpsFrames) / (now - fpsTime));
                fpsFrames = frames;
                fpsTime = now;
            }
            if (!showFps)
                return;
            char text[32];
            std::snprintf(text, sizeof(text), "%.0f FPS", fps);
            const int x = static_cast<int>(picture.x) + 10;
            const int y = static_cast<int>(picture.y) + 8;
            DrawText(text, x + 2, y + 2, 20, BLACK);
            DrawText(text, x, y, 20, fps >= 58.0f ? GREEN : fps >= 45.0f ? YELLOW : RED);
        }
    };
}

void PS2Runtime::run()
{
    m_stopRequested.store(false, std::memory_order_relaxed);
    ps2StartSamplingProfilerFromEnv();
    ps2_stubs::resetSifState();
    resetIop();
    ps2_stubs::resetAudioStubState();
    ps2_stubs::resetMpegStubState();
    initializeEeKernelState(m_memory.getRDRAM());
    s_watchRdram = m_memory.getRDRAM();
    s_watchGs = &m_memory.gs();
    m_cpuContext.r[4] = _mm_setzero_si128();
    m_cpuContext.r[5] = _mm_setzero_si128();
    m_cpuContext.r[29] = _mm_set_epi64x(0, static_cast<int64_t>(PS2_RAM_SIZE - 0x10u));
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)), std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[29], 0)), std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[28], 0)), std::memory_order_relaxed);

    RUNTIME_LOG("Starting execution at address 0x" << std::hex << m_cpuContext.pc << std::dec);

    // A blank image to use as a framebuffer
    Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLANK);
    Texture2D frameTex = LoadTextureFromImage(blank);
    UnloadImage(blank);
    PresentationSettings presentation;

    std::atomic<bool> gameThreadFinished{false};

    std::thread gameThread([&]()
                           {
        ThreadNaming::SetCurrentThreadName("GameThread");
        try
        {
            m_eeScheduler->reset(m_memory.getRDRAM(), m_cpuContext);
            m_eeScheduler->run();
            uint32_t pc = m_debugPc.load(std::memory_order_relaxed);
            RUNTIME_LOG("Game thread returned. PC=0x" << std::hex << pc
                      << " RA=0x" << static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)) << std::dec << std::endl);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during program execution: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Error during program execution: unknown exception" << std::endl;
        }
        gameThreadFinished.store(true, std::memory_order_release); });

    uint64_t tick = 0;
    while (!isStopRequested() && !gameThreadFinished.load(std::memory_order_acquire))
    {
        PS2_IF_AGRESSIVE_LOGS({
            tick++;
            if ((tick % 120) == 0)
            {
                uint64_t curDma = m_memory.dmaStartCount();
                uint64_t curGif = m_memory.gifCopyCount();
                uint64_t curGs = m_memory.gsWriteCount();
                uint64_t curVif = m_memory.vifWriteCount();
                const GSRegisters &gs = m_memory.gs();
                const uint32_t dbgPc = m_debugPc.load(std::memory_order_relaxed);
                const uint32_t dbgRa = m_debugRa.load(std::memory_order_relaxed);
                const uint32_t dbgSp = m_debugSp.load(std::memory_order_relaxed);
                const uint32_t dbgGp = m_debugGp.load(std::memory_order_relaxed);
                const auto eeSnapshot = m_eeScheduler->snapshot();

                RUNTIME_LOG("[run:tick] tick=" << tick
                                               << " pc=0x" << std::hex << dbgPc
                                               << " ra=0x" << dbgRa
                                               << " sp=0x" << dbgSp
                                               << " gp=0x" << dbgGp
                                               << " dispfb1=0x" << gs.dispfb1
                                               << " display1=0x" << gs.display1
                                               << std::dec
                                               << " activeThreads=" << eeSnapshot.threads.size()
                                               << " dma=" << curDma
                                               << " gif=" << curGif
                                               << " gsw=" << curGs
                                               << " vif=" << curVif
                                               << std::endl);

            }
        });
        uint32_t presentWidth = FB_WIDTH;
        uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
        UploadFrame(frameTex, this, presentWidth, presentHeight);
        Texture2D picture = frameTex;
        if (ps2x::gles::active())
        {
            unsigned glesTexture = 0u;
            uint32_t glesWidth = 0u;
            uint32_t glesHeight = 0u;
            if (ps2x::gles::displayTexture(glesTexture, glesWidth, glesHeight))
            {
                picture = Texture2D{glesTexture, static_cast<int>(glesWidth), static_cast<int>(glesHeight), 1,
                                    PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
                presentWidth = glesWidth;
                presentHeight = glesHeight;
                DumpGlesFrame(picture, m_eeScheduler->currentVSyncTick());
            }
        }

        presentation.apply(frameTex);
        if (picture.id != frameTex.id)
            SetTextureFilter(picture, presentation.filter == "sharp" ? TEXTURE_FILTER_POINT : TEXTURE_FILTER_BILINEAR);
        BeginDrawing();
        ClearBackground(BLACK);
        const float srcWidth = static_cast<float>(std::max<uint32_t>(1u, presentWidth));
        const float srcHeight = static_cast<float>(std::max<uint32_t>(1u, presentHeight));
        const float screenWidth = static_cast<float>(GetScreenWidth());
        const float screenHeight = static_cast<float>(GetScreenHeight());
        // A PS2 picture fills a 4:3 TV whatever the size of its frame buffer.
        // 16:9 and 21:9 show the picture a game override widened (PS2X_WIDESCREEN) at that shape;
        // a 2D screen it did not widen (menu, map, movie) stays 4:3 when the HUD is laid out (ps2x::hud).
        float dstWidth = screenWidth;
        float dstHeight = screenHeight;
        const bool wide = (presentation.aspect == "16:9" || presentation.aspect == "21:9") && !ps2x::hud::pillarbox();
        // In portrait with the touch controls on, the picture goes to the top and the controls below.
        const bool atTop = ps2x::touch::pictureAtTop(static_cast<int>(screenWidth), static_cast<int>(screenHeight));
        const float shape = !wide && (presentation.aspect != "stretch" || atTop) ? 4.0f / 3.0f
                            : presentation.aspect == "16:9"                     ? 16.0f / 9.0f
                            : presentation.aspect == "21:9"                     ? 21.0f / 9.0f
                                                                                : 0.0f;
        if (shape > 0.0f)
        {
            dstWidth = std::min(screenWidth, screenHeight * shape);
            dstHeight = dstWidth / shape;
        }
        const Rectangle srcRect{0.0f, 0.0f, srcWidth, srcHeight};
        const Rectangle dstRect{
            (screenWidth - dstWidth) * 0.5f,
            atTop ? 0.0f : (screenHeight - dstHeight) * 0.5f,
            dstWidth,
            dstHeight};
        presentation.drawFrame(picture, srcRect, dstRect);
        presentation.drawFps(dstRect);
        ps2x::touch::frame(static_cast<int>(screenWidth), static_cast<int>(screenHeight));
        if (m_debugUiInitialized && m_debugUiDrawCallback)
        {
            m_debugUiDrawCallback(*this, m_debugUiUserData);
        }
        EndDrawing();

        if (WindowShouldClose())
        {
            RUNTIME_LOG("[run] window close requested, breaking out of loop");
            requestStop();
            break;
        }
    }

    requestStop();
    if (gameThread.joinable())
    {
        gameThread.join();
        // The game thread may have been waiting for the VIF1 thread until it stopped.
        m_memory.shutdownVif1();
    }
    ps2x::gles::shutdown();

    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }
    UnloadTexture(frameTex);
    CloseWindow();

    RUNTIME_LOG("[run] exiting loop");
}
