#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu1_compiled.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"

#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <ps2_log.h>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace
{
    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    // std::countr_zero goes through a runtime CPU check in MSVC's library; this inlines.
    inline uint32_t lowestSetBit(uint32_t mask)
    {
#if defined(__GNUC__) || defined(__clang__)
        return static_cast<uint32_t>(__builtin_ctz(mask));
#else
        unsigned long index = 0;
        _BitScanForward(&index, mask);
        return static_cast<uint32_t>(index);
#endif
    }

    // Diagnostics: PS2X_VU1_STATS prints VU1 work and host time about once a second.
    bool vu1StatsEnabled()
    {
        static const bool s_enabled = std::getenv("PS2X_VU1_STATS") != nullptr;
        return s_enabled;
    }

    void recordVu1Stats(uint64_t pairs, uint64_t cycles, std::chrono::steady_clock::duration busy)
    {
        using Clock = std::chrono::steady_clock;
        static Clock::time_point s_windowStart = Clock::now();
        static uint64_t s_runs = 0;
        static uint64_t s_pairs = 0;
        static uint64_t s_cycles = 0;
        static Clock::duration s_busy{};
        static uint32_t s_lines = 0;
        ++s_runs;
        s_pairs += pairs;
        s_cycles += cycles;
        s_busy += busy;
        const Clock::time_point now = Clock::now();
        const double window = std::chrono::duration<double>(now - s_windowStart).count();
        if (window < 1.0)
            return;
        if (s_lines++ < 4000u)
        {
            const double busyMs = std::chrono::duration<double, std::milli>(s_busy).count();
            std::fprintf(stderr, "[vu1-stats] runs/s=%.0f pairs/s=%.0f cycles/s=%.0f busy=%.0f%% ns/pair=%.1f\n",
                         s_runs / window, s_pairs / window, s_cycles / window, busyMs / (window * 10.0),
                         s_pairs ? busyMs * 1.0e6 / static_cast<double>(s_pairs) : 0.0);
        }
        s_windowStart = now;
        s_runs = s_pairs = s_cycles = 0;
        s_busy = {};
    }

    // Diagnostics: PS2X_VU1_CAPTURE=<dir> identifies the VU1 micro memory image behind every
    // MSCAL by content hash, saves each new image once as <dir>/vu1code_<hash>.bin and
    // prints, every few seconds, how often each (image, start PC) pair ran.
    const std::string &vu1CaptureDir()
    {
        static const std::string s_dir = []()
        {
            const char *value = std::getenv("PS2X_VU1_CAPTURE");
            return std::string(value ? value : "");
        }();
        return s_dir;
    }

    uint64_t fnv1a64(const uint8_t *data, size_t size)
    {
        uint64_t hash = 1469598103934665603ull;
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= data[i];
            hash *= 1099511628211ull;
        }
        return hash;
    }

    void captureVu1Program(const uint8_t *code, uint32_t codeSize, uint64_t generation, uint32_t startPC)
    {
        using Clock = std::chrono::steady_clock;
        static uint64_t s_generation = ~0ull;
        static uint64_t s_hash = 0;
        static uint64_t s_generationsSeen = 0;
        static std::unordered_map<uint64_t, bool> s_saved;
        static std::map<std::pair<uint64_t, uint32_t>, uint64_t> s_calls;
        static Clock::time_point s_lastReport = Clock::now();
        static uint32_t s_reports = 0;

        if (generation != s_generation)
        {
            s_generation = generation;
            s_hash = fnv1a64(code, codeSize);
            ++s_generationsSeen;
            if (!s_saved[s_hash])
            {
                s_saved[s_hash] = true;
                // A translated program matching the image up to the end of its code
                // already runs it (the rest is leftovers of earlier uploads).
                if (const ps2x::vu1c::Program *program = ps2x::vu1c::findProgram(code, codeSize))
                {
                    std::fprintf(stderr, "[vu1-capture] image %016llx runs as translated program %s\n",
                                 static_cast<unsigned long long>(s_hash), program->name);
                }
                else
                {
                    char name[64];
                    std::snprintf(name, sizeof(name), "vu1code_%016llx.bin", static_cast<unsigned long long>(s_hash));
                    const std::string path = vu1CaptureDir() + "/" + name;
                    if (FILE *file = std::fopen(path.c_str(), "wb"))
                    {
                        std::fwrite(code, 1, codeSize, file);
                        std::fclose(file);
                    }
                    std::fprintf(stderr, "[vu1-capture] new image %016llx (generation %llu)\n",
                                 static_cast<unsigned long long>(s_hash), static_cast<unsigned long long>(generation));
                }
            }
        }
        ++s_calls[{s_hash, startPC}];

        const Clock::time_point now = Clock::now();
        if (now - s_lastReport < std::chrono::seconds(5) || s_reports >= 200u)
            return;
        s_lastReport = now;
        ++s_reports;
        std::fprintf(stderr, "[vu1-capture] images=%zu generations=%llu entries=%zu\n", s_saved.size(),
                     static_cast<unsigned long long>(s_generationsSeen), s_calls.size());
        std::vector<std::pair<uint64_t, std::pair<uint64_t, uint32_t>>> ranked;
        for (const auto &[key, count] : s_calls)
            ranked.push_back({count, key});
        std::sort(ranked.rbegin(), ranked.rend());
        for (size_t i = 0; i < ranked.size() && i < 12u; ++i)
        {
            std::fprintf(stderr, "[vu1-capture]   %016llx pc=%04x calls=%llu\n",
                         static_cast<unsigned long long>(ranked[i].second.first), ranked[i].second.second,
                         static_cast<unsigned long long>(ranked[i].first));
        }
    }

    // PS2X_VU1_COMPILED=0 forces the interpreter even when a translated program exists.
    bool compiledProgramsEnabled()
    {
        static const bool s_enabled = []()
        {
            const char *value = std::getenv("PS2X_VU1_COMPILED");
            return !(value && value[0] == '0');
        }();
        return s_enabled;
    }

    // Diagnostics: PS2X_VU1_VERIFY runs every translated program and then the interpreter
    // on the same input, keeps the interpreter's result and reports any difference.
    bool compiledVerificationEnabled()
    {
        static const bool s_enabled = std::getenv("PS2X_VU1_VERIFY") != nullptr;
        return s_enabled;
    }

    void captureGifPacket(void *user, const uint8_t *data, uint32_t size)
    {
        auto *packets = static_cast<std::vector<uint8_t> *>(user);
        packets->insert(packets->end(), data, data + size);
    }

    uint32_t floatBits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    // MADD/MSUB-style upper instructions: ACC plus or minus a product.
    bool isFmacProductSum(uint32_t upper)
    {
        const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
        const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
        return (op >= 0x08u && op <= 0x0Fu) ||
               op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
               op == 0x29u || op == 0x2Du || op == 0x2Eu ||
               (special >= 0x08u && special <= 0x0Fu) ||
               special == 0x21u || special == 0x23u || special == 0x25u ||
               special == 0x27u || special == 0x29u || special == 0x2Du;
    }
}

void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

struct VU1Interpreter::CompiledVerification
{
    bool active = false;
    const ps2x::vu1c::Program *program = nullptr;
    uint32_t entryPc = 0;
    ps2x::vu1c::Exit compiledExit = ps2x::vu1c::Exit::Refused;
    VU1State state{};
    uint64_t cycle = 0;
    uint32_t workingClip = 0;
    std::array<std::array<uint64_t, 4>, 32> vfReady{};
    std::array<uint64_t, 16> viReady{};
    std::array<uint64_t, 4> accReady{};
    bool backupValid = false;
    uint8_t backupReg = 0;
    int32_t backupValue = 0;
    std::vector<uint8_t> memoryBefore;
    VU1State compiledState{};
    uint64_t compiledCycle = 0;
    uint32_t compiledWorkingClip = 0;
    std::vector<uint8_t> compiledMemory;
    std::vector<uint8_t> compiledPackets;
    std::vector<uint8_t> interpreterPackets;
    uint64_t verified = 0;
    uint64_t mismatches = 0;
    uint64_t handoffs = 0;
};

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    reset();
}

VU1Interpreter::~VU1Interpreter() = default;

void VU1Interpreter::resetScheduler()
{
    m_flagPipeline = {};
    m_fdiv = {};
    m_efu = {};
    m_storePipeline = {};
    m_vfWritePipeline = {};
    m_viWritePipeline = {};
    m_accWritePipeline = {};
    m_flagSlots = 0u;
    m_storeSlots = 0u;
    m_vfWriteSlots = 0u;
    m_viWriteSlots = 0u;
    m_accWriteSlots = 0u;
    resetXgkick();
    m_nextCommitCycle = ~0ull;
    m_vfReady = {};
    m_viReady = {};
    m_accReady = {};
    m_vfLatestWrite = {};
    m_viLatestWrite = {};
    m_accLatestWrite = {};
    m_nextWriteSequence = 0;
    m_efuResourceReady = 0;
    m_workingClip = m_state.clip;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    resetScheduler();
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4])
{
    const bool productSum = isFmacProductSum(m_currentUpperInstruction);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        // Same flags as the exact path below, without recomputing the result: with an
        // exponent well inside the float range nothing under- or overflowed, and rounding
        // toward zero cannot flip the sign unless a product-sum cancelled against an ACC
        // more than 2^16 times larger than the result.
        const uint32_t bits = floatBits(result[component]);
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent >= 2u && exponent <= 253u &&
            (!productSum || ((floatBits(m_state.acc[component]) >> 23) & 0xFFu) <= exponent + 16u))
        {
            laneFlags[component] = (bits & 0x80000000u) != 0u ? 0x2u : 0u;
            continue;
        }

        long double exactResult = 0.0L;
        if (calculateFmacExactResult(component, exactResult))
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
            continue;
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               long double &result) const
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<long double>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const long double q = operand(m_state.q);
    const long double i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0L
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0L
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  long double exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const long double magnitude = std::fabs(exactResult);
    const long double maximum = static_cast<long double>(std::numeric_limits<float>::max());
    const long double minimum = static_cast<long double>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0L)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest) const
{
    uint32_t extraSticky = 0u;
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = normalizeOperand(m_state.q);
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = normalizeOperand(m_state.i);
        }
        else if (op == 0x2Eu)
        {
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        }
        else
        {
            right = normalizeOperand(m_state.vf[ft][component]);
        }

        float product = left * right;
        // A single rounding toward zero keeps the sign and, away from the range limits,
        // cannot under- or overflow, so only the sign can be set.
        const uint32_t productBits = floatBits(product);
        const uint32_t productExponent = (productBits >> 23) & 0xFFu;
        if (productExponent >= 2u && productExponent <= 253u)
        {
            extraSticky |= (productBits & 0x80000000u) != 0u ? 0x2u : 0u;
            continue;
        }
        const long double exactProduct = static_cast<long double>(left) * static_cast<long double>(right);
        const uint8_t productFlags = normalizeFmacExactResult(product, exactProduct);
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    FlagPipelineEntry *entry = nullptr;
    for (FlagPipelineEntry &candidate : m_flagPipeline)
    {
        if (!candidate.valid)
        {
            entry = &candidate;
            break;
        }
    }
    if (!entry)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return;
    }

    *entry = {};
    entry->valid = true;
    m_flagSlots |= 1u << static_cast<uint32_t>(entry - m_flagPipeline.data());
    entry->issueCycle = m_cycle;
    entry->readyCycle = m_cycle + kFmacLatency;
    noteCommitCycle(entry->readyCycle);
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDest(dst, result, dest);
}

void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDestAcc(result, dest);
}

void VU1Interpreter::queueFsset(uint16_t immediate)
{
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (entry.valid && entry.issueCycle == m_cycle)
            entry.writesStatus = false;
    }

    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (!entry.valid)
        {
            entry = {};
            entry.valid = true;
            m_flagSlots |= 1u << static_cast<uint32_t>(&entry - m_flagPipeline.data());
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            noteCommitCycle(entry.readyCycle);
            entry.status = static_cast<uint32_t>(immediate) & 0xFC0u;
            entry.writesSticky = true;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFEu);
}

void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (!entry.valid)
        {
            entry = {};
            entry.valid = true;
            m_flagSlots |= 1u << static_cast<uint32_t>(&entry - m_flagPipeline.data());
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            noteCommitCycle(entry.readyCycle);
            entry.clip = m_workingClip;
            entry.writesClip = true;
            return;
        }
    }
    reportReservedInstruction(true, 0xFFFFFFFDu);
}

void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (entry.valid && entry.issueCycle == m_cycle)
            entry.writesClip = false;
    }
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (!entry.valid)
        {
            entry = {};
            entry.valid = true;
            m_flagSlots |= 1u << static_cast<uint32_t>(&entry - m_flagPipeline.data());
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            noteCommitCycle(entry.readyCycle);
            entry.clip = m_workingClip;
            entry.writesClip = true;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFAu);
}

void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    noteCommitCycle(m_fdiv.readyCycle);
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            noteCommitCycle(entry.readyCycle);
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    for (PendingStore &store : m_storePipeline)
    {
        if (!store.valid)
        {
            store.valid = true;
            m_storeSlots |= 1u << static_cast<uint32_t>(&store - m_storePipeline.data());
            store.readyCycle = m_cycle + 1u;
            noteCommitCycle(store.readyCycle);
            store.address = address;
            store.laneMask = laneMask;
            std::copy(words, words + 4, store.words.begin());
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFCu);
}

void VU1Interpreter::queueVfWrite(uint8_t reg, uint8_t laneMask,
                                  const float value[4], uint32_t latency)
{
    if (reg == 0u || laneMask == 0u)
        return;
    for (PendingVfWrite &write : m_vfWritePipeline)
    {
        if (!write.valid)
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            noteCommitCycle(write.readyCycle);
            m_vfWriteSlots |= 1u << static_cast<uint32_t>(&write - m_vfWritePipeline.data());
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_vfLatestWrite[reg][component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF7u);
}

void VU1Interpreter::queueViWrite(uint8_t reg, int32_t value, uint32_t latency)
{
    if (reg == 0u)
        return;
    for (PendingViWrite &write : m_viWritePipeline)
    {
        if (!write.valid)
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            noteCommitCycle(write.readyCycle);
            m_viWriteSlots |= 1u << static_cast<uint32_t>(&write - m_viWritePipeline.data());
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.value = value;
            m_viLatestWrite[reg] = write.sequence;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF6u);
}

void VU1Interpreter::queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency)
{
    if (laneMask == 0u)
        return;
    for (PendingAccWrite &write : m_accWritePipeline)
    {
        if (!write.valid)
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            noteCommitCycle(write.readyCycle);
            m_accWriteSlots |= 1u << static_cast<uint32_t>(&write - m_accWritePipeline.data());
            write.sequence = ++m_nextWriteSequence;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_accLatestWrite[component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(true, 0xFFFFFFF5u);
}

void VU1Interpreter::commitReadyPipelines()
{
    if (m_cycle < m_nextCommitCycle)
        return;
    uint64_t next = ~0ull;
    const auto pending = [&](bool valid, uint64_t readyCycle)
    {
        if (!valid)
            return false;
        if (readyCycle > m_cycle)
        {
            next = std::min(next, readyCycle);
            return false;
        }
        return true;
    };

    for (uint32_t slots = m_flagSlots; slots != 0u; slots &= slots - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(lowestSetBit(slots));
        FlagPipelineEntry &entry = m_flagPipeline[index];
        if (!pending(entry.valid, entry.readyCycle))
            continue;

        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
        {
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        }
        if (entry.writesClip)
            m_state.clip = entry.clip;
        entry = {};
        m_flagSlots &= ~(1u << index);
    }

    if (pending(m_fdiv.valid, m_fdiv.readyCycle))
    {
        m_state.q = m_fdiv.value;
        const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
        m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
        m_fdiv = {};
    }

    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (pending(entry.valid, entry.readyCycle))
        {
            m_state.p = entry.value;
            entry = {};
        }
    }

    for (uint32_t slots = m_storeSlots; slots != 0u; slots &= slots - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(lowestSetBit(slots));
        PendingStore &store = m_storePipeline[index];
        if (!pending(store.valid, store.readyCycle))
            continue;
        if (m_activeVuData && store.address + 16u <= m_activeVuDataSize)
        {
            uint32_t oldWords[4]{};
            std::memcpy(oldWords, m_activeVuData + store.address, sizeof(oldWords));
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((store.laneMask & laneForComponent(component)) != 0u)
                    oldWords[component] = store.words[component];
            }
            std::memcpy(m_activeVuData + store.address, oldWords, sizeof(oldWords));
        }
        store = {};
        m_storeSlots &= ~(1u << index);
    }

    for (uint32_t slots = m_vfWriteSlots; slots != 0u; slots &= slots - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(lowestSetBit(slots));
        PendingVfWrite &write = m_vfWritePipeline[index];
        if (!pending(write.valid, write.readyCycle))
            continue;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_vfLatestWrite[write.reg][component] == write.sequence)
            {
                m_state.vf[write.reg][component] = write.value[component];
            }
        }
        write = {};
        m_vfWriteSlots &= ~(1u << index);
    }

    for (uint32_t slots = m_viWriteSlots; slots != 0u; slots &= slots - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(lowestSetBit(slots));
        PendingViWrite &write = m_viWritePipeline[index];
        if (!pending(write.valid, write.readyCycle))
            continue;
        if (m_viLatestWrite[write.reg] == write.sequence)
            m_state.vi[write.reg] = static_cast<int16_t>(write.value);
        write = {};
        m_viWriteSlots &= ~(1u << index);
    }

    for (uint32_t slots = m_accWriteSlots; slots != 0u; slots &= slots - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(lowestSetBit(slots));
        PendingAccWrite &write = m_accWritePipeline[index];
        if (!pending(write.valid, write.readyCycle))
            continue;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_accLatestWrite[component] == write.sequence)
            {
                m_state.acc[component] = write.value[component];
            }
        }
        write = {};
        m_accWriteSlots &= ~(1u << index);
    }
    m_nextCommitCycle = next;
}

void VU1Interpreter::progressXgkick()
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    ++m_xgkick.cycleCredit;
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        for (uint32_t i = 0; i < 16u; ++i)
        {
            const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes + i) % m_activeVuDataSize;
            m_xgkick.packet[m_xgkick.copiedBytes + i] = m_activeVuData[source];
        }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else if (format == 2u)
                tagBytes += static_cast<uint64_t>(nloop) * 16u;
            else
            {
                reportReservedInstruction(false, 0xFFFFFFF8u);
                m_xgkick.active = false;
                return;
            }

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    if (m_gifCapture)
        m_gifCapture->insert(m_gifCapture->end(), m_xgkick.packet.data(), m_xgkick.packet.data() + m_xgkick.totalBytes);
    if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
    m_xgkick.active = false;
}

void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    resetXgkick();
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
}

void VU1Interpreter::resetXgkick()
{
    // Clearing the 64 KB packet buffer here used to cost more than whole micro programs.
    m_xgkick.sourceAddress = 0u;
    m_xgkick.totalBytes = 0u;
    m_xgkick.copiedBytes = 0u;
    m_xgkick.currentTagEnd = 0u;
    m_xgkick.cycleCredit = 0u;
    m_xgkick.issueCycle = 0u;
    m_xgkick.active = false;
    m_xgkick.currentTagEop = false;
}

void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    m_state.cycles = m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    if (m_cycle >= m_nextCommitCycle)
        commitReadyPipelines();
    progressXgkick();
}

void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
        advanceOneCycle();
}

bool VU1Interpreter::pipelinesPending() const
{
    if (m_fdiv.valid || m_xgkick.active)
        return true;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    return (m_flagSlots | m_storeSlots | m_vfWriteSlots | m_viWriteSlots | m_accWriteSlots) != 0u;
}

void VU1Interpreter::flushPipelines()
{
    while (pipelinesPending())
        advanceOneCycle();
}

uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    uint64_t ready = m_cycle;
    const InstructionUsage *usages[2] = {
        &decoded.upperUsage,
        &decoded.lowerUsage};
    for (const InstructionUsage *usage : usages)
    {
        if (!usage)
            continue;
        for (uint32_t index = 0; index < usage->vfReadCount; ++index)
        {
            const VfAccess &access = usage->vfRead[index];
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((access.lanes & laneForComponent(component)) != 0u)
                    ready = std::max(ready, m_vfReady[access.reg][component]);
            }
        }
        for (uint32_t mask = usage->viRead & 0xFFFEu; mask != 0u; mask &= mask - 1u)
            ready = std::max(ready, m_viReady[lowestSetBit(mask)]);
        if (usage->accRead != 0u)
        {
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((usage->accRead & laneForComponent(component)) != 0u)
                    ready = std::max(ready, m_accReady[component]);
            }
        }
    }

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u &&
        decoded.suppressedLowerVf != lowerWrite.reg)
    {
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((lowerWrite.lanes & laneForComponent(component)) != 0u)
                m_vfReady[lowerWrite.reg][component] = m_cycle + latency;
        }
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                     ? decoded.upperUsage.vfLatency
                                     : decoded.upperUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((upperWrite.lanes & laneForComponent(component)) != 0u)
                m_vfReady[upperWrite.reg][component] = m_cycle + latency;
        }
    }

    for (uint32_t mask = decoded.lowerUsage.viWrite & 0xFFFEu; mask != 0u; mask &= mask - 1u)
        m_viReady[lowestSetBit(mask)] = m_cycle + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((decoded.upperUsage.accWrite & laneForComponent(component)) != 0u)
            m_accReady[component] = m_cycle + kAccForwardLatency;
    }
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper) const
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        if (op == 0x08u || op == 0x09u || op == 0x0Au || op == 0x0Bu ||
            op == 0x0Cu || op == 0x0Du || op == 0x0Eu || op == 0x0Fu ||
            op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
            op == 0x29u || op == 0x2Du || op == 0x2Eu)
        {
            usage.accRead = dest;
        }
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
            usage.accWrite = dest;
            if ((special >= 0x08u && special <= 0x0Fu) ||
                special == 0x21u || special == 0x23u || special == 0x25u ||
                special == 0x27u || special == 0x29u || special == 0x2Du)
            {
                usage.accRead = dest;
            }
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special != 0x2Fu && special != 0x30u)
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(uint32_t lower) const
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x77:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7C:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x79:
            usage.latency = 29u;
            break;
        case 0x7D:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(decoded.lower);

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }
    return decoded;
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
    for (uint32_t i = 0; i < pairCount; ++i)
        m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u);

    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;
}

const VU1Interpreter::DecodedInstructionPair &VU1Interpreter::getDecodedInstructionPairForPc(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
{
    if ((pc & 7u) != 0u)
    {
        m_uncachedDecodedPair = decodeInstructionPair(vuCode, pc);
        return m_uncachedDecodedPair;
    }

    const bool trackedVu1Code = memory != nullptr &&
                                ((m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                                 (m_unit == Unit::VU0 && vuCode == memory->getVU0Code()));
    if (!trackedVu1Code)
    {
        m_uncachedDecodedPair = decodeInstructionPair(vuCode, pc);
        return m_uncachedDecodedPair;
    }

    const uint64_t generation = m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration();
    if (!m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize ||
        m_cachedCodeGeneration != generation)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }
    const uint32_t pairIndex = pc / 8u;
    if (pairIndex >= kMaxDecodedPairs)
    {
        m_uncachedDecodedPair = decodeInstructionPair(vuCode, pc);
        return m_uncachedDecodedPair;
    }
    return m_decodedCodeCache[pairIndex];
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    m_stopRequested = true;
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    if (m_unit == Unit::VU1 && memory && !vu1CaptureDir().empty())
        captureVu1Program(vuCode, codeSize, memory->getVU1CodeGeneration(), startPC & microAddressMask());
    resetScheduler();
    m_state.pc = startPC & microAddressMask();
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    const int previousRoundingMode = std::fegetround();
    const bool useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    const uint64_t budgetEnd = m_cycle + maxCycles;
    bool programEnded = false;
    const bool collectStats = m_unit == Unit::VU1 && vu1StatsEnabled();
    const auto statsStart = collectStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const uint64_t statsStartCycle = m_cycle;
    uint64_t executedPairs = 0;
    if (m_unit == Unit::VU1 && memory && compiledProgramsEnabled() && !pipelinesPending() &&
        !m_state.branchPending && !m_state.ebit && !m_state.haltAfterDelaySlot)
    {
        if (const ps2x::vu1c::Program *program = compiledProgramFor(vuCode, codeSize, memory))
        {
            if (compiledVerificationEnabled())
                beginCompiledVerification(*program, vuData, dataSize, budgetEnd);
            else if (runCompiled(*program, vuData, dataSize, budgetEnd, nullptr))
                programEnded = true;
        }
    }
    while (!programEnded && m_cycle < budgetEnd && !m_stopRequested)
    {
        if (m_cycle >= m_nextCommitCycle)
            commitReadyPipelines();
        if (m_state.pc + 8u > codeSize)
            break;

        const DecodedInstructionPair &decoded = getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
        if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
        {
            reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
            break;
        }

        uint64_t readyCycle = calculatePairReadyCycle(decoded);
        while (readyCycle > m_cycle)
        {
            if (readyCycle >= budgetEnd)
            {
                advanceTo(budgetEnd);
                break;
            }
            advanceTo(readyCycle);
            readyCycle = calculatePairReadyCycle(decoded);
        }
        if (m_cycle >= budgetEnd)
            break;

        ++executedPairs;
        uint8_t writtenVi = 0u;
        int32_t oldVi = 0;
        if (const uint32_t viWrites = decoded.lowerUsage.viWrite & 0xFFFEu; viWrites != 0u)
        {
            writtenVi = static_cast<uint8_t>(lowestSetBit(viWrites));
            oldVi = m_state.vi[writtenVi];
        }

        const VfAccess upperWrite = decoded.upperUsage.vfWrite;
        const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
        const bool hasUpperWrite = upperWrite.reg != 0u;
        const bool hasLowerWrite = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
        const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
        float oldUpperVf[4]{};
        float newUpperVf[4]{};
        float oldLowerVf[4]{};
        float newLowerVf[4]{};
        float oldAcc[4]{};
        float newAcc[4]{};
        if (hasUpperWrite)
            std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
        if (hasDistinctLowerWrite)
            std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
        if (decoded.upperUsage.accWrite != 0u)
            std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));

        if (decoded.iBit)
        {
            execUpper(decoded.upper);
            float immediate = 0.0f;
            std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
            m_state.i = normalizeOperand(immediate);
        }
        else if (decoded.upperVfShadowReg != 0u)
        {
            float oldVf[4]{};
            float upperVf[4]{};
            std::memcpy(oldVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(oldVf));
            execUpper(decoded.upper);
            std::memcpy(upperVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(upperVf));
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        oldVf,
                        sizeof(oldVf));
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        upperVf,
                        sizeof(upperVf));
        }
        else
        {
            execUpper(decoded.upper);
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        }

        m_viBranchBackupValid = false;

        if (hasUpperWrite)
        {
            std::memcpy(newUpperVf, m_state.vf[upperWrite.reg], sizeof(newUpperVf));
            std::memcpy(m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
            const uint32_t latency =
                decoded.upperUsage.vfLatency != 0u
                    ? decoded.upperUsage.vfLatency
                    : decoded.upperUsage.latency;
            queueVfWrite(upperWrite.reg, upperWrite.lanes, newUpperVf, latency);
        }
        if (hasDistinctLowerWrite)
        {
            std::memcpy(newLowerVf, m_state.vf[lowerWrite.reg], sizeof(newLowerVf));
            std::memcpy(m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
            const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                         ? decoded.lowerUsage.vfLatency
                                         : decoded.lowerUsage.latency;
            queueVfWrite(lowerWrite.reg, lowerWrite.lanes, newLowerVf, latency);
        }
        if (decoded.upperUsage.accWrite != 0u)
        {
            std::memcpy(newAcc, m_state.acc, sizeof(newAcc));
            std::memcpy(m_state.acc, oldAcc, sizeof(oldAcc));
            // ACC is forwarded to the next upper instruction. Its arithmetic
            // flags still use the normal four-cycle FMAC timeline.
            queueAccWrite(decoded.upperUsage.accWrite, newAcc,
                          kAccForwardLatency);
        }
        int32_t tracedNewVi = 0;
        if (writtenVi != 0u)
        {
            const int32_t newVi = m_state.vi[writtenVi];
            tracedNewVi = newVi;
            m_state.vi[writtenVi] = oldVi;
            const uint32_t latency =
                decoded.lowerUsage.viLatency != 0u
                    ? decoded.lowerUsage.viLatency
                    : decoded.lowerUsage.latency;
            queueViWrite(writtenVi, newVi, latency);
        }

        markPairWrites(decoded);
        if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
            recordViWriteForBranch(writtenVi, oldVi);

        {
            // Diagnostics: PS2X_VU1_TRACE=<pc>:<pairs>[:<skip>[:<lower>]] logs <pairs>
            // executed pairs once VU1 reaches micro address <pc> (hex) for the
            // (<skip>+1)th time, optionally only where the lower word is <lower> (hex).
            const uint8_t upperReg = hasUpperWrite ? upperWrite.reg : 0u;
            const float *upperValue = newUpperVf;
            const uint8_t lowerReg = hasDistinctLowerWrite ? lowerWrite.reg : 0u;
            const float *lowerValue = newLowerVf;
            const uint8_t viReg = writtenVi;
            const int32_t viValue = tracedNewVi;
            do
            {
                struct TraceConfig
                {
                    bool enabled = false;
                    uint32_t pc = 0;
                    uint32_t pairs = 0;
                    uint32_t skip = 0;
                    uint32_t lower = 0;
                    bool matchLower = false;
                };
                static const TraceConfig config = []()
                {
                    TraceConfig c;
                    if (const char *value = std::getenv("PS2X_VU1_TRACE"))
                    {
                        char *end = nullptr;
                        c.pc = static_cast<uint32_t>(std::strtoul(value, &end, 16));
                        if (end && *end == ':')
                            c.pairs = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
                        if (end && *end == ':')
                            c.skip = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
                        if (end && *end == ':')
                        {
                            c.lower = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
                            c.matchLower = true;
                        }
                        c.enabled = c.pairs != 0u;
                    }
                    return c;
                }();
                static uint32_t s_remaining = 0;
                static uint32_t s_arrivals = 0;
                if (!config.enabled || m_unit != Unit::VU1)
                    break;
                if (s_remaining == 0u && m_state.pc == config.pc && (!config.matchLower || decoded.lower == config.lower))
                {
                    if (s_arrivals++ == config.skip)
                        s_remaining = config.pairs;
                }
                if (s_remaining == 0u)
                    break;
                --s_remaining;
                std::fprintf(stderr, "[vu1] c=%llu pc=%04x lo=%08x up=%08x clip=%06x q=%g vi:",
                             static_cast<unsigned long long>(m_cycle), m_state.pc, decoded.lower, decoded.upper,
                             m_state.clip & 0xFFFFFFu, m_state.q);
                for (uint32_t reg = 1; reg < 16u; ++reg)
                    std::fprintf(stderr, " %d", static_cast<int>(static_cast<int16_t>(m_state.vi[reg])));
                if (upperReg != 0u)
                    std::fprintf(stderr, " | U vf%u=(%g,%g,%g,%g)", upperReg, upperValue[0], upperValue[1], upperValue[2], upperValue[3]);
                if (lowerReg != 0u)
                    std::fprintf(stderr, " | L vf%u=(%g,%g,%g,%g)", lowerReg, lowerValue[0], lowerValue[1], lowerValue[2], lowerValue[3]);
                if (viReg != 0u)
                    std::fprintf(stderr, " | vi%u<-%d", viReg, static_cast<int>(static_cast<int16_t>(viValue)));
                std::fputc('\n', stderr);
            } while (false);
        }

        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        m_state.vi[0] = 0;

        uint32_t nextPc = m_state.pc + 8u;
        if (nextPc >= codeSize)
            nextPc = 0u;
        m_state.pc = nextPc;

        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0u)
            {
                m_state.pc = m_state.branchTarget & microAddressMask();
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        const bool dHalt = decoded.dBit && m_state.dBitEnabled;
        const bool tHalt = decoded.tBit && m_state.tBitEnabled;
        const bool haltBit = dHalt || tHalt;
        const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

        if (m_state.haltAfterDelaySlot)
        {
            m_state.stoppedByD = m_pendingHaltD;
            m_state.stoppedByT = m_pendingHaltT;
            programEnded = true;
        }
        else if (m_state.ebit)
            programEnded = true;
        else if (haltBit && !haltBranch)
        {
            m_state.stoppedByD = dHalt;
            m_state.stoppedByT = tHalt;
            programEnded = true;
        }
        else if (decoded.eBit)
            m_state.ebit = true;
        else if (haltBranch)
        {
            m_state.haltAfterDelaySlot = true;
            m_pendingHaltD = dHalt;
            m_pendingHaltT = tHalt;
        }

        advanceOneCycle();
        if (programEnded)
            break;
    }

    if (programEnded)
    {
        flushPipelines();
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    m_state.cycles = m_cycle;
    if (m_compiledVerification && m_compiledVerification->active)
        finishCompiledVerification(vuData, dataSize, programEnded);
    if (useVuRounding && previousRoundingMode != -1)
        std::fesetround(previousRoundingMode);
    if (collectStats)
        recordVu1Stats(executedPairs, m_cycle - statsStartCycle, std::chrono::steady_clock::now() - statsStart);
}

const ps2x::vu1c::Program *VU1Interpreter::compiledProgramFor(const uint8_t *vuCode, uint32_t codeSize,
                                                              PS2Memory *memory)
{
    // Micro memory only changes through VIF MPG uploads, which bump the generation.
    const uint64_t generation = memory->getVU1CodeGeneration();
    if (generation != m_compiledGeneration)
    {
        m_compiledGeneration = generation;
        m_compiledProgram = ps2x::vu1c::findProgram(vuCode, codeSize);
    }
    return m_compiledProgram;
}

void VU1Interpreter::exportCompiledState(ps2x::vu1c::Exec &x, uint8_t *vuData, uint32_t dataSize, uint64_t budgetEnd)
{
    std::memcpy(x.vf, m_state.vf, sizeof(x.vf));
    std::memcpy(x.vi, m_state.vi, sizeof(x.vi));
    std::memcpy(x.acc, m_state.acc, sizeof(x.acc));
    x.q = m_state.q;
    x.i = m_state.i;
    x.mac = m_state.mac;
    x.status = m_state.status;
    x.clip = m_state.clip;
    x.top = m_state.top;
    x.itop = m_state.itop;

    x.cycle = m_cycle;
    x.budgetEnd = budgetEnd;
    x.lastReady = m_cycle;
    for (uint32_t reg = 0; reg < 32u; ++reg)
        for (uint32_t component = 0; component < 4u; ++component)
            x.vfReady[reg][component] = m_vfReady[reg][component];
    for (uint32_t reg = 0; reg < 16u; ++reg)
        x.viReady[reg] = m_viReady[reg];
    for (uint32_t component = 0; component < 4u; ++component)
        x.accReady[component] = m_accReady[component];

    x.flagHead = 0u;
    x.flagCount = 0u;
    x.workingClip = m_workingClip;
    x.fdivPending = false;
    x.fdivReady = 0u;
    x.fdivValue = 0.0f;
    x.fdivStatusDi = 0u;
    x.nextCommit = ps2x::vu1c::kNever;
    x.viBackupValid = m_viBranchBackupValid;
    x.viBackupReg = m_viBranchBackupReg;
    x.viBackupValue = m_viBranchBackupValue;

    x.xgActive = false;
    x.xgSource = 0u;
    x.xgCopied = 0u;
    x.xgTagEnd = 0u;
    x.xgTotal = 0u;
    x.xgTagEop = false;
    x.xgIssue = 0u;
    x.xgPacket = m_xgkick.packet.data();

    x.mem = vuData;
    x.memSize = dataSize;
    static const bool s_noClip = std::getenv("PS2X_VU1_NOCLIP") != nullptr;
    x.noClip = s_noClip;
    x.submitGif = [](void *user, const uint8_t *data, uint32_t size)
    {
        auto *self = static_cast<VU1Interpreter *>(user);
        if (self->m_activeMemory)
            self->m_activeMemory->submitGifPacket(GifPathId::Path1, data, size);
        else if (self->m_activeGs)
            self->m_activeGs->processGIFPacket(data, size);
    };
    x.gifUser = this;
    x.reportError = [](void *user, uint32_t code)
    { static_cast<VU1Interpreter *>(user)->reportReservedInstruction(false, code); };
    x.errorUser = this;
    x.exit = ps2x::vu1c::Exit::Refused;
    x.exitPc = m_state.pc;
}

void VU1Interpreter::importCompiledState(const ps2x::vu1c::Exec &x)
{
    std::memcpy(m_state.vf, x.vf, sizeof(x.vf));
    std::memcpy(m_state.vi, x.vi, sizeof(x.vi));
    std::memcpy(m_state.acc, x.acc, sizeof(x.acc));
    m_state.q = x.q;
    m_state.i = x.i;
    m_state.mac = x.mac;
    m_state.status = x.status;
    m_state.clip = x.clip;
    m_cycle = x.cycle;
    m_state.cycles = m_cycle;
    m_workingClip = x.workingClip;
    for (uint32_t reg = 0; reg < 32u; ++reg)
        for (uint32_t component = 0; component < 4u; ++component)
            m_vfReady[reg][component] = x.vfReady[reg][component];
    for (uint32_t reg = 0; reg < 16u; ++reg)
        m_viReady[reg] = x.viReady[reg];
    for (uint32_t component = 0; component < 4u; ++component)
        m_accReady[component] = x.accReady[component];
    m_viBranchBackupValid = x.viBackupValid;
    m_viBranchBackupReg = x.viBackupReg;
    m_viBranchBackupValue = x.viBackupValue;
    m_state.pc = x.exitPc & microAddressMask();
    m_state.branchPending = false;
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;

    if (x.exit != ps2x::vu1c::Exit::Handoff)
        return;

    // Hand the in-flight work to the interpreter's pipelines so its stalls, commits
    // and final flush come out exactly as if it had run the whole program.
    for (uint32_t index = 0; index < x.flagCount; ++index)
    {
        const ps2x::vu1c::FlagEntry &pending = x.flags[(x.flagHead + index) & 15u];
        FlagPipelineEntry &entry = m_flagPipeline[index];
        entry = {};
        entry.valid = true;
        entry.issueCycle = pending.issue;
        entry.readyCycle = pending.ready;
        entry.mac = pending.mac;
        entry.status = pending.status;
        entry.extraSticky = pending.extraSticky;
        entry.clip = pending.clip;
        entry.writesMac = pending.writesMac;
        entry.writesStatus = pending.writesStatus;
        entry.writesSticky = pending.writesSticky;
        entry.writesClip = pending.writesClip;
        m_flagSlots |= 1u << index;
        noteCommitCycle(entry.readyCycle);
    }
    if (x.fdivPending)
    {
        m_fdiv.valid = true;
        m_fdiv.readyCycle = x.fdivReady;
        m_fdiv.value = x.fdivValue;
        m_fdiv.statusDi = x.fdivStatusDi;
        noteCommitCycle(m_fdiv.readyCycle);
    }
    for (uint32_t reg = 1; reg < 32u; ++reg)
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            const uint64_t ready = x.vfReady[reg][component];
            if (ready <= m_cycle)
                continue;
            uint8_t lanes = 0u;
            for (uint32_t other = component; other < 4u; ++other)
            {
                if (x.vfReady[reg][other] == ready)
                    lanes |= laneForComponent(other);
            }
            bool alreadyQueued = false;
            for (uint32_t earlier = 0; earlier < component; ++earlier)
                alreadyQueued = alreadyQueued || x.vfReady[reg][earlier] == ready;
            if (!alreadyQueued)
                queueVfWrite(static_cast<uint8_t>(reg), lanes, m_state.vf[reg], static_cast<uint32_t>(ready - m_cycle));
        }
    }
    for (uint32_t reg = 1; reg < 16u; ++reg)
    {
        if (x.viReady[reg] > m_cycle)
            queueViWrite(static_cast<uint8_t>(reg), m_state.vi[reg], static_cast<uint32_t>(x.viReady[reg] - m_cycle));
    }
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint64_t ready = x.accReady[component];
        if (ready <= m_cycle)
            continue;
        uint8_t lanes = 0u;
        for (uint32_t other = component; other < 4u; ++other)
        {
            if (x.accReady[other] == ready)
                lanes |= laneForComponent(other);
        }
        bool alreadyQueued = false;
        for (uint32_t earlier = 0; earlier < component; ++earlier)
            alreadyQueued = alreadyQueued || x.accReady[earlier] == ready;
        if (!alreadyQueued)
            queueAccWrite(lanes, m_state.acc, static_cast<uint32_t>(ready - m_cycle));
    }
    if (x.xgActive)
    {
        m_xgkick.active = true;
        m_xgkick.sourceAddress = x.xgSource;
        m_xgkick.copiedBytes = x.xgCopied;
        m_xgkick.currentTagEnd = x.xgTagEnd;
        m_xgkick.totalBytes = x.xgTotal;
        m_xgkick.currentTagEop = x.xgTagEop;
        m_xgkick.issueCycle = x.xgIssue;
        // Credit after the copy of cycle m_cycle: odd distance = a qword was just taken.
        m_xgkick.cycleCredit = ((m_cycle - x.xgIssue) % 2u) == 0u ? 1u : 0u;
    }
}

bool VU1Interpreter::runCompiled(const ps2x::vu1c::Program &program, uint8_t *vuData, uint32_t dataSize,
                                 uint64_t budgetEnd, std::vector<uint8_t> *gifCapture)
{
    if (!m_compiledExec)
        m_compiledExec = std::make_unique<ps2x::vu1c::Exec>();
    ps2x::vu1c::Exec &x = *m_compiledExec;
    exportCompiledState(x, vuData, dataSize, budgetEnd);
    if (gifCapture)
    {
        x.submitGif = captureGifPacket;
        x.gifUser = gifCapture;
    }
    program.run(x, m_state.pc);
    if (x.exit == ps2x::vu1c::Exit::Refused)
    {
        // The image matched, but the translator did not see this entry point.
        static std::mutex s_refusedMutex;
        static std::set<std::pair<const ps2x::vu1c::Program *, uint32_t>> s_refused;
        std::lock_guard<std::mutex> lock(s_refusedMutex);
        if (s_refused.insert({&program, m_state.pc}).second)
            std::fprintf(stderr, "[vu1] %s has no entry at pc=%04x; the interpreter runs it\n", program.name,
                         m_state.pc);
        return false;
    }
    importCompiledState(x);
    return x.exit == ps2x::vu1c::Exit::Ended;
}

void VU1Interpreter::beginCompiledVerification(const ps2x::vu1c::Program &program, uint8_t *vuData,
                                               uint32_t dataSize, uint64_t budgetEnd)
{
    if (!m_compiledVerification)
        m_compiledVerification = std::make_unique<CompiledVerification>();
    CompiledVerification &v = *m_compiledVerification;
    v.program = &program;
    v.entryPc = m_state.pc;
    v.state = m_state;
    v.cycle = m_cycle;
    v.workingClip = m_workingClip;
    v.vfReady = m_vfReady;
    v.viReady = m_viReady;
    v.accReady = m_accReady;
    v.backupValid = m_viBranchBackupValid;
    v.backupReg = m_viBranchBackupReg;
    v.backupValue = m_viBranchBackupValue;
    v.memoryBefore.assign(vuData, vuData + dataSize);
    v.compiledPackets.clear();

    runCompiled(program, vuData, dataSize, budgetEnd, &v.compiledPackets);
    v.compiledExit = m_compiledExec->exit;
    if (v.compiledExit == ps2x::vu1c::Exit::Refused)
    {
        v.active = false;
        return;
    }
    v.compiledState = m_state;
    v.compiledCycle = m_cycle;
    v.compiledWorkingClip = m_workingClip;
    v.compiledMemory.assign(vuData, vuData + dataSize);

    // Rewind and let the interpreter produce the reference result (and the real GIF output).
    m_state = v.state;
    m_cycle = v.cycle;
    m_workingClip = v.workingClip;
    resetScheduler();
    m_workingClip = v.workingClip;
    m_vfReady = v.vfReady;
    m_viReady = v.viReady;
    m_accReady = v.accReady;
    m_viBranchBackupValid = v.backupValid;
    m_viBranchBackupReg = v.backupReg;
    m_viBranchBackupValue = v.backupValue;
    std::memcpy(vuData, v.memoryBefore.data(), dataSize);
    v.interpreterPackets.clear();
    m_gifCapture = &v.interpreterPackets;
    v.active = true;
}

void VU1Interpreter::finishCompiledVerification(uint8_t *vuData, uint32_t dataSize, bool programEnded)
{
    CompiledVerification &v = *m_compiledVerification;
    m_gifCapture = nullptr;
    v.active = false;
    if (v.compiledExit != ps2x::vu1c::Exit::Ended || !programEnded)
    {
        ++v.handoffs;
        return;
    }
    ++v.verified;

    char what[160] = {};
    const VU1State &c = v.compiledState;
    const VU1State &r = m_state;
    if (std::memcmp(c.vf, r.vf, sizeof(r.vf)) != 0)
    {
        for (uint32_t reg = 0; reg < 32u && !what[0]; ++reg)
            for (uint32_t component = 0; component < 4u && !what[0]; ++component)
                if (floatBits(c.vf[reg][component]) != floatBits(r.vf[reg][component]))
                    std::snprintf(what, sizeof(what), "vf%u[%u] compiled=%08x interpreter=%08x", reg, component,
                                  floatBits(c.vf[reg][component]), floatBits(r.vf[reg][component]));
    }
    else if (std::memcmp(c.vi, r.vi, sizeof(r.vi)) != 0)
    {
        for (uint32_t reg = 0; reg < 16u && !what[0]; ++reg)
            if (c.vi[reg] != r.vi[reg])
                std::snprintf(what, sizeof(what), "vi%u compiled=%d interpreter=%d", reg, c.vi[reg], r.vi[reg]);
    }
    else if (std::memcmp(c.acc, r.acc, sizeof(r.acc)) != 0)
        std::snprintf(what, sizeof(what), "acc");
    else if (floatBits(c.q) != floatBits(r.q))
        std::snprintf(what, sizeof(what), "q compiled=%08x interpreter=%08x", floatBits(c.q), floatBits(r.q));
    else if (floatBits(c.i) != floatBits(r.i))
        std::snprintf(what, sizeof(what), "i");
    else if (c.mac != r.mac)
        std::snprintf(what, sizeof(what), "mac compiled=%04x interpreter=%04x", c.mac, r.mac);
    else if (c.status != r.status)
        std::snprintf(what, sizeof(what), "status compiled=%03x interpreter=%03x", c.status, r.status);
    else if (c.clip != r.clip)
        std::snprintf(what, sizeof(what), "clip compiled=%06x interpreter=%06x", c.clip, r.clip);
    else if (v.compiledWorkingClip != m_workingClip)
        std::snprintf(what, sizeof(what), "working clip");
    else if (c.pc != r.pc)
        std::snprintf(what, sizeof(what), "pc compiled=%04x interpreter=%04x", c.pc, r.pc);
    else if (v.compiledCycle != m_cycle)
        std::snprintf(what, sizeof(what), "cycles compiled=%llu interpreter=%llu",
                      static_cast<unsigned long long>(v.compiledCycle - v.cycle),
                      static_cast<unsigned long long>(m_cycle - v.cycle));
    else if (v.compiledMemory.size() != dataSize || std::memcmp(v.compiledMemory.data(), vuData, dataSize) != 0)
    {
        uint32_t offset = 0;
        while (offset < dataSize && v.compiledMemory[offset] == vuData[offset])
            ++offset;
        std::snprintf(what, sizeof(what), "vu memory at 0x%04x", offset);
    }
    else if (v.compiledPackets != v.interpreterPackets)
        std::snprintf(what, sizeof(what), "gif packets compiled=%zu bytes interpreter=%zu bytes",
                      v.compiledPackets.size(), v.interpreterPackets.size());

    if (what[0])
    {
        ++v.mismatches;
        if (v.mismatches <= 200u)
            std::fprintf(stderr, "[vu1-verify] MISMATCH %s entry=%04x: %s\n", v.program->name, v.entryPc, what);
    }
    if ((v.verified & 0xFFFu) == 0u)
        std::fprintf(stderr, "[vu1-verify] verified=%llu mismatches=%llu handoffs=%llu\n",
                     static_cast<unsigned long long>(v.verified), static_cast<unsigned long long>(v.mismatches),
                     static_cast<unsigned long long>(v.handoffs));
}
