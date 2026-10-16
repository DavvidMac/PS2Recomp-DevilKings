#include "runtime/gs/gs_pgs.h"
#include "runtime/ps2_memory.h"

#include "context.hpp"
#include "device.hpp"
#include "gs_interface.hpp"
#include "thread_id.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace ps2x::pgs
{
    namespace
    {
        using namespace Vulkan;
        using namespace ParallelGS;

        // The CPU backend parses the same stream and raises SIGNAL, FINISH and LABEL in CSR,
        // so the backend's own events are not forwarded to the guest.
        struct IgnoreSignals final : SignalInterface
        {
            bool on_signal(uint64_t) override { return false; }
            bool on_finish(uint64_t) override { return false; }
            bool on_label(uint64_t) override { return false; }
        };

        struct Readback
        {
            BufferHandle buffer;
            Fence fence;
            ImageHandle image;
            uint32_t width = 0;
            uint32_t height = 0;
            VkFormat format = VK_FORMAT_UNDEFINED;
            bool field = false;
            bool pending = false;
        };

        struct State
        {
            std::mutex mutex;
            bool started = false;
            bool failed = false;
            Context context;
            Device device;
            GSInterface gs;
            IgnoreSignals signals;
            GSRegisters *regs = nullptr;
            uint32_t scanoutShift = 0;
            uint32_t field = 0;
            Readback ring[3];
            uint32_t next = 0;
            std::vector<uint8_t> frame;
            uint32_t frameWidth = 0;
            uint32_t frameHeight = 0;
            bool frameField = false;
            bool frameFresh = false;
        };

        // Never destroyed: emulation threads may still send packets while the process exits.
        State &state()
        {
            static State *s = new State;
            return *s;
        }

        // -1 until the first packet starts the backend, then 1 (running) or 0 (failed).
        std::atomic<int> g_started{-1};

        bool wanted()
        {
            static const bool s_wanted = []()
            {
                const char *value = std::getenv("PS2X_RENDERER");
                return value && std::strcmp(value, "vulkan") == 0;
            }();
            return s_wanted;
        }

        // PS2X_PGS_SCALE: the scanout is 1, 2 or 4 times the native size (paraLLEl-GS scans out
        // at power-of-two sizes); "aa" draws more samples per pixel at the same size, which
        // smooths edges. A 2x scanout needs 4 samples per pixel and a 4x one 16.
        struct Quality
        {
            SuperSampling samples;
            uint32_t scanoutShift;
            const char *name;
        };

        Quality internalQuality()
        {
            const char *value = std::getenv("PS2X_PGS_SCALE");
            const std::string scale = value ? value : "1";
            if (scale == "1aa")
                return {SuperSampling::X4, 0u, "1x with 4 samples per pixel"};
            if (scale == "2aa")
                return {SuperSampling::X8, 1u, "2x with 8 samples per pixel"};
            const int factor = std::atoi(scale.c_str());
            if (factor >= 4)
                return {SuperSampling::X16, 2u, "4x"};
            if (factor >= 2)
                return {SuperSampling::X4, 1u, "2x"};
            return {SuperSampling::X1, 0u, "1x"};
        }

        // Granite keys its command pools by a per-thread index. Every call is serialised by the
        // state mutex, so all the threads that feed the GS share index 0.
        void registerThread()
        {
            static thread_local bool s_registered = false;
            if (!s_registered)
            {
                Util::register_thread_index(0);
                s_registered = true;
            }
        }

        bool startLocked(State &s)
        {
            registerThread();
            if (s.started)
                return true;
            if (s.failed)
                return false;
            s.failed = true;
            const auto fail = [](const char *why)
            {
                std::fprintf(stderr, "[pgs] %s: the Vulkan renderer is unavailable, using the software renderer\n", why);
                g_started.store(0);
                return false;
            };
            if (!Context::init_loader(nullptr))
                return fail("Vulkan loader init failed");
            s.context.set_num_thread_indices(1);
            if (!s.context.init_instance_and_device(nullptr, 0, nullptr, 0,
                                                    CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT |
                                                        CONTEXT_CREATION_ENABLE_DESCRIPTOR_HEAP_BIT |
                                                        CONTEXT_CREATION_ENABLE_DESCRIPTOR_BUFFER_BIT))
                return fail("Vulkan instance/device init failed");
            s.device.set_context(s.context);
            s.device.init_frame_contexts(4);

            const Quality quality = internalQuality();
            GSOptions options = {};
            options.vram_size = 4u * 1024u * 1024u;
            options.super_sampling = quality.samples;
            options.super_sampled_textures = true;
            s.scanoutShift = quality.scanoutShift;
            if (!s.gs.init(&s.device, options))
                return fail("GSInterface init failed");
            s.gs.set_signal_interface(&s.signals);

            s.failed = false;
            s.started = true;
            g_started.store(1);
            std::fprintf(stderr, "[pgs] Vulkan renderer (paraLLEl-GS) on %s, internal resolution %s\n",
                         s.device.get_gpu_properties().deviceName, quality.name);
            return true;
        }

        // DISPLAY1/2 as the CRTC reads them. The runtime's sceGs display stubs write the frame
        // buffer's size: DW in pixels with MAGH 0, and DH in lines of the buffer. The CPU backend
        // reads them that way; the CRTC paraLLEl-GS emulates draws such a picture a fifth of a line
        // wide and, in field mode, half as tall. The console counts DW in video clocks (VCK, 2560
        // across an NTSC line) with MAGH+1 clocks per pixel, and in field mode (SMODE2 INT and
        // FFMD) DH in lines of the whole frame, each field showing half of them.
        uint64_t displayForCrtc(uint64_t display, uint64_t smode2)
        {
            constexpr uint32_t kLineClocks = 2560u;
            const uint32_t magh = static_cast<uint32_t>((display >> 23) & 0xFu);
            const uint32_t width = static_cast<uint32_t>((display >> 32) & 0xFFFu) + 1u;
            if (magh == 0u && width <= kLineClocks / 2u)
            {
                // The magnification the console's library picks for that width.
                const uint32_t clocksPerPixel = std::max(1u, std::min(16u, kLineClocks / width));
                const uint64_t dw = static_cast<uint64_t>(width * clocksPerPixel - 1u);
                display &= ~((0xFull << 23) | (0xFFFull << 32));
                display |= (static_cast<uint64_t>(clocksPerPixel - 1u) << 23) | (dw << 32);
            }
            const bool fieldMode = (smode2 & 3u) == 3u;
            const uint32_t height = static_cast<uint32_t>((display >> 44) & 0x7FFu) + 1u;
            if (fieldMode && height <= 256u)
            {
                display &= ~(0x7FFull << 44);
                display |= static_cast<uint64_t>(height * 2u - 1u) << 44;
            }
            return display;
        }

        void copyPrivilegedLocked(State &s)
        {
            const GSRegisters *r = s.regs;
            if (!r)
                return;
            PrivRegisterState &p = s.gs.get_priv_register_state();
            const auto put = [](auto &dst, uint64_t value) { std::memcpy(&dst, &value, sizeof(value)); };
            // NTSC when the game never programmed the CRTC itself.
            constexpr uint64_t kNtscSmode1 = 0x0000000740814504ull;
            put(p.pmode, r->pmode);
            put(p.smode1, r->smode1 ? r->smode1 : kNtscSmode1);
            put(p.smode2, r->smode2);
            put(p.srfsh, r->srfsh);
            put(p.synch1, r->synch1);
            put(p.synch2, r->synch2);
            put(p.syncv, r->syncv);
            put(p.dispfb1, r->dispfb1);
            put(p.display1, displayForCrtc(r->display1, r->smode2));
            put(p.dispfb2, r->dispfb2);
            put(p.display2, displayForCrtc(r->display2, r->smode2));
            put(p.extbuf, r->extbuf);
            put(p.extdata, r->extdata);
            put(p.extwrite, r->extwrite);
            put(p.bgcolor, r->bgcolor);
            put(p.csr, r->csr.load(std::memory_order_relaxed));
            put(p.imr, r->imr);
            put(p.busdir, r->busdir);
            put(p.siglblid, r->siglblid);
        }

        void consumeLocked(State &s, Readback &slot)
        {
            const auto *src = static_cast<const uint32_t *>(s.device.map_host_buffer(*slot.buffer, MEMORY_ACCESS_READ_BIT));
            const size_t count = static_cast<size_t>(slot.width) * slot.height;
            s.frame.resize(count * 4u);
            uint32_t *dst = reinterpret_cast<uint32_t *>(s.frame.data());
            const bool bgra = slot.format == VK_FORMAT_B8G8R8A8_UNORM || slot.format == VK_FORMAT_B8G8R8A8_SRGB;
            for (size_t i = 0; i < count; ++i)
            {
                const uint32_t pixel = src[i];
                dst[i] = bgra ? (((pixel >> 16) & 0xFFu) | (pixel & 0xFF00u) | ((pixel & 0xFFu) << 16) | 0xFF000000u)
                              : (pixel | 0xFF000000u);
            }
            s.device.unmap_host_buffer(*slot.buffer, MEMORY_ACCESS_READ_BIT);
            s.frameWidth = slot.width;
            s.frameHeight = slot.height;
            s.frameField = slot.field;
            s.frameFresh = true;
            slot.pending = false;
            slot.image.reset();
            slot.fence.reset();
        }

        // Copies the scanout to a host buffer and takes whichever earlier copy has landed,
        // without waiting: a copy still in flight two swaps later drops this frame instead.
        void readbackLocked(State &s, const ScanoutResult &result, bool field)
        {
            for (uint32_t k = 1; k <= 2; ++k)
            {
                Readback &older = s.ring[(s.next + k) % 3u];
                if (older.pending && older.fence && older.fence->wait_timeout(0))
                    consumeLocked(s, older);
            }
            Readback &slot = s.ring[s.next];
            if (slot.pending)
            {
                if (slot.fence && slot.fence->wait_timeout(0))
                    consumeLocked(s, slot);
                else
                    return;
            }
            const uint32_t width = result.image->get_width();
            const uint32_t height = result.image->get_height();
            if (!slot.buffer || slot.width != width || slot.height != height)
            {
                BufferCreateInfo info = {};
                info.size = VkDeviceSize(width) * height * 4u;
                info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                info.domain = BufferDomain::CachedHost;
                slot.buffer = s.device.create_buffer(info);
                slot.width = width;
                slot.height = height;
            }
            slot.format = result.image->get_format();
            slot.field = field;
            auto cmd = s.device.request_command_buffer();
            cmd->image_barrier(*result.image, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                               VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            cmd->copy_image_to_buffer(*slot.buffer, *result.image, 0, {}, {width, height, 1}, 0, 0,
                                      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
            cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
            Fence fence;
            s.device.submit(cmd, &fence);
            slot.fence = std::move(fence);
            slot.image = result.image;
            slot.pending = true;
            s.next = (s.next + 1u) % 3u;
        }
    }

    bool active()
    {
        const int started = g_started.load(std::memory_order_relaxed);
        return started < 0 ? wanted() : started == 1;
    }

    bool exclusive()
    {
        static const bool s_dual = []()
        {
            const char *value = std::getenv("PS2X_PGS_DUAL");
            return value && value[0] == '1';
        }();
        return !s_dual && active();
    }

    void setRegisters(GSRegisters *regs)
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.regs = regs;
    }

    void gifTransfer(uint32_t path, const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes < 16u || path > PathHost || !active())
            return;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!startLocked(s))
            return;
        s.gs.gif_transfer(path, data, sizeBytes & ~15u);
    }

    void vsync()
    {
        if (!active())
            return;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!startLocked(s) || !s.regs)
            return;
        s.gs.flush();
        copyPrivilegedLocked(s);
        VSyncInfo info = {};
        info.phase = s.field ^= 1u;
        info.dst_layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
        info.dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        info.dst_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        info.force_progressive = true;
        info.anti_blur = true;
        info.overscan = false;
        info.crtc_offsets = false;
        info.adapt_to_internal_horizontal_resolution = false;
        info.raw_circuit_scanout = true;
        info.high_resolution_scanout = s.scanoutShift >= 1u;
        info.high_resolution_scanout_shift = s.scanoutShift >= 2u ? 2u : 1u;
        // Field mode: hand the field over as drawn and let the presentation weave it, which
        // finds the field order from the picture. paraLLEl-GS's own deinterlacer needs the
        // phase of each field, and a game running below 60 frames per second does not show
        // its fields in VSync order, so the picture bobs.
        const bool field = (s.regs->smode2 & 3u) == 3u;
        info.skip_deinterlace = field;
        const ScanoutResult result = s.gs.vsync(info);
        static const bool s_trace = std::getenv("PS2X_PGS_TRACE") != nullptr;
        static uint32_t s_swaps = 0;
        if (s_trace && (s_swaps++ % 300u) == 0u)
        {
            std::fprintf(stderr,
                         "[pgs] swap %u: pmode=%llx smode1=%llx smode2=%llx dispfb1=%llx display1=%llx dispfb2=%llx "
                         "display2=%llx -> image %ux%u internal %ux%u mode %ux%u\n",
                         s_swaps, (unsigned long long)s.regs->pmode, (unsigned long long)s.regs->smode1,
                         (unsigned long long)s.regs->smode2, (unsigned long long)s.regs->dispfb1,
                         (unsigned long long)s.regs->display1, (unsigned long long)s.regs->dispfb2,
                         (unsigned long long)s.regs->display2, result.image ? result.image->get_width() : 0u,
                         result.image ? result.image->get_height() : 0u, result.internal_width, result.internal_height,
                         result.mode_width, result.mode_height);
        }
        if (result.image)
            readbackLocked(s, result, field);
    }

    bool takeFrame(std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height, bool &field)
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.frameFresh)
            return false;
        rgba.swap(s.frame);
        width = s.frameWidth;
        height = s.frameHeight;
        field = s.frameField;
        s.frameFresh = false;
        return true;
    }

    bool readVram(uint8_t *dst, uint32_t sizeBytes)
    {
        if (!dst || !active())
            return false;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!startLocked(s))
            return false;
        s.gs.flush();
        const size_t size = std::min<size_t>(sizeBytes, 4u * 1024u * 1024u);
        const void *vram = s.gs.map_vram_read(0, size);
        if (!vram)
            return false;
        std::memcpy(dst, vram, size);
        return true;
    }

    void shutdown()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.started)
            s.device.wait_idle();
    }
}
