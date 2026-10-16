#include "runtime/gs/gs_hud_layout.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace ps2x::hud
{
    namespace
    {
        enum class Layout
        {
            Off,
            Centre,
            Edges,
        };

        // Written once by enable() at start-up, before the game draws.
        Layout g_layout = Layout::Off;
        float g_factor = 1.0f;
        Rule g_rule{};

        // One vertex write inside the packet: where its X sits and the offset it is relative to.
        struct Vertex
        {
            uint32_t offset;
            uint16_t x;
            uint16_t ofx;
        };

        // The vertices between two PRIM writes, decided together so a quad is never torn apart.
        struct Run
        {
            uint32_t first = 0;
            uint32_t count = 0;
            uint32_t width = 0;
            int32_t x0 = INT32_MAX;   // screen X range, 12.4
            int32_t x1 = INT32_MIN;
            uint32_t fbp = 0;
            uint32_t tbp = 0;
            bool tme = false;
            bool hud = true;
        };

        struct Walker
        {
            GsState state;
            std::vector<Vertex> &vertices;
            std::vector<Run> &runs;
            bool open = false;

            void closeRun() { open = false; }

            void vertex(uint32_t offset, uint16_t x, uint32_t z)
            {
                const uint32_t ctx = (state.prmodeCont ? state.primCtxt : state.prmodeCtxt) ? 1u : 0u;
                if (!open)
                {
                    Run run;
                    run.first = static_cast<uint32_t>(vertices.size());
                    run.width = state.width[ctx];
                    run.fbp = state.fbp[ctx];
                    run.tbp = state.tbp[ctx];
                    run.tme = state.prmodeCont ? state.primTme : state.prmodeTme;
                    run.hud = run.tme ? run.tbp >= g_rule.minTextureBase : g_rule.untexturedHud;
                    runs.push_back(run);
                    open = true;
                }
                Run &run = runs.back();
                const int32_t screenX = static_cast<int32_t>(x) - static_cast<int32_t>(state.ofx[ctx]);
                run.x0 = std::min(run.x0, screenX);
                run.x1 = std::max(run.x1, screenX);
                if (z < g_rule.minZ)
                    run.hud = false;
                vertices.push_back({offset, x, state.ofx[ctx]});
                ++run.count;
            }

            // A register written whole (A+D, REGLIST): `offset` is where its 64-bit value sits.
            void reg(uint8_t address, uint64_t value, uint32_t offset)
            {
                switch (address)
                {
                case 0x00:   // PRIM
                    state.primCtxt = ((value >> 9) & 1u) != 0;
                    state.primTme = ((value >> 4) & 1u) != 0;
                    closeRun();
                    break;
                case 0x06:   // TEX0_1
                case 0x07:   // TEX0_2
                    state.tbp[address - 0x06u] = static_cast<uint32_t>(value & 0x3FFFu);
                    break;
                case 0x04:   // XYZF2
                case 0x0C:   // XYZF3
                    vertex(offset, static_cast<uint16_t>(value), static_cast<uint32_t>((value >> 32) & 0xFFFFFFu));
                    break;
                case 0x05:   // XYZ2
                case 0x0D:   // XYZ3
                    vertex(offset, static_cast<uint16_t>(value), static_cast<uint32_t>(value >> 32));
                    break;
                case 0x18:   // XYOFFSET_1
                case 0x19:   // XYOFFSET_2
                    state.ofx[address - 0x18u] = static_cast<uint16_t>(value);
                    break;
                case 0x1A:   // PRMODECONT
                    state.prmodeCont = (value & 1u) != 0;
                    break;
                case 0x1B:   // PRMODE
                    state.prmodeCtxt = ((value >> 9) & 1u) != 0;
                    state.prmodeTme = ((value >> 4) & 1u) != 0;
                    break;
                case 0x4C:   // FRAME_1
                case 0x4D:   // FRAME_2
                    state.width[address - 0x4Cu] = static_cast<uint32_t>((value >> 16) & 0x3Fu) * 64u;
                    state.fbp[address - 0x4Cu] = static_cast<uint32_t>(value & 0x1FFu);
                    break;
                default:
                    break;
                }
            }

            void packed(uint8_t desc, uint64_t lo, uint64_t hi, uint32_t offset)
            {
                switch (desc)
                {
                case 0x00:
                    reg(0x00, lo & 0x7FFu, offset);
                    break;
                case 0x04:   // XYZF2
                case 0x0C:   // XYZF3
                    vertex(offset, static_cast<uint16_t>(lo), static_cast<uint32_t>((hi >> 4) & 0xFFFFFFu));
                    break;
                case 0x05:   // XYZ2
                case 0x0D:   // XYZ3
                    vertex(offset, static_cast<uint16_t>(lo), static_cast<uint32_t>(hi));
                    break;
                case 0x0E:   // A+D
                    reg(static_cast<uint8_t>(hi & 0xFFu), lo, offset);
                    break;
                case 0x01:
                case 0x02:
                case 0x03:
                case 0x0A:
                case 0x0F:
                    break;
                default:
                    reg(desc, lo, offset);
                    break;
                }
            }

            void walk(const uint8_t *data, uint32_t sizeBytes)
            {
                const auto load64 = [data](uint32_t offset)
                {
                    uint64_t value;
                    std::memcpy(&value, data + offset, 8);
                    return value;
                };
                uint32_t offset = 0;
                while (offset + 16u <= sizeBytes)
                {
                    const uint64_t tagLo = load64(offset);
                    const uint64_t tagHi = load64(offset + 8u);
                    offset += 16u;
                    const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
                    const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 3u);
                    uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
                    if (nreg == 0u)
                        nreg = 16u;
                    if ((tagLo >> 46) & 1u)
                        reg(0x00, (tagLo >> 47) & 0x7FFu, 0);
                    if (flg == 0u)
                    {
                        for (uint32_t loop = 0; loop < nloop; ++loop)
                            for (uint32_t r = 0; r < nreg; ++r)
                            {
                                if (offset + 16u > sizeBytes)
                                    return;
                                packed(static_cast<uint8_t>((tagHi >> (r * 4u)) & 0xFu), load64(offset),
                                       load64(offset + 8u), offset);
                                offset += 16u;
                            }
                    }
                    else if (flg == 1u)
                    {
                        for (uint32_t loop = 0; loop < nloop; ++loop)
                            for (uint32_t r = 0; r < nreg; ++r)
                            {
                                if (offset + 8u > sizeBytes)
                                    return;
                                reg(static_cast<uint8_t>((tagHi >> (r * 4u)) & 0xFu), load64(offset), offset);
                                offset += 8u;
                            }
                        if ((nloop * nreg) & 1u)
                            offset += 8u;
                    }
                    else
                    {
                        offset += std::min(nloop * 16u, sizeBytes - offset);
                    }
                }
            }
        };

        enum class Anchor
        {
            Left,
            Centre,
            Right,
        };

        // Where screen X (pixels, frame of width w) goes in the squeezed HUD.
        float mapX(float x, float w, Anchor anchor)
        {
            const float k = g_factor;
            switch (anchor)
            {
            case Anchor::Left:
                return x * k;
            case Anchor::Right:
                return w - (w - x) * k;
            default:
                return w * 0.5f + (x - w * 0.5f) * k;
            }
        }

        // Per frame buffer (FBP), the last vsync + 1 with a scene packet drawing into it (0 = never).
        // The HUD over it is squeezed, and the frame is shown wide, only while it has a scene.
        constexpr uint64_t kSceneTicks = 4;
        std::array<std::atomic<uint64_t>, 512> g_sceneTick{};
        std::atomic<bool> g_sceneShown{true};

        bool sceneRecent(uint32_t fbp, uint64_t tick)
        {
            const uint64_t mark = g_sceneTick[fbp & 511u].load(std::memory_order_relaxed);
            return mark != 0 && tick + 1 - mark <= kSceneTicks;
        }

        // Last vsync with a menu-sized panel on the HUD path: the edges layout centres everything
        // for a few vsyncs after one, so a menu is never split between the two sides.
        uint64_t g_lastPanelTick = 0;
        bool g_seenPanel = false;
        constexpr uint64_t kPanelHoldTicks = 4;
    }

    void enable(float factor, const Rule &rule)
    {
        const char *value = std::getenv("PS2X_WIDESCREEN_HUD");
        const std::string mode = value ? value : "";
        g_rule = rule;
        g_factor = factor;
        g_layout = (mode == "stretch" || mode == "off" || !(factor > 0.0f && factor < 1.0f)) ? Layout::Off
                   : mode == "edges"                                                         ? Layout::Edges
                                                                                             : Layout::Centre;
        std::fprintf(stderr, "[hud] widescreen HUD: %s, x%.3f\n",
                     g_layout == Layout::Off ? "stretched" : g_layout == Layout::Edges ? "edges" : "centred", factor);
    }

    bool active()
    {
        return g_layout != Layout::Off;
    }

    void noteSwap(uint32_t displayFbp, uint64_t tick)
    {
        if (g_layout == Layout::Off)
            return;
        const bool scene = sceneRecent(displayFbp, tick);
        static const bool s_trace = std::getenv("PS2X_WIDESCREEN_HUD_TRACE") != nullptr;
        if (s_trace && scene != g_sceneShown.load(std::memory_order_relaxed))
            std::fprintf(stderr, "[hud] t=%llu shown at %s (fbp %u)\n", static_cast<unsigned long long>(tick),
                         scene ? "wide" : "4:3", displayFbp);
        g_sceneShown.store(scene, std::memory_order_relaxed);
    }

    bool pillarbox()
    {
        return g_layout != Layout::Off && !g_sceneShown.load(std::memory_order_relaxed);
    }

    const uint8_t *apply(uint8_t path, const uint8_t *data, uint32_t sizeBytes, const GsState &state,
                         std::vector<uint8_t> &scratch)
    {
        if (g_layout == Layout::Off || path > 7u)
            return data;
        if ((g_rule.scenePathMask >> path) & 1u)
        {
            const bool ctx = state.prmodeCont ? state.primCtxt : state.prmodeCtxt;
            g_sceneTick[state.fbp[ctx ? 1 : 0] & 511u].store(state.tick + 1, std::memory_order_relaxed);
        }
        if (!((g_rule.pathMask >> path) & 1u))
            return data;
        thread_local std::vector<Vertex> vertices;
        thread_local std::vector<Run> runs;
        vertices.clear();
        runs.clear();
        Walker walker{state, vertices, runs};
        walker.walk(data, sizeBytes);

        // PS2X_WIDESCREEN_HUD_TRACE=<first vsync>:<count>[,...]: the runs of each packet in those windows.
        static const auto s_trace = []()
        {
            std::vector<std::pair<uint64_t, uint64_t>> windows;
            const char *value = std::getenv("PS2X_WIDESCREEN_HUD_TRACE");
            while (value && *value)
            {
                unsigned long long first = 0, count = 0;
                if (std::sscanf(value, "%llu:%llu", &first, &count) == 2)
                    windows.emplace_back(first, first + count);
                value = std::strchr(value, ',');
                if (value)
                    ++value;
            }
            return windows;
        }();
        for (const auto &window : s_trace)
        {
            if (state.tick < window.first || state.tick >= window.second || runs.empty())
                continue;
            std::fprintf(stderr, "[hud] t=%llu path=%u size=%u runs=%zu:", static_cast<unsigned long long>(state.tick),
                         static_cast<uint32_t>(path), sizeBytes, runs.size());
            for (const Run &run : runs)
                std::fprintf(stderr, " %c%u[%.0f..%.0f f%u t%d]", run.hud ? 'H' : '-', run.count, run.x0 / 16.0f,
                             run.x1 / 16.0f, run.fbp, run.tme ? static_cast<int>(run.tbp) : -1);
            std::fprintf(stderr, "\n");
        }

        // Runs that span the whole frame (a fade, a band behind the text) keep covering it.
        const auto fullWidth = [](const Run &run)
        {
            const int32_t width16 = static_cast<int32_t>(run.width) * 16;
            return run.x0 <= 4 * 16 && run.x1 >= width16 - 4 * 16;
        };
        bool centreAll = g_layout == Layout::Centre;
        if (!centreAll)
        {
            for (const Run &run : runs)
                if (run.hud && run.width && !fullWidth(run) && sceneRecent(run.fbp, state.tick) &&
                    run.x1 - run.x0 >= static_cast<int32_t>(g_rule.panelWidth * static_cast<float>(run.width) * 16.0f))
                {
                    g_lastPanelTick = state.tick;
                    g_seenPanel = true;
                }
            centreAll = g_seenPanel && state.tick - g_lastPanelTick <= kPanelHoldTicks;
        }

        bool copied = false;
        for (const Run &run : runs)
        {
            if (!run.hud || run.count == 0u || run.width == 0u || fullWidth(run) || !sceneRecent(run.fbp, state.tick))
                continue;
            if (!copied)
            {
                scratch.assign(data, data + sizeBytes);
                copied = true;
            }
            const float w = static_cast<float>(run.width);
            // A piece of HUD moves whole to its side of the screen.
            const float centre = (static_cast<float>(run.x0) + static_cast<float>(run.x1)) / 32.0f;
            const Anchor anchor = centreAll ? Anchor::Centre : centre < g_rule.edgeSplit * w ? Anchor::Left : Anchor::Right;
            for (uint32_t i = 0; i < run.count; ++i)
            {
                const Vertex &v = vertices[run.first + i];
                const float x = (static_cast<float>(v.x) - static_cast<float>(v.ofx)) / 16.0f;
                const float mapped = std::round(mapX(x, w, anchor) * 16.0f) + static_cast<float>(v.ofx);
                const uint16_t nx = static_cast<uint16_t>(std::clamp(mapped, 0.0f, 65535.0f));
                std::memcpy(scratch.data() + v.offset, &nx, 2);
            }
        }
        return copied ? scratch.data() : data;
    }
}
