#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace GSInternal;

namespace
{
    float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    // Rows are dealt to band threads in groups of 1 << kBandShift.
    constexpr uint32_t kBandShift = 2u;
    thread_local uint32_t s_bandIndex = 0u;
    thread_local uint32_t s_bandCount = 1u;
    thread_local bool s_inBand = false;

    inline bool ownsRow(int y)
    {
        return s_bandCount <= 1u || ((static_cast<uint32_t>(y) >> kBandShift) % s_bandCount) == s_bandIndex;
    }

    // Whether this band owns any row in [y0, y1] (rows are never negative here).
    inline bool ownsAnyRow(int y0, int y1)
    {
        if (s_bandCount <= 1u)
            return true;
        if (y1 < y0)
            return false;
        const uint32_t group0 = static_cast<uint32_t>(y0) >> kBandShift;
        const uint32_t group1 = static_cast<uint32_t>(y1) >> kBandShift;
        if (group1 - group0 + 1u >= s_bandCount)
            return true;
        for (uint32_t group = group0; group <= group1; ++group)
        {
            if (group % s_bandCount == s_bandIndex)
                return true;
        }
        return false;
    }

    // Diagnostics: PS2X_GS_STATS prints, about once a second, how queued draws
    // were batched for the band threads and where the GS threads spent time.
    bool gsStatsEnabled()
    {
        static const bool s_enabled = std::getenv("PS2X_GS_STATS") != nullptr;
        return s_enabled;
    }

    using GsStatsClock = std::chrono::steady_clock;

    uint64_t gsStatsNanoseconds(GsStatsClock::time_point start)
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(GsStatsClock::now() - start).count());
    }

    struct GsStats
    {
        static constexpr size_t kReasons = 8u;
        std::atomic<uint64_t> draws{0};
        std::atomic<uint64_t> triangles{0};
        std::atomic<uint64_t> sprites{0};
        std::atomic<uint64_t> area{0};
        std::atomic<uint64_t> batches{0};
        std::atomic<uint64_t> batchItems{0};
        std::atomic<uint64_t> flushes{0};
        std::atomic<uint64_t> inlineFlushes{0};
        std::atomic<uint64_t> flushedDraws{0};
        std::atomic<uint64_t> parallelDraws{0};
        std::atomic<uint64_t> reasons[kReasons]{};
        std::atomic<uint64_t> predecodeDraws{0};
        std::atomic<uint64_t> predecodeTiles{0};
        std::atomic<uint64_t> predecodeNs{0};
        std::atomic<uint64_t> lazyTilesWorker{0};
        std::atomic<uint64_t> lazyTilesBand{0};
        std::atomic<uint64_t> decodeWaitNs{0};
        std::atomic<uint64_t> workerBusyNs{0};
        std::atomic<uint64_t> workerRasterNs{0};
        std::atomic<uint64_t> bandWaitNs{0};
        std::atomic<uint64_t> presentNs{0};
        std::atomic<uint64_t> bandBusyNs{0};
        std::atomic<uint64_t> producerQueueWaitNs{0};
        std::atomic<uint64_t> producerPresentWaitNs{0};
        std::atomic<uint64_t> producerDrainWaitNs{0};
        std::atomic<uint64_t> producerDrains{0};
        std::atomic<uint64_t> cacheHits{0};
        std::atomic<uint64_t> cacheEpochResets{0};
        std::atomic<uint64_t> cacheMisses{0};
        std::atomic<uint64_t> cacheMissesClut{0};
        std::atomic<uint64_t> cacheEvictions{0};
        std::atomic<uint64_t> listActiveNs{0};
        std::atomic<uint64_t> lists{0};
    };

    GsStats &gsStats()
    {
        static GsStats s_stats;
        return s_stats;
    }

    thread_local bool s_onGsWorker = false;

    void maybePrintGsStats(uint32_t bandCount)
    {
        static GsStatsClock::time_point s_last = GsStatsClock::now();
        const GsStatsClock::time_point now = GsStatsClock::now();
        const double seconds = std::chrono::duration<double>(now - s_last).count();
        if (seconds < 1.0)
            return;
        s_last = now;
        GsStats &s = gsStats();
        const auto take = [](std::atomic<uint64_t> &value) { return value.exchange(0u, std::memory_order_relaxed); };
        const uint64_t draws = take(s.draws);
        const uint64_t triangles = take(s.triangles);
        const uint64_t sprites = take(s.sprites);
        const uint64_t area = take(s.area);
        const uint64_t batches = take(s.batches);
        const uint64_t batchItems = take(s.batchItems);
        const uint64_t flushes = take(s.flushes);
        const uint64_t inlineFlushes = take(s.inlineFlushes);
        const uint64_t flushedDraws = take(s.flushedDraws);
        const uint64_t parallelDraws = take(s.parallelDraws);
        uint64_t reasons[GsStats::kReasons];
        for (size_t i = 0; i < GsStats::kReasons; ++i)
            reasons[i] = take(s.reasons[i]);
        const uint64_t predecodeDraws = take(s.predecodeDraws);
        const uint64_t predecodeTiles = take(s.predecodeTiles);
        const double ms = 1.0e-6;
        const double predecodeMs = static_cast<double>(take(s.predecodeNs)) * ms;
        const uint64_t lazyTilesWorker = take(s.lazyTilesWorker);
        const uint64_t lazyTilesBand = take(s.lazyTilesBand);
        const double decodeWaitMs = static_cast<double>(take(s.decodeWaitNs)) * ms;
        const double workerBusyMs = static_cast<double>(take(s.workerBusyNs)) * ms;
        const double workerRasterMs = static_cast<double>(take(s.workerRasterNs)) * ms;
        const double bandWaitMs = static_cast<double>(take(s.bandWaitNs)) * ms;
        const double presentMs = static_cast<double>(take(s.presentNs)) * ms;
        const double bandBusyMs = static_cast<double>(take(s.bandBusyNs)) * ms;
        const double queueWaitMs = static_cast<double>(take(s.producerQueueWaitNs)) * ms;
        const double presentWaitMs = static_cast<double>(take(s.producerPresentWaitNs)) * ms;
        const double drainWaitMs = static_cast<double>(take(s.producerDrainWaitNs)) * ms;
        const uint64_t drains = take(s.producerDrains);
        const uint64_t cacheHits = take(s.cacheHits);
        const uint64_t cacheEpochResets = take(s.cacheEpochResets);
        const uint64_t cacheMisses = take(s.cacheMisses);
        const uint64_t cacheMissesClut = take(s.cacheMissesClut);
        const uint64_t cacheEvictions = take(s.cacheEvictions);
        const double listActiveMs = static_cast<double>(take(s.listActiveNs)) * ms;
        const uint64_t lists = take(s.lists);
        const double wall = seconds * 1000.0;
        std::fprintf(stderr,
                     "[gs-stats] %.2fs draws=%llu (tri=%llu sprite=%llu) area=%llu batches=%llu items=%llu "
                     "flushes=%llu inline=%llu draws/flush=%.1f parallelDraws=%llu "
                     "why[hazard=%llu clut=%llu size=%llu pinned=%llu xfer=%llu upload=%llu present=%llu end=%llu] "
                     "predecode=%llu draws/%llu tiles/%.1fms lazyTiles worker=%llu band=%llu decodeWait=%.1fms "
                     "worker busy=%.0f%% raster=%.0f%% bandWait=%.0f%% present=%.0f%% bands busy=%.0f%% of %u "
                     "| EE waits queue=%.0f%% present=%.0f%% drain=%.0f%% (%llu) "
                     "| texcache hit=%llu reset=%llu miss=%llu (clut %llu) evict=%llu "
                     "| lists=%llu in flight %.0f%% (bands busy %.0f%% of that)\n",
                     seconds,
                     static_cast<unsigned long long>(draws), static_cast<unsigned long long>(triangles),
                     static_cast<unsigned long long>(sprites), static_cast<unsigned long long>(area),
                     static_cast<unsigned long long>(batches), static_cast<unsigned long long>(batchItems),
                     static_cast<unsigned long long>(flushes), static_cast<unsigned long long>(inlineFlushes),
                     flushes ? static_cast<double>(flushedDraws) / static_cast<double>(flushes) : 0.0,
                     static_cast<unsigned long long>(parallelDraws),
                     static_cast<unsigned long long>(reasons[0]), static_cast<unsigned long long>(reasons[1]),
                     static_cast<unsigned long long>(reasons[2]), static_cast<unsigned long long>(reasons[3]),
                     static_cast<unsigned long long>(reasons[4]), static_cast<unsigned long long>(reasons[5]),
                     static_cast<unsigned long long>(reasons[6]), static_cast<unsigned long long>(reasons[7]),
                     static_cast<unsigned long long>(predecodeDraws), static_cast<unsigned long long>(predecodeTiles),
                     predecodeMs, static_cast<unsigned long long>(lazyTilesWorker),
                     static_cast<unsigned long long>(lazyTilesBand), decodeWaitMs,
                     100.0 * workerBusyMs / wall, 100.0 * workerRasterMs / wall, 100.0 * bandWaitMs / wall,
                     100.0 * presentMs / wall,
                     bandCount > 1u ? 100.0 * bandBusyMs / (wall * static_cast<double>(bandCount)) : 0.0,
                     bandCount > 1u ? bandCount : 0u,
                     100.0 * queueWaitMs / wall, 100.0 * presentWaitMs / wall, 100.0 * drainWaitMs / wall,
                     static_cast<unsigned long long>(drains), static_cast<unsigned long long>(cacheHits),
                     static_cast<unsigned long long>(cacheEpochResets), static_cast<unsigned long long>(cacheMisses),
                     static_cast<unsigned long long>(cacheMissesClut), static_cast<unsigned long long>(cacheEvictions),
                     static_cast<unsigned long long>(lists), 100.0 * listActiveMs / wall,
                     listActiveMs > 0.0 && bandCount > 1u
                         ? 100.0 * bandBusyMs / (listActiveMs * static_cast<double>(bandCount))
                         : 0.0);
    }

    int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    bool isFourBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    bool isEightBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H;
    }

}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = dh + 1u;
        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
                sourceY = height - 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        source.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }
}

namespace
{
    // Diagnostics: PS2X_GS_RECORD=<path>[:<vsync>[:<presents>]] writes every call the
    // GS frontend makes on the backend to <path>, from the first presentation at or
    // after VSync <vsync> (default 0) for <presents> presentations (default 30). The
    // file starts with VRAM and the palette, so ps2_gs_replay can rasterise the same
    // frames outside the game.
    enum class GsRecordKind : uint8_t
    {
        Submit = 1,
        LoadClut = 2,
        TextureFlush = 3,
        BeginTransfer = 4,
        UploadImage = 5,
        Present = 6,
        ClearFramebuffer = 7,
        WriteVram = 8,
        ConsumeLocalToHost = 9,
        Reset = 10,
    };

    static_assert(std::is_trivially_copyable_v<GSPrimitiveBatch>);
    static_assert(std::is_trivially_copyable_v<GSTex0Reg>);
    static_assert(std::is_trivially_copyable_v<GSTexClutReg>);
    static_assert(std::is_trivially_copyable_v<GSTransferCommand>);
    static_assert(std::is_trivially_copyable_v<GSPresentationRequest>);
    static_assert(std::is_trivially_copyable_v<GSContext>);

    struct GsRecorder
    {
        std::mutex mutex;
        std::FILE *file = nullptr;
        const void *owner = nullptr;
        std::string path;
        uint64_t fromVsync = 0u;
        uint64_t presents = 30u;
        uint64_t written = 0u;
        bool done = false;
    };

    GsRecorder *gsRecorder()
    {
        static GsRecorder *const s_recorder = []() -> GsRecorder *
        {
            const char *value = std::getenv("PS2X_GS_RECORD");
            if (!value || !*value)
                return nullptr;
            auto *recorder = new GsRecorder();
            std::string spec(value);
            // Up to two numeric fields after the path (which may hold a drive colon).
            uint64_t numbers[2] = {};
            size_t count = 0u;
            while (count < 2u)
            {
                const size_t colon = spec.rfind(':');
                if (colon == std::string::npos || colon + 1u >= spec.size() ||
                    spec.find_first_not_of("0123456789", colon + 1u) != std::string::npos)
                    break;
                numbers[count++] = std::strtoull(spec.c_str() + colon + 1u, nullptr, 10);
                spec.resize(colon);
            }
            if (count == 2u)
            {
                recorder->fromVsync = numbers[1];
                recorder->presents = std::max<uint64_t>(numbers[0], 1u);
            }
            else if (count == 1u)
            {
                recorder->fromVsync = numbers[0];
            }
            recorder->path = spec;
            return recorder;
        }();
        return s_recorder;
    }

    void gsRecordWrite(const void *owner, GsRecordKind kind, const void *data, size_t size,
                       const void *extra = nullptr, size_t extraSize = 0u)
    {
        GsRecorder *recorder = gsRecorder();
        if (!recorder)
            return;
        std::lock_guard<std::mutex> lock(recorder->mutex);
        if (!recorder->file || recorder->owner != owner)
            return;
        const uint8_t tag = static_cast<uint8_t>(kind);
        std::fwrite(&tag, 1u, 1u, recorder->file);
        if (size != 0u)
            std::fwrite(data, 1u, size, recorder->file);
        if (extraSize != 0u)
            std::fwrite(extra, 1u, extraSize, recorder->file);
    }
}

void GSCpuBackend::NoteRecordedPresent(const GSPresentationRequest &request)
{
    GsRecorder *recorder = gsRecorder();
    if (!recorder)
        return;
    {
        std::lock_guard<std::mutex> recordLock(recorder->mutex);
        if (recorder->done)
            return;
        if (!recorder->owner)
            recorder->owner = this;
        if (recorder->owner != this)
            return;
        if (recorder->file)
        {
            const uint8_t tag = static_cast<uint8_t>(GsRecordKind::Present);
            std::fwrite(&tag, 1u, 1u, recorder->file);
            std::fwrite(&request, 1u, sizeof(request), recorder->file);
            if (++recorder->written >= recorder->presents)
            {
                std::fclose(recorder->file);
                recorder->file = nullptr;
                recorder->done = true;
                std::fprintf(stderr, "[gs-record] wrote %llu presentations to %s\n",
                             static_cast<unsigned long long>(recorder->written), recorder->path.c_str());
            }
            return;
        }
        if (request.vsyncTick < recorder->fromVsync)
            return;
    }

    // Every queued call has landed, so VRAM matches the call stream from here on.
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    std::lock_guard<std::mutex> recordLock(recorder->mutex);
    std::FILE *file = m_vram ? std::fopen(recorder->path.c_str(), "wb") : nullptr;
    if (!file)
    {
        std::fprintf(stderr, "[gs-record] cannot write %s\n", recorder->path.c_str());
        recorder->done = true;
        return;
    }
    const char magic[8] = {'P', 'S', '2', 'X', 'G', 'S', 'R', '1'};
    const uint32_t header[8] = {
        1u,
        m_vramSize,
        static_cast<uint32_t>(sizeof(GSPrimitiveBatch)),
        static_cast<uint32_t>(sizeof(GSTex0Reg)),
        static_cast<uint32_t>(sizeof(GSTexClutReg)),
        static_cast<uint32_t>(sizeof(GSTransferCommand)),
        static_cast<uint32_t>(sizeof(GSPresentationRequest)),
        static_cast<uint32_t>(sizeof(GSContext)),
    };
    std::fwrite(magic, 1u, sizeof(magic), file);
    std::fwrite(header, 1u, sizeof(header), file);
    std::fwrite(m_vram, 1u, m_vramSize, file);
    std::fwrite(m_clut.data(), 1u, sizeof(m_clut), file);
    std::fwrite(m_clutCbp.data(), 1u, sizeof(m_clutCbp), file);
    if (m_transferState.direction == 0u && m_transferState.copiedPixels < m_transferState.totalPixels)
        std::fprintf(stderr, "[gs-record] warning: an image transfer is in progress at the start\n");
    recorder->file = file;
    std::fprintf(stderr, "[gs-record] recording %llu presentations to %s\n",
                 static_cast<unsigned long long>(recorder->presents), recorder->path.c_str());
}

void GSCpuBackend::RestoreClutForReplay(const std::array<uint16_t, 512> &clut, const std::array<uint32_t, 2> &clutCbp)
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    m_clut = clut;
    m_clutCbp = clutCbp;
    uint64_t hash = 0xcbf29ce484222325ull;
    for (const uint16_t value : m_clut)
        hash = (hash ^ value) * 0x100000001b3ull;
    m_clutVersion = hash;
}

GSCpuBackend::GSCpuBackend()
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            m_readVramFuncs[i] = ReadNull;
            m_writeVramFuncs[i] = WriteNull;
            break;
        }
    }
    Reset();
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    if (vram && vramSize < GSMem::MEMORY_SIZE)
        throw std::invalid_argument("GS CPU backend requires at least 4 MiB of VRAM");

    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    m_vram = vram;
    m_vramSize = vramSize;
    m_depthModeResolved = false;
    m_separateDepth = false;
    m_depthPlane.clear();
    ResetUnlocked();
}

void GSCpuBackend::Reset()
{
    gsRecordWrite(this, GsRecordKind::Reset, nullptr, 0u);
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    m_clut.fill(0u);
    m_clutCbp.fill(0u);
    m_texturePageCache.Invalidate();
    InvalidateDecodedTexturesUnlocked();
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    if (batch.vertexCount == 0u)
        return;
    gsRecordWrite(this, GsRecordKind::Submit, &batch, sizeof(batch));
    if (QueueDraw(batch))
        return;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram)
        return;
    DrawPrimitive(batch);
}

bool GSCpuBackend::AsyncEnabled()
{
    if (!m_asyncResolved)
    {
        m_asyncResolved = true;
        const char *value = std::getenv("PS2X_GS_ASYNC");
        m_async = value && value[0] == '1';
    }
    return m_async;
}

bool GSCpuBackend::EnqueueWork(const std::function<void(QueuedWork &)> &fill)
{
    if (!AsyncEnabled())
        return false;
    std::unique_lock<std::mutex> queueLock(m_queueMutex);
    if (!m_worker.joinable())
        m_worker = std::thread([this]() { WorkerLoop(); });
    auto hasRoom = [this]()
    { return m_queue.size() + m_drawQueue.size() + m_inFlightWork < kMaxQueuedWork || m_stopWorker; };
    if (!hasRoom())
    {
        m_unnotifiedDraws = 0u;
        m_queueCv.notify_one();
        const GsStatsClock::time_point waitStart = gsStatsEnabled() ? GsStatsClock::now() : GsStatsClock::time_point{};
        m_presentCv.wait(queueLock, hasRoom);
        if (gsStatsEnabled())
            gsStats().producerQueueWaitNs.fetch_add(gsStatsNanoseconds(waitStart), std::memory_order_relaxed);
    }
    m_queue.emplace_back();
    fill(m_queue.back());
    m_queue.back().drawIndex = m_drawQueue.size();
    m_unnotifiedDraws = 0u;
    queueLock.unlock();
    m_queueCv.notify_one();
    return true;
}

bool GSCpuBackend::QueueDraw(const GSPrimitiveBatch &batch)
{
    if (!AsyncEnabled())
        return false;
    std::unique_lock<std::mutex> queueLock(m_queueMutex);
    if (!m_worker.joinable())
        m_worker = std::thread([this]() { WorkerLoop(); });
    auto hasRoom = [this]()
    { return m_queue.size() + m_drawQueue.size() + m_inFlightWork < kMaxQueuedWork || m_stopWorker; };
    if (!hasRoom())
    {
        m_unnotifiedDraws = 0u;
        m_queueCv.notify_one();
        const GsStatsClock::time_point waitStart = gsStatsEnabled() ? GsStatsClock::now() : GsStatsClock::time_point{};
        m_presentCv.wait(queueLock, hasRoom);
        if (gsStatsEnabled())
            gsStats().producerQueueWaitNs.fetch_add(gsStatsNanoseconds(waitStart), std::memory_order_relaxed);
    }
    m_drawQueue.push_back(batch);
    // A busy worker takes every queued draw when it finishes its batch; a sleeping one is
    // woken only every kDrawsPerWake draws (and by Flush), not for each triangle.
    const bool wake = ++m_unnotifiedDraws >= kDrawsPerWake && m_workerWaiting;
    if (wake)
        m_unnotifiedDraws = 0u;
    queueLock.unlock();
    if (wake)
        m_queueCv.notify_one();
    return true;
}

GSCpuBackend::~GSCpuBackend()
{
    {
        std::lock_guard<std::mutex> queueLock(m_queueMutex);
        m_stopWorker = true;
    }
    m_queueCv.notify_all();
    m_presentCv.notify_all();
    if (m_worker.joinable())
        m_worker.join();
    {
        std::lock_guard<std::mutex> jobLock(m_presentJobMutex);
        m_stopPresenter = true;
    }
    m_presentJobCv.notify_all();
    if (m_presenter.joinable())
        m_presenter.join();
    {
        std::lock_guard<std::mutex> bandLock(m_bandMutex);
        m_stopBands = true;
    }
    m_bandCv.notify_all();
    for (std::thread &thread : m_bandThreads)
        thread.join();
}

void GSCpuBackend::WorkerLoop()
{
    s_onGsWorker = true;
    std::vector<QueuedWork> batches;
    std::vector<GSPrimitiveBatch> draws;
    // The batch before: a list handed to the band threads may still point into it.
    std::vector<QueuedWork> previousBatches;
    std::vector<GSPrimitiveBatch> previousDraws;
    for (;;)
    {
        {
            std::unique_lock<std::mutex> queueLock(m_queueMutex);
            // Draws can be queued without a wake-up (see QueueDraw): never sleep long on them.
            m_workerWaiting = true;
            while (!m_stopWorker && m_queue.empty() && m_drawQueue.empty())
                m_queueCv.wait_for(queueLock, std::chrono::milliseconds(2));
            m_workerWaiting = false;
            if (m_queue.empty() && m_drawQueue.empty() && m_stopWorker)
                return;
            batches.swap(m_queue);
            draws.swap(m_drawQueue);
            m_unnotifiedDraws = 0u;
            m_inFlightWork = batches.size() + draws.size();
            m_workerBusy = true;
        }
        const bool collectStats = gsStatsEnabled();
        const GsStatsClock::time_point busyStart = collectStats ? GsStatsClock::now() : GsStatsClock::time_point{};
        if (collectStats)
        {
            gsStats().batches.fetch_add(1u, std::memory_order_relaxed);
            gsStats().batchItems.fetch_add(batches.size() + draws.size(), std::memory_order_relaxed);
        }
        const uint64_t submittedBefore = m_submittedLists;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            size_t nextDraw = 0u;
            auto drawUpTo = [&](size_t end)
            {
                for (; nextDraw < end && nextDraw < draws.size(); ++nextDraw)
                {
                    if (m_vram)
                        QueueDrawUnlocked(draws[nextDraw]);
                }
            };
            for (QueuedWork &work : batches)
            {
                drawUpTo(work.drawIndex);
                switch (work.kind)
                {
                case QueuedWork::Kind::Draw:
                    if (m_vram)
                        QueueDrawUnlocked(work.batch);
                    continue;
                case QueuedWork::Kind::Clut:
                    QueueClutUnlocked(work.tex0, work.texclut);
                    continue;
                case QueuedWork::Kind::TextureFlush:
                    TextureFlushUnlocked();
                    continue;
                case QueuedWork::Kind::Transfer:
                    FlushPendingDrawsUnlocked(FlushReason::Transfer);
                    JoinInflightUnlocked();
                    BeginTransferUnlocked(work.transfer);
                    continue;
                case QueuedWork::Kind::Upload:
                    FlushPendingDrawsUnlocked(FlushReason::Upload);
                    JoinInflightUnlocked();
                    UploadImageUnlocked(work.payload.data(), static_cast<uint32_t>(work.payload.size()));
                    continue;
                case QueuedWork::Kind::Present:
                    FlushPendingDrawsUnlocked(FlushReason::Present);
                    JoinInflightUnlocked();
                    break;
                }
                const GsStatsClock::time_point presentStart =
                    collectStats ? GsStatsClock::now() : GsStatsClock::time_point{};
                PresentJob job;
                {
                    std::lock_guard<std::mutex> jobLock(m_presentJobMutex);
                    if (!m_presentBuffers.empty())
                    {
                        job.vram = std::move(m_presentBuffers.back());
                        m_presentBuffers.pop_back();
                    }
                    if (!m_presenter.joinable())
                        m_presenter = std::thread([this]() { PresentLoop(); });
                }
                if (m_vram && m_vramSize != 0u)
                    job.vram.assign(m_vram, m_vram + m_vramSize);
                else
                    job.vram.clear();
                job.request = work.request;
                job.sink = std::move(work.sink);
                {
                    std::lock_guard<std::mutex> jobLock(m_presentJobMutex);
                    m_presentJobs.push_back(std::move(job));
                }
                m_presentJobCv.notify_one();
                if (collectStats)
                    gsStats().presentNs.fetch_add(gsStatsNanoseconds(presentStart), std::memory_order_relaxed);
            }
            drawUpTo(draws.size());
            FlushPendingDrawsUnlocked(FlushReason::BatchEnd);
            // Going idle means every queued draw is in VRAM. Otherwise the list in flight
            // can keep going while the next batch is queued, unless it was handed over
            // during the previous batch, whose storage is dropped below.
            bool moreWork = false;
            {
                std::lock_guard<std::mutex> queueLock(m_queueMutex);
                moreWork = !m_queue.empty() || !m_drawQueue.empty();
            }
            if (!moreWork || m_submittedLists == submittedBefore)
                JoinInflightUnlocked();
        }
        if (collectStats)
        {
            gsStats().workerBusyNs.fetch_add(gsStatsNanoseconds(busyStart), std::memory_order_relaxed);
            maybePrintGsStats(m_bandCount);
        }
        previousBatches.swap(batches);
        batches.clear();
        previousDraws.swap(draws);
        draws.clear();
        {
            std::lock_guard<std::mutex> queueLock(m_queueMutex);
            m_workerBusy = false;
            m_inFlightWork = 0u;
            if (m_queue.empty() && m_drawQueue.empty())
                m_idleCv.notify_all();
            m_presentCv.notify_all();
        }
    }
}

void GSCpuBackend::PresentLoop()
{
    GSCpuBackend snapshotBackend;
    for (;;)
    {
        PresentJob job;
        {
            std::unique_lock<std::mutex> jobLock(m_presentJobMutex);
            m_presentJobCv.wait(jobLock, [this]() { return m_stopPresenter || !m_presentJobs.empty(); });
            if (m_presentJobs.empty())
                return;
            job = std::move(m_presentJobs.front());
            m_presentJobs.erase(m_presentJobs.begin());
        }
        PresentationFrame frame{};
        if (!job.vram.empty())
        {
            snapshotBackend.Initialize(job.vram.data(), static_cast<uint32_t>(job.vram.size()));
            frame = snapshotBackend.PresentFromLocalMemory(job.request);
        }
        if (job.sink)
            job.sink(std::move(frame));
        {
            std::lock_guard<std::mutex> jobLock(m_presentJobMutex);
            m_presentBuffers.push_back(std::move(job.vram));
        }
        {
            std::lock_guard<std::mutex> queueLock(m_queueMutex);
            if (m_pendingPresents != 0u)
                --m_pendingPresents;
        }
        m_presentCv.notify_all();
    }
}

void GSCpuBackend::QueuePresent(const GSPresentationRequest &request, PresentSink sink)
{
    NoteRecordedPresent(request);
    {
        std::unique_lock<std::mutex> queueLock(m_queueMutex);
        if (m_worker.joinable())
        {
            if (gsStatsEnabled() && m_pendingPresents >= kMaxPendingPresents)
            {
                const GsStatsClock::time_point waitStart = GsStatsClock::now();
                m_presentCv.wait(queueLock, [this]()
                                 { return m_pendingPresents < kMaxPendingPresents || m_stopWorker; });
                gsStats().producerPresentWaitNs.fetch_add(gsStatsNanoseconds(waitStart), std::memory_order_relaxed);
            }
            if (m_pendingPresents >= kMaxPendingPresents)
                m_queueCv.notify_one();
            m_presentCv.wait(queueLock, [this]()
                             { return m_pendingPresents < kMaxPendingPresents || m_stopWorker; });
            ++m_pendingPresents;
            QueuedWork work{};
            work.kind = QueuedWork::Kind::Present;
            work.request = request;
            work.sink = std::move(sink);
            work.drawIndex = m_drawQueue.size();
            m_queue.push_back(std::move(work));
            m_unnotifiedDraws = 0u;
            queueLock.unlock();
            m_queueCv.notify_one();
            return;
        }
    }
    PresentationFrame frame = PresentSnapshot(request);
    if (sink)
        sink(std::move(frame));
}

void GSCpuBackend::DrainQueue() const
{
    std::unique_lock<std::mutex> queueLock(m_queueMutex);
    auto idle = [this]() { return m_queue.empty() && m_drawQueue.empty() && !m_workerBusy; };
    if (!idle())
    {
        m_unnotifiedDraws = 0u;
        m_queueCv.notify_one();
    }
    if (gsStatsEnabled() && !idle())
    {
        const GsStatsClock::time_point waitStart = GsStatsClock::now();
        m_idleCv.wait(queueLock, idle);
        gsStats().producerDrainWaitNs.fetch_add(gsStatsNanoseconds(waitStart), std::memory_order_relaxed);
        gsStats().producerDrains.fetch_add(1u, std::memory_order_relaxed);
    }
    m_idleCv.wait(queueLock, idle);
}

void GSCpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    gsRecordWrite(this, GsRecordKind::LoadClut, &tex0, sizeof(tex0), &texclut, sizeof(texclut));
    if (EnqueueWork([&](QueuedWork &work)
                    {
                        work.kind = QueuedWork::Kind::Clut;
                        work.tex0 = tex0;
                        work.texclut = texclut; }))
        return;
    std::lock_guard<std::mutex> lock(m_mutex);
    LoadClutStateUnlocked(tex0, texclut);
}

void GSCpuBackend::LoadClutStateUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    if (!m_vram || (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm)))
        return;

    switch (tex0.cld)
    {
    case 0u:
    case 6u:
    case 7u:
        return;
    case 1u:
        break;
    case 2u:
        m_clutCbp[0] = tex0.cbp;
        break;
    case 3u:
        m_clutCbp[1] = tex0.cbp;
        break;
    case 4u:
        if (m_clutCbp[0] == tex0.cbp)
            return;
        m_clutCbp[0] = tex0.cbp;
        break;
    case 5u:
        if (m_clutCbp[1] == tex0.cbp)
            return;
        m_clutCbp[1] = tex0.cbp;
        break;
    default:
        return;
    }

    LoadClutUnlocked(tex0, texclut);
}

void GSCpuBackend::LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return;

    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;

    const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwoBit && !fourBit;
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;

    for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
    {
        uint32_t sourceX = 0u;
        uint32_t sourceY = 0u;
        uint32_t sourceWidth = 1u;

        if (tex0.csm == 0u)
        {
            const uint32_t sourceIndex = swizzleClutIndexCSM1(entry);
            sourceX = sourceIndex & 0x0Fu;
            sourceY = sourceIndex >> 4u;
        }
        else
        {
            sourceWidth = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
            sourceX = (static_cast<uint32_t>(texclut.cou) << 4u) + entry;
            sourceY = static_cast<uint32_t>(texclut.cov);
        }

        const uint32_t raw = ReadTextureVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY); 
        const uint32_t destination = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteenBit ? 0x1FFu : 0x0FFu);
        if (sixteenBit)
        {
            m_clut[destination] = static_cast<uint16_t>(raw);
        }
        else
        {
            m_clut[destination] = static_cast<uint16_t>(raw & 0xFFFFu);
            m_clut[destination + 256u] = static_cast<uint16_t>(raw >> 16u);
        }
    }

    // Decoded textures key on the palette contents, so reloading an unchanged
    // CLUT (games do it before every draw) keeps them.
    uint64_t hash = 0xcbf29ce484222325ull;
    for (const uint16_t value : m_clut)
        hash = (hash ^ value) * 0x100000001b3ull;
    m_clutVersion = hash;
}

void GSCpuBackend::Flush()
{
    // Wakes the worker for the draws Submit queued without waking it.
    bool wake = false;
    {
        std::lock_guard<std::mutex> queueLock(m_queueMutex);
        wake = m_unnotifiedDraws != 0u && m_workerWaiting;
        m_unnotifiedDraws = 0u;
    }
    if (wake)
        m_queueCv.notify_one();
}

void GSCpuBackend::TextureFlush()
{
    gsRecordWrite(this, GsRecordKind::TextureFlush, nullptr, 0u);
    if (EnqueueWork([](QueuedWork &work) { work.kind = QueuedWork::Kind::TextureFlush; }))
        return;
    std::lock_guard<std::mutex> lock(m_mutex);
    TextureFlushUnlocked();
}

void GSCpuBackend::TextureFlushUnlocked()
{
    JoinInflightUnlocked();
    m_texturePageCache.Invalidate();
    // Decoded tiles were read through the texture page buffer, so tiles whose
    // pages were written since the previous flush may hold data the flush now
    // exposes.
    for (DecodedTexture &entry : m_decodedTextures)
    {
        if (!entry.valid)
            continue;
        for (uint32_t i = 0; i < entry.pageCount; ++i)
        {
            if (m_pageWriteEpoch[(entry.firstPage + i) & 511u] >= m_flushEpoch)
            {
                std::fill(entry.tileReady.begin(), entry.tileReady.end(), uint8_t{0});
                entry.readyTiles = 0u;
                entry.epoch = ++m_writeEpoch;
                break;
            }
        }
    }
    m_flushEpoch = ++m_writeEpoch;
}

void GSCpuBackend::Sync(GSSyncReason reason)
{
    // FINISH only raises a CSR flag; anything that observes VRAM drains on
    // its own, so the worker may keep drawing.
    if (reason != GSSyncReason::Finish)
        DrainQueue();
    // CPU backend is immediate. GPU backends may wait on fences/readbacks here.
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    return ReadVramUnlocked(psm, base, bw, x, y);
}

uint8_t *GSCpuBackend::DepthMemoryUnlocked()
{
    if (!m_depthModeResolved)
    {
        m_depthModeResolved = true;
        const char *value = std::getenv("PS2X_SEPARATE_ZBUFFER");
        m_separateDepth = value && value[0] == '1';
        if (m_separateDepth && m_vram && m_vramSize != 0u)
            m_depthPlane.assign(m_vram, m_vram + m_vramSize);
    }
    return (m_separateDepth && !m_depthPlane.empty()) ? m_depthPlane.data() : m_vram;
}

uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    if (!m_vram)
        return 0u;

    return GSMem::ReadTexture(m_texturePageCache, m_vram, psm, base, bw, x, y);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    const uint32_t recorded[6] = {psm, base, bw, x, y, value};
    gsRecordWrite(this, GsRecordKind::WriteVram, recorded, sizeof(recorded));
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    WriteVramUnlocked(psm, base, bw, x, y, value);
}

void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (!m_vram)
        return;
    // Band threads leave page marking to QueueDrawUnlocked.
    if (!s_inBand)
        MarkVramWriteUnlocked(psm, base, bw, x, y);
    m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value);
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch)
{
    s_activeTexture = PrepareTextureCacheUnlocked(batch.state, false);
    RasterPrimitive(batch);
}

void GSCpuBackend::RasterPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << batch.vertices[0].x << "," << batch.vertices[0].y << ")"
                      << " uv0=(" << (batch.vertices[0].u >> 4) << "," << (batch.vertices[0].v >> 4) << ")"
                      << " stq0=(" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q << ")"
                      << " v1=(" << batch.vertices[1].x << "," << batch.vertices[1].y << ")"
                      << " uv1=(" << (batch.vertices[1].u >> 4) << "," << (batch.vertices[1].v >> 4) << ")"
                      << " stq1=(" << batch.vertices[1].s << "," << batch.vertices[1].t << "," << batch.vertices[1].q << ")"
                      << " v2=(" << batch.vertices[2].x << "," << batch.vertices[2].y << ")"
                      << " uv2=(" << (batch.vertices[2].u >> 4) << "," << (batch.vertices[2].v >> 4) << ")"
                      << " stq2=(" << batch.vertices[2].s << "," << batch.vertices[2].t << "," << batch.vertices[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(batch.vertices[0].r) << ","
                      << static_cast<uint32_t>(batch.vertices[0].g) << ","
                      << static_cast<uint32_t>(batch.vertices[0].b) << ","
                      << static_cast<uint32_t>(batch.vertices[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(batch.vertices[1].r) << ","
                      << static_cast<uint32_t>(batch.vertices[1].g) << ","
                      << static_cast<uint32_t>(batch.vertices[1].b) << ","
                      << static_cast<uint32_t>(batch.vertices[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(batch.vertices[2].r) << ","
                      << static_cast<uint32_t>(batch.vertices[2].g) << ","
                      << static_cast<uint32_t>(batch.vertices[2].b) << ","
                      << static_cast<uint32_t>(batch.vertices[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((state.prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    switch (state.prim.type)
    {
    case GS_PRIM_SPRITE:
        DrawSprite(batch);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        DrawTriangle(batch);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        DrawLine(batch);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        const auto &ctx = state.context;
        int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
        if (ownsRow(py))
            WritePixel(state, px, py, static_cast<u32>(v.z), v.r, v.g, v.b, v.a, v.fog);
        break;
    }
    default:
        break;
    }
}

void GSCpuBackend::WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    PixelWriter writer;
    SetupPixelWriter(state, writer);
    WritePixel(writer, x, y, z, r, g, b, a, fog);
}

void GSCpuBackend::SetupPixelWriter(const GSDrawState &state, PixelWriter &writer)
{
    const auto &ctx = state.context;
    writer.vram = m_vram;
    writer.depthMemory = DepthMemoryUnlocked();
    writer.test = ctx.test;
    writer.scissorX0 = ctx.scissor.x0;
    writer.scissorX1 = ctx.scissor.x1;
    writer.scissorY0 = ctx.scissor.y0;
    writer.scissorY1 = ctx.scissor.y1;
    writer.fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    writer.fbw = std::max<u32>(ctx.frame.fbw, 1u);
    writer.fpsm = ctx.frame.psm;
    writer.zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    writer.zpsm = ctx.zbuf.psm;
    writer.fbmsk = ctx.frame.fbmsk;
    writer.readFrame = m_readVramFuncs[writer.fpsm & 0x3Fu];
    writer.writeFrame = m_writeVramFuncs[writer.fpsm & 0x3Fu];
    writer.readDepth = m_readVramFuncs[writer.zpsm & 0x3Fu];
    writer.writeDepth = m_writeVramFuncs[writer.zpsm & 0x3Fu];

    // TEST.ZTE=0 disables the depth test and depth writes (GS manual; PCSX2
    // treats it the same way). Games rely on it to draw 2D layers without
    // touching a Z buffer that may alias other VRAM.
    const bool depthTestEnabled = ((ctx.test >> 16) & 1u) != 0u;
    writer.ztestMethod = depthTestEnabled ? static_cast<uint32_t>((ctx.test >> 17) & 3u) : 1u;
    writer.depthWrites = !ctx.zbuf.zmask && depthTestEnabled;
    writer.fog = state.prim.fge;
    writer.blend = state.prim.abe;
    writer.pabe = state.pabe;
    writer.forceAlphaMsb = (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24;
    writer.frame16 = bitsPerPixel(static_cast<uint8_t>(writer.fpsm)) == 16;
    writer.frame24 = writer.fpsm == GS_PSM_CT24;
    writer.dateNeedsRead = ((ctx.test >> 14) & 0x1u) != 0u &&
                           (writer.fpsm == GS_PSM_CT32 || writer.fpsm == GS_PSM_CT16 || writer.fpsm == GS_PSM_CT16S);
    // Band threads leave page marking to QueueDrawUnlocked.
    writer.markFrameWrites = !s_inBand;
    writer.markDepthWrites = writer.depthMemory == m_vram && !s_inBand;
    const uint64_t alphaReg = ctx.alpha;
    writer.asel = alphaReg & 3;
    writer.bsel = (alphaReg >> 2) & 3;
    writer.csel = (alphaReg >> 4) & 3;
    writer.dsel = (alphaReg >> 6) & 3;
    writer.fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);
    writer.fogR = state.fogR;
    writer.fogG = state.fogG;
    writer.fogB = state.fogB;
}

void GSCpuBackend::WritePixel(const PixelWriter &w, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                              uint8_t fog)
{
    if (x < w.scissorX0 || x > w.scissorX1 || y < w.scissorY0 || y > w.scissorY1)
        return;

    if (w.fog)
    {
        const uint32_t inverseFog = 255u - fog;
        auto applyFog = [&](uint8_t input, uint8_t fogColor) -> uint8_t
        {
            return static_cast<uint8_t>(((static_cast<uint32_t>(fog) * input) >> 8) + ((inverseFog * fogColor) >> 8));
        };

        r = applyFog(r, w.fogR);
        g = applyFog(g, w.fogG);
        b = applyFog(b, w.fogB);
    }

    const u32 fpsm = w.fpsm;
    const PixelWriteMask writeMask = classifyAlphaTest(w.test, a, static_cast<uint8_t>(fpsm));
    if (!writeMask.writesAnything())
    {
        return;
    }

    const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && fpsm == GS_PSM_CT32;

    // small optimization, avoid reading the framebuffer for simple draws
    const bool frmw = w.dateNeedsRead || (writeMask.writesFramebuffer() && (w.fbmsk != 0 || w.blend || preserveDestinationAlpha));

    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        rawFramebufferPixel = w.readFrame(w.vram, w.fbp, w.fbw, x, y);
        fbrgba = rawFramebufferPixel;

        if (w.frame16)
        {
            fbrgba = Rgba5551ToRgba8888(fbrgba);
        }
        else if (w.frame24)
        {
            // The GS supplies 0x80 as destination alpha for RGB24 blending.
            fbrgba |= 0x80000000u;
        }
    }

    if (!passesDestinationAlphaTest(w.test, static_cast<uint8_t>(fpsm), rawFramebufferPixel))
    {
        return;
    }

    switch (w.ztestMethod)
    {
    case 0:
        return;
    case 1:
        break;
    case 2:
        if (!(static_cast<uint32_t>(z) >= w.readDepth(w.depthMemory, w.zbp, w.fbw, x, y)))
            return;
        break;
    case 3:
        if (!(static_cast<uint32_t>(z) > w.readDepth(w.depthMemory, w.zbp, w.fbw, x, y)))
            return;
        break;
    }

    if (writeMask.writesFramebuffer())
    {
        const u8 srcR = r;
        const u8 srcG = g;
        const u8 srcB = b;

        if (w.blend)
        {
            uint8_t dr = fbrgba & 0xFF;
            uint8_t dg = (fbrgba >> 8) & 0xFF;
            uint8_t db = (fbrgba >> 16) & 0xFF;
            uint8_t da = (fbrgba >> 24) & 0xFF;

            // PABE disables alpha blending when the source alpha MSB is clear.
            if (!(w.pabe && (a & 0x80u) == 0u))
            {
                auto pickRGB = [&](uint8_t sel, int cs, int cd) -> int
                {
                    if (sel == 0)
                        return cs;
                    if (sel == 1)
                        return cd;
                    return 0;
                };
                int cAlpha = (w.csel == 0) ? a : (w.csel == 1) ? da
                                                               : w.fix;

                r = clampU8(((pickRGB(w.asel, r, dr) - pickRGB(w.bsel, r, dr)) * cAlpha >> 7) + pickRGB(w.dsel, r, dr));
                g = clampU8(((pickRGB(w.asel, g, dg) - pickRGB(w.bsel, g, dg)) * cAlpha >> 7) + pickRGB(w.dsel, g, dg));
                b = clampU8(((pickRGB(w.asel, b, db) - pickRGB(w.bsel, b, db)) * cAlpha >> 7) + pickRGB(w.dsel, b, db));
            }
            else
            {
                r = srcR;
                g = srcG;
                b = srcB;
            }
        }

        if (writeMask.writeAlpha && w.forceAlphaMsb)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(r, g, b, a);

        if (w.fbmsk != 0)
        {
            pixel = (pixel & ~w.fbmsk) | (fbrgba & w.fbmsk);
        }

        if (preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        // format conversion
        if (w.frame16)
        {
            pixel = Rgba8888ToRgba5551(pixel);
        }

        if (w.markFrameWrites)
            MarkVramWriteUnlocked(fpsm, w.fbp, w.fbw, x, y);
        w.writeFrame(w.vram, w.fbp, w.fbw, x, y, pixel);
    }

    if (writeMask.writeDepth && w.depthWrites)
    {
        if (w.markDepthWrites)
            MarkVramWriteUnlocked(w.zpsm, w.zbp, w.fbw, x, y);
        w.writeDepth(w.depthMemory, w.zbp, w.fbw, x, y, z);
    }
}

uint32_t GSCpuBackend::LookupCLUT(const GSDrawState &state,
                                  const std::array<uint16_t, 512> &m_clut,
                                  uint8_t index,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    const bool sixteenBit = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;
    const uint32_t sourceIndex = isFourBitIndexedPsm(sourcePsm)
                                     ? (static_cast<uint32_t>(index) & 0x0Fu)
                                     : static_cast<uint32_t>(index);

    uint32_t clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
    if (!sixteenBit && csm == 0u && isEightBitIndexedPsm(sourcePsm))
    {
        const uint32_t block = std::min((sourceIndex & 0xF0u) + clutBase, 240u);
        clutIndex = block + (sourceIndex & 0x0Fu);
    }

    switch (cpsm)
    {
    case GS_PSM_CT32:
    {
        const uint32_t raw = static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw);
    }
    case GS_PSM_CT24:
    {
        const uint32_t raw = static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw & 0x00FFFFFFu);
    }
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(m_clut[clutIndex]));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

uint32_t GSCpuBackend::DecodeTexelUnlocked(const GSDrawState &state, const std::array<uint16_t, 512> &clut, int u, int v)
{
    const auto &tex = state.context.tex0;
    const u32 out = ReadTextureVramUnlocked(tex.psm, tex.tbp0, tex.tbw, u, v);
    switch (tex.psm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        return applyTexa(state.texa, tex.psm, out);
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        return applyTexa(state.texa, tex.psm, Rgba5551ToRgba8888(out));
    case GS_PSM_T8:
    case GS_PSM_T8H:
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
        return LookupCLUT(state, clut, static_cast<u8>(out), tex.cpsm, tex.csm, tex.csa, tex.psm);
    }
    return 0xFFFF00FFu;
}

void GSCpuBackend::ConvertTexelsUnlocked(const GSDrawState &state, const std::array<uint16_t, 512> &clut,
                                         const uint32_t *raw, uint32_t *out, uint32_t count)
{
    // The colour half of DecodeTexelUnlocked, with the format switch hoisted.
    const auto &tex = state.context.tex0;
    switch (tex.psm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        for (uint32_t i = 0; i < count; ++i)
            out[i] = applyTexa(state.texa, tex.psm, raw[i]);
        return;
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        for (uint32_t i = 0; i < count; ++i)
            out[i] = applyTexa(state.texa, tex.psm, Rgba5551ToRgba8888(static_cast<u16>(raw[i])));
        return;
    case GS_PSM_T8:
    case GS_PSM_T8H:
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
        for (uint32_t i = 0; i < count; ++i)
            out[i] = LookupCLUT(state, clut, static_cast<u8>(raw[i]), tex.cpsm, tex.csm, tex.csa, tex.psm);
        return;
    default:
        for (uint32_t i = 0; i < count; ++i)
            out[i] = 0xFFFF00FFu;
        return;
    }
}

namespace
{
    // Tile states in DecodedTexture::tileReady.
    constexpr uint8_t kTileEmpty = 0u;
    constexpr uint8_t kTileReady = 1u;
    constexpr uint8_t kTileDecoding = 2u;
}

uint32_t GSCpuBackend::FetchDecodedTexelUnlocked(const GSDrawState &state, int u, int v)
{
    DecodedTexture &entry = *s_activeTexture;
    const uint32_t tileX = static_cast<uint32_t>(u) >> 5u;
    const uint32_t tileY = static_cast<uint32_t>(v) >> 5u;
    std::atomic_ref<uint8_t> ready(entry.tileReady[tileY * entry.tilesX + tileX]);
    if (ready.load(std::memory_order_acquire) != kTileReady)
        WaitForTileUnlocked(state, entry, tileX, tileY);
    return entry.texels[static_cast<uint32_t>(v) * entry.width + static_cast<uint32_t>(u)];
}

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void GSCpuBackend::WaitForTileUnlocked(const GSDrawState &state, DecodedTexture &entry, uint32_t tileX, uint32_t tileY)
{
    std::atomic_ref<uint8_t> ready(entry.tileReady[tileY * entry.tilesX + tileX]);
    const bool collectStats = gsStatsEnabled();
    if (s_inBand && m_parallelDecode)
    {
        // The first band thread to need the tile decodes it; the others wait for it.
        uint8_t expected = kTileEmpty;
        if (ready.compare_exchange_strong(expected, kTileDecoding, std::memory_order_acq_rel))
        {
            if (collectStats)
                (s_onGsWorker ? gsStats().lazyTilesWorker : gsStats().lazyTilesBand)
                    .fetch_add(1u, std::memory_order_relaxed);
            DecodeTileUnlocked(state, entry, tileX, tileY, nullptr);
            ready.store(kTileReady, std::memory_order_release);
            std::atomic_ref<uint32_t>(entry.readyTiles).fetch_add(1u, std::memory_order_acq_rel);
            return;
        }
        const GsStatsClock::time_point waitStart = collectStats ? GsStatsClock::now() : GsStatsClock::time_point{};
        while (ready.load(std::memory_order_acquire) != kTileReady)
            std::this_thread::yield();
        if (collectStats)
            gsStats().decodeWaitNs.fetch_add(gsStatsNanoseconds(waitStart), std::memory_order_relaxed);
        return;
    }

    const GsStatsClock::time_point waitStart = collectStats ? GsStatsClock::now() : GsStatsClock::time_point{};
    std::lock_guard<std::mutex> decodeLock(m_decodeMutex);
    if (collectStats)
        gsStats().decodeWaitNs.fetch_add(gsStatsNanoseconds(waitStart), std::memory_order_relaxed);
    if (ready.load(std::memory_order_relaxed) != kTileReady)
    {
        if (collectStats)
            (s_onGsWorker ? gsStats().lazyTilesWorker : gsStats().lazyTilesBand)
                .fetch_add(1u, std::memory_order_relaxed);
        DecodeTileUnlocked(state, entry, tileX, tileY, &m_texturePageCache);
        ready.store(kTileReady, std::memory_order_release);
        std::atomic_ref<uint32_t>(entry.readyTiles).fetch_add(1u, std::memory_order_acq_rel);
    }
}

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void GSCpuBackend::DecodeTileUnlocked(const GSDrawState &state, DecodedTexture &entry, uint32_t tileX, uint32_t tileY,
                                      GSMem::TexturePageCache *cache)
{
    const uint32_t x0 = tileX << 5u;
    const uint32_t y0 = tileY << 5u;
    const uint32_t columns = std::min(x0 + 32u, entry.width) - x0;
    const uint32_t rows = std::min(y0 + 32u, entry.height) - y0;
    // Same texels, in the same order, as DecodeTexelUnlocked over the tile (through
    // the texture page buffer when one is given).
    const auto &tex = state.context.tex0;
    uint32_t raw[32u * 32u];
    GSMem::ReadTextureRect(cache, m_vram, tex.psm, tex.tbp0, tex.tbw, x0, y0, columns, rows, raw, 32u);
    for (uint32_t row = 0; row < rows; ++row)
        ConvertTexelsUnlocked(state, entry.clut, raw + row * 32u, entry.texels.data() + (y0 + row) * entry.width + x0,
                              columns);
}

namespace
{
    // Page touched by a pixel write (GS pages are 8 KiB).
    uint32_t vramPageOf(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
    {
        uint32_t pageWidth = 64u;
        uint32_t pageHeight = 32u;
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            pageHeight = 64u;
            break;
        case GS_PSM_T8:
            pageWidth = 128u;
            pageHeight = 64u;
            break;
        case GS_PSM_T4:
            pageWidth = 128u;
            pageHeight = 128u;
            break;
        default:
            break;
        }
        const uint32_t pagesPerRow = std::max<uint32_t>(1u, (bw * 64u) / pageWidth);
        return base / 32u + (y / pageHeight) * pagesPerRow + x / pageWidth;
    }
}

thread_local GSCpuBackend::DecodedTexture *GSCpuBackend::s_activeTexture = nullptr;

namespace
{
    // Pages from the first row touched to the end of the last one, so the
    // range is contiguous (pages are laid out row-major from the base).
    void markPageRows(std::bitset<512> &pages, uint32_t psm, uint32_t base, uint32_t bw, int y0, int y1,
                      uint64_t *epochs = nullptr, uint64_t epoch = 0u)
    {
        const uint32_t width = std::max<uint32_t>(bw, 1u) * 64u;
        const uint32_t first = vramPageOf(psm, base, bw, 0u, static_cast<uint32_t>(y0));
        const uint32_t last = vramPageOf(psm, base, bw, width - 1u, static_cast<uint32_t>(y1));
        for (uint32_t page = first; page <= last && page - first < 512u; ++page)
        {
            pages.set(page & 511u);
            if (epochs)
                epochs[page & 511u] = epoch;
        }
    }

    // Last page a texture can touch. A base in the middle of a page (TBP0 not a
    // multiple of 32 blocks) shifts the block layout, so texels spill into the
    // page after the last one addressed; a page-aligned base cannot.
    uint32_t lastTexturePage(const GSTex0Reg &tex, uint32_t width, uint32_t height)
    {
        const uint32_t spanWidth = std::max<uint32_t>(width, std::max<uint32_t>(1u, tex.tbw) * 64u);
        const uint32_t spill = (tex.tbp0 % 32u) != 0u ? 1u : 0u;
        return vramPageOf(tex.psm, tex.tbp0, tex.tbw, spanWidth - 1u, height - 1u) + spill;
    }

    void markTexturePages(std::bitset<512> &pages, const GSTex0Reg &tex, uint32_t width, uint32_t height)
    {
        // Same span PrepareTextureCacheUnlocked covers.
        const uint32_t firstPage = tex.tbp0 / 32u;
        const uint32_t lastPage = lastTexturePage(tex, width, height);
        for (uint32_t page = firstPage; page <= lastPage && page - firstPage < 512u; ++page)
            pages.set(page & 511u);
    }

    // Pixel rectangle a primitive can touch, clipped to the scissor.
    bool primitiveBounds(const GSPrimitiveBatch &batch, int &x0, int &y0, int &x1, int &y1)
    {
        const auto &ctx = batch.state.context;
        const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        uint32_t count = 1u;
        switch (batch.state.prim.type)
        {
        case GS_PRIM_TRIANGLE:
        case GS_PRIM_TRISTRIP:
        case GS_PRIM_TRIFAN:
            count = 3u;
            break;
        case GS_PRIM_SPRITE:
        case GS_PRIM_LINE:
        case GS_PRIM_LINESTRIP:
            count = 2u;
            break;
        default:
            break;
        }
        float minX = batch.vertices[0].x - ofx, maxX = minX;
        float minY = batch.vertices[0].y - ofy, maxY = minY;
        for (uint32_t i = 1u; i < count; ++i)
        {
            minX = std::min(minX, batch.vertices[i].x - ofx);
            maxX = std::max(maxX, batch.vertices[i].x - ofx);
            minY = std::min(minY, batch.vertices[i].y - ofy);
            maxY = std::max(maxY, batch.vertices[i].y - ofy);
        }
        x0 = std::max(static_cast<int>(std::floor(minX)) - 1, static_cast<int>(ctx.scissor.x0));
        y0 = std::max(static_cast<int>(std::floor(minY)) - 1, static_cast<int>(ctx.scissor.y0));
        x1 = std::min(static_cast<int>(std::ceil(maxX)) + 1, static_cast<int>(ctx.scissor.x1));
        y1 = std::min(static_cast<int>(std::ceil(maxY)) + 1, static_cast<int>(ctx.scissor.y1));
        return x0 <= x1 && y0 <= y1;
    }
}

void GSCpuBackend::QueueDrawUnlocked(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!primitiveBounds(batch, x0, y0, x1, y1))
        return;

    std::bitset<512> writes;
    markPageRows(writes, ctx.frame.psm, GSInternal::framePageBaseToBlock(ctx.frame.fbp), ctx.frame.fbw, y0, y1);
    const bool depthWrites = ((ctx.test >> 16) & 1u) != 0u && !ctx.zbuf.zmask;
    if (depthWrites && DepthMemoryUnlocked() == m_vram)
        markPageRows(writes, ctx.zbuf.psm, GSInternal::framePageBaseToBlock(ctx.zbuf.zbp), ctx.frame.fbw, y0, y1);

    std::bitset<512> reads;
    if (state.prim.tme)
        markTexturePages(reads, ctx.tex0, state.textureWidth, state.textureHeight);

    // Read after write and write after read between queued draws.
    if ((reads & m_pendingWritePages).any() || (writes & m_pendingReadPages).any())
        FlushPendingDrawsUnlocked(FlushReason::Hazard);

    const bool collectStats = gsStatsEnabled();
    DecodedTexture *texture = PrepareTextureCacheUnlocked(state, true);
    if (texture && (reads & writes).any())
    {
        // The draw samples what it renders: decode every tile now, as drawing
        // inline would on first use, so band threads never decode it late.
        JoinInflightUnlocked();
        const GsStatsClock::time_point predecodeStart =
            collectStats ? GsStatsClock::now() : GsStatsClock::time_point{};
        std::fill(texture->tileReady.begin(), texture->tileReady.end(), uint8_t{0});
        texture->readyTiles = 0u;
        s_activeTexture = texture;
        for (uint32_t ty = 0; ty < texture->height; ty += 32u)
            for (uint32_t tx = 0; tx < texture->width; tx += 32u)
                (void)FetchDecodedTexelUnlocked(state, static_cast<int>(tx), static_cast<int>(ty));
        if (collectStats)
        {
            gsStats().predecodeDraws.fetch_add(1u, std::memory_order_relaxed);
            gsStats().predecodeTiles.fetch_add(texture->tileReady.size(), std::memory_order_relaxed);
            gsStats().predecodeNs.fetch_add(gsStatsNanoseconds(predecodeStart), std::memory_order_relaxed);
        }
    }
    if (texture)
        ++texture->pendingRefs;

    // Page write epochs follow submission order, as drawing inline marks them.
    {
        std::bitset<512> marked;
        markPageRows(marked, ctx.frame.psm, GSInternal::framePageBaseToBlock(ctx.frame.fbp), ctx.frame.fbw, y0, y1,
                     m_pageWriteEpoch.data(), m_writeEpoch);
        if (depthWrites && DepthMemoryUnlocked() == m_vram)
            markPageRows(marked, ctx.zbuf.psm, GSInternal::framePageBaseToBlock(ctx.zbuf.zbp), ctx.frame.fbw, y0, y1,
                         m_pageWriteEpoch.data(), m_writeEpoch);
    }
    m_pendingWritePages |= writes;
    m_pendingReadPages |= reads;
    m_pendingDraws.push_back(PendingDraw{&batch, texture});
    const uint64_t area = static_cast<uint64_t>(x1 - x0 + 1) * static_cast<uint64_t>(y1 - y0 + 1);
    m_pendingArea += area;
    if (collectStats)
    {
        GsStats &stats = gsStats();
        stats.draws.fetch_add(1u, std::memory_order_relaxed);
        stats.area.fetch_add(area, std::memory_order_relaxed);
        if (state.prim.type == GS_PRIM_SPRITE)
            stats.sprites.fetch_add(1u, std::memory_order_relaxed);
        else if (state.prim.type == GS_PRIM_TRIANGLE || state.prim.type == GS_PRIM_TRISTRIP ||
                 state.prim.type == GS_PRIM_TRIFAN)
            stats.triangles.fetch_add(1u, std::memory_order_relaxed);
    }
    if (m_pendingDraws.size() >= 8192u)
        FlushPendingDrawsUnlocked(FlushReason::Size);
}

void GSCpuBackend::QueueClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    if (tex0.cld >= 1u && tex0.cld <= 5u && m_inflight && !m_parallelDecode)
        JoinInflightUnlocked(); // the band threads read through the page buffer
    if (tex0.cld >= 1u && tex0.cld <= 5u && (m_pendingWritePages.any() || m_inflight))
    {
        // The palette is read from VRAM a pending draw may still write.
        std::bitset<512> reads;
        if (tex0.csm == 0u)
        {
            for (uint32_t page = tex0.cbp / 32u; page <= tex0.cbp / 32u + 1u; ++page)
                reads.set(page & 511u);
        }
        else
        {
            const uint32_t cbw = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
            const int row = static_cast<int>(texclut.cov);
            markPageRows(reads, tex0.cpsm, tex0.cbp, cbw, row, row);
        }
        if ((reads & m_pendingWritePages).any())
            FlushPendingDrawsUnlocked(FlushReason::Clut);
        if ((reads & m_inflightWritePages).any())
            JoinInflightUnlocked();
    }
    LoadClutStateUnlocked(tex0, texclut);
}

void GSCpuBackend::RunPendingBand(const std::vector<PendingDraw> &draws, uint32_t band, uint32_t bandCount)
{
    s_bandIndex = band;
    s_bandCount = bandCount;
    s_inBand = true;
    for (const PendingDraw &draw : draws)
    {
        s_activeTexture = draw.texture;
        RasterPrimitive(*draw.batch);
    }
    s_inBand = false;
    s_bandIndex = 0u;
    s_bandCount = 1u;
}

void GSCpuBackend::BandLoop(uint32_t band)
{
    uint64_t seen = 0u;
    for (;;)
    {
        uint32_t bandCount = 0u;
        {
            std::unique_lock<std::mutex> bandLock(m_bandMutex);
            m_bandCv.wait(bandLock, [&]() { return m_stopBands || m_bandGeneration != seen; });
            if (m_stopBands)
                return;
            seen = m_bandGeneration;
            bandCount = m_bandCount;
        }
        const bool collectStats = gsStatsEnabled();
        const GsStatsClock::time_point busyStart = collectStats ? GsStatsClock::now() : GsStatsClock::time_point{};
        RunPendingBand(m_inflightDraws, band, bandCount);
        if (collectStats)
            gsStats().bandBusyNs.fetch_add(gsStatsNanoseconds(busyStart), std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> bandLock(m_bandMutex);
            if (--m_bandsRunning == 0u)
            {
                if (collectStats)
                    m_inflightFinished = GsStatsClock::now();
                m_bandDoneCv.notify_one();
            }
        }
    }
}

void GSCpuBackend::FlushPendingDrawsUnlocked(FlushReason reason)
{
    if (m_pendingDraws.empty())
        return;
    // Lists are rasterised one after another.
    JoinInflightUnlocked();
    (void)DepthMemoryUnlocked();
    const bool collectStats = gsStatsEnabled();
    if (collectStats)
    {
        GsStats &stats = gsStats();
        stats.flushes.fetch_add(1u, std::memory_order_relaxed);
        stats.flushedDraws.fetch_add(m_pendingDraws.size(), std::memory_order_relaxed);
        stats.reasons[static_cast<size_t>(reason) % GsStats::kReasons].fetch_add(1u, std::memory_order_relaxed);
    }

    if (m_bandCount == 0u)
    {
        // Band threads draw while the worker queues the next list and the EE thread runs
        // the game; three quarters of the logical CPUs leaves room for those. A phone has
        // about four fast cores among its eight, so three bands there.
#if defined(__aarch64__) || defined(__arm__)
        uint32_t count = std::min(3u, std::max(1u, std::thread::hardware_concurrency() * 3u / 4u));
#else
        uint32_t count = std::max(1u, std::thread::hardware_concurrency() * 3u / 4u);
#endif
        if (const char *value = std::getenv("PS2X_GS_THREADS"))
            count = static_cast<uint32_t>(std::max(1, std::atoi(value)));
        m_bandCount = std::min(count, 16u);
        if (m_bandCount > 1u)
        {
            for (uint32_t band = 0u; band < m_bandCount; ++band)
                m_bandThreads.emplace_back([this, band]() { BandLoop(band); });
        }
    }

    // Small lists are not worth waking the band threads for.
    if (m_bandCount <= 1u || m_pendingArea < 16384u)
    {
        const GsStatsClock::time_point rasterStart = collectStats ? GsStatsClock::now() : GsStatsClock::time_point{};
        RunPendingBand(m_pendingDraws, 0u, 1u);
        if (collectStats)
        {
            gsStats().inlineFlushes.fetch_add(1u, std::memory_order_relaxed);
            gsStats().workerRasterNs.fetch_add(gsStatsNanoseconds(rasterStart), std::memory_order_relaxed);
        }
        for (const PendingDraw &draw : m_pendingDraws)
            if (draw.texture)
                --draw.texture->pendingRefs;
    }
    else
    {
        // Band threads decode texture tiles in parallel, straight from VRAM, unless the
        // texture page buffer holds a page that VRAM has since changed under: reads
        // through it must then see the old data, one decode at a time. No draw in the
        // list writes a page that a draw in it samples, so sampled pages hold still.
        const uint32_t bufferedPage = m_texturePageCache.PageBase();
        m_parallelDecode = bufferedPage == UINT32_MAX ||
                           std::memcmp(m_texturePageCache.Bytes(), m_vram + bufferedPage,
                                       GSMem::TexturePageCache::kPageSize) == 0;
        // Decoding one tile at a time would leave the buffer holding one of the pages
        // the list reads; the worker reads on from VRAM instead.
        if (m_parallelDecode)
            m_texturePageCache.Invalidate();
        m_inflightDraws.swap(m_pendingDraws);
        for (const PendingDraw &draw : m_inflightDraws)
        {
            if (draw.texture)
            {
                --draw.texture->pendingRefs;
                ++draw.texture->inflightRefs;
            }
        }
        m_inflightWritePages = m_pendingWritePages;
        m_inflight = true;
        ++m_submittedLists;
        if (collectStats)
            m_inflightSubmitted = GsStatsClock::now();
        if (collectStats)
            gsStats().parallelDraws.fetch_add(m_inflightDraws.size(), std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> bandLock(m_bandMutex);
            m_bandsRunning = m_bandCount;
            ++m_bandGeneration;
        }
        m_bandCv.notify_all();
    }

    m_pendingDraws.clear();
    m_pendingWritePages.reset();
    m_pendingReadPages.reset();
    m_pendingArea = 0u;
}

void GSCpuBackend::JoinInflightUnlocked()
{
    if (!m_inflight)
        return;
    const bool collectStats = gsStatsEnabled();
    const GsStatsClock::time_point waitStart = collectStats ? GsStatsClock::now() : GsStatsClock::time_point{};
    {
        std::unique_lock<std::mutex> bandLock(m_bandMutex);
        m_bandDoneCv.wait(bandLock, [this]() { return m_bandsRunning == 0u; });
    }
    if (collectStats)
    {
        gsStats().bandWaitNs.fetch_add(gsStatsNanoseconds(waitStart), std::memory_order_relaxed);
        gsStats().lists.fetch_add(1u, std::memory_order_relaxed);
        gsStats().listActiveNs.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(m_inflightFinished - m_inflightSubmitted).count()),
            std::memory_order_relaxed);
    }
    m_inflight = false;
    m_parallelDecode = false;
    for (const PendingDraw &draw : m_inflightDraws)
        if (draw.texture)
            --draw.texture->inflightRefs;
    m_inflightDraws.clear();
    m_inflightWritePages.reset();
}

void GSCpuBackend::DecodeTextureRgba(const GSDrawState &state, uint32_t width, uint32_t height,
                                     std::vector<uint32_t> &out)
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto &tex = state.context.tex0;
    out.resize(static_cast<size_t>(width) * height);
    if (!m_vram || out.empty())
        return;
    std::vector<uint32_t> raw(out.size());
    GSMem::ReadTextureRect(nullptr, m_vram, tex.psm, tex.tbp0, tex.tbw, 0u, 0u, width, height, raw.data(), width);
    ConvertTexelsUnlocked(state, m_clut, raw.data(), out.data(), static_cast<uint32_t>(out.size()));
}

uint64_t GSCpuBackend::NewTextureEpoch()
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    return ++m_writeEpoch;
}

bool GSCpuBackend::TextureWrittenSince(const GSTex0Reg &tex, uint32_t width, uint32_t height, uint64_t epoch) const
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    const uint32_t firstPage = tex.tbp0 / 32u;
    const uint32_t lastPage = lastTexturePage(tex, width, height);
    const uint32_t pageCount = std::min<uint32_t>(512u, lastPage - firstPage + 1u);
    for (uint32_t i = 0; i < pageCount; ++i)
        if (m_pageWriteEpoch[(firstPage + i) & 511u] >= epoch)
            return true;
    return false;
}

uint64_t GSCpuBackend::ClutVersion() const
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_clutVersion;
}

void GSCpuBackend::ReadRectRaw(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t width,
                               uint32_t height, std::vector<uint32_t> &out) const
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    out.resize(static_cast<size_t>(width) * height);
    if (!m_vram || out.empty())
        return;
    const ReadVramFunc read = m_readVramFuncs[psm & 0x3Fu];
    uint32_t *dst = out.data();
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            *dst++ = read(m_vram, base, bw, x0 + x, y0 + y);
}

void GSCpuBackend::WriteRectRaw(uint32_t psm, uint32_t base, uint32_t bw, uint32_t width, uint32_t height,
                                const uint32_t *values)
{
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || !values)
        return;
    const WriteVramFunc write = m_writeVramFuncs[psm & 0x3Fu];
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
        {
            MarkVramWriteUnlocked(psm, base, bw, x, y);
            write(m_vram, base, bw, x, y, *values++);
        }
}

void GSCpuBackend::SetSynchronous()
{
    DrainQueue();
    std::lock_guard<std::mutex> queueLock(m_queueMutex);
    m_asyncResolved = true;
    m_async = false;
}

void GSCpuBackend::MarkVramWriteUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    m_pageWriteEpoch[vramPageOf(psm, base, bw, x, y) & 511u] = m_writeEpoch;
}

void GSCpuBackend::InvalidateDecodedTexturesUnlocked()
{
    for (DecodedTexture &entry : m_decodedTextures)
    {
        entry.valid = false;
        entry.pendingRefs = 0u;
        entry.inflightRefs = 0u;
    }
    s_activeTexture = nullptr;
    ++m_clutVersion;
}

GSCpuBackend::DecodedTexture *GSCpuBackend::PrepareTextureCacheUnlocked(const GSDrawState &state, bool cacheSmall)
{
    if (!m_vram || !state.prim.tme)
        return nullptr;
    const auto &tex = state.context.tex0;
    const uint32_t width = static_cast<uint32_t>(state.textureWidth);
    const uint32_t height = static_cast<uint32_t>(state.textureHeight);
    // Small textures stay on the direct path when drawing inline (cheap, and
    // it keeps the texture page buffer semantics for tiny textures). Band
    // threads always sample through an entry.
    if ((!cacheSmall && width * height < 4096u) || width == 0u || height == 0u || width > 1024u || height > 1024u)
        return nullptr;
    const bool indexed = isFourBitIndexedPsm(tex.psm) || isEightBitIndexedPsm(tex.psm);

    DecodedTexture *entry = nullptr;
    DecodedTexture *victim = nullptr;
    const bool collectStats = gsStatsEnabled();
    bool clutOnlyMismatch = false;
    const auto sameTexture = [&](const DecodedTexture &candidate)
    {
        return candidate.valid && candidate.tbp == tex.tbp0 && candidate.tbw == tex.tbw && candidate.psm == tex.psm &&
               candidate.width == width && candidate.height == height && candidate.ta0 == state.texa.ta0 &&
               candidate.ta1 == state.texa.ta1 && candidate.aem == state.texa.aem &&
               (!indexed || (candidate.cpsm == tex.cpsm && candidate.csm == tex.csm && candidate.csa == tex.csa));
    };
    if (m_lastPreparedTexture && sameTexture(*m_lastPreparedTexture) &&
        (!indexed || m_lastPreparedTexture->clutVersion == m_clutVersion))
    {
        entry = m_lastPreparedTexture;
    }
    else
    {
        for (DecodedTexture &candidate : m_decodedTextures)
        {
            if (sameTexture(candidate))
            {
                if (!indexed || candidate.clutVersion == m_clutVersion)
                {
                    entry = &candidate;
                    break;
                }
                clutOnlyMismatch = true;
            }
            if (candidate.pendingRefs != 0u || candidate.inflightRefs != 0u)
                continue;
            if (!victim || (!candidate.valid && victim->valid) ||
                (candidate.valid == victim->valid && candidate.lastUse < victim->lastUse))
                victim = &candidate;
        }
    }
    if (!entry && !victim)
    {
        // Every entry is pinned by a pending draw.
        FlushPendingDrawsUnlocked(FlushReason::Pinned);
        JoinInflightUnlocked();
        return PrepareTextureCacheUnlocked(state, cacheSmall);
    }

    if (entry)
    {
        if (collectStats)
            gsStats().cacheHits.fetch_add(1u, std::memory_order_relaxed);
        for (uint32_t i = 0; i < entry->pageCount; ++i)
        {
            if (m_pageWriteEpoch[(entry->firstPage + i) & 511u] >= entry->epoch)
            {
                if (entry->inflightRefs != 0u)
                    JoinInflightUnlocked();
                std::fill(entry->tileReady.begin(), entry->tileReady.end(), uint8_t{0});
                entry->readyTiles = 0u;
                entry->epoch = ++m_writeEpoch;
                if (collectStats)
                    gsStats().cacheEpochResets.fetch_add(1u, std::memory_order_relaxed);
                break;
            }
        }
    }
    else
    {
        if (collectStats)
        {
            gsStats().cacheMisses.fetch_add(1u, std::memory_order_relaxed);
            if (clutOnlyMismatch)
                gsStats().cacheMissesClut.fetch_add(1u, std::memory_order_relaxed);
            if (victim->valid)
                gsStats().cacheEvictions.fetch_add(1u, std::memory_order_relaxed);
        }
        entry = victim;
        entry->valid = true;
        entry->tbp = tex.tbp0;
        entry->tbw = tex.tbw;
        entry->psm = tex.psm;
        entry->width = width;
        entry->height = height;
        entry->cpsm = tex.cpsm;
        entry->csm = tex.csm;
        entry->csa = tex.csa;
        entry->ta0 = state.texa.ta0;
        entry->ta1 = state.texa.ta1;
        entry->aem = state.texa.aem;
        entry->clutVersion = m_clutVersion;
        entry->clut = m_clut;
        entry->tilesX = (width + 31u) / 32u;
        entry->texels.assign(static_cast<size_t>(width) * height, 0u);
        entry->tileReady.assign(static_cast<size_t>(entry->tilesX) * ((height + 31u) / 32u), uint8_t{0});
        entry->readyTiles = 0u;
        const uint32_t firstPage = tex.tbp0 / 32u;
        const uint32_t lastPage = lastTexturePage(tex, width, height);
        entry->firstPage = firstPage;
        entry->pageCount = std::min<uint32_t>(512u, lastPage - firstPage + 1u);
        entry->epoch = ++m_writeEpoch;
    }
    entry->lastUse = ++m_decodeUseCounter;
    m_lastPreparedTexture = entry;
    return entry;
}

void GSCpuBackend::SetupSampler(const GSDrawState &state, TextureSampler &sampler)
{
    const uint64_t clamp = state.context.clamp;
    sampler.texture = s_activeTexture;
    sampler.texels = nullptr;
    sampler.width = 0u;
    sampler.height = 0u;
    if (DecodedTexture *texture = sampler.texture)
    {
        sampler.width = texture->width;
        sampler.height = texture->height;
        // Tiles are only reset between lists, so once all are decoded they stay.
        if (std::atomic_ref<uint32_t>(texture->readyTiles).load(std::memory_order_acquire) ==
            texture->tileReady.size())
            sampler.texels = texture->texels.data();
    }
    sampler.texW = state.textureWidth;
    sampler.texH = state.textureHeight;
    sampler.wrapU = static_cast<uint8_t>(clamp & 0x3u);
    sampler.wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    sampler.minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
    sampler.maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
    sampler.minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
    sampler.maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);
}

uint32_t GSCpuBackend::SampleTexture(const GSDrawState &state, const TextureSampler &sampler, float s, float t,
                                     float q, uint16_t u, uint16_t v)
{
    const int texW = sampler.texW;
    const int texH = sampler.texH;

    float texUf, texVf;
    if (state.prim.fst)
    {
        texUf = static_cast<float>(u) / 16.0f;
        texVf = static_cast<float>(v) / 16.0f;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        texUf = s * invQ * static_cast<float>(texW);
        texVf = t * invQ * static_cast<float>(texH);
    }

    // Fetches the texel at already wrapped coordinates.
    auto samplePoint = [&](int sampleU, int sampleV) -> uint32_t
    {
        DecodedTexture *active = sampler.texture;
        if (active && sampleU >= 0 && sampleV >= 0 &&
            static_cast<uint32_t>(sampleU) < sampler.width && static_cast<uint32_t>(sampleV) < sampler.height)
        {
            if (sampler.texels)
                return sampler.texels[static_cast<uint32_t>(sampleV) * sampler.width + static_cast<uint32_t>(sampleU)];
            return FetchDecodedTexelUnlocked(state, sampleU, sampleV);
        }
        if (!active)
            return DecodeTexelUnlocked(state, m_clut, sampleU, sampleV);
        if (s_inBand && m_parallelDecode)
        {
            const auto &tex = state.context.tex0;
            uint32_t raw = 0u;
            uint32_t texel = 0u;
            GSMem::ReadTextureRect(nullptr, m_vram, tex.psm, tex.tbp0, tex.tbw, static_cast<uint32_t>(sampleU),
                                   static_cast<uint32_t>(sampleV), 1u, 1u, &raw, 1u);
            ConvertTexelsUnlocked(state, active->clut, &raw, &texel, 1u);
            return texel;
        }
        std::lock_guard<std::mutex> decodeLock(m_decodeMutex);
        return DecodeTexelUnlocked(state, active->clut, sampleU, sampleV);
    };

    const auto wrapU = [&](int coordinate)
    { return wrapTextureCoordinate(coordinate, texW, sampler.wrapU, sampler.minU, sampler.maxU); };
    const auto wrapV = [&](int coordinate)
    { return wrapTextureCoordinate(coordinate, texH, sampler.wrapV, sampler.minV, sampler.maxV); };

    if (!state.linearFilter)
    {
        return samplePoint(wrapU(static_cast<int>(texUf)), wrapV(static_cast<int>(texVf)));
    }

    const float sampleU = texUf - 0.5f;
    const float sampleV = texVf - 0.5f;
    const int u0 = static_cast<int>(std::floor(sampleU));
    const int v0 = static_cast<int>(std::floor(sampleV));
    const int u1 = u0 + 1;
    const int v1 = v0 + 1;
    const float fx = sampleU - static_cast<float>(u0);
    const float fy = sampleV - static_cast<float>(v0);

    // Each axis is wrapped once for the four taps.
    const int wrappedU0 = wrapU(u0);
    const int wrappedU1 = wrapU(u1);
    const int wrappedV0 = wrapV(v0);
    const int wrappedV1 = wrapV(v1);
    const uint32_t c00 = samplePoint(wrappedU0, wrappedV0);
    const uint32_t c10 = samplePoint(wrappedU1, wrappedV0);
    const uint32_t c01 = samplePoint(wrappedU0, wrappedV1);
    const uint32_t c11 = samplePoint(wrappedU1, wrappedV1);

    // Fixed-point bilinear blend (8-bit fractions) of all four channels.
    const int ifx = static_cast<int>(fx * 256.0f + 0.5f);
    const int ify = static_cast<int>(fy * 256.0f + 0.5f);
    const auto blend = [&](uint32_t shift) -> uint8_t
    {
        const int a00 = static_cast<int>((c00 >> shift) & 0xFFu);
        const int a10 = static_cast<int>((c10 >> shift) & 0xFFu);
        const int a01 = static_cast<int>((c01 >> shift) & 0xFFu);
        const int a11 = static_cast<int>((c11 >> shift) & 0xFFu);
        const int top = a00 * 256 + (a10 - a00) * ifx;
        const int bottom = a01 * 256 + (a11 - a01) * ifx;
        return clampU8((top * 256 + (bottom - top) * ify + 32768) >> 16);
    };
    const uint8_t r = blend(0u);
    const uint8_t g = blend(8u);
    const uint8_t b = blend(16u);
    const uint8_t a = blend(24u);

    return static_cast<uint32_t>(r) |
           (static_cast<uint32_t>(g) << 8) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(a) << 24);
}

void GSCpuBackend::DrawSprite(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;
    u32 z1 = static_cast<u32>(v1.z);

    if (x0 > x1)
        std::swap(x0, x1);
    if (y0 > y1)
        std::swap(y0, y1);

    const int unclippedX0 = x0;
    const int unclippedY0 = y0;
    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int unclippedX1 = unclippedX0 + spanX - 1;
    const int unclippedY1 = unclippedY0 + spanY - 1;

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return;

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);
    if (!ownsAnyRow(drawY0, drawY1))
        return;

    PixelWriter writer;
    SetupPixelWriter(state, writer);

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;

    if (state.prim.tme)
    {
        const auto &tex = ctx.tex0;
        const int texW = state.textureWidth;
        const int texH = state.textureHeight;
        TextureSampler sampler;
        SetupSampler(state, sampler);

        float u0f, v0f, u1f, v1f;
        if (state.prim.fst)
        {
            u0f = static_cast<float>(v0.u >> 4);
            v0f = static_cast<float>(v0.v >> 4);
            u1f = static_cast<float>(v1.u >> 4);
            v1f = static_cast<float>(v1.v >> 4);
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }

        float spriteW = static_cast<float>(spanX);
        float spriteH = static_cast<float>(spanY);
        if (spriteW < 1.0f)
            spriteW = 1.0f;
        if (spriteH < 1.0f)
            spriteH = 1.0f;

        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!ownsRow(y))
                continue;
            float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / spriteH;
            float texVf = v0f + (v1f - v0f) * ty;

            for (int x = drawX0; x <= drawX1; ++x)
            {
                float tx = (static_cast<float>(x - unclippedX0) + 0.5f) / spriteW;
                float texUf = u0f + (u1f - u0f) * tx;
                uint32_t texel = 0xFFFF00FFu;
                if (state.prim.fst)
                {
                    const int fixedU = static_cast<int>((texUf * 16.0f) + 0.5f);
                    const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                    const uint16_t sampleU = static_cast<uint16_t>(clampInt(fixedU, 0, 0xFFFF));
                    const uint16_t sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
                    texel = SampleTexture(state, sampler, 0.0f, 0.0f, 1.0f, sampleU, sampleV);
                }
                else
                {
                    texel = SampleTexture(state, sampler, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH),
                                          1.0f, 0u, 0u);
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                WritePixel(writer, x, y, z1, color.r, color.g, color.b, color.a, v1.fog);
            }
        }
    }
    else
    {
        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!ownsRow(y))
                continue;
            for (int x = drawX0; x <= drawX1; ++x)
                WritePixel(writer, x, y, z1, r, g, b, a, v1.fog);
        }
    }
}

void GSCpuBackend::DrawTriangle(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    float fx2 = v2.x - static_cast<float>(ofx);
    float fy2 = v2.y - static_cast<float>(ofy);

    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));

    minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
    maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
    minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
    maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);

    float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return;
    if (!ownsAnyRow(minY, maxY))
        return;

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;

    // Each barycentric weight is linear in px along a scanline; solve for the
    // span where all three are non-negative so pixels outside the triangle are
    // skipped. The span is widened by a pixel and every pixel still runs the
    // exact edge test below.
    TextureSampler sampler;
    if (state.prim.tme)
        SetupSampler(state, sampler);
    PixelWriter writer;
    SetupPixelWriter(state, writer);

    const float k = winding * invAbsDenom;
    const float a0 = (fy1 - fy2) * k;
    const float a1 = (fy2 - fy0) * k;
    const float a2 = -(a0 + a1);
    for (int y = minY; y <= maxY; ++y)
    {
        if (!ownsRow(y))
            continue;
        float py = static_cast<float>(y) + 0.5f;
        const float b0 = (-(fy1 - fy2) * fx2 + (fx2 - fx1) * (py - fy2)) * k;
        const float b1 = (-(fy2 - fy0) * fx2 + (fx0 - fx2) * (py - fy2)) * k;
        const float b2 = 1.0f - b0 - b1;
        float spanMin = static_cast<float>(minX);
        float spanMax = static_cast<float>(maxX) + 1.0f;
        bool emptyRow = false;
        const float coefficients[3][2] = {{a0, b0}, {a1, b1}, {a2, b2}};
        for (const auto &edge : coefficients)
        {
            if (edge[0] > 1.0e-12f)
                spanMin = std::max(spanMin, (-kEdgeEpsilon - edge[1]) / edge[0]);
            else if (edge[0] < -1.0e-12f)
                spanMax = std::min(spanMax, (-kEdgeEpsilon - edge[1]) / edge[0]);
            else if (edge[1] < -kEdgeEpsilon)
                emptyRow = true;
        }
        if (emptyRow || spanMin > spanMax + 1.0f)
            continue;
        const int rowMinX = std::max(minX, static_cast<int>(std::floor(spanMin - 0.5f)) - 1);
        const int rowMaxX = std::min(maxX, static_cast<int>(std::ceil(spanMax - 0.5f)) + 1);
        for (int x = rowMinX; x <= rowMaxX; ++x)
        {
            float px = static_cast<float>(x) + 0.5f;

            float w0 = (((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
            float w1 = (((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                continue;

            double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

            uint8_t r, g, b, a;
            if (state.prim.iip)
            {
                r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (state.prim.tme)
            {
                float is, it, iq;
                uint16_t iu, iv;
                if (state.prim.fst)
                {
                    iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                    iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    // The GS DDA interpolates the homogeneous S, T and Q
                    // values. Texel coordinates are calculated from S/Q and
                    // T/Q only after interpolation.
                    is = v0.s * w0 + v1.s * w1 + v2.s * w2;
                    it = v0.t * w0 + v1.t * w1 + v2.t * w2;
                    iq = v0.q * w0 + v1.q * w1 + v2.q * w2;
                    iu = 0;
                    iv = 0;
                }

                uint32_t texel = SampleTexture(state, sampler, is, it, iq, iu, iv);

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const auto &tex = ctx.tex0;
                const uint8_t shadeR = r;
                const uint8_t shadeG = g;
                const uint8_t shadeB = b;
                const uint8_t shadeA = a;
                const TextureCombineResult color = combineTexture(tex, shadeR, shadeG, shadeB, shadeA, tr, tg, tb, ta);

                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            const uint8_t fog = clampU8(static_cast<int>(v0.fog * w0 + v1.fog * w1 + v2.fog * w2));
            WritePixel(writer, x, y, static_cast<u32>(z + 0.5), r, g, b, a, fog);
        }
    }
}

void GSCpuBackend::DrawLine(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r, g, b, a;
        if (state.prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        double z = (v0.z + (v1.z - v0.z) * t);
        const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
        if (ownsRow(y0))
            WritePixel(state, x0, y0, static_cast<u32>(z), r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
        ++step;
    }
}

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    gsRecordWrite(this, GsRecordKind::BeginTransfer, &command, sizeof(command));
    if (EnqueueWork([&](QueuedWork &work)
                    {
                        work.kind = QueuedWork::Kind::Transfer;
                        work.transfer = command; }))
        return;
    std::lock_guard<std::mutex> lock(m_mutex);
    BeginTransferUnlocked(command);
}

void GSCpuBackend::BeginTransferUnlocked(const GSTransferCommand &command)
{
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    const uint32_t recordedSize = data ? sizeBytes : 0u;
    gsRecordWrite(this, GsRecordKind::UploadImage, &recordedSize, sizeof(recordedSize), data, recordedSize);
    if (data && sizeBytes != 0u &&
        EnqueueWork([&](QueuedWork &work)
                    {
                        work.kind = QueuedWork::Kind::Upload;
                        work.payload.assign(data, data + sizeBytes); }))
        return;
    std::lock_guard<std::mutex> lock(m_mutex);
    UploadImageUnlocked(data, sizeBytes);
}

void GSCpuBackend::UploadImageUnlocked(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    if (m_transfer.trxreg.rrw == 0u || m_transfer.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
        return;

    const uint32_t dbp = m_transfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = m_transfer.bitbltbuf.dpsm;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t dsax = m_transfer.trxpos.dsax;
    uint32_t offset = 0u;

    auto advancePixel = [&](uint32_t count)
    {
        const uint32_t totalPixels = m_transferState.totalPixels;
        m_transferState.copiedPixels =
            std::min<uint32_t>(totalPixels, m_transferState.copiedPixels + count);

        if (m_transferState.copiedPixels >= totalPixels)
        {
            m_transferState.direction = 3u;
            m_transferState.totalPixels = 0u;
            return;
        }

        m_transferState.x = dsax + (m_transferState.copiedPixels % rrw);
        m_transferState.y = m_transfer.trxpos.dsay + (m_transferState.copiedPixels / rrw);
    };

    while (offset < sizeBytes && m_transferState.direction == 0u)
    {
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            if (sizeBytes - offset < 4u)
                return;
            uint32_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 4u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            if (sizeBytes - offset < 3u)
                return;
            const uint32_t value = static_cast<uint32_t>(data[offset]) |
                                   (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
                                   (static_cast<uint32_t>(data[offset + 2u]) << 16u);
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 3u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            if (sizeBytes - offset < 2u)
                return;
            uint16_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 2u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_T8:
        case GS_PSM_T8H:
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, data[offset++]);
            advancePixel(1u);
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint8_t packed = data[offset++];
            const uint32_t firstPixel = m_transferState.copiedPixels;
            WriteVramUnlocked(dpsm, dbp, dbw,
                              dsax + (firstPixel % rrw),
                              m_transfer.trxpos.dsay + (firstPixel / rrw),
                              packed & 0x0Fu);
            if (firstPixel + 1u < m_transferState.totalPixels)
            {
                const uint32_t secondPixel = firstPixel + 1u;
                WriteVramUnlocked(dpsm, dbp, dbw,
                                  dsax + (secondPixel % rrw),
                                  m_transfer.trxpos.dsay + (secondPixel / rrw),
                                  (packed >> 4u) & 0x0Fu);
            }
            advancePixel(std::min<uint32_t>(2u, m_transferState.totalPixels - firstPixel));
            break;
        }
        default:
            return;
        }
    }
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }

    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    const uint32_t recordedBytes = dst ? maxBytes : 0u;
    gsRecordWrite(this, GsRecordKind::ConsumeLocalToHost, &recordedBytes, sizeof(recordedBytes));
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    gsRecordWrite(this, GsRecordKind::ClearFramebuffer, &context, sizeof(context), &rgba, sizeof(rgba));
    DrainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || context.frame.fbw == 0u)
        return false;

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(context.frame.fbw, 1u);
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }
    return false;
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    NoteRecordedPresent(request);
    return PresentSnapshot(request);
}

PresentationFrame GSCpuBackend::PresentSnapshot(const GSPresentationRequest &request)
{
    // Snapshot local memory under the backend lock, then perform the expensive
    // display conversion without holding the producer-side raster lock.
    thread_local std::vector<uint8_t> snapshot;
    SnapshotVram(snapshot);
    if (snapshot.empty())
        return {};

    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    const bool fieldMode = smode2.interlaced && !smode2.frameMode;
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    if (!valid1 && !valid2)
        return result;

    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t height,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        selected = displayFrame;
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(displayFrame, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        return true;
    };

    if (valid1 && valid2)
    {
        GSFrameReg selected1{}, selected2{};
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copySource(displayFrame1, origin1, width1, height1, false, true, selected1, crt1, preferred1) &&
            copySource(displayFrame2, origin2, width2, height2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (!pmode.slbg)
                for (uint32_t y = 0; y < height2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            for (uint32_t y = 0; y < height1; ++y)
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    const uint32_t factor = pmode.mmod ? pmode.alp : std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u);
                    dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                    dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                    dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    dst[3] = pmode.amod ? dst[3] : src[3];
                }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            if (fieldMode)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            return result;
        }
    }

    const GSFrameReg &displayFrame = valid1 ? displayFrame1 : displayFrame2;
    const GSDisplayReadOrigin &origin = valid1 ? origin1 : origin2;
    result.width = valid1 ? width1 : width2;
    result.height = valid1 ? height1 : height2;
    GSFrameReg selected = displayFrame;
    if (!copySource(displayFrame, origin, result.width, result.height, true, false, selected, result.pixels, result.usedPreferred))
        return {};
    if (fieldMode)
        applyFieldPresentation(result.pixels, result.width, result.height, oddField);
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return result;
}
