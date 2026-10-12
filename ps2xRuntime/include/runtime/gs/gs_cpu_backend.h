#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <bitset>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();
    ~GSCpuBackend() override;

    // Presentation snapshot taken in queue order: the sink receives the frame
    // as VRAM stood after every primitive submitted before this call.
    using PresentSink = std::function<void(PresentationFrame &&)>;
    void QueuePresent(const GSPresentationRequest &request, PresentSink sink);

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

private:
    void ResetUnlocked();
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    void RasterPrimitive(const GSPrimitiveBatch &batch);
    void DrawSprite(const GSPrimitiveBatch &batch);
    void DrawTriangle(const GSPrimitiveBatch &batch);
    void DrawLine(const GSPrimitiveBatch &batch);
    void WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v);
    uint32_t LookupCLUT(const GSDrawState &state, const std::array<uint16_t, 512> &clut,
                        uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    // PS2X_SEPARATE_ZBUFFER=1 keeps depth in its own plane, like hardware
    // renderers do. Some games place the Z buffer over memory that aliases a
    // displayed frame buffer and rely on real GS timing to hide it. Off by
    // default (canonical local memory); resolved on first depth access so a
    // game override can enable it.
    std::vector<uint8_t> m_depthPlane;
    bool m_separateDepth = false;
    bool m_depthModeResolved = false;
    uint8_t *DepthMemoryUnlocked();
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    std::array<uint16_t, 512> m_clut{};
    std::array<uint32_t, 2> m_clutCbp{};
    GSMem::TexturePageCache m_texturePageCache;

    // Textures decoded to RGBA8888 in 32x32 tiles on first use, so sampling
    // avoids swizzled reads and CLUT lookups per texel. Tiles are dropped when
    // a VRAM page they cover is written (at the next use, and again at the
    // next TEXFLUSH so writes hidden by the texture page buffer show up).
    // Indexed entries are keyed by a hash of the CLUT contents.
    struct DecodedTexture
    {
        bool valid = false;
        uint32_t tbp = 0;
        uint32_t tbw = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint8_t psm = 0;
        uint8_t cpsm = 0;
        uint8_t csm = 0;
        uint8_t csa = 0;
        uint8_t ta0 = 0;
        uint8_t ta1 = 0;
        bool aem = false;
        uint64_t clutVersion = 0;
        uint64_t epoch = 0;
        uint64_t lastUse = 0;
        uint32_t firstPage = 0;
        uint32_t pageCount = 0;
        uint32_t tilesX = 0;
        // Draws queued for the band threads that sample this entry; a pinned
        // entry is never evicted.
        uint32_t pendingRefs = 0;
        std::vector<uint32_t> texels;
        std::vector<uint8_t> tileReady;
        // Palette the tiles decode with, copied when the entry is created.
        std::array<uint16_t, 512> clut{};
    };
    std::array<DecodedTexture, 32> m_decodedTextures{};
    std::array<uint64_t, 512> m_pageWriteEpoch{};
    uint64_t m_writeEpoch = 1;
    uint64_t m_clutVersion = 0;
    uint64_t m_flushEpoch = 0;
    uint64_t m_decodeUseCounter = 0;
    // Entry sampled by the primitive being drawn on this thread.
    static thread_local DecodedTexture *s_activeTexture;
    // Serialises tile decodes (and the single-page texture buffer they read
    // through) when band threads decode lazily.
    std::mutex m_decodeMutex;

    // Parallel rasterisation (async mode only). The worker prepares draws in
    // submission order (CLUT loads, texture cache, page tracking) into
    // m_pendingDraws, and band threads then rasterise the list together, each
    // owning interleaved groups of rows. A draw that samples pages a pending
    // draw writes, or writes pages a pending draw samples, flushes the list
    // first, so the result matches drawing one primitive at a time.
    struct PendingDraw
    {
        const GSPrimitiveBatch *batch = nullptr;
        DecodedTexture *texture = nullptr;
    };
    void QueueDrawUnlocked(const GSPrimitiveBatch &batch);
    void QueueClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    void FlushPendingDrawsUnlocked();
    void RunPendingBand(uint32_t band, uint32_t bandCount);
    void BandLoop(uint32_t band);
    std::vector<PendingDraw> m_pendingDraws;
    std::bitset<512> m_pendingWritePages;
    std::bitset<512> m_pendingReadPages;
    uint64_t m_pendingArea = 0;
    uint32_t m_bandCount = 0;
    std::vector<std::thread> m_bandThreads;
    std::mutex m_bandMutex;
    std::condition_variable m_bandCv;
    std::condition_variable m_bandDoneCv;
    uint64_t m_bandGeneration = 0;
    uint32_t m_bandsRunning = 0;
    bool m_stopBands = false;

    // Asynchronous rasterisation: Submit queues primitives for a worker thread
    // and every other entry point drains the queue first, so the observable
    // order of GS operations is unchanged. Opt in with PS2X_GS_ASYNC=1 (callers
    // that read VRAM memory directly, like unit tests, need inline drawing).
    void DrainQueue() const;
    void WorkerLoop();
    bool m_asyncResolved = false;
    bool m_async = false;
    mutable std::mutex m_queueMutex;
    mutable std::condition_variable m_queueCv;
    mutable std::condition_variable m_idleCv;
    struct QueuedWork
    {
        enum class Kind : uint8_t
        {
            Draw,
            Present,
            Clut,
            TextureFlush,
            Transfer,
            Upload
        };
        Kind kind = Kind::Draw;
        GSPrimitiveBatch batch{};
        GSPresentationRequest request{};
        PresentSink sink;
        GSTex0Reg tex0{};
        GSTexClutReg texclut{};
        GSTransferCommand transfer{};
        std::vector<uint8_t> payload;
    };
    bool EnqueueWork(const std::function<void(QueuedWork &)> &fill);
    void LoadClutStateUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    void TextureFlushUnlocked();
    void BeginTransferUnlocked(const GSTransferCommand &command);
    void UploadImageUnlocked(const uint8_t *data, uint32_t sizeBytes);
    std::vector<QueuedWork> m_queue;
    bool m_workerBusy = false;
    // Presentation snapshots queued but not yet taken. The producer may run at
    // most kMaxPendingPresents frames ahead of the rasteriser.
    static constexpr uint32_t kMaxPendingPresents = 2u;
    uint32_t m_pendingPresents = 0u;
    // Bound on queued plus in-flight work items, for scenes that draw without
    // flipping (loading screens).
    static constexpr size_t kMaxQueuedWork = 32768u;
    size_t m_inFlightWork = 0u;
    std::condition_variable m_presentCv;
    bool m_stopWorker = false;
    std::thread m_worker;
    void MarkVramWriteUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void InvalidateDecodedTexturesUnlocked();
    DecodedTexture *PrepareTextureCacheUnlocked(const GSDrawState &state, bool cacheSmall);
    uint32_t DecodeTexelUnlocked(const GSDrawState &state, const std::array<uint16_t, 512> &clut, int u, int v);
    uint32_t FetchDecodedTexelUnlocked(const GSDrawState &state, int u, int v);

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;
};
