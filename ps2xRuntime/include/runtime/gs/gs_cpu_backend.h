#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
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

    // Replay support (ps2_gs_replay): restores the palette a PS2X_GS_RECORD
    // recording started with.
    void RestoreClutForReplay(const std::array<uint16_t, 512> &clut, const std::array<uint32_t, 2> &clutCbp);

private:
    PresentationFrame PresentSnapshot(const GSPresentationRequest &request);
    void NoteRecordedPresent(const GSPresentationRequest &request);
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
    struct PixelWriter;
    void SetupPixelWriter(const GSDrawState &state, PixelWriter &writer);
    void WritePixel(const PixelWriter &writer, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                    uint8_t fog);
    struct TextureSampler;
    uint32_t SampleTexture(const GSDrawState &state, const TextureSampler &sampler, float s, float t, float q,
                           uint16_t u, uint16_t v);
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
        // Draws of the list the band threads are rasterising that sample this entry.
        uint32_t inflightRefs = 0;
        std::vector<uint32_t> texels;
        std::vector<uint8_t> tileReady;
        // Tiles marked ready since the last reset (updated atomically by band threads).
        uint32_t readyTiles = 0;
        // Palette the tiles decode with, copied when the entry is created.
        std::array<uint16_t, 512> clut{};
    };
    // A Devil Kings battle frame samples about 60 textures; with 32 entries every
    // frame evicted and decoded dozens of them again.
    std::array<DecodedTexture, 128> m_decodedTextures{};
    // Entry the previous draw prepared; consecutive draws mostly share one.
    DecodedTexture *m_lastPreparedTexture = nullptr;
    // Texture state one primitive samples with, gathered once per primitive.
    struct TextureSampler
    {
        // Entry sampled by the primitive (s_activeTexture), or nullptr.
        DecodedTexture *texture = nullptr;
        // texture->texels when every tile was already decoded at setup.
        const uint32_t *texels = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        int texW = 1;
        int texH = 1;
        uint8_t wrapU = 0;
        uint8_t wrapV = 0;
        uint16_t minU = 0;
        uint16_t maxU = 0;
        uint16_t minV = 0;
        uint16_t maxV = 0;
    };
    static void SetupSampler(const GSDrawState &state, TextureSampler &sampler);
    // Everything WritePixel needs that only changes between primitives.
    struct PixelWriter
    {
        uint8_t *vram = nullptr;
        uint8_t *depthMemory = nullptr;
        ReadVramFunc readFrame = nullptr;
        WriteVramFunc writeFrame = nullptr;
        ReadVramFunc readDepth = nullptr;
        WriteVramFunc writeDepth = nullptr;
        uint64_t test = 0;
        int scissorX0 = 0;
        int scissorX1 = 0;
        int scissorY0 = 0;
        int scissorY1 = 0;
        uint32_t fbp = 0;
        uint32_t fbw = 1;
        uint32_t fpsm = 0;
        uint32_t zbp = 0;
        uint32_t zpsm = 0;
        uint32_t fbmsk = 0;
        uint32_t ztestMethod = 1;
        bool depthWrites = false;
        bool fog = false;
        bool blend = false;
        bool pabe = false;
        bool forceAlphaMsb = false;
        bool frame16 = false;
        bool frame24 = false;
        bool dateNeedsRead = false;
        bool markFrameWrites = false;
        bool markDepthWrites = false;
        uint8_t asel = 0;
        uint8_t bsel = 0;
        uint8_t csel = 0;
        uint8_t dsel = 0;
        uint8_t fix = 0;
        uint8_t fogR = 0;
        uint8_t fogG = 0;
        uint8_t fogB = 0;
    };
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
    // Why the pending list was rasterised (PS2X_GS_STATS).
    enum class FlushReason : uint8_t
    {
        Hazard,
        Clut,
        Size,
        Pinned,
        Transfer,
        Upload,
        Present,
        BatchEnd,
        Count
    };
    void QueueDrawUnlocked(const GSPrimitiveBatch &batch);
    void QueueClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    void FlushPendingDrawsUnlocked(FlushReason reason);
    // Waits for the list the band threads are rasterising (if any) and releases it.
    void JoinInflightUnlocked();
    void RunPendingBand(const std::vector<PendingDraw> &draws, uint32_t band, uint32_t bandCount);
    void BandLoop(uint32_t band);
    std::vector<PendingDraw> m_pendingDraws;
    // Handed to the band threads by FlushPendingDrawsUnlocked; the worker queues the
    // next list meanwhile and joins before anything that needs VRAM up to date.
    std::vector<PendingDraw> m_inflightDraws;
    std::bitset<512> m_inflightWritePages;
    bool m_inflight = false;
    // PS2X_GS_STATS: when the in-flight list was handed over, and when its last band finished.
    std::chrono::steady_clock::time_point m_inflightSubmitted{};
    std::chrono::steady_clock::time_point m_inflightFinished{};
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
    void ConvertTexelsUnlocked(const GSDrawState &state, const std::array<uint16_t, 512> &clut,
                               const uint32_t *raw, uint32_t *out, uint32_t count);
    uint32_t FetchDecodedTexelUnlocked(const GSDrawState &state, int u, int v);
    void WaitForTileUnlocked(const GSDrawState &state, DecodedTexture &entry, uint32_t tileX, uint32_t tileY);
    void DecodeTileUnlocked(const GSDrawState &state, DecodedTexture &entry, uint32_t tileX, uint32_t tileY,
                            GSMem::TexturePageCache *cache);
    // Set while band threads rasterise a list in which texture tiles may be decoded
    // in parallel, straight from VRAM (see FlushPendingDrawsUnlocked).
    bool m_parallelDecode = false;

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;
};
