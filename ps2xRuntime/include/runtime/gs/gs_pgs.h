#ifndef PS2_GS_PGS_H
#define PS2_GS_PGS_H

#include <cstdint>
#include <vector>

struct GSRegisters;

// Vulkan renderer: paraLLEl-GS (Arntzen Software, LGPL-3.0+) emulates the GS in Vulkan
// compute and is fed the same GIF stream as the CPU backend. Built when the runtime is
// configured with PS2X_PGS_DIR pointing at a paraLLEl-GS checkout (PS2X_HAVE_PGS), and
// selected with PS2X_RENDERER=vulkan (settings: Video > Renderer). PS2X_PGS_SCALE=1|2|4 sets
// the internal resolution (super-sampling 1x/4x/16x, scanned out at 1x/2x/4x).
//
// The GS frontend keeps parsing the stream (registers, SIGNAL/FINISH, uploads into its VRAM) but
// the CPU backend draws nothing; a GS->EE transfer first copies paraLLEl-GS's VRAM into the CPU
// backend's. PS2X_PGS_DUAL=1 keeps the CPU backend drawing too, for comparing the two. At each
// display swap the scanout is copied back to the host through a ring of three buffers, so the
// window shows it two swaps later without the game thread waiting for the GPU.
// Adapted from the BT3-Recomp integration (github.com/z3xox/BT3-Recomp, GPL-3.0).
namespace ps2x::pgs
{
    // GIF paths as paraLLEl-GS tracks them: each keeps its own GIF tag state across transfers.
    enum Path : uint32_t
    {
        PathVu1 = 0,   // PATH1, XGKICK
        PathVif1 = 1,  // PATH2
        PathGif = 2,   // PATH3
        PathHost = 3,  // writes the runtime makes itself (image uploads, stub register writes)
    };

#if defined(PS2X_HAVE_PGS)
    // PS2X_RENDERER=vulkan and the backend came up (false again after a failed start).
    bool active();
    // Active, and the CPU backend does not draw (unless PS2X_PGS_DUAL=1).
    bool exclusive();
    void setRegisters(GSRegisters *regs);
    void gifTransfer(uint32_t path, const uint8_t *data, uint32_t sizeBytes);
    // The game flipped the display (or the host latched it): scan out and queue the readback.
    void vsync();
    // The newest scanout copied back, RGBA8; false when none arrived since the last call. In
    // field mode (SMODE2 INT and FFMD) it is the field as drawn, for the caller to weave.
    bool takeFrame(std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height, bool &field);
    // Waits for every queued draw and copies GS local memory (sizeBytes from address 0) to dst.
    bool readVram(uint8_t *dst, uint32_t sizeBytes);
    void shutdown();
#else
    inline bool active() { return false; }
    inline bool exclusive() { return false; }
    inline void setRegisters(GSRegisters *) {}
    inline void gifTransfer(uint32_t, const uint8_t *, uint32_t) {}
    inline void vsync() {}
    inline bool takeFrame(std::vector<uint8_t> &, uint32_t &, uint32_t &, bool &) { return false; }
    inline bool readVram(uint8_t *, uint32_t) { return false; }
    inline void shutdown() {}
#endif
}

#endif
