#ifndef PS2_GS_GLES_H
#define PS2_GS_GLES_H

#include "runtime/gs/gs_backend.h"

#include <cstdint>
#include <memory>

// OpenGL renderer: draws the GS primitives with OpenGL ES 3.0 (OpenGL 3.3 on the desktop),
// for the phones whose Vulkan cannot run paraLLEl-GS. Selected with PS2X_RENDERER=opengl
// (settings: Video > Renderer).
//
// The backend keeps a CPU backend for GS local memory: uploads, transfers and palette loads
// land there, and textures are decoded from it in stream order. The frame buffers the game
// draws to live on the GPU, as textures of the size of the buffer (alpha in 1/128 units, so GS
// blending maps to GL blend factors). A texture that is one of them is sampled from the GPU; a
// texture inside one in another layout (32-bit memory read as 16-bit or 8-bit indices, the
// usual post-processing tricks) is read by the shader through GS memory, with page tables
// built from the CPU backend's own swizzles. What GL blending and colour masks cannot express
// (partial FBMSK, DATE, PABE, some blend forms) the shader does reading a copy of the frame
// buffer. Anything else that reads memory a GPU buffer covers, or an upload over it, copies the
// buffer back into local memory first.
//
// The GS threads only queue calls; a worker runs them in order and records draws, and the
// renderer's own GL thread replays them in a context that shares objects with the window's.
// The picture is composed (and half-height fields woven) on the GPU at each display swap, and
// the main thread draws it from that texture: no copy through the CPU.
//
// PS2X_GLES_TRACE=1 logs what made the renderer copy a buffer back, once each, and a line every
// 300 swaps with the draws, texture uploads and copies back. PS2X_GLES_FRAMELOG=<swap> logs every
// draw, palette load and transfer of two swaps from that one. PS2X_GLES_COMPARE=<dir> feeds a
// software renderer the same calls, compares every picture with it (a line every 300 swaps) and
// saves the ones that differ side by side, with the difference, as PPM files in <dir>.
namespace ps2x::gles
{
    // PS2X_RENDERER=opengl.
    bool requested();
    std::unique_ptr<GSRasterBackend> createBackend();

    // Main thread, once the window exists: creates the renderer's context and starts its
    // thread. False when the GL context could not be created (the software renderer stays).
    bool attachToWindow();
    // attachToWindow() succeeded.
    bool active();

    // Main thread: the newest picture, a texture of the window's context (RGBA8, row 0 at the
    // top). False until the first display swap.
    bool displayTexture(unsigned &textureId, uint32_t &width, uint32_t &height);

    void shutdown();
}

#endif
