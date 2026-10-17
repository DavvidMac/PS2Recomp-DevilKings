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
// land there, and textures are decoded from it on the submitting thread, in stream order. The
// frame buffers the game draws to live on the GPU, as textures of the size of the buffer; a
// texture that is one of them (same base, width and format) is sampled straight from the GPU.
// Before the guest or a texture reads other memory a GPU buffer covers, or an upload writes
// there, the buffer is copied back into local memory.
//
// Draws are recorded on the threads that feed the GS and replayed by a thread of the renderer's
// own, in a GL context that shares objects with the window's. The picture is composed on the
// GPU at each display swap, and the main thread draws it from that texture: no copy through the
// CPU.
//
// PS2X_GLES_TRACE=1 logs what the renderer approximates (blend formulas, DATE, PABE) and what
// made it copy a buffer back, once each, and a line every 300 swaps with the draws, texture
// uploads and copies back. PS2X_GLES_FRAMELOG=<swap> logs every draw, palette load and transfer
// of two swaps from that one.
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
