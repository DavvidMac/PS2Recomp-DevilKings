#pragma once

#include <cstdint>
#include <vector>

// Widescreen HUD: when a game override widens the 3D view (PS2X_WIDESCREEN), the 2D HUD
// would stretch with the picture. This rewrites the X of HUD vertices inside each GIF
// packet, on a copy, before the GS front end and paraLLEl-GS read it, so the HUD keeps
// its 4:3 proportions. PS2X_WIDESCREEN_HUD picks the layout: centred (default), "edges"
// (pinned to the sides of the screen) or "stretch" (unchanged). Only frames with a 3D
// scene are widened: a 2D screen (menu, map, movie) is left as drawn and shown at 4:3.
namespace ps2x::hud
{
    // How a game's HUD is told apart from the rest; set by its override.
    struct Rule
    {
        uint8_t pathMask = 1u << 2;   // GIF paths that can carry HUD (bit n = PATH n)
        uint8_t scenePathMask = 0x3u; // paths of the 3D scene (0 = XGKICK straight from VU1)
        uint32_t minZ = 0xFFFFu;      // every HUD vertex has at least this Z
        uint32_t minTextureBase = 0;  // HUD textures sit at or above this TBP0 (frame buffers below)
        bool untexturedHud = true;    // untextured primitives can be HUD
        // The edges layout moves each piece of HUD (the vertices between two PRIM writes) whole to
        // the left side when its middle is left of edgeSplit, else to the right side. A piece at
        // least panelWidth wide is a menu panel: then everything is centred for a few vsyncs.
        // Fractions of the frame width.
        float edgeSplit = 0.5f;
        float panelWidth = 0.55f;
    };

    // GS registers the walker needs, as they are when a packet starts.
    struct GsState
    {
        bool primCtxt = false;
        bool prmodeCtxt = false;
        bool prmodeCont = true;
        bool primTme = false;
        bool prmodeTme = false;
        uint16_t ofx[2] = {0, 0};      // XYOFFSET OFX, 12.4
        uint32_t fbp[2] = {0, 0};      // FRAME FBP
        uint32_t tbp[2] = {0, 0};      // TEX0 TBP0
        uint32_t width[2] = {640, 640}; // FRAME FBW * 64
        uint64_t tick = 0;              // vsync count, for PS2X_WIDESCREEN_HUD_TRACE
    };

    // factor: HUD width / picture width, 0.75 for 16:9. Reads PS2X_WIDESCREEN_HUD.
    void enable(float factor, const Rule &rule);
    bool active();

    // At a display swap: whether the frame now shown had a 3D scene in the last few vsyncs.
    void noteSwap(uint32_t displayFbp, uint64_t tick);
    // True while the frame shown is a 2D screen: the presenter shows it at 4:3 with bars.
    bool pillarbox();

    // Returns data itself, or `scratch` holding a copy with the HUD squeezed.
    const uint8_t *apply(uint8_t path, const uint8_t *data, uint32_t sizeBytes, const GsState &state,
                         std::vector<uint8_t> &scratch);
}
