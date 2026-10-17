#pragma once

#include <cstdint>

// On-screen controls for phones: a stick, the face and shoulder buttons, Start/Select and a
// menu button, drawn over the picture and read by the pad like a gamepad. The [touch]
// settings (Toque tab) turn them on, size them and set how see-through they are.
namespace ps2x::touch
{
    // Once per presented frame, on the window's thread, after the picture is drawn: reads the
    // touches (the mouse on a desktop, for testing) and draws the controls.
    void frame(int screenWidth, int screenHeight);

    // Whether the controls are on (touch.mode, and on "auto" a phone with no gamepad).
    bool visible();

    // In portrait the picture goes to the top of the screen and the controls below it.
    bool pictureAtTop(int screenWidth, int screenHeight);

    // Buttons held on the screen as PS2 pad bits (set = held), and the stick (-1..1) when held.
    uint16_t buttons();
    bool stick(float &x, float &y);

    // True once after the menu button was tapped.
    bool takeMenuRequest();
}
