#pragma once

#include <cstdint>

// On-screen controls for phones: both sticks, a d-pad, the face and shoulder buttons, L3/R3,
// Start/Select and a menu button, drawn over the picture and read by the pad like a gamepad.
// The [touch] settings (Toque tab) turn them on, pick which ones show, size them and set how
// see-through they are; the position editor moves and resizes each one, per orientation.
namespace ps2x::touch
{
    // Once per presented frame, on the window's thread, after the picture is drawn: reads the
    // touches (the mouse on a desktop, for testing) and draws the controls (or the editor).
    void frame(int screenWidth, int screenHeight);

    // Whether the controls are on (touch.mode, and on "auto" a phone with no gamepad).
    bool visible();

    // In portrait the picture goes to the top of the screen and the controls below it.
    bool pictureAtTop(int screenWidth, int screenHeight);

    // Buttons held on the screen as PS2 pad bits (set = held), and each stick (-1..1) when held.
    uint16_t buttons();
    bool stick(float &x, float &y);
    bool rightStick(float &x, float &y);

    // True once after the menu button was tapped.
    bool takeMenuRequest();

    // The position editor (Toque tab > Editar posições): every control shown can be dragged
    // and resized for the current orientation; the game sees no input until it is saved or
    // cancelled.
    void beginEditing();
    void cancelEditing();
    bool editing();
}
