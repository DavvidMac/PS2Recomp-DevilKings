#ifndef PS2_SETTINGS_MENU_H
#define PS2_SETTINGS_MENU_H

class PS2Runtime;

// In-game settings menu (F1) over the game, editing ps2x::settings (settings.ini).
class PS2SettingsMenu
{
public:
    // After rlImGuiSetup: adds a font with accented letters.
    void initialize();
    void draw(PS2Runtime &runtime);

    bool isVisible() const { return m_visible; }

private:
    void drawSettings(const char *section);
    void drawControls();
    void drawTouch();
    void drawHacks();
    void close();

    bool m_initialized = false;
    bool m_visible = false;
    void *m_font = nullptr; // ImFont*
    float m_scale = 1.0f;   // UI size: 1 on a desktop, larger on a phone's dense screen
    char m_pathBuffer[512] = {};
    int m_capture = -1; // pad button waiting for a key
};

#endif // PS2_SETTINGS_MENU_H
