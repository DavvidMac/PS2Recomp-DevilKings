#ifndef PS2_SETTINGS_H
#define PS2_SETTINGS_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Player settings: settings.ini next to the game's ELF (PS2X_SETTINGS=<path> picks another
// file, PS2X_SETTINGS=off ignores it), edited by the in-game menu (ps2_settings_menu.cpp).
// A setting backed by a PS2X_* variable reaches the runtime through that variable: set in
// the environment it wins over the file, which wins over the default. Settings marked
// `live` are read by the runtime while it runs; the others take effect at the next start.
namespace ps2x::settings
{
    enum class Kind : uint8_t
    {
        Bool,
        Choice, // one of `choices`
        Number, // integer in [minimum, maximum]; 0 may mean "auto" (see autoLabel)
        Path,
        Keys,   // keyboard keys, comma-separated raylib names (controls section)
    };

    struct Definition
    {
        const char *section;
        const char *key;
        const char *label;
        const char *help;
        Kind kind;
        const char *defaultValue;
        const char *environment;            // PS2X_* variable it maps to, or nullptr
        std::vector<std::string> choices;   // Choice: values as written to the file
        std::vector<std::string> choiceLabels;
        int minimum = 0;
        int maximum = 0;
        const char *autoLabel = nullptr;     // Number: shown for 0
        bool live = false;
    };

    // Every setting, in menu order.
    const std::vector<Definition> &definitions();

    // Loads the file (once; later calls return the loaded state) and sets the PS2X_* variables
    // the environment does not already set. Call before the runtime starts.
    void load(const std::filesystem::path &elfPath);
    // Writes every setting to the file (no-op when the file is disabled).
    bool save();
    std::filesystem::path filePath();

    // Value of section.key as written in the file (the default when absent).
    std::string get(const char *section, const char *key);
    bool getBool(const char *section, const char *key);
    int getNumber(const char *section, const char *key);
    void set(const char *section, const char *key, const std::string &value);
    // The environment overrides the file for this setting (the menu shows it locked).
    bool overriddenByEnvironment(const char *section, const char *key);
    // Back to the defaults (the file is rewritten on save()).
    void resetToDefaults();
    // A setting that only takes effect at the next start changed since load().
    bool restartNeeded();

    // ---------------------------------------------------------------- cheats

    struct FrameWrite
    {
        uint32_t address; // EE physical address
        uint32_t value;
        uint8_t size;     // 1, 2 or 4 bytes
    };

    // A [section] of a .pnach file in cheats/ next to the ELF: writes applied on every VSync
    // while it is enabled, like PCSX2 patches with place=1.
    struct CheatGroup
    {
        std::string name;
        std::string description;
        std::string file;
        std::vector<FrameWrite> writes;
        bool enabled = false;
    };

    std::vector<CheatGroup> cheatGroups();
    void setCheatEnabled(const std::string &name, bool enabled);
    // The writes of every enabled group (cheap to call once per VSync).
    std::shared_ptr<const std::vector<FrameWrite>> activeFrameWrites();

    // Increments on every change, for readers that cache what they parse from the settings.
    uint64_t version();

    // ---------------------------------------------------------------- keyboard and saves

    struct PadButton
    {
        const char *key;   // setting in [controls]
        const char *label; // shown in the menu
        uint16_t bit;      // pad button bit (active low in the pad report)
    };
    const std::vector<PadButton> &padButtons();
    int keyCode(const std::string &name); // raylib key, 0 when unknown
    std::string keyName(int code);         // empty when the key has no name here

    // Copies the memory card folders (mcRoot and its sibling mc1) to saves_backup/<date-time>
    // next to the ELF, keeping the newest `keep` copies; nothing when keep is 0 or the cards
    // are empty.
    void backupMemoryCards(const std::filesystem::path &mcRoot, int keep);

    // ---------------------------------------------------------------- runtime state for the UI

    // The menu is open: the game does not see the keyboard or the gamepad.
    void setInputBlocked(bool blocked);
    bool inputBlocked();
    // Frames the game finished (display buffer swaps), for the FPS counter.
    void countGameFrame();
    uint64_t gameFrames();
}

#endif
