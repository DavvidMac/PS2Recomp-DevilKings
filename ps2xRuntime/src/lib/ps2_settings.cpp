#include "runtime/ps2_settings.h"

#include "raylib.h"

#include <algorithm>
#include <atomic>
#include <ctime>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>

namespace ps2x::settings
{
    namespace
    {
        std::string trim(const std::string &text)
        {
            size_t begin = 0;
            size_t end = text.size();
            while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])))
                ++begin;
            while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
                --end;
            return text.substr(begin, end - begin);
        }

        std::string lower(std::string text)
        {
            std::transform(text.begin(), text.end(), text.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return text;
        }

        void setEnvironment(const char *name, const std::string &value)
        {
#ifdef _WIN32
            _putenv_s(name, value.c_str());
#else
            setenv(name, value.c_str(), 1);
#endif
        }

        std::string keyOf(const char *section, const char *key)
        {
            return std::string(section) + "." + key;
        }

        std::atomic<uint64_t> g_version{1};

        struct KeyName
        {
            int code;
            const char *name;
        };

        // Keys a binding may use (F1 and F2 open the menu and the debug panel).
        const std::vector<KeyName> &keyNames()
        {
            static const std::vector<KeyName> s_names = []()
            {
                std::vector<KeyName> names = {
                    {KEY_UP, "UP"}, {KEY_DOWN, "DOWN"}, {KEY_LEFT, "LEFT"}, {KEY_RIGHT, "RIGHT"},
                    {KEY_SPACE, "SPACE"}, {KEY_ENTER, "ENTER"}, {KEY_TAB, "TAB"}, {KEY_ESCAPE, "ESCAPE"},
                    {KEY_BACKSPACE, "BACKSPACE"}, {KEY_INSERT, "INSERT"}, {KEY_DELETE, "DELETE"},
                    {KEY_HOME, "HOME"}, {KEY_END, "END"}, {KEY_PAGE_UP, "PAGE_UP"}, {KEY_PAGE_DOWN, "PAGE_DOWN"},
                    {KEY_LEFT_SHIFT, "LEFT_SHIFT"}, {KEY_RIGHT_SHIFT, "RIGHT_SHIFT"},
                    {KEY_LEFT_CONTROL, "LEFT_CONTROL"}, {KEY_RIGHT_CONTROL, "RIGHT_CONTROL"},
                    {KEY_LEFT_ALT, "LEFT_ALT"}, {KEY_RIGHT_ALT, "RIGHT_ALT"},
                    {KEY_COMMA, "COMMA"}, {KEY_PERIOD, "PERIOD"}, {KEY_SLASH, "SLASH"}, {KEY_SEMICOLON, "SEMICOLON"},
                    {KEY_MINUS, "MINUS"}, {KEY_EQUAL, "EQUAL"}, {KEY_APOSTROPHE, "APOSTROPHE"}, {KEY_GRAVE, "GRAVE"},
                    {KEY_LEFT_BRACKET, "LEFT_BRACKET"}, {KEY_RIGHT_BRACKET, "RIGHT_BRACKET"},
                    {KEY_BACKSLASH, "BACKSLASH"}, {KEY_KP_ENTER, "KP_ENTER"},
                };
                static const char *const kLetters[] = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                                                       "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
                for (int index = 0; index < 26; ++index)
                    names.push_back({KEY_A + index, kLetters[index]});
                static const char *const kDigits[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
                static const char *const kPad[] = {"KP_0", "KP_1", "KP_2", "KP_3", "KP_4",
                                                   "KP_5", "KP_6", "KP_7", "KP_8", "KP_9"};
                for (int index = 0; index < 10; ++index)
                {
                    names.push_back({KEY_ZERO + index, kDigits[index]});
                    names.push_back({KEY_KP_0 + index, kPad[index]});
                }
                static const char *const kFunctions[] = {"F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"};
                for (int index = 0; index < 10; ++index)
                    names.push_back({KEY_F3 + index, kFunctions[index]});
                return names;
            }();
            return s_names;
        }

        struct State
        {
            std::mutex mutex;
            bool loaded = false;
            bool disabled = false;
            std::filesystem::path file;
            std::filesystem::path elfDirectory;
            std::map<std::string, std::string> values;    // section.key -> value
            std::map<std::string, std::string> atLoad;    // values the runtime started with
            std::map<std::string, bool> fromEnvironment;  // section.key -> the environment set it
            std::vector<CheatGroup> cheats;
            std::shared_ptr<const std::vector<FrameWrite>> frameWrites =
                std::make_shared<const std::vector<FrameWrite>>();
        };

        State &state()
        {
            static State s_state;
            return s_state;
        }

        const Definition *find(const char *section, const char *key)
        {
            for (const Definition &definition : definitions())
            {
                if (std::string(definition.section) == section && std::string(definition.key) == key)
                    return &definition;
            }
            return nullptr;
        }

        // The PS2X_* value for a setting, or empty for "leave the runtime default".
        std::string environmentValue(const Definition &definition, const std::string &value)
        {
            if (value == "auto")
                return std::string();
            if (definition.kind == Kind::Choice && !definition.choiceEnvironment.empty())
            {
                for (size_t index = 0; index < definition.choices.size(); ++index)
                {
                    if (definition.choices[index] == value && index < definition.choiceEnvironment.size())
                        return definition.choiceEnvironment[index];
                }
                return std::string();
            }
            if (definition.kind == Kind::Bool)
                return value == "true" ? "1" : "0";
            if (definition.kind == Kind::Number)
                return value == "0" && definition.autoLabel ? std::string() : value;
            return value;
        }

        std::string fromEnvironmentValue(const Definition &definition, const char *value)
        {
            if (definition.kind == Kind::Bool)
                return (value[0] == '0' || value[0] == '\0') ? "false" : "true";
            if (definition.kind == Kind::Choice && !definition.choiceEnvironment.empty())
            {
                for (size_t index = 0; index < definition.choiceEnvironment.size(); ++index)
                {
                    if (!definition.choiceEnvironment[index].empty() && definition.choiceEnvironment[index] == value)
                        return definition.choices[index];
                }
                return definition.defaultValue;
            }
            return value;
        }

        // PCSX2 .pnach patches: patch=<place>,EE,<address>,<byte|short|word|extended>,<value>.
        bool parsePatch(const std::string &line, FrameWrite &write)
        {
            std::vector<std::string> fields;
            std::stringstream stream(line.substr(line.find('=') + 1));
            std::string field;
            while (std::getline(stream, field, ','))
                fields.push_back(trim(field));
            if (fields.size() < 5 || lower(fields[1]) != "ee")
                return false;
            uint32_t address = static_cast<uint32_t>(std::strtoul(fields[2].c_str(), nullptr, 16));
            const uint32_t value = static_cast<uint32_t>(std::strtoul(fields[4].c_str(), nullptr, 16));
            const std::string type = lower(fields[3]);
            uint8_t size = 0;
            if (type == "byte")
                size = 1;
            else if (type == "short")
                size = 2;
            else if (type == "word")
                size = 4;
            else if (type == "extended")
            {
                // CodeBreaker-style constant writes: 0aaaaaaa (8 bit), 1aaaaaaa (16), 2aaaaaaa (32).
                const uint32_t code = address >> 28;
                size = code == 0u ? 1u : code == 1u ? 2u : code == 2u ? 4u : 0u;
                address &= 0x0FFFFFFFu;
            }
            if (size == 0u)
                return false;
            write.address = address & 0x01FFFFFFu & ~(static_cast<uint32_t>(size) - 1u);
            write.value = value;
            write.size = size;
            return true;
        }

        void loadCheats(State &s)
        {
            s.cheats.clear();
            const std::filesystem::path folder = s.elfDirectory / "cheats";
            std::error_code error;
            if (!std::filesystem::is_directory(folder, error))
                return;
            std::vector<std::filesystem::path> files;
            for (const auto &entry : std::filesystem::directory_iterator(folder, error))
            {
                if (entry.is_regular_file() && lower(entry.path().extension().string()) == ".pnach")
                    files.push_back(entry.path());
            }
            std::sort(files.begin(), files.end());
            for (const std::filesystem::path &path : files)
            {
                std::ifstream in(path);
                std::string line;
                CheatGroup current;
                current.name = path.stem().string();
                current.file = path.filename().string();
                auto flush = [&]()
                {
                    if (!current.writes.empty())
                        s.cheats.push_back(current);
                };
                while (std::getline(in, line))
                {
                    line = trim(line);
                    if (line.empty() || line[0] == ';' || line.rfind("//", 0) == 0)
                        continue;
                    if (line.front() == '[' && line.back() == ']')
                    {
                        flush();
                        current = CheatGroup{};
                        current.name = trim(line.substr(1, line.size() - 2));
                        current.file = path.filename().string();
                        continue;
                    }
                    const std::string lowered = lower(line);
                    if (lowered.rfind("description=", 0) == 0)
                        current.description = trim(line.substr(12));
                    else if (lowered.rfind("patch=", 0) == 0)
                    {
                        FrameWrite write{};
                        if (parsePatch(line, write))
                            current.writes.push_back(write);
                        else
                            std::cerr << "[settings] " << path.filename().string() << ": ignored '" << line << "'"
                                      << std::endl;
                    }
                }
                flush();
            }
        }

        // Rebuilds the enabled flags and the frame writes from [hacks] enabled.
        void refreshCheats(State &s)
        {
            const std::string list = s.values[keyOf("hacks", "enabled")];
            auto enabled = [&](const std::string &name)
            {
                std::stringstream stream(list);
                std::string item;
                while (std::getline(stream, item, '|'))
                {
                    if (trim(item) == name)
                        return true;
                }
                return false;
            };
            auto writes = std::make_shared<std::vector<FrameWrite>>();
            for (CheatGroup &group : s.cheats)
            {
                group.enabled = enabled(group.name);
                if (group.enabled)
                    writes->insert(writes->end(), group.writes.begin(), group.writes.end());
            }
            s.frameWrites = writes;
        }
    }

    const std::vector<Definition> &definitions()
    {
        static const std::vector<Definition> s_definitions = []()
        {
            std::vector<Definition> list;
            auto add = [&](Definition definition) { list.push_back(std::move(definition)); };
            // Vídeo
            {
                // Vulkan is paraLLEl-GS, in builds configured with PS2X_PGS_DIR (gs_pgs.h).
#if defined(PS2X_HAVE_PGS)
                constexpr bool kHaveVulkan = true;
#else
                constexpr bool kHaveVulkan = false;
#endif
                Definition renderer{"video", "renderer", "Renderizador",
                                    "Software desenha na CPU. Vulkan desenha na placa de vídeo (paraLLEl-GS, tão fiel "
                                    "quanto o Software) e permite resolução interna maior.",
                                    Kind::Choice, "software", "PS2X_RENDERER", {"software", "vulkan"},
                                    {"Software (CPU)", kHaveVulkan ? "Vulkan (GPU)" : "Vulkan (não incluído nesta build)"},
                                    0, 0, nullptr, false};
                renderer.choiceEnvironment = {"", "vulkan"};
                renderer.choiceAvailable = {true, kHaveVulkan};
                add(renderer);
                Definition scale{"video", "internal_resolution", "Resolução interna",
                                 "Só no Vulkan. 2x e 4x desenham com 4 e 16 amostras por pixel: bordas suaves e "
                                 "detalhe nítido em telas grandes, com mais trabalho para a placa de vídeo.",
                                 Kind::Choice, "1", "PS2X_PGS_SCALE", {"1", "2", "4"}, {"1x (nativa)", "2x", "4x"},
                                 0, 0, nullptr, false};
                scale.choiceEnvironment = {"", "2", "4"};
                scale.choiceAvailable = {true, kHaveVulkan, kHaveVulkan};
                add(scale);
            }
            add({"video", "window", "Janela", "Como o jogo ocupa o monitor.", Kind::Choice, "windowed", nullptr,
                 {"windowed", "fullscreen", "borderless"}, {"Janela", "Tela cheia", "Tela cheia sem borda"}, 0, 0, nullptr,
                 true});
            {
                Definition aspect{"video", "aspect", "Proporção",
                                  "4:3 é a TV para a qual o jogo foi feito. 16:9 e 21:9 abrem a cena dos lados nos jogos "
                                  "que suportam (o HUD 2D ainda estica).",
                                  Kind::Choice, "4:3", "PS2X_WIDESCREEN", {"4:3", "16:9", "21:9", "stretch"},
                                  {"4:3", "16:9", "21:9", "Esticar na janela"}, 0, 0, nullptr, false};
                aspect.choiceEnvironment = {"", "16:9", "21:9", ""};
                add(aspect);
            }
            add({"video", "filter", "Filtro da imagem",
                 "Nítido mostra os pixels; suave os mistura ao ampliar; scanlines e CRT imitam a TV.", Kind::Choice,
                 "smooth", nullptr, {"sharp", "smooth", "scanlines", "crt"}, {"Nítido", "Suave", "Scanlines", "CRT"}, 0,
                 0, nullptr, true});
            add({"video", "antialias", "Antisserrilhado (FXAA)", "Suaviza as bordas serrilhadas dos objetos.",
                 Kind::Bool, "false", nullptr, {}, {}, 0, 0, nullptr, true});
            add({"video", "vsync", "VSync", "Sincroniza com o monitor (evita imagem rasgada).", Kind::Bool, "true",
                 nullptr, {}, {}, 0, 0, nullptr, true});
            add({"video", "deinterlace", "Desentrelaçamento", "Junta os dois campos entrelaçados como uma TV faz.",
                 Kind::Bool, "true", "PS2X_DEINTERLACE", {}, {}, 0, 0, nullptr, false});
            // Desempenho
            add({"performance", "gs_threads", "Threads do renderizador",
                 "Threads que desenham a imagem na CPU. Automático usa 3/4 dos núcleos lógicos.", Kind::Number, "0",
                 "PS2X_GS_THREADS", {}, {}, 0, 32, "Automático", false});
            add({"performance", "vif1_thread", "VIF1 em thread própria",
                 "Processa a geometria em paralelo com a lógica do jogo, como no console.", Kind::Bool, "true",
                 "PS2X_VIF1_THREAD", {}, {}, 0, 0, nullptr, false});
            add({"performance", "vu1_threads", "Threads do VU1",
                 "Programas do VU1 em paralelo, mesmo resultado; cerca de +10% em batalha. Automático usa 3 com 8 "
                 "ou mais núcleos lógicos.", Kind::Choice, "auto", "PS2X_VU1_THREADS", {"auto", "0", "2", "3", "4"},
                 {"Automático", "Desligado", "2", "3", "4"}, 0, 0, nullptr, false});
            add({"performance", "vu1_depth", "Execuções do VU1 adiantadas",
                 "Quantas execuções do VU1 podem começar antes das anteriores terminarem.", Kind::Number, "0",
                 "PS2X_VU1_DEPTH", {}, {}, 0, 64, "Automático", false});
            add({"performance", "show_fps", "Mostrar FPS", "Contador de quadros por segundo no canto da tela.",
                 Kind::Bool, "false", nullptr, {}, {}, 0, 0, nullptr, true});
            // Áudio
            add({"audio", "volume", "Volume geral", "Volume de saída do jogo.", Kind::Number, "100", nullptr, {}, {}, 0,
                 100, nullptr, true});
            // Saves
            add({"saves", "memcard_folder", "Pasta do memory card 1",
                 "Pasta dos saves do slot 1 (o slot 2 usa a pasta irmã mc1). Relativa à pasta do jogo.", Kind::Path,
                 "mc0", nullptr, {}, {}, 0, 0, nullptr, false});
            // Jogo (ganchos que o override do jogo implementa)
            add({"game", "skip_intro", "Pular logos da abertura",
                 "Vai direto à abertura e ao título, nos jogos que suportam.", Kind::Bool, "false",
                 "PS2X_SKIP_INTRO", {}, {}, 0, 0, nullptr, false});
            // Saves: cópias de segurança
            add({"saves", "backups", "Cópias de segurança",
                 "Ao abrir o jogo, copia os memory cards para saves_backup/ e guarda as mais recentes.",
                 Kind::Number, "5", nullptr, {}, {}, 0, 20, "Desligado", false});
            // Controles (teclado): uma lista de teclas por botão
            for (const PadButton &button : padButtons())
            {
                static const std::map<std::string, std::string> kDefaults = {
                    {"up", "UP,W"},       {"down", "DOWN,S"},     {"left", "LEFT,A"},       {"right", "RIGHT,D"},
                    {"cross", "X,SPACE"}, {"circle", "C,ESCAPE"}, {"square", "Z,KP_0"},     {"triangle", "V,KP_1"},
                    {"l1", "Q"},          {"r1", "E"},            {"l2", "LEFT_SHIFT"},     {"r2", "RIGHT_SHIFT"},
                    {"start", "ENTER"},   {"select", "TAB"},      {"l3", ""},               {"r3", ""},
                };
                const auto found = kDefaults.find(button.key);
                Definition definition{"controls", button.key, button.label, "", Kind::Keys, "", nullptr, {}, {}};
                definition.defaultValue = found != kDefaults.end() ? found->second.c_str() : "";
                definition.live = true;
                add(definition);
            }
            // Avançado
            add({"advanced", "debug_panel", "Painel de depuração", "CPU, threads, GS, DMA e memória (F2).", Kind::Bool,
                 "false", nullptr, {}, {}, 0, 0, nullptr, true});
            add({"advanced", "frame_dump_dir", "Salvar quadros em",
                 "Pasta para salvar quadros (para comparar com o PCSX2). Vazio desliga.", Kind::Path, "",
                 "PS2X_FRAME_DUMP_DIR", {}, {}, 0, 0, nullptr, false});
            return list;
        }();
        return s_definitions;
    }

    void load(const std::filesystem::path &elfPath)
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.loaded)
            return;
        s.loaded = true;
        std::error_code error;
        s.elfDirectory = std::filesystem::absolute(elfPath, error).parent_path();
        const char *choice = std::getenv("PS2X_SETTINGS");
        if (choice && lower(choice) == "off")
            s.disabled = true;
        else if (choice && choice[0] != '\0')
            s.file = choice;
        else
            s.file = s.elfDirectory / "settings.ini";

        std::map<std::string, std::string> fileValues;
        if (!s.disabled)
        {
            std::ifstream in(s.file);
            std::string line;
            std::string section;
            while (std::getline(in, line))
            {
                line = trim(line);
                if (line.empty() || line[0] == ';' || line[0] == '#')
                    continue;
                if (line.front() == '[' && line.back() == ']')
                {
                    section = lower(trim(line.substr(1, line.size() - 2)));
                    continue;
                }
                const size_t equals = line.find('=');
                if (equals == std::string::npos || section.empty())
                    continue;
                std::string value = trim(line.substr(equals + 1));
                const size_t comment = value.find(" ;");
                if (comment != std::string::npos)
                    value = trim(value.substr(0, comment));
                fileValues[section + "." + lower(trim(line.substr(0, equals)))] = value;
            }
        }

        for (const Definition &definition : definitions())
        {
            const std::string key = keyOf(definition.section, definition.key);
            const auto inFile = fileValues.find(key);
            std::string value = inFile != fileValues.end() ? inFile->second : definition.defaultValue;
            if (definition.environment)
            {
                if (const char *env = std::getenv(definition.environment))
                {
                    value = fromEnvironmentValue(definition, env);
                    s.fromEnvironment[key] = true;
                }
                else if (inFile != fileValues.end())
                {
                    const std::string env = environmentValue(definition, value);
                    if (!env.empty())
                        setEnvironment(definition.environment, env);
                }
            }
            s.values[key] = value;
        }
        s.values[keyOf("hacks", "enabled")] = fileValues[keyOf("hacks", "enabled")];
        s.atLoad = s.values;
        loadCheats(s);
        refreshCheats(s);
        std::cerr << "[settings] " << (s.disabled ? std::string("off") : s.file.string()) << ", " << s.cheats.size()
                  << " cheat groups" << std::endl;
    }

    bool save()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.disabled || s.file.empty())
            return false;
        std::ofstream out(s.file, std::ios::trunc);
        if (!out)
            return false;
        out << "; Configurações do jogo, gravadas pelo menu (F1).\n";
        out << "; Uma variável PS2X_* definida no ambiente vale mais que este arquivo.\n";
        std::string section;
        for (const Definition &definition : definitions())
        {
            if (section != definition.section)
            {
                section = definition.section;
                out << "\n[" << section << "]\n";
            }
            out << definition.key << " = " << s.values[keyOf(definition.section, definition.key)] << "\n";
        }
        out << "\n[hacks]\n; grupos de cheats/*.pnach ligados, separados por |\n";
        out << "enabled = " << s.values[keyOf("hacks", "enabled")] << "\n";
        return static_cast<bool>(out);
    }

    std::filesystem::path filePath()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        return s.disabled ? std::filesystem::path() : s.file;
    }

    std::string get(const char *section, const char *key)
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto it = s.values.find(keyOf(section, key));
        if (it != s.values.end())
            return it->second;
        const Definition *definition = find(section, key);
        return definition ? definition->defaultValue : std::string();
    }

    bool getBool(const char *section, const char *key)
    {
        return get(section, key) == "true";
    }

    int getNumber(const char *section, const char *key)
    {
        return std::atoi(get(section, key).c_str());
    }

    void set(const char *section, const char *key, const std::string &value)
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.values[keyOf(section, key)] = value;
        if (std::string(section) == "hacks")
            refreshCheats(s);
        g_version.fetch_add(1u, std::memory_order_relaxed);
    }

    bool overriddenByEnvironment(const char *section, const char *key)
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        return s.fromEnvironment.count(keyOf(section, key)) != 0;
    }

    void resetToDefaults()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        for (const Definition &definition : definitions())
        {
            const std::string key = keyOf(definition.section, definition.key);
            if (!s.fromEnvironment.count(key))
                s.values[key] = definition.defaultValue;
        }
        s.values[keyOf("hacks", "enabled")].clear();
        refreshCheats(s);
        g_version.fetch_add(1u, std::memory_order_relaxed);
    }

    bool restartNeeded()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        for (const Definition &definition : definitions())
        {
            const std::string key = keyOf(definition.section, definition.key);
            if (!definition.live && s.values[key] != s.atLoad[key])
                return true;
        }
        return false;
    }

    std::vector<CheatGroup> cheatGroups()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        return s.cheats;
    }

    void setCheatEnabled(const std::string &name, bool enabled)
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        std::string list;
        for (const CheatGroup &group : s.cheats)
        {
            const bool on = group.name == name ? enabled : group.enabled;
            if (!on)
                continue;
            if (!list.empty())
                list += " | ";
            list += group.name;
        }
        s.values[keyOf("hacks", "enabled")] = list;
        refreshCheats(s);
    }

    std::shared_ptr<const std::vector<FrameWrite>> activeFrameWrites()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        return s.frameWrites;
    }

    uint64_t version()
    {
        return g_version.load(std::memory_order_relaxed);
    }

    const std::vector<PadButton> &padButtons()
    {
        static const std::vector<PadButton> s_buttons = {
            {"up", "Direcional ↑", 0x0010u},   {"down", "Direcional ↓", 0x0040u},
            {"left", "Direcional ←", 0x0080u}, {"right", "Direcional →", 0x0020u},
            {"cross", "X (Xis)", 0x4000u},     {"circle", "O (Bola)", 0x2000u},
            {"square", "Quadrado", 0x8000u},   {"triangle", "Triângulo", 0x1000u},
            {"l1", "L1", 0x0400u},             {"r1", "R1", 0x0800u},
            {"l2", "L2", 0x0100u},             {"r2", "R2", 0x0200u},
            {"start", "Start", 0x0008u},       {"select", "Select", 0x0001u},
            {"l3", "L3", 0x0002u},             {"r3", "R3", 0x0004u},
        };
        return s_buttons;
    }

    int keyCode(const std::string &name)
    {
        for (const KeyName &key : keyNames())
        {
            if (name == key.name)
                return key.code;
        }
        return 0;
    }

    std::string keyName(int code)
    {
        for (const KeyName &key : keyNames())
        {
            if (code == key.code)
                return key.name;
        }
        return std::string();
    }

    void backupMemoryCards(const std::filesystem::path &mcRoot, int keep)
    {
        if (keep <= 0)
            return;
        std::error_code error;
        std::vector<std::filesystem::path> cards;
        for (const std::filesystem::path &folder : {mcRoot, mcRoot.parent_path() / "mc1"})
        {
            if (std::filesystem::is_directory(folder, error) && !std::filesystem::is_empty(folder, error))
                cards.push_back(folder);
        }
        if (cards.empty())
            return;
        std::filesystem::path backups;
        {
            State &s = state();
            std::lock_guard<std::mutex> lock(s.mutex);
            backups = s.elfDirectory / "saves_backup";
        }
        char stamp[32];
        const std::time_t now = std::time(nullptr);
        std::tm local{};
#ifdef _WIN32
        localtime_s(&local, &now);
#else
        localtime_r(&now, &local);
#endif
        std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
        for (const std::filesystem::path &card : cards)
        {
            const std::filesystem::path target = backups / stamp / card.filename();
            std::filesystem::create_directories(target, error);
            std::filesystem::copy(card, target,
                                  std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing,
                                  error);
            if (error)
                std::cerr << "[settings] backup of " << card.string() << " failed: " << error.message() << std::endl;
        }
        // Keep the newest copies (the names sort by date).
        std::vector<std::filesystem::path> copies;
        for (const auto &entry : std::filesystem::directory_iterator(backups, error))
        {
            if (entry.is_directory())
                copies.push_back(entry.path());
        }
        std::sort(copies.begin(), copies.end());
        while (copies.size() > static_cast<size_t>(keep))
        {
            std::filesystem::remove_all(copies.front(), error);
            copies.erase(copies.begin());
        }
        std::cerr << "[settings] memory cards copied to " << (backups / stamp).string() << std::endl;
    }

    namespace
    {
        std::atomic<bool> g_inputBlocked{false};
        std::atomic<uint64_t> g_gameFrames{0};
    }

    void setInputBlocked(bool blocked)
    {
        g_inputBlocked.store(blocked, std::memory_order_relaxed);
    }

    bool inputBlocked()
    {
        return g_inputBlocked.load(std::memory_order_relaxed);
    }

    void countGameFrame()
    {
        g_gameFrames.fetch_add(1u, std::memory_order_relaxed);
    }

    uint64_t gameFrames()
    {
        return g_gameFrames.load(std::memory_order_relaxed);
    }
}
