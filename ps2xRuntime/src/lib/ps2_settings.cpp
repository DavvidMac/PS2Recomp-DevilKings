#include "runtime/ps2_settings.h"

#include <algorithm>
#include <atomic>
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
            add({"video", "window", "Janela", "Como o jogo ocupa o monitor.", Kind::Choice, "windowed", nullptr,
                 {"windowed", "fullscreen", "borderless"}, {"Janela", "Tela cheia", "Tela cheia sem borda"}, 0, 0, nullptr,
                 true});
            add({"video", "aspect", "Proporção", "4:3 é a proporção da TV para a qual o jogo foi feito.", Kind::Choice,
                 "4:3", nullptr, {"4:3", "stretch"}, {"4:3", "Esticar na janela"}, 0, 0, nullptr, true});
            add({"video", "filter", "Filtro da imagem", "Nítido mostra os pixels; suave os mistura ao ampliar.",
                 Kind::Choice, "smooth", nullptr, {"sharp", "smooth"}, {"Nítido", "Suave"}, 0, 0, nullptr, true});
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
