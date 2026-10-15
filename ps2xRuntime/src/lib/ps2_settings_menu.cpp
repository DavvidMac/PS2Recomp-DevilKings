#include "ps2_settings_menu.h"

#include "ps2_runtime.h"
#include "runtime/ps2_settings.h"

#include "imgui.h"
#include "raylib.h"
#include "rlImGui.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace
{
    namespace settings = ps2x::settings;

    struct Tab
    {
        const char *title;
        const char *section; // settings section, or nullptr for a custom tab
    };

    constexpr Tab kTabs[] = {
        {"Vídeo", "video"},       {"Desempenho", "performance"}, {"Áudio", "audio"}, {"Controles", nullptr},
        {"Saves", "saves"},       {"Hacks", nullptr},            {"Avançado", "advanced"},
    };

    void helpLine(const char *text)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("%s", text);
        ImGui::PopStyleColor();
    }

    // Path settings are relative to the folder of the game's ELF.
    std::filesystem::path resolvePath(const std::string &value)
    {
        std::filesystem::path path(value);
        if (path.is_relative())
            path = PS2Runtime::getIoPaths().elfDirectory / path;
        return path.lexically_normal();
    }
}

void PS2SettingsMenu::initialize()
{
    if (m_initialized)
        return;
    m_initialized = true;
    ImGuiIO &io = ImGui::GetIO();
    // The default font has no accented letters; ImGui 1.92 loads glyphs on demand.
    for (const char *candidate : {"C:\\Windows\\Fonts\\segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  "/System/Library/Fonts/Supplemental/Arial.ttf"})
    {
        std::error_code error;
        if (std::filesystem::exists(candidate, error))
        {
            m_font = io.Fonts->AddFontFromFileTTF(candidate, 18.0f);
            break;
        }
    }
}

void PS2SettingsMenu::close()
{
    m_visible = false;
    settings::save();
    settings::setInputBlocked(false);
}

void PS2SettingsMenu::draw(PS2Runtime &runtime)
{
    (void)runtime;
    if (IsKeyPressed(KEY_F1))
    {
        if (m_visible)
            close();
        else
        {
            m_visible = true;
            settings::setInputBlocked(true);
        }
    }
    if (!m_visible)
        return;

    rlImGuiBegin();
    if (m_font)
        ImGui::PushFont(static_cast<ImFont *>(m_font), 18.0f);

    const ImVec2 screen(static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight()));
    const ImVec2 size(std::min(760.0f, screen.x - 40.0f), std::min(560.0f, screen.y - 40.0f));
    ImGui::SetNextWindowPos(ImVec2((screen.x - size.x) * 0.5f, (screen.y - size.y) * 0.5f), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(size, ImGuiCond_Appearing);
    bool open = true;
    if (ImGui::Begin("Opções  (F1 fecha)", &open, ImGuiWindowFlags_NoCollapse))
    {
        if (ImGui::BeginTabBar("settings-tabs"))
        {
            for (const Tab &tab : kTabs)
            {
                if (!ImGui::BeginTabItem(tab.title))
                    continue;
                ImGui::BeginChild("tab-body", ImVec2(0.0f, -ImGui::GetFrameHeightWithSpacing() * 1.6f));
                if (tab.section)
                    drawSettings(tab.section);
                else if (std::strcmp(tab.title, "Controles") == 0)
                    drawControls();
                else
                    drawHacks();
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::Separator();
        if (settings::restartNeeded())
        {
            ImGui::TextColored(ImVec4(0.9f, 0.75f, 0.35f, 1.0f), "Algumas mudanças valem quando o jogo for reiniciado.");
            ImGui::SameLine();
        }
        const float buttons = 220.0f;
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowWidth() - buttons));
        if (ImGui::Button("Restaurar padrões"))
            settings::resetToDefaults();
        ImGui::SameLine();
        if (ImGui::Button("Fechar"))
            open = false;
    }
    ImGui::End();
    if (!open)
        close();

    if (m_font)
        ImGui::PopFont();
    rlImGuiEnd();
}

void PS2SettingsMenu::drawSettings(const char *section)
{
    if (std::strcmp(section, "video") == 0)
        helpLine("Em breve: renderizador por GPU, filtros CRT e scanlines, 16:9 e 21:9 sem esticar o HUD.");
    for (const settings::Definition &definition : settings::definitions())
    {
        if (std::strcmp(definition.section, section) != 0)
            continue;
        ImGui::PushID(definition.key);
        const bool locked = settings::overriddenByEnvironment(definition.section, definition.key);
        const std::string value = settings::get(definition.section, definition.key);
        ImGui::Spacing();
        ImGui::TextUnformatted(definition.label);
        if (!definition.live)
        {
            ImGui::SameLine();
            ImGui::TextDisabled("(ao reiniciar)");
        }
        ImGui::BeginDisabled(locked);
        switch (definition.kind)
        {
        case settings::Kind::Bool:
        {
            bool on = value == "true";
            if (ImGui::Checkbox("Ligado", &on))
                settings::set(definition.section, definition.key, on ? "true" : "false");
            break;
        }
        case settings::Kind::Choice:
            for (size_t index = 0; index < definition.choices.size(); ++index)
            {
                if (index != 0)
                    ImGui::SameLine();
                if (ImGui::RadioButton(definition.choiceLabels[index].c_str(), value == definition.choices[index]))
                    settings::set(definition.section, definition.key, definition.choices[index]);
            }
            break;
        case settings::Kind::Number:
        {
            int number = std::atoi(value.c_str());
            const char *format = (number == 0 && definition.autoLabel) ? definition.autoLabel : "%d";
            ImGui::SetNextItemWidth(260.0f);
            if (ImGui::SliderInt("##value", &number, definition.minimum, definition.maximum, format))
                settings::set(definition.section, definition.key, std::to_string(number));
            break;
        }
        case settings::Kind::Path:
        {
            std::snprintf(m_pathBuffer, sizeof(m_pathBuffer), "%s", value.c_str());
            ImGui::SetNextItemWidth(360.0f);
            if (ImGui::InputText("##value", m_pathBuffer, sizeof(m_pathBuffer)))
                settings::set(definition.section, definition.key, m_pathBuffer);
            if (!value.empty())
            {
                ImGui::SameLine();
                if (ImGui::Button("Abrir pasta"))
                {
                    std::error_code error;
                    const std::filesystem::path folder = resolvePath(value);
                    std::filesystem::create_directories(folder, error);
                    OpenURL(folder.string().c_str());
                }
            }
            break;
        }
        }
        ImGui::EndDisabled();
        if (locked)
            helpLine((std::string("Definido por ") + definition.environment + " no ambiente.").c_str());
        else
            helpLine(definition.help);
        ImGui::PopID();
    }
    if (std::strcmp(section, "advanced") == 0)
    {
        ImGui::Spacing();
        ImGui::Separator();
        const std::filesystem::path file = settings::filePath();
        helpLine(file.empty() ? "Arquivo de configurações desligado (PS2X_SETTINGS=off)."
                              : (std::string("Arquivo: ") + file.string()).c_str());
    }
}

void PS2SettingsMenu::drawControls()
{
    helpLine("Um controle conectado tem prioridade sobre o teclado. Remapear: em breve.");
    static const char *const kKeys[][2] = {
        {"Direcional", "Setas ou W A S D"}, {"X (Xis)", "X ou Espaço"},  {"O (Bola)", "C ou Esc"},
        {"Quadrado", "Z ou 0 do teclado numérico"}, {"Triângulo", "V ou 1 do teclado numérico"},
        {"L1 / R1", "Q / E"},                {"L2 / R2", "Shift esquerdo / direito"}, {"Start / Select", "Enter / Tab"},
        {"Opções / Depuração", "F1 / F2"},
    };
    if (ImGui::BeginTable("keys", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV))
    {
        ImGui::TableSetupColumn("Botão do PS2");
        ImGui::TableSetupColumn("Teclado");
        ImGui::TableHeadersRow();
        for (const auto &row : kKeys)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row[0]);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row[1]);
        }
        ImGui::EndTable();
    }
}

void PS2SettingsMenu::drawHacks()
{
    helpLine("Cada código escreve na memória do jogo a cada quadro, como um CodeBreaker. Começam desligados. "
             "Desbloqueios podem ficar gravados no memory card.");
    const std::vector<settings::CheatGroup> groups = settings::cheatGroups();
    if (groups.empty())
    {
        ImGui::Spacing();
        ImGui::TextWrapped("Nenhum código encontrado. Coloque arquivos .pnach (formato do PCSX2) na pasta cheats/ "
                           "ao lado do jogo; cada [seção] vira uma opção aqui.");
        return;
    }
    for (const settings::CheatGroup &group : groups)
    {
        ImGui::PushID(group.name.c_str());
        ImGui::Spacing();
        bool on = group.enabled;
        if (ImGui::Checkbox(group.name.c_str(), &on))
            settings::setCheatEnabled(group.name, on);
        std::string detail = group.description.empty() ? std::string() : group.description + "  ";
        detail += "(" + group.file + ", " + std::to_string(group.writes.size()) + " escrita" +
                  (group.writes.size() == 1 ? "" : "s") + ")";
        helpLine(detail.c_str());
        ImGui::PopID();
    }
}
