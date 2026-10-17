#include "ps2_settings_menu.h"

#include "ps2_runtime.h"
#include "runtime/ps2_settings.h"
#include "runtime/ps2_touch_controls.h"

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
        {"Vídeo", "video"},   {"Desempenho", "performance"}, {"Áudio", "audio"},       {"Controles", nullptr},
        {"Toque", nullptr},   {"Saves", "saves"},            {"Hacks", nullptr},       {"Avançado", "advanced"},
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
#if defined(__ANDROID__)
    // A phone packs ~400 pixels per inch: size the menu from the screen's short side.
    m_scale = std::clamp(static_cast<float>(std::min(GetScreenWidth(), GetScreenHeight())) / 540.0f, 1.0f, 3.0f);
    ImGui::GetStyle().ScaleAllSizes(m_scale);
#endif
    // The default font has no accented letters; ImGui 1.92 loads glyphs on demand.
    for (const char *candidate : {"C:\\Windows\\Fonts\\segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  "/System/Library/Fonts/Supplemental/Arial.ttf", "/system/fonts/Roboto-Regular.ttf"})
    {
        std::error_code error;
        if (std::filesystem::exists(candidate, error))
        {
            m_font = io.Fonts->AddFontFromFileTTF(candidate, 18.0f * m_scale);
            break;
        }
    }
}

void PS2SettingsMenu::close()
{
    std::fprintf(stderr, "[menu] closed\n");
    m_visible = false;
    settings::save();
    settings::setInputBlocked(false);
}

void PS2SettingsMenu::draw(PS2Runtime &runtime)
{
    (void)runtime;
    // The touch position editor owns the screen; F1 or Back leave it without saving.
    if (ps2x::touch::editing())
    {
        if (IsKeyPressed(KEY_F1)
#if defined(__ANDROID__)
            || IsKeyPressed(KEY_BACK)
#endif
        )
            ps2x::touch::cancelEditing();
        return;
    }
    // F1 on a keyboard; the menu button of the touch controls or Back on a phone.
    const bool toggle = IsKeyPressed(KEY_F1) || ps2x::touch::takeMenuRequest()
#if defined(__ANDROID__)
                        || IsKeyPressed(KEY_BACK)
#endif
        ;
    if (toggle)
    {
        if (m_visible)
            close();
        else
        {
            std::fprintf(stderr, "[menu] opened\n");
            m_visible = true;
            settings::setInputBlocked(true);
        }
    }
    if (!m_visible)
        return;

    rlImGuiBegin();
    if (m_font)
        ImGui::PushFont(static_cast<ImFont *>(m_font), 18.0f * m_scale);

    const ImVec2 screen(static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight()));
    const ImVec2 size(std::min(760.0f * m_scale, screen.x - 20.0f * m_scale),
                      std::min(560.0f * m_scale, screen.y - 20.0f * m_scale));
    ImGui::SetNextWindowPos(ImVec2((screen.x - size.x) * 0.5f, (screen.y - size.y) * 0.5f), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(size, ImGuiCond_Appearing);
    bool open = true;
#if defined(__ANDROID__)
    const char *title = "Opções";
#else
    const char *title = "Opções  (F1 fecha)";
#endif
    if (ImGui::Begin(title, &open, ImGuiWindowFlags_NoCollapse))
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
                else if (std::strcmp(tab.title, "Toque") == 0)
                    drawTouch();
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
        const float buttons = 220.0f * m_scale;
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
    for (const settings::Definition &definition : settings::definitions())
    {
        if (std::strcmp(definition.section, section) != 0 || definition.kind == settings::Kind::Text ||
            std::strncmp(definition.key, "show_", 5) == 0)
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
                const bool available =
                    index >= definition.choiceAvailable.size() || definition.choiceAvailable[index];
                ImGui::BeginDisabled(!available);
                if (ImGui::RadioButton(definition.choiceLabels[index].c_str(), value == definition.choices[index]))
                    settings::set(definition.section, definition.key, definition.choices[index]);
                ImGui::EndDisabled();
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
    helpLine("Teclado: clique em Alterar e aperte a tecla nova. Um controle conectado tem prioridade "
             "sobre o teclado e usa o mapa padrão. F1 abre este menu e F2 o painel de depuração.");
    if (m_capture >= 0)
    {
        // The first key pressed (F1/F2 aside) becomes the button's key; Backspace clears it.
        for (int key = GetKeyPressed(); key != 0; key = GetKeyPressed())
        {
            if (key == KEY_F1 || key == KEY_F2)
                continue;
            const auto &buttons = settings::padButtons();
            const std::string name = key == KEY_BACKSPACE ? std::string() : settings::keyName(key);
            if (key == KEY_BACKSPACE || !name.empty())
            {
                settings::set("controls", buttons[static_cast<size_t>(m_capture)].key, name);
                m_capture = -1;
                break;
            }
        }
    }
    if (ImGui::BeginTable("keys", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV))
    {
        ImGui::TableSetupColumn("Botão do PS2");
        ImGui::TableSetupColumn("Teclas");
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableHeadersRow();
        const auto &buttons = settings::padButtons();
        for (size_t index = 0; index < buttons.size(); ++index)
        {
            ImGui::PushID(static_cast<int>(index));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(buttons[index].label);
            ImGui::TableNextColumn();
            if (m_capture == static_cast<int>(index))
                ImGui::TextColored(ImVec4(0.9f, 0.75f, 0.35f, 1.0f), "aperte uma tecla (Backspace limpa)");
            else
            {
                const std::string keys = settings::get("controls", buttons[index].key);
                ImGui::TextUnformatted(keys.empty() ? "-" : keys.c_str());
            }
            ImGui::TableNextColumn();
            if (ImGui::SmallButton(m_capture == static_cast<int>(index) ? "Cancelar" : "Alterar"))
                m_capture = m_capture == static_cast<int>(index) ? -1 : static_cast<int>(index);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void PS2SettingsMenu::drawTouch()
{
    drawSettings("touch");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Controles na tela");
    helpLine("Escolha quais aparecem. O analógico direito move a câmera; L3 e R3 são o clique dos analógicos.");
    if (ImGui::BeginTable("touch-controls", 3))
    {
        for (const settings::Definition &definition : settings::definitions())
        {
            if (std::strcmp(definition.section, "touch") != 0 || std::strncmp(definition.key, "show_", 5) != 0)
                continue;
            ImGui::TableNextColumn();
            bool on = settings::getBool(definition.section, definition.key);
            if (ImGui::Checkbox(definition.label, &on))
                settings::set(definition.section, definition.key, on ? "true" : "false");
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Posições");
    helpLine("Editar abre a tela de jogo com os controles: arraste cada um para onde quiser, use - e + para o "
             "tamanho e toque em Salvar. Retrato e paisagem guardam posições separadas.");
    if (ImGui::Button("Editar posições"))
    {
        close();
        ps2x::touch::beginEditing();
    }
    ImGui::SameLine();
    if (ImGui::Button("Posições padrão"))
    {
        settings::set("touch", "layout_landscape", "");
        settings::set("touch", "layout_portrait", "");
    }
}

void PS2SettingsMenu::drawHacks()
{
    drawSettings("game");
    ImGui::Separator();
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
