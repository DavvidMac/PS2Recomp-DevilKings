#include "runtime/ps2_touch_controls.h"

#include "runtime/ps2_settings.h"

#include "raylib.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ps2x::touch
{
    namespace
    {
        namespace settings = ps2x::settings;

        // PS2 pad bits, as in ps2_pad.cpp.
        constexpr uint16_t kSelect = 0x0001u;
        constexpr uint16_t kL3 = 0x0002u;
        constexpr uint16_t kR3 = 0x0004u;
        constexpr uint16_t kStart = 0x0008u;
        constexpr uint16_t kUp = 0x0010u;
        constexpr uint16_t kRight = 0x0020u;
        constexpr uint16_t kDown = 0x0040u;
        constexpr uint16_t kLeft = 0x0080u;
        constexpr uint16_t kL2 = 0x0100u;
        constexpr uint16_t kR2 = 0x0200u;
        constexpr uint16_t kL1 = 0x0400u;
        constexpr uint16_t kR1 = 0x0800u;
        constexpr uint16_t kTriangle = 0x1000u;
        constexpr uint16_t kCircle = 0x2000u;
        constexpr uint16_t kCross = 0x4000u;
        constexpr uint16_t kSquare = 0x8000u;

        // Written on the window's thread, read by the pad on the EE thread.
        std::atomic<uint16_t> g_buttons{0};
        std::atomic<bool> g_stickHeld{false};
        std::atomic<float> g_stickX{0.0f};
        std::atomic<float> g_stickY{0.0f};
        std::atomic<bool> g_rightHeld{false};
        std::atomic<float> g_rightX{0.0f};
        std::atomic<float> g_rightY{0.0f};
        std::atomic<bool> g_menuRequest{false};
        std::atomic<bool> g_visible{false};

        // ------------------------------------------------------------------ the controls

        enum Id : int
        {
            LeftStick,
            RightStick,
            Dpad,
            Face,
            L1,
            L2,
            R1,
            R2,
            L3,
            R3,
            Select,
            Start,
            Menu,
            Count
        };

        enum class Shape : uint8_t
        {
            Stick, // a: radius
            Dpad,  // a: radius
            Face,  // a: distance of each button from the centre, b: button radius
            Pill,  // a x b: width and height
            Round, // a: radius
        };

        struct Info
        {
            const char *id;   // in the layout settings
            const char *show; // [touch] setting that turns it on (nullptr: always)
            Shape shape;
            float a;          // sizes in units of a tenth of the screen's short side
            float b;
            uint16_t bit;
            const char *label;
        };

        constexpr Info kInfo[Count] = {
            {"ls", "show_left_stick", Shape::Stick, 1.45f, 0.0f, 0, ""},
            {"rs", "show_right_stick", Shape::Stick, 1.25f, 0.0f, 0, ""},
            {"dpad", "show_dpad", Shape::Dpad, 1.35f, 0.0f, 0, ""},
            {"face", "show_face", Shape::Face, 1.15f, 0.62f, 0, ""},
            {"l1", "show_l1", Shape::Pill, 1.7f, 0.75f, kL1, "L1"},
            {"l2", "show_l2", Shape::Pill, 1.7f, 0.75f, kL2, "L2"},
            {"r1", "show_r1", Shape::Pill, 1.7f, 0.75f, kR1, "R1"},
            {"r2", "show_r2", Shape::Pill, 1.7f, 0.75f, kR2, "R2"},
            {"l3", "show_l3", Shape::Pill, 1.1f, 0.6f, kL3, "L3"},
            {"r3", "show_r3", Shape::Pill, 1.1f, 0.6f, kR3, "R3"},
            {"select", "show_select", Shape::Pill, 1.3f, 0.55f, kSelect, "SELECT"},
            {"start", "show_start", Shape::Pill, 1.3f, 0.55f, kStart, "START"},
            {"menu", nullptr, Shape::Round, 0.42f, 0.0f, 0, ""},
        };

        // A control moved or resized by the editor: centre in fractions of the screen.
        struct Override
        {
            bool set = false;
            float x = 0.0f;
            float y = 0.0f;
            float scale = 1.0f;
        };
        using Overrides = std::array<Override, Count>;

        Overrides parseLayout(const std::string &text)
        {
            Overrides out{};
            size_t begin = 0;
            while (begin < text.size())
            {
                size_t end = text.find(';', begin);
                if (end == std::string::npos)
                    end = text.size();
                const std::string item = text.substr(begin, end - begin);
                begin = end + 1;
                const size_t equals = item.find('=');
                if (equals == std::string::npos)
                    continue;
                const std::string id = item.substr(0, equals);
                float x = 0.0f, y = 0.0f, scale = 1.0f;
                if (std::sscanf(item.c_str() + equals + 1, "%f,%f,%f", &x, &y, &scale) < 2)
                    continue;
                for (int i = 0; i < Count; ++i)
                    if (id == kInfo[i].id)
                        out[static_cast<size_t>(i)] = {true, std::clamp(x, 0.0f, 1.0f), std::clamp(y, 0.0f, 1.0f),
                                                       std::clamp(scale, 0.5f, 2.0f)};
            }
            return out;
        }

        std::string formatLayout(const Overrides &layout)
        {
            std::string out;
            char item[64];
            for (int i = 0; i < Count; ++i)
            {
                const Override &o = layout[static_cast<size_t>(i)];
                if (!o.set)
                    continue;
                std::snprintf(item, sizeof(item), "%s=%.4f,%.4f,%.2f;", kInfo[i].id, o.x, o.y, o.scale);
                out += item;
            }
            return out;
        }

        struct Config
        {
            std::string mode = "auto";
            float opacity = 0.55f;
            float scale = 1.0f;
            std::array<bool, Count> shown{};
            Overrides landscape{};
            Overrides portrait{};
        };

        const Config &config()
        {
            static Config s_config;
            static uint64_t s_version = ~0ull;
            if (settings::version() != s_version)
            {
                s_version = settings::version();
                s_config.mode = settings::get("touch", "mode");
                s_config.opacity = std::clamp(settings::getNumber("touch", "opacity"), 10, 100) / 100.0f;
                s_config.scale = std::clamp(settings::getNumber("touch", "size"), 50, 200) / 100.0f;
                for (int i = 0; i < Count; ++i)
                    s_config.shown[static_cast<size_t>(i)] = !kInfo[i].show || settings::getBool("touch", kInfo[i].show);
                s_config.landscape = parseLayout(settings::get("touch", "layout_landscape"));
                s_config.portrait = parseLayout(settings::get("touch", "layout_portrait"));
            }
            return s_config;
        }

        bool wanted()
        {
            const Config &c = config();
            if (c.mode == "on")
                return true;
            if (c.mode != "auto")
                return false;
#if defined(__ANDROID__)
            return !IsGamepadAvailable(0);
#else
            return false;
#endif
        }

        // Where every control sits on this screen, in pixels.
        struct Layout
        {
            float u = 0.0f; // size unit: a tenth of the short side, times touch.size
            float w = 0.0f;
            float h = 0.0f;
            bool portrait = false;
            float top = 0.0f; // portrait: where the controls' area starts
            std::array<Vector2, Count> centre{};
            std::array<float, Count> scale{};
            std::array<bool, Count> shown{};
            float unit(int i) const { return u * scale[static_cast<size_t>(i)]; }
        };

        std::array<Vector2, Count> defaultCentres(const Layout &l)
        {
            const float u = l.u;
            const float w = l.w;
            const float h = l.h;
            std::array<Vector2, Count> c{};
            if (!l.portrait)
            {
                c[LeftStick] = {2.25f * u, h - 1.95f * u};
                c[RightStick] = {w - 5.6f * u, h - 1.5f * u};
                c[Dpad] = {2.25f * u, h - 5.3f * u};
                c[Face] = {w - 2.27f * u, h - 1.95f * u};
                c[L2] = {1.25f * u, 0.775f * u};
                c[L1] = {1.25f * u, 1.725f * u};
                c[R2] = {w - 1.25f * u, 0.775f * u};
                c[R1] = {w - 1.25f * u, 1.725f * u};
                c[L3] = {2.25f * u, h - 3.95f * u};
                c[R3] = {w - 5.6f * u, h - 3.35f * u};
                c[Select] = {w * 0.5f - 0.8f * u, h - 0.575f * u};
                c[Start] = {w * 0.5f + 0.8f * u, h - 0.575f * u};
                c[Menu] = {w * 0.5f, 0.65f * u};
            }
            else
            {
                // The picture fills the top (pictureAtTop); the controls take the rest.
                const float top = l.top;
                c[LeftStick] = {2.25f * u, h - 3.05f * u};
                c[RightStick] = {w * 0.5f, top + 2.65f * u};
                c[Dpad] = {w * 0.5f, h - 2.3f * u};
                c[Face] = {w - 2.27f * u, h - 3.05f * u};
                c[L2] = {1.25f * u, top + 0.675f * u};
                c[L1] = {3.15f * u, top + 0.675f * u};
                c[R2] = {w - 1.25f * u, top + 0.675f * u};
                c[R1] = {w - 3.15f * u, top + 0.675f * u};
                c[L3] = {2.25f * u, top + 2.2f * u};
                c[R3] = {w * 0.5f, top + 4.25f * u};
                c[Select] = {w * 0.5f - 0.8f * u, h - 0.575f * u};
                c[Start] = {w * 0.5f + 0.8f * u, h - 0.575f * u};
                c[Menu] = {w * 0.5f, top + 0.7f * u};
            }
            return c;
        }

        Layout layoutFor(int width, int height, const Config &config, const Overrides &overrides)
        {
            Layout l;
            l.w = static_cast<float>(width);
            l.h = static_cast<float>(height);
            l.portrait = l.h > l.w;
            l.u = std::min(l.w, l.h) / 10.0f * config.scale;
            l.top = std::max(l.w * 0.75f + 0.3f * l.u, l.h - 7.2f * l.u);
            l.centre = defaultCentres(l);
            for (int i = 0; i < Count; ++i)
            {
                const Override &o = overrides[static_cast<size_t>(i)];
                l.scale[static_cast<size_t>(i)] = o.set ? o.scale : 1.0f;
                if (o.set)
                    l.centre[static_cast<size_t>(i)] = {o.x * l.w, o.y * l.h};
                l.shown[static_cast<size_t>(i)] = config.shown[static_cast<size_t>(i)];
            }
            return l;
        }

        // ------------------------------------------------------------------ geometry and drawing

        Rectangle pillRect(const Layout &l, int i)
        {
            const float u = l.unit(i);
            const Vector2 c = l.centre[static_cast<size_t>(i)];
            return {c.x - 0.5f * kInfo[i].a * u, c.y - 0.5f * kInfo[i].b * u, kInfo[i].a * u, kInfo[i].b * u};
        }

        // How far a touch reaches the control from its centre (round shapes).
        float reach(const Layout &l, int i)
        {
            const float u = l.unit(i);
            switch (kInfo[i].shape)
            {
            case Shape::Face:
                return (kInfo[i].a + kInfo[i].b) * u;
            default:
                return kInfo[i].a * u;
            }
        }

        Vector2 faceCentre(const Layout &l, int which)
        {
            const Vector2 c = l.centre[Face];
            const float gap = kInfo[Face].a * l.unit(Face);
            switch (which)
            {
            case 0:
                return {c.x, c.y - gap}; // triangle
            case 1:
                return {c.x + gap, c.y}; // circle
            case 2:
                return {c.x, c.y + gap}; // cross
            default:
                return {c.x - gap, c.y}; // square
            }
        }
        constexpr uint16_t kFaceBits[4] = {kTriangle, kCircle, kCross, kSquare};

        Rectangle grown(const Rectangle &r, float by)
        {
            return {r.x - by, r.y - by, r.width + 2.0f * by, r.height + 2.0f * by};
        }

        float distance(Vector2 a, Vector2 b)
        {
            return std::hypot(a.x - b.x, a.y - b.y);
        }

        bool hits(const Layout &l, int i, Vector2 p, float slack)
        {
            if (kInfo[i].shape == Shape::Pill)
                return CheckCollisionPointRec(p, grown(pillRect(l, i), slack));
            return distance(p, l.centre[static_cast<size_t>(i)]) <= reach(l, i) + slack;
        }

        Color shade(Color c, float alpha)
        {
            c.a = static_cast<unsigned char>(std::clamp(alpha, 0.0f, 1.0f) * 255.0f);
            return c;
        }

        void drawPill(const Rectangle &r, const char *label, bool held, float opacity, float u)
        {
            DrawRectangleRounded(r, 0.5f, 8, held ? shade(WHITE, opacity * 0.45f) : shade(BLACK, opacity * 0.35f));
            DrawRectangleRoundedLinesEx(r, 0.5f, 8, std::max(1.0f, 0.04f * u), shade(WHITE, opacity * 0.7f));
            const int size = std::max(10, static_cast<int>(std::min(0.36f * u, r.height * 0.6f)));
            const int textWidth = MeasureText(label, size);
            const int x = static_cast<int>(r.x + (r.width - textWidth) * 0.5f);
            const int y = static_cast<int>(r.y + (r.height - size) * 0.5f);
            const int offset = std::max(1, size / 12);
            DrawText(label, x + offset, y + offset, size, shade(BLACK, opacity * 0.8f));
            DrawText(label, x, y, size, shade(WHITE, opacity * 0.95f));
        }

        void drawFaceButton(Vector2 c, uint16_t bit, float radius, bool held, float opacity, float u)
        {
            DrawCircleV(c, radius, held ? shade(WHITE, opacity * 0.45f) : shade(BLACK, opacity * 0.35f));
            DrawRing(c, radius - std::max(1.0f, 0.04f * u), radius, 0.0f, 360.0f, 48, shade(WHITE, opacity * 0.7f));
            const float s = radius * 0.45f;
            const float line = std::max(1.5f, 0.06f * u);
            switch (bit)
            {
            case kTriangle:
            {
                const Color green = shade({64, 226, 160, 255}, opacity);
                const Vector2 top{c.x, c.y - s}, left{c.x - s * 1.1f, c.y + s * 0.7f}, right{c.x + s * 1.1f, c.y + s * 0.7f};
                DrawLineEx(top, left, line, green);
                DrawLineEx(left, right, line, green);
                DrawLineEx(right, top, line, green);
                break;
            }
            case kCircle:
                DrawRing(c, s - line, s, 0.0f, 360.0f, 32, shade({255, 102, 102, 255}, opacity));
                break;
            case kCross:
            {
                const Color blue = shade({124, 178, 232, 255}, opacity);
                DrawLineEx({c.x - s, c.y - s}, {c.x + s, c.y + s}, line, blue);
                DrawLineEx({c.x - s, c.y + s}, {c.x + s, c.y - s}, line, blue);
                break;
            }
            default:
                DrawRectangleLinesEx({c.x - s * 0.85f, c.y - s * 0.85f, s * 1.7f, s * 1.7f}, line,
                                     shade({255, 105, 248, 255}, opacity));
                break;
            }
        }

        void drawStick(Vector2 c, float radius, Vector2 knob, bool held, float opacity, float u)
        {
            DrawCircleV(c, radius, shade(BLACK, opacity * 0.25f));
            DrawRing(c, radius - std::max(1.0f, 0.04f * u), radius, 0.0f, 360.0f, 64, shade(WHITE, opacity * 0.6f));
            const Vector2 at{c.x + knob.x * radius * 0.6f, c.y + knob.y * radius * 0.6f};
            DrawCircleV(at, radius * 0.38f, shade(WHITE, opacity * (held ? 0.55f : 0.3f)));
        }

        void drawDpad(Vector2 c, float radius, uint16_t held, float opacity, float u)
        {
            DrawCircleV(c, radius, shade(BLACK, opacity * 0.2f));
            const float arm = radius * 0.36f;
            const struct
            {
                uint16_t bit;
                Vector2 dir;
            } arms[] = {{kUp, {0.0f, -1.0f}}, {kRight, {1.0f, 0.0f}}, {kDown, {0.0f, 1.0f}}, {kLeft, {-1.0f, 0.0f}}};
            for (const auto &a : arms)
            {
                const Vector2 centre{c.x + a.dir.x * radius * 0.58f, c.y + a.dir.y * radius * 0.58f};
                const Rectangle r{centre.x - arm, centre.y - arm, 2.0f * arm, 2.0f * arm};
                DrawRectangleRounded(r, 0.25f, 6, (held & a.bit) ? shade(WHITE, opacity * 0.45f) : shade(BLACK, opacity * 0.35f));
                DrawRectangleRoundedLinesEx(r, 0.25f, 6, std::max(1.0f, 0.04f * u), shade(WHITE, opacity * 0.7f));
                // An arrow pointing out.
                const float s = arm * 0.45f;
                const Vector2 tip{centre.x + a.dir.x * s, centre.y + a.dir.y * s};
                const Vector2 side{-a.dir.y * s, a.dir.x * s};
                const Vector2 back{centre.x - a.dir.x * s, centre.y - a.dir.y * s};
                DrawTriangle(tip, {back.x + side.x, back.y + side.y}, {back.x - side.x, back.y - side.y},
                             shade(WHITE, opacity * 0.8f));
                DrawTriangle(tip, {back.x - side.x, back.y - side.y}, {back.x + side.x, back.y + side.y},
                             shade(WHITE, opacity * 0.8f));
            }
        }

        void drawMenu(Vector2 c, float radius, float opacity, float u)
        {
            DrawCircleV(c, radius, shade(BLACK, opacity * 0.35f));
            DrawRing(c, radius - std::max(1.0f, 0.04f * u), radius, 0.0f, 360.0f, 32, shade(WHITE, opacity * 0.7f));
            const float bar = radius * 0.5f;
            const float line = std::max(1.5f, 0.05f * u);
            for (int i = -1; i <= 1; ++i)
            {
                const float y = c.y + i * bar * 0.55f;
                DrawLineEx({c.x - bar, y}, {c.x + bar, y}, line, shade(WHITE, opacity * 0.9f));
            }
        }

        struct Sticks
        {
            bool held[2] = {false, false};
            Vector2 value[2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        };

        void drawControls(const Layout &l, uint16_t held, const Sticks &sticks, float opacity)
        {
            for (int i = 0; i < Count; ++i)
            {
                if (!l.shown[static_cast<size_t>(i)])
                    continue;
                const float u = l.unit(i);
                const Vector2 c = l.centre[static_cast<size_t>(i)];
                switch (kInfo[i].shape)
                {
                case Shape::Stick:
                {
                    const int which = i == LeftStick ? 0 : 1;
                    drawStick(c, kInfo[i].a * u, sticks.value[which], sticks.held[which], opacity, u);
                    break;
                }
                case Shape::Dpad:
                    drawDpad(c, kInfo[i].a * u, held, opacity, u);
                    break;
                case Shape::Face:
                    for (int b = 0; b < 4; ++b)
                        drawFaceButton(faceCentre(l, b), kFaceBits[b], kInfo[i].b * u, (held & kFaceBits[b]) != 0, opacity,
                                       u);
                    break;
                case Shape::Pill:
                    drawPill(pillRect(l, i), kInfo[i].label, (held & kInfo[i].bit) != 0, opacity, u);
                    break;
                case Shape::Round:
                    drawMenu(c, kInfo[i].a * u, opacity, u);
                    break;
                }
            }
        }

        // ------------------------------------------------------------------ touches

        struct Touch
        {
            int id;
            Vector2 p;
        };

        int readTouches(Touch *out, int capacity)
        {
            int count = std::min(GetTouchPointCount(), capacity);
            for (int i = 0; i < count; ++i)
                out[i] = {GetTouchPointId(i), GetTouchPosition(i)};
#if !defined(__ANDROID__)
            // On a desktop the mouse stands in for a finger, for testing.
            if (count == 0 && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
                out[count++] = {1000, GetMousePosition()};
#endif
            return count;
        }

        // ------------------------------------------------------------------ the position editor

        struct Editor
        {
            bool active = false;
            Overrides landscape{};
            Overrides portrait{};
            int selected = -1;
            int dragId = -1;
            Vector2 dragOffset{};
            std::unordered_set<int> down; // touches seen in the last frame
        } g_editor;

        enum class Tool
        {
            Save,
            Cancel,
            Restore,
            Smaller,
            Bigger,
            Count
        };

        struct ToolButton
        {
            Tool tool;
            Rectangle area;
            const char *label;
        };

        std::array<ToolButton, static_cast<size_t>(Tool::Count)> toolbar(const Layout &l)
        {
            const float u = l.u;
            const float y = l.portrait ? std::max(0.3f * u, l.top * 0.45f - 0.4f * u) : l.h * 0.42f - 0.4f * u;
            const float wide = 2.1f * u, narrow = 0.9f * u, height = 0.8f * u, gap = 0.2f * u;
            const float total = 3.0f * wide + 2.0f * narrow + 4.0f * gap;
            float x = (l.w - total) * 0.5f;
            std::array<ToolButton, static_cast<size_t>(Tool::Count)> out{};
            const struct
            {
                Tool tool;
                float width;
                const char *label;
            } items[] = {{Tool::Save, wide, "Salvar"},
                         {Tool::Cancel, wide, "Cancelar"},
                         {Tool::Restore, wide, "Restaurar"},
                         {Tool::Smaller, narrow, "-"},
                         {Tool::Bigger, narrow, "+"}};
            for (size_t i = 0; i < out.size(); ++i)
            {
                out[i] = {items[i].tool, {x, y, items[i].width, height}, items[i].label};
                x += items[i].width + gap;
            }
            return out;
        }

        void endEditing()
        {
            g_editor.active = false;
            g_editor.selected = -1;
            g_editor.dragId = -1;
            g_editor.down.clear();
            settings::setInputBlocked(false);
            std::fprintf(stderr, "[touch] position editor closed\n");
        }

        void editFrame(int width, int height)
        {
            const Config &c = config();
            Overrides &edits = height > width ? g_editor.portrait : g_editor.landscape;
            const Layout l = layoutFor(width, height, c, edits);
            Touch touches[10];
            const int count = readTouches(touches, 10);
            std::unordered_set<int> now;
            const auto tools = toolbar(l);
            const auto store = [&](int i, Vector2 centre, float scale)
            {
                edits[static_cast<size_t>(i)] = {true, std::clamp(centre.x / l.w, 0.0f, 1.0f),
                                                 std::clamp(centre.y / l.h, 0.0f, 1.0f), std::clamp(scale, 0.5f, 2.0f)};
            };
            for (int t = 0; t < count; ++t)
            {
                const Touch &touch = touches[t];
                now.insert(touch.id);
                if (touch.id == g_editor.dragId)
                {
                    const int i = g_editor.selected;
                    if (i >= 0)
                        store(i, {touch.p.x + g_editor.dragOffset.x, touch.p.y + g_editor.dragOffset.y},
                              l.scale[static_cast<size_t>(i)]);
                    continue;
                }
                if (g_editor.down.count(touch.id))
                    continue; // held since an earlier frame on nothing
                // A new touch: a tool button, else the control under it (the last drawn is on top).
                bool used = false;
                for (const ToolButton &button : tools)
                {
                    if (!CheckCollisionPointRec(touch.p, grown(button.area, 0.1f * l.u)))
                        continue;
                    used = true;
                    const int i = g_editor.selected;
                    switch (button.tool)
                    {
                    case Tool::Save:
                        settings::set("touch", "layout_landscape", formatLayout(g_editor.landscape));
                        settings::set("touch", "layout_portrait", formatLayout(g_editor.portrait));
                        settings::save();
                        endEditing();
                        return;
                    case Tool::Cancel:
                        endEditing();
                        return;
                    case Tool::Restore:
                        edits = Overrides{};
                        g_editor.selected = -1;
                        break;
                    case Tool::Smaller:
                    case Tool::Bigger:
                        if (i >= 0)
                            store(i, l.centre[static_cast<size_t>(i)],
                                  l.scale[static_cast<size_t>(i)] + (button.tool == Tool::Bigger ? 0.1f : -0.1f));
                        break;
                    default:
                        break;
                    }
                }
                if (used)
                    continue;
                for (int i = Count - 1; i >= 0; --i)
                {
                    if (!l.shown[static_cast<size_t>(i)] || !hits(l, i, touch.p, 0.1f * l.u))
                        continue;
                    g_editor.selected = i;
                    g_editor.dragId = touch.id;
                    const Vector2 centre = l.centre[static_cast<size_t>(i)];
                    g_editor.dragOffset = {centre.x - touch.p.x, centre.y - touch.p.y};
                    break;
                }
            }
            if (!now.count(g_editor.dragId))
                g_editor.dragId = -1;
            g_editor.down = now;

            // The controls as they are, the chosen one outlined, and the tools.
            const Layout shown = layoutFor(width, height, c, edits);
            drawControls(shown, 0, Sticks{}, std::max(0.6f, c.opacity));
            if (g_editor.selected >= 0)
            {
                const int i = g_editor.selected;
                const float line = std::max(2.0f, 0.06f * l.u);
                const Color yellow{255, 214, 64, 255};
                if (kInfo[i].shape == Shape::Pill)
                    DrawRectangleLinesEx(grown(pillRect(shown, i), 0.1f * l.u), line, yellow);
                else
                    DrawRing(shown.centre[static_cast<size_t>(i)], reach(shown, i) + 0.06f * l.u,
                             reach(shown, i) + 0.06f * l.u + line, 0.0f, 360.0f, 64, yellow);
            }
            for (const ToolButton &button : tools)
            {
                const bool needsSelection = button.tool == Tool::Smaller || button.tool == Tool::Bigger;
                const float alpha = needsSelection && g_editor.selected < 0 ? 0.35f : 0.9f;
                DrawRectangleRounded(button.area, 0.3f, 8, shade({30, 30, 36, 255}, 0.85f));
                DrawRectangleRoundedLinesEx(button.area, 0.3f, 8, std::max(1.0f, 0.04f * l.u), shade(WHITE, alpha));
                const int size = std::max(12, static_cast<int>(0.4f * l.u));
                const int textWidth = MeasureText(button.label, size);
                DrawText(button.label, static_cast<int>(button.area.x + (button.area.width - textWidth) * 0.5f),
                         static_cast<int>(button.area.y + (button.area.height - size) * 0.5f), size, shade(WHITE, alpha));
            }
            const char *hint = "Arraste um controle para mover. - e + mudam o tamanho do escolhido.";
            const int hintSize = std::max(10, static_cast<int>(0.3f * l.u));
            const int hintWidth = MeasureText(hint, hintSize);
            const float hintY = tools[0].area.y + tools[0].area.height + 0.25f * l.u;
            DrawText(hint, static_cast<int>((l.w - hintWidth) * 0.5f) + 1, static_cast<int>(hintY) + 1, hintSize, BLACK);
            DrawText(hint, static_cast<int>((l.w - hintWidth) * 0.5f), static_cast<int>(hintY), hintSize, WHITE);
        }
    }

    void frame(int screenWidth, int screenHeight)
    {
        if (g_editor.active)
        {
            g_visible.store(true, std::memory_order_relaxed);
            g_buttons.store(0, std::memory_order_relaxed);
            g_stickHeld.store(false, std::memory_order_relaxed);
            g_rightHeld.store(false, std::memory_order_relaxed);
            if (screenWidth > 0 && screenHeight > 0)
                editFrame(screenWidth, screenHeight);
            return;
        }
        const bool on = wanted();
        g_visible.store(on, std::memory_order_relaxed);
        {
            // Why the controls show or not, once per change (logcat on a phone).
            const char *reason = on ? (settings::inputBlocked() ? "hidden: the menu is open" : "shown")
                                 : config().mode == "off" ? "hidden: turned off"
                                 : config().mode == "auto" && IsGamepadAvailable(0) ? "hidden: a gamepad is connected"
                                                                                     : "hidden: auto on a desktop";
            static const char *s_reason = nullptr;
            if (reason != s_reason)
            {
                std::fprintf(stderr, "[touch] controls %s (%dx%d)\n", reason, screenWidth, screenHeight);
                s_reason = reason;
            }
        }
        static std::unordered_map<int, int> s_stickOf; // touch id -> stick (0 left, 1 right)
        if (!on || settings::inputBlocked() || screenWidth <= 0 || screenHeight <= 0)
        {
            // The menu is open (it takes the touches) or the controls are off.
            g_buttons.store(0, std::memory_order_relaxed);
            g_stickHeld.store(false, std::memory_order_relaxed);
            g_rightHeld.store(false, std::memory_order_relaxed);
            s_stickOf.clear();
            return;
        }

        const Config &c = config();
        const Layout l = layoutFor(screenWidth, screenHeight, c, screenHeight > screenWidth ? c.portrait : c.landscape);
        Touch touches[10];
        const int count = readTouches(touches, 10);

        uint16_t held = 0;
        Sticks sticks;
        bool menuDown = false;
        const float slack = 0.2f * l.u;
        std::unordered_map<int, int> stickOf;
        for (int t = 0; t < count; ++t)
        {
            const Touch &touch = touches[t];
            const Vector2 p = touch.p;
            // A finger that took a stick keeps it until it lifts, wherever it slides.
            int which = -1;
            if (const auto found = s_stickOf.find(touch.id); found != s_stickOf.end())
                which = found->second;
            else
                for (int s = 0; s < 2; ++s)
                {
                    const int i = s == 0 ? LeftStick : RightStick;
                    if (l.shown[static_cast<size_t>(i)] && !sticks.held[s] &&
                        distance(p, l.centre[static_cast<size_t>(i)]) <= reach(l, i) * 1.35f)
                        which = s;
                }
            if (which >= 0)
            {
                const int i = which == 0 ? LeftStick : RightStick;
                const float radius = reach(l, i);
                float dx = (p.x - l.centre[static_cast<size_t>(i)].x) / radius;
                float dy = (p.y - l.centre[static_cast<size_t>(i)].y) / radius;
                const float length = std::hypot(dx, dy);
                if (length > 1.0f)
                {
                    dx /= length;
                    dy /= length;
                }
                sticks.value[which] = {dx, dy};
                sticks.held[which] = true;
                stickOf[touch.id] = which;
                continue;
            }
            if (l.shown[Menu] && hits(l, Menu, p, reach(l, Menu) * 0.4f))
            {
                menuDown = true;
                continue;
            }
            if (l.shown[Dpad] && hits(l, Dpad, p, 0.15f * reach(l, Dpad)))
            {
                const Vector2 centre = l.centre[Dpad];
                const float dx = p.x - centre.x, dy = p.y - centre.y;
                const float length = std::hypot(dx, dy);
                if (length > 0.2f * reach(l, Dpad))
                {
                    if (dx < -0.4f * length)
                        held |= kLeft;
                    if (dx > 0.4f * length)
                        held |= kRight;
                    if (dy < -0.4f * length)
                        held |= kUp;
                    if (dy > 0.4f * length)
                        held |= kDown;
                }
                continue;
            }
            if (l.shown[Face])
                for (int b = 0; b < 4; ++b)
                    if (distance(p, faceCentre(l, b)) <= kInfo[Face].b * l.unit(Face) * 1.25f)
                        held |= kFaceBits[b];
            for (int i = 0; i < Count; ++i)
                if (kInfo[i].shape == Shape::Pill && l.shown[static_cast<size_t>(i)] && hits(l, i, p, slack))
                    held |= kInfo[i].bit;
        }
        s_stickOf = stickOf;
        if (sticks.held[0])
        {
            // The left stick also presses the d-pad, which the menus read.
            const Vector2 s = sticks.value[0];
            if (s.x < -0.5f)
                held |= kLeft;
            if (s.x > 0.5f)
                held |= kRight;
            if (s.y < -0.5f)
                held |= kUp;
            if (s.y > 0.5f)
                held |= kDown;
        }

        static bool s_menuWasDown = false;
        if (menuDown && !s_menuWasDown)
            g_menuRequest.store(true, std::memory_order_relaxed);
        s_menuWasDown = menuDown;

        g_stickX.store(sticks.value[0].x, std::memory_order_relaxed);
        g_stickY.store(sticks.value[0].y, std::memory_order_relaxed);
        g_stickHeld.store(sticks.held[0], std::memory_order_relaxed);
        g_rightX.store(sticks.value[1].x, std::memory_order_relaxed);
        g_rightY.store(sticks.value[1].y, std::memory_order_relaxed);
        g_rightHeld.store(sticks.held[1], std::memory_order_relaxed);
        g_buttons.store(held, std::memory_order_relaxed);

        drawControls(l, held, sticks, c.opacity);
    }

    bool visible()
    {
        return g_visible.load(std::memory_order_relaxed);
    }

    bool pictureAtTop(int screenWidth, int screenHeight)
    {
        return visible() && screenHeight > screenWidth;
    }

    uint16_t buttons()
    {
        return g_buttons.load(std::memory_order_relaxed);
    }

    bool stick(float &x, float &y)
    {
        if (!g_stickHeld.load(std::memory_order_relaxed))
            return false;
        x = g_stickX.load(std::memory_order_relaxed);
        y = g_stickY.load(std::memory_order_relaxed);
        return true;
    }

    bool rightStick(float &x, float &y)
    {
        if (!g_rightHeld.load(std::memory_order_relaxed))
            return false;
        x = g_rightX.load(std::memory_order_relaxed);
        y = g_rightY.load(std::memory_order_relaxed);
        return true;
    }

    bool takeMenuRequest()
    {
        return g_menuRequest.exchange(false, std::memory_order_relaxed);
    }

    void beginEditing()
    {
        const Config &c = config();
        g_editor = Editor{};
        g_editor.active = true;
        g_editor.landscape = c.landscape;
        g_editor.portrait = c.portrait;
        settings::setInputBlocked(true);
        std::fprintf(stderr, "[touch] position editor open\n");
    }

    void cancelEditing()
    {
        if (g_editor.active)
            endEditing();
    }

    bool editing()
    {
        return g_editor.active;
    }
}
