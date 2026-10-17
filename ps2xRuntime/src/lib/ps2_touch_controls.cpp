#include "runtime/ps2_touch_controls.h"

#include "runtime/ps2_settings.h"

#include "raylib.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <string>

namespace ps2x::touch
{
    namespace
    {
        namespace settings = ps2x::settings;

        // PS2 pad bits, as in ps2_pad.cpp.
        constexpr uint16_t kSelect = 0x0001u;
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
        std::atomic<bool> g_menuRequest{false};
        std::atomic<bool> g_visible{false};

        struct Config
        {
            std::string mode = "auto";
            float opacity = 0.55f;
            float scale = 1.0f;
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

        struct FaceButton
        {
            Vector2 centre;
            uint16_t bit;
        };

        struct Layout
        {
            float u = 0.0f;             // size unit
            Vector2 stick{};
            float stickRadius = 0.0f;
            Rectangle stickZone{};      // a touch anywhere here moves the stick
            FaceButton face[4]{};
            float buttonRadius = 0.0f;
            Rectangle l1{}, l2{}, r1{}, r2{}, select{}, start{};
            Vector2 menu{};
            float menuRadius = 0.0f;
        };

        Layout layoutFor(int width, int height, float scale)
        {
            Layout l;
            const float w = static_cast<float>(width);
            const float h = static_cast<float>(height);
            const bool portrait = h > w;
            l.u = std::min(w, h) / 10.0f * scale;
            const float u = l.u;
            l.stickRadius = 1.45f * u;
            l.buttonRadius = 0.62f * u;
            const float faceGap = 1.15f * u;
            const Vector2 shoulder{1.7f * u, 0.75f * u};
            const Vector2 small{1.3f * u, 0.55f * u};
            if (!portrait)
            {
                l.stick = {0.5f * u + l.stickRadius + 0.3f * u, h - 0.5f * u - l.stickRadius};
                l.stickZone = {0.0f, 2.3f * u, w * 0.45f, h - 2.3f * u};
                const Vector2 face{w - 0.5f * u - faceGap - l.buttonRadius, l.stick.y};
                l.face[0] = {{face.x, face.y - faceGap}, kTriangle};
                l.face[1] = {{face.x + faceGap, face.y}, kCircle};
                l.face[2] = {{face.x, face.y + faceGap}, kCross};
                l.face[3] = {{face.x - faceGap, face.y}, kSquare};
                l.l2 = {0.4f * u, 0.4f * u, shoulder.x, shoulder.y};
                l.l1 = {0.4f * u, 1.35f * u, shoulder.x, shoulder.y};
                l.r2 = {w - 0.4f * u - shoulder.x, 0.4f * u, shoulder.x, shoulder.y};
                l.r1 = {w - 0.4f * u - shoulder.x, 1.35f * u, shoulder.x, shoulder.y};
                l.select = {w * 0.5f - 0.15f * u - small.x, h - 0.85f * u, small.x, small.y};
                l.start = {w * 0.5f + 0.15f * u, h - 0.85f * u, small.x, small.y};
                l.menu = {w * 0.5f, 0.65f * u};
            }
            else
            {
                // The picture fills the top (pictureAtTop); the controls take the rest.
                const float top = std::max(w * 0.75f + 0.3f * u, h - 7.2f * u);
                l.stick = {0.5f * u + l.stickRadius + 0.3f * u, h - 1.6f * u - l.stickRadius};
                l.stickZone = {0.0f, top + 1.5f * u, w * 0.5f, h - top - 1.5f * u};
                const Vector2 face{w - 0.5f * u - faceGap - l.buttonRadius, l.stick.y};
                l.face[0] = {{face.x, face.y - faceGap}, kTriangle};
                l.face[1] = {{face.x + faceGap, face.y}, kCircle};
                l.face[2] = {{face.x, face.y + faceGap}, kCross};
                l.face[3] = {{face.x - faceGap, face.y}, kSquare};
                l.l2 = {0.4f * u, top + 0.3f * u, shoulder.x, shoulder.y};
                l.l1 = {0.6f * u + shoulder.x, top + 0.3f * u, shoulder.x, shoulder.y};
                l.r2 = {w - 0.4f * u - shoulder.x, top + 0.3f * u, shoulder.x, shoulder.y};
                l.r1 = {w - 0.6f * u - 2.0f * shoulder.x, top + 0.3f * u, shoulder.x, shoulder.y};
                l.select = {w * 0.5f - 0.15f * u - small.x, h - 0.85f * u, small.x, small.y};
                l.start = {w * 0.5f + 0.15f * u, h - 0.85f * u, small.x, small.y};
                l.menu = {w * 0.5f, top + 0.7f * u};
            }
            l.menuRadius = 0.42f * u;
            return l;
        }

        Rectangle grown(const Rectangle &r, float by)
        {
            return {r.x - by, r.y - by, r.width + 2.0f * by, r.height + 2.0f * by};
        }

        float distance(Vector2 a, Vector2 b)
        {
            return std::hypot(a.x - b.x, a.y - b.y);
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
            const int size = std::max(10, static_cast<int>(0.36f * u));
            const int textWidth = MeasureText(label, size);
            const int x = static_cast<int>(r.x + (r.width - textWidth) * 0.5f);
            const int y = static_cast<int>(r.y + (r.height - size) * 0.5f);
            const int offset = std::max(1, size / 12);
            DrawText(label, x + offset, y + offset, size, shade(BLACK, opacity * 0.8f));
            DrawText(label, x, y, size, shade(WHITE, opacity * 0.95f));
        }

        void drawFace(const FaceButton &b, float radius, bool held, float opacity, float u)
        {
            DrawCircleV(b.centre, radius, held ? shade(WHITE, opacity * 0.45f) : shade(BLACK, opacity * 0.35f));
            DrawRing(b.centre, radius - std::max(1.0f, 0.04f * u), radius, 0.0f, 360.0f, 48, shade(WHITE, opacity * 0.7f));
            const float s = radius * 0.45f;
            const float line = std::max(1.5f, 0.06f * u);
            const Vector2 c = b.centre;
            switch (b.bit)
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

        void drawControls(const Layout &l, uint16_t held, bool stickHeld, Vector2 stick, float opacity)
        {
            const float u = l.u;
            // Stick: the base, and the knob where the thumb is.
            DrawCircleV(l.stick, l.stickRadius, shade(BLACK, opacity * 0.25f));
            DrawRing(l.stick, l.stickRadius - std::max(1.0f, 0.04f * u), l.stickRadius, 0.0f, 360.0f, 64,
                     shade(WHITE, opacity * 0.6f));
            const Vector2 knob{l.stick.x + stick.x * l.stickRadius * 0.6f, l.stick.y + stick.y * l.stickRadius * 0.6f};
            DrawCircleV(knob, 0.55f * u, shade(WHITE, opacity * (stickHeld ? 0.55f : 0.3f)));
            for (const FaceButton &b : l.face)
                drawFace(b, l.buttonRadius, (held & b.bit) != 0, opacity, u);
            drawPill(l.l1, "L1", (held & kL1) != 0, opacity, u);
            drawPill(l.l2, "L2", (held & kL2) != 0, opacity, u);
            drawPill(l.r1, "R1", (held & kR1) != 0, opacity, u);
            drawPill(l.r2, "R2", (held & kR2) != 0, opacity, u);
            drawPill(l.select, "SELECT", (held & kSelect) != 0, opacity, u);
            drawPill(l.start, "START", (held & kStart) != 0, opacity, u);
            // Menu: three lines in a circle.
            DrawCircleV(l.menu, l.menuRadius, shade(BLACK, opacity * 0.35f));
            DrawRing(l.menu, l.menuRadius - std::max(1.0f, 0.04f * u), l.menuRadius, 0.0f, 360.0f, 32,
                     shade(WHITE, opacity * 0.7f));
            const float bar = l.menuRadius * 0.5f;
            const float line = std::max(1.5f, 0.05f * u);
            for (int i = -1; i <= 1; ++i)
            {
                const float y = l.menu.y + i * bar * 0.55f;
                DrawLineEx({l.menu.x - bar, y}, {l.menu.x + bar, y}, line, shade(WHITE, opacity * 0.9f));
            }
        }
    }

    void frame(int screenWidth, int screenHeight)
    {
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
        if (!on || settings::inputBlocked() || screenWidth <= 0 || screenHeight <= 0)
        {
            // The menu is open (it takes the touches) or the controls are off.
            g_buttons.store(0, std::memory_order_relaxed);
            g_stickHeld.store(false, std::memory_order_relaxed);
            return;
        }

        const Config &c = config();
        const Layout l = layoutFor(screenWidth, screenHeight, c.scale);

        // Every finger on the screen; on a desktop the mouse stands in for one, for testing.
        Vector2 points[10];
        int count = std::min(GetTouchPointCount(), 10);
        for (int i = 0; i < count; ++i)
            points[i] = GetTouchPosition(i);
#if !defined(__ANDROID__)
        if (count == 0 && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
            points[count++] = GetMousePosition();
#endif

        uint16_t held = 0;
        bool stickHeld = false;
        Vector2 stick{0.0f, 0.0f};
        bool menuDown = false;
        const float slack = 0.2f * l.u;
        for (int i = 0; i < count; ++i)
        {
            const Vector2 p = points[i];
            if (distance(p, l.menu) <= l.menuRadius * 1.4f)
            {
                menuDown = true;
                continue;
            }
            if (CheckCollisionPointRec(p, l.stickZone))
            {
                float dx = (p.x - l.stick.x) / l.stickRadius;
                float dy = (p.y - l.stick.y) / l.stickRadius;
                const float length = std::hypot(dx, dy);
                if (length > 1.0f)
                {
                    dx /= length;
                    dy /= length;
                }
                stick = {dx, dy};
                stickHeld = true;
                continue;
            }
            for (const FaceButton &b : l.face)
                if (distance(p, b.centre) <= l.buttonRadius * 1.25f)
                    held |= b.bit;
            const struct
            {
                Rectangle area;
                uint16_t bit;
            } pills[] = {{l.l1, kL1}, {l.l2, kL2}, {l.r1, kR1}, {l.r2, kR2}, {l.select, kSelect}, {l.start, kStart}};
            for (const auto &pill : pills)
                if (CheckCollisionPointRec(p, grown(pill.area, slack)))
                    held |= pill.bit;
        }
        if (stickHeld)
        {
            // The stick also presses the d-pad, which the menus read.
            if (stick.x < -0.5f)
                held |= kLeft;
            if (stick.x > 0.5f)
                held |= kRight;
            if (stick.y < -0.5f)
                held |= kUp;
            if (stick.y > 0.5f)
                held |= kDown;
        }

        static bool s_menuWasDown = false;
        if (menuDown && !s_menuWasDown)
            g_menuRequest.store(true, std::memory_order_relaxed);
        s_menuWasDown = menuDown;

        g_stickX.store(stick.x, std::memory_order_relaxed);
        g_stickY.store(stick.y, std::memory_order_relaxed);
        g_stickHeld.store(stickHeld, std::memory_order_relaxed);
        g_buttons.store(held, std::memory_order_relaxed);

        drawControls(l, held, stickHeld, stick, c.opacity);
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

    bool takeMenuRequest()
    {
        return g_menuRequest.exchange(false, std::memory_order_relaxed);
    }
}
