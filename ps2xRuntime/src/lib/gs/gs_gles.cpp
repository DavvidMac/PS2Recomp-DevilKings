#include "runtime/gs/gs_gles.h"

#if defined(PLATFORM_VITA)
// The Vita build has no OpenGL ES 3 to draw with.
namespace ps2x::gles
{
    bool requested() { return false; }
    std::unique_ptr<GSRasterBackend> createBackend() { return nullptr; }
    bool attachToWindow() { return false; }
    bool active() { return false; }
    bool displayTexture(unsigned &, uint32_t &, uint32_t &) { return false; }
    void shutdown() {}
}
#else

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"

#if defined(__ANDROID__)
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#define PS2X_GLES_EGL 1
#else
// The window's GL 3.3 core context belongs to raylib, whose glad has already loaded the
// function pointers this file calls through.
#include "external/glad.h"
#define GLFW_INCLUDE_NONE
#include "external/glfw/include/GLFW/glfw3.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ps2x::gles
{
    namespace
    {
        bool traceEnabled()
        {
            static const bool s_trace = []()
            {
                const char *value = std::getenv("PS2X_GLES_TRACE");
                return value && value[0] == '1';
            }();
            return s_trace;
        }

        uint64_t g_frameLogSwap = []() -> uint64_t
        {
            const char *value = std::getenv("PS2X_GLES_FRAMELOG");
            return value ? std::strtoull(value, nullptr, 10) : 0u;
        }();
        std::atomic<uint64_t> g_swaps{0};
        bool frameLogging()
        {
            const uint64_t swaps = g_swaps.load(std::memory_order_relaxed);
            return g_frameLogSwap != 0u && swaps >= g_frameLogSwap && swaps < g_frameLogSwap + 2u;
        }

        // What the renderer draws differently from the console, said once per kind.
        void noteOnce(const std::string &what)
        {
            if (!traceEnabled())
                return;
            static std::mutex s_mutex;
            static std::set<std::string> s_said;
            std::lock_guard<std::mutex> lock(s_mutex);
            if (s_said.insert(what).second)
                std::fprintf(stderr, "[gles] %s\n", what.c_str());
        }

        // ------------------------------------------------------------------ recorded work

        // A vertex as the GL draws it: x, y in pixels of the target, z in [0, 1], s, t in texels
        // and q (1 for UV and sprites), colour and fog as bytes.
        struct Vertex
        {
            float x, y, z;
            float s, t, q;
            uint8_t r, g, b, a;
            float fog;
        };
        static_assert(sizeof(Vertex) == 32, "Vertex is uploaded as is");

        enum TextureSource : uint8_t
        {
            SourceNone = 0,
            SourceCache = 1,       // decoded from local memory
            SourceTarget = 2,      // a frame buffer the GPU drew
            SourceTargetCopy = 3,  // the frame buffer being drawn to (copied first)
        };

        // Everything the GL state of a draw depends on: consecutive draws with the same key
        // share one call. Filled after a memset, compared with memcmp.
        struct DrawKey
        {
            uint32_t target;
            uint32_t depth;      // 0: none
            uint32_t texture;
            uint16_t targetWidth, targetHeight;
            uint16_t texWidth, texHeight;      // 2^TW, 2^TH
            uint16_t storeWidth, storeHeight;  // the GL texture's size
            uint16_t minU, maxU, minV, maxV;
            uint16_t scissorX0, scissorY0, scissorX1, scissorY1;
            uint16_t blendEquation, blendSrc, blendDst;
            uint8_t source;
            uint8_t tme, tfx, tcc, fge, linear;
            int8_t atst;   // -1: no alpha test
            uint8_t afail;
            uint8_t aref;
            uint8_t wms, wmt;
            uint8_t texaMode;  // 0: texel alpha, 1: 24-bit target read with TEXA, 2: 16-bit target
            uint8_t ta0, ta1, aem;
            uint8_t alphaScale;  // 255: alpha as stored (decoded textures), 128: target alpha
            uint8_t frame16, forceAlpha, keepBlendAlpha;
            uint8_t blend, blendFix;
            uint8_t depthFunc;  // 0 off, 1 always, 2 gequal, 3 greater
            uint8_t depthWrite;
            uint8_t colorMask;  // bit 0 r .. bit 3 a
            uint8_t fogR, fogG, fogB;
            uint8_t pad[2];
        };

        struct PresentParams
        {
            uint32_t target[2];
            int originX[2];
            int originY[2];
            uint32_t width[2];
            uint32_t height[2];
            bool enable[2];
            bool slbg;
            bool mmod;
            uint8_t alp;
            uint32_t bgcolor;
            bool field;
            bool oddField;
            uint32_t outWidth;
            uint32_t outHeight;
        };

        // A GPU frame buffer copied back for local memory.
        struct ReadbackJob
        {
            uint32_t target = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            std::vector<uint32_t> pixels;
            bool done = false;
        };

        struct Command
        {
            enum class Kind : uint8_t
            {
                Draw,
                CreateTarget,
                DeleteTarget,
                UploadTarget,
                UploadTexture,
                DeleteTexture,
                Readback,
                Present,
            };
            Kind kind = Kind::Draw;
            uint32_t id = 0;
            uint32_t x = 0;  // UploadTarget: where the pixels go
            uint32_t y = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            bool depth = false;
            size_t blob = 0;  // first word of the pixels in Batch::blob
            DrawKey key{};
            uint32_t first = 0;
            uint32_t count = 0;
            ReadbackJob *job = nullptr;
            PresentParams present{};
        };

        struct Batch
        {
            std::vector<Vertex> vertices;
            std::vector<Command> commands;
            std::vector<uint32_t> blob;
            uint32_t presents = 0;
            bool empty() const { return commands.empty(); }
        };

        // ------------------------------------------------------------------ hand-off to the GL thread

        struct Channel
        {
            std::mutex mutex;
            std::condition_variable workCv;
            std::condition_variable doneCv;
            std::deque<Batch> queue;
            uint32_t queuedPresents = 0;
            bool running = false;
            bool stop = false;
            std::thread thread;
        };

        Channel &channel()
        {
            static Channel *s_channel = new Channel;
            return *s_channel;
        }

        std::atomic<bool> g_active{false};

        // The picture handed to the main thread: three textures, so the GL thread draws into
        // one the main thread neither shows nor is about to take.
        struct DisplaySlot
        {
            unsigned texture = 0;
            unsigned framebuffer = 0;
            uint32_t width = 0;
            uint32_t height = 0;
        };
        std::mutex g_displayMutex;
        std::array<DisplaySlot, 3> g_slots{};
        int g_latestSlot = -1;
        int g_shownSlot = -1;

        // A GS frame can be drawn this many swaps ahead of the GPU.
        constexpr uint32_t kMaxQueuedPresents = 2u;

        // ------------------------------------------------------------------ the renderer's GL context

#if defined(PS2X_GLES_EGL)
        EGLDisplay g_eglDisplay = EGL_NO_DISPLAY;
        EGLContext g_eglContext = EGL_NO_CONTEXT;
        EGLSurface g_eglSurface = EGL_NO_SURFACE;

        bool createContext()
        {
            g_eglDisplay = eglGetCurrentDisplay();
            const EGLContext shared = eglGetCurrentContext();
            if (g_eglDisplay == EGL_NO_DISPLAY || shared == EGL_NO_CONTEXT)
                return false;
            constexpr EGLint kOpenGlEs3Bit = 0x40; // EGL_OPENGL_ES3_BIT_KHR
            const EGLint configAttribs[] = {EGL_RENDERABLE_TYPE, kOpenGlEs3Bit, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                                            EGL_NONE};
            EGLConfig config = nullptr;
            EGLint configs = 0;
            if (!eglChooseConfig(g_eglDisplay, configAttribs, &config, 1, &configs) || configs < 1)
                return false;
            const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
            g_eglContext = eglCreateContext(g_eglDisplay, config, shared, contextAttribs);
            if (g_eglContext == EGL_NO_CONTEXT)
                return false;
            const EGLint surfaceAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
            g_eglSurface = eglCreatePbufferSurface(g_eglDisplay, config, surfaceAttribs);
            return g_eglSurface != EGL_NO_SURFACE;
        }

        bool makeContextCurrent()
        {
            return eglMakeCurrent(g_eglDisplay, g_eglSurface, g_eglSurface, g_eglContext) == EGL_TRUE;
        }

        const char *kShaderHeader = "#version 300 es\nprecision highp float;\nprecision highp int;\n"
                                    "precision highp sampler2D;\n";
        void clearDepthToZero() { glClearDepthf(0.0f); }
#else
        GLFWwindow *g_contextWindow = nullptr;

        bool createContext()
        {
            // GLFW creates windows on the main thread only: a hidden one, sharing objects with
            // raylib's, whose context the GL thread then makes current.
            GLFWwindow *shared = glfwGetCurrentContext();
            if (!shared)
                return false;
            glfwDefaultWindowHints();
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
            glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
            g_contextWindow = glfwCreateWindow(16, 16, "ps2x-gles", nullptr, shared);
            glfwMakeContextCurrent(shared);
            return g_contextWindow != nullptr;
        }

        bool makeContextCurrent()
        {
            glfwMakeContextCurrent(g_contextWindow);
            return glfwGetCurrentContext() == g_contextWindow;
        }

        const char *kShaderHeader = "#version 330 core\n";
        void clearDepthToZero() { glClearDepth(0.0); }
#endif

        // ------------------------------------------------------------------ shaders

        const char *kDrawVertex = R"(
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aTex;
layout(location = 2) in vec4 aColor;
layout(location = 3) in float aFog;
uniform vec2 uTargetSize;
out vec3 vTex;
out vec4 vColor;
out float vFog;
void main()
{
    // Row 0 of a target is line 0 of the GS buffer: no flip anywhere.
    vec2 ndc = aPos.xy / uTargetSize * 2.0 - 1.0;
    gl_Position = vec4(ndc, aPos.z * 2.0 - 1.0, 1.0);
    vTex = aTex;
    vColor = aColor;
    vFog = aFog;
}
)";

        // The texture function, fog, alpha test and output of the GS, in the 0..255 units the
        // GS computes in. Targets store alpha in units of 1/128 (0x80 is 1.0), so the blend
        // reads source and destination alpha as factors directly.
        const char *kDrawFragment = R"(
uniform sampler2D uTex;
uniform ivec4 uMode;     // tme, tfx, tcc, linear
uniform ivec4 uWrap;     // wms, wmt, texture width, texture height
uniform ivec4 uRegion;   // minu, maxu, minv, maxv
uniform ivec4 uStore;    // stored width, stored height, texa mode, 16-bit frame
uniform vec4 uTexa;      // ta0, ta1, aem, alpha scale
uniform vec4 uFog;       // fog colour, aref
uniform ivec4 uTest;     // fge, atst (-1 none), pass (0 passing, 1 failing), alpha out mode
in vec3 vTex;
in vec4 vColor;
in float vFog;
out vec4 oColor;

int wrapCoord(int c, int size, int mode, int lo, int hi)
{
    if (mode == 0) return c & (size - 1);
    if (mode == 1) return clamp(c, 0, size - 1);
    if (mode == 2) return clamp(c, lo, hi);
    return (c & lo) | hi;
}

vec4 fetchTexel(int u, int v)
{
    u = wrapCoord(u, uWrap.z, uWrap.x, uRegion.x, uRegion.y);
    v = wrapCoord(v, uWrap.w, uWrap.y, uRegion.z, uRegion.w);
    vec4 t = vec4(0.0);
    if (u >= 0 && v >= 0 && u < uStore.x && v < uStore.y)
        t = texelFetch(uTex, ivec2(u, v), 0);
    vec4 c = vec4(floor(t.rgb * 255.0 + 0.5), floor(t.a * uTexa.w + 0.5));
    bool black = c.r == 0.0 && c.g == 0.0 && c.b == 0.0;
    if (uStore.z == 1)
        c.a = (uTexa.z > 0.5 && black) ? 0.0 : uTexa.x;
    else if (uStore.z == 2)
        c.a = t.a > 0.99 ? uTexa.y : ((uTexa.z > 0.5 && black) ? 0.0 : uTexa.x);
    return c;
}

vec4 sampleTexture()
{
    float q = abs(vTex.z) > 1e-8 ? vTex.z : 1.0;
    vec2 uv = vTex.xy / q;
    if (uMode.w == 0)
        return fetchTexel(int(floor(uv.x)), int(floor(uv.y)));
    uv -= 0.5;
    vec2 base = floor(uv);
    vec2 f = uv - base;
    int u0 = int(base.x);
    int v0 = int(base.y);
    vec4 c00 = fetchTexel(u0, v0);
    vec4 c10 = fetchTexel(u0 + 1, v0);
    vec4 c01 = fetchTexel(u0, v0 + 1);
    vec4 c11 = fetchTexel(u0 + 1, v0 + 1);
    return floor(mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y) + 0.5);
}

bool alphaPasses(float a)
{
    float aref = uFog.w;
    int atst = uTest.y;
    if (atst == 0) return false;
    if (atst == 1) return true;
    if (atst == 2) return a < aref;
    if (atst == 3) return a <= aref;
    if (atst == 4) return a == aref;
    if (atst == 5) return a >= aref;
    if (atst == 6) return a > aref;
    return a != aref;
}

void main()
{
    vec4 c = floor(vColor * 255.0 + 0.5);
    if (uMode.x == 1)
    {
        vec4 t = sampleTexture();
        bool tcc = uMode.z == 1;
        int tfx = uMode.y;
        vec3 rgb;
        float a;
        if (tfx == 0)
        {
            rgb = min(floor(t.rgb * c.rgb / 128.0), 255.0);
            a = tcc ? min(floor(t.a * c.a / 128.0), 255.0) : c.a;
        }
        else if (tfx == 1)
        {
            rgb = t.rgb;
            a = tcc ? t.a : c.a;
        }
        else if (tfx == 2)
        {
            rgb = min(floor(t.rgb * c.rgb / 128.0) + c.a, 255.0);
            a = tcc ? min(t.a + c.a, 255.0) : c.a;
        }
        else
        {
            rgb = min(floor(t.rgb * c.rgb / 128.0) + c.a, 255.0);
            a = tcc ? t.a : c.a;
        }
        c = vec4(rgb, a);
    }
    if (uTest.x == 1)
    {
        float f = floor(vFog * 255.0 + 0.5);
        c.rgb = floor(f * c.rgb / 256.0) + floor((255.0 - f) * uFog.rgb / 256.0);
    }
    if (uTest.y >= 0)
    {
        bool passes = alphaPasses(c.a);
        if (uTest.z == 0 ? !passes : passes)
            discard;
    }
    vec3 rgb = c.rgb;
    float a = min(c.a / 128.0, 1.0);
    if (uStore.w == 1)
    {
        // A 16-bit frame keeps 5 bits per colour and the top bit of alpha.
        rgb = floor(rgb / 8.0) * 8.0;
        if (uTest.w != 2)
            a = c.a >= 128.0 ? 1.0 : 0.0;
    }
    if (uTest.w == 1)
        a = 1.0;  // FBA
    oColor = vec4(rgb / 255.0, a);
}
)";

        const char *kPresentVertex = R"(
uniform vec2 uOutSize;
out vec2 vPos;
void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vPos = p * uOutSize;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

        // The read circuits, as the CPU backend's PresentFromLocalMemory merges them.
        const char *kPresentFragment = R"(
uniform sampler2D uCrt1;
uniform sampler2D uCrt2;
uniform ivec4 uOrigin;   // x1, y1, x2, y2
uniform ivec4 uSize;     // w1, h1, w2, h2
uniform ivec4 uStore;    // stored sizes of 1 and 2
uniform ivec4 uEnable;   // crt1, crt2, slbg, mmod
uniform vec4 uBg;        // background, alp
uniform ivec2 uField;    // field mode, odd field
in vec2 vPos;
out vec4 oColor;

vec4 fetchCrt(sampler2D t, ivec2 p, ivec2 origin, ivec2 store)
{
    ivec2 q = p + origin;
    if (q.x < 0 || q.y < 0 || q.x >= store.x || q.y >= store.y)
        return vec4(0.0);
    return texelFetch(t, q, 0);
}

void main()
{
    ivec2 p = ivec2(floor(vPos));
    if (uField.x == 1)
        p.y = ((p.y >> 1) << 1) + uField.y;
    bool one = uEnable.x == 1 && p.x < uSize.x && p.y < uSize.y;
    bool two = uEnable.y == 1 && p.x < uSize.z && p.y < uSize.w;
    vec3 rgb;
    if (uEnable.x == 1 && uEnable.y == 1)
    {
        vec3 back = uBg.rgb;
        if (uEnable.z == 0 && two)
            back = fetchCrt(uCrt2, p, uOrigin.zw, uStore.zw).rgb;
        rgb = back;
        if (one)
        {
            vec4 c1 = fetchCrt(uCrt1, p, uOrigin.xy, uStore.xy);
            float factor = uEnable.w == 1 ? uBg.a : min(1.0, c1.a * 256.0 / 255.0);
            rgb = mix(back, c1.rgb, factor);
        }
    }
    else if (uEnable.x == 1)
        rgb = one ? fetchCrt(uCrt1, p, uOrigin.xy, uStore.xy).rgb : vec3(0.0);
    else
        rgb = two ? fetchCrt(uCrt2, p, uOrigin.zw, uStore.zw).rgb : vec3(0.0);
    oColor = vec4(rgb, 1.0);
}
)";

        GLuint compileShader(GLenum type, const char *body)
        {
            const GLuint shader = glCreateShader(type);
            const char *sources[2] = {kShaderHeader, body};
            glShaderSource(shader, 2, sources, nullptr);
            glCompileShader(shader);
            GLint ok = GL_FALSE;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
            if (!ok)
            {
                char log[2048] = {};
                glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
                std::fprintf(stderr, "[gles] shader compile failed: %s\n", log);
                glDeleteShader(shader);
                return 0;
            }
            return shader;
        }

        GLuint linkProgram(const char *vertex, const char *fragment)
        {
            const GLuint vs = compileShader(GL_VERTEX_SHADER, vertex);
            const GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragment);
            if (!vs || !fs)
                return 0;
            const GLuint program = glCreateProgram();
            glAttachShader(program, vs);
            glAttachShader(program, fs);
            glLinkProgram(program);
            glDeleteShader(vs);
            glDeleteShader(fs);
            GLint ok = GL_FALSE;
            glGetProgramiv(program, GL_LINK_STATUS, &ok);
            if (!ok)
            {
                char log[2048] = {};
                glGetProgramInfoLog(program, sizeof(log), nullptr, log);
                std::fprintf(stderr, "[gles] program link failed: %s\n", log);
                glDeleteProgram(program);
                return 0;
            }
            return program;
        }

        // ------------------------------------------------------------------ GL thread

        class Renderer
        {
        public:
            bool init();
            void execute(Batch &batch);

        private:
            struct Target
            {
                GLuint texture = 0;
                uint32_t width = 0;
                uint32_t height = 0;
                bool depth = false;
                bool needsClear = false;
            };
            struct Texture
            {
                GLuint texture = 0;
                uint32_t width = 0;
                uint32_t height = 0;
            };

            GLuint framebufferFor(uint32_t color, uint32_t depth);
            void forgetFramebuffers(uint32_t target);
            void draw(const Command &command);
            void present(const PresentParams &params);
            void readback(ReadbackJob &job);
            GLuint newTexture(GLint internalFormat, GLenum format, GLenum type, uint32_t width, uint32_t height,
                              const void *pixels);

            std::unordered_map<uint32_t, Target> m_targets;
            std::unordered_map<uint32_t, Texture> m_textures;
            std::unordered_map<uint64_t, GLuint> m_framebuffers;
            GLuint m_drawProgram = 0;
            GLuint m_presentProgram = 0;
            GLuint m_vao = 0;
            GLuint m_vbo = 0;
            GLuint m_emptyVao = 0;
            GLuint m_copyTexture = 0;
            uint32_t m_copyWidth = 0;
            uint32_t m_copyHeight = 0;
            GLuint m_copyFramebuffer = 0;
            struct DrawUniforms
            {
                GLint targetSize, tex, mode, wrap, region, store, texa, fog, test;
            } m_du{};
            struct PresentUniforms
            {
                GLint outSize, crt1, crt2, origin, size, store, enable, bg, field;
            } m_pu{};
        };

        GLuint Renderer::newTexture(GLint internalFormat, GLenum format, GLenum type, uint32_t width,
                                    uint32_t height, const void *pixels)
        {
            GLuint texture = 0;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, static_cast<GLsizei>(width), static_cast<GLsizei>(height),
                         0, format, type, pixels);
            return texture;
        }

        bool Renderer::init()
        {
            m_drawProgram = linkProgram(kDrawVertex, kDrawFragment);
            m_presentProgram = linkProgram(kPresentVertex, kPresentFragment);
            if (!m_drawProgram || !m_presentProgram)
                return false;
            const auto du = [&](const char *name) { return glGetUniformLocation(m_drawProgram, name); };
            m_du = {du("uTargetSize"), du("uTex"), du("uMode"), du("uWrap"), du("uRegion"),
                    du("uStore"), du("uTexa"), du("uFog"), du("uTest")};
            const auto pu = [&](const char *name) { return glGetUniformLocation(m_presentProgram, name); };
            m_pu = {pu("uOutSize"), pu("uCrt1"), pu("uCrt2"), pu("uOrigin"), pu("uSize"),
                    pu("uStore"), pu("uEnable"), pu("uBg"), pu("uField")};

            glGenVertexArrays(1, &m_vao);
            glGenBuffers(1, &m_vbo);
            glBindVertexArray(m_vao);
            glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(0));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(12));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), reinterpret_cast<void *>(24));
            glEnableVertexAttribArray(3);
            glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(28));
            glGenVertexArrays(1, &m_emptyVao);
            glGenFramebuffers(1, &m_copyFramebuffer);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glPixelStorei(GL_PACK_ALIGNMENT, 4);
            return glGetError() == GL_NO_ERROR;
        }

        GLuint Renderer::framebufferFor(uint32_t color, uint32_t depth)
        {
            const uint64_t key = (static_cast<uint64_t>(color) << 32) | depth;
            const auto found = m_framebuffers.find(key);
            if (found != m_framebuffers.end())
                return found->second;
            GLuint framebuffer = 0;
            glGenFramebuffers(1, &framebuffer);
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            const auto colorTarget = m_targets.find(color);
            if (colorTarget != m_targets.end())
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTarget->second.texture, 0);
            if (depth != 0u)
            {
                const auto depthTarget = m_targets.find(depth);
                if (depthTarget != m_targets.end())
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTarget->second.texture, 0);
            }
            const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE)
                noteOnce("incomplete framebuffer (status " + std::to_string(status) + ")");
            m_framebuffers.emplace(key, framebuffer);
            return framebuffer;
        }

        void Renderer::forgetFramebuffers(uint32_t target)
        {
            for (auto it = m_framebuffers.begin(); it != m_framebuffers.end();)
            {
                if (static_cast<uint32_t>(it->first >> 32) == target || static_cast<uint32_t>(it->first) == target)
                {
                    glDeleteFramebuffers(1, &it->second);
                    it = m_framebuffers.erase(it);
                }
                else
                    ++it;
            }
        }

        void Renderer::draw(const Command &command)
        {
            const DrawKey &k = command.key;
            const auto colorTarget = m_targets.find(k.target);
            if (colorTarget == m_targets.end() || command.count == 0u)
                return;
            const GLuint framebuffer = framebufferFor(k.target, k.depth);
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            if (k.depth != 0u)
            {
                auto depthTarget = m_targets.find(k.depth);
                if (depthTarget != m_targets.end() && depthTarget->second.needsClear)
                {
                    glDisable(GL_SCISSOR_TEST);
                    glDepthMask(GL_TRUE);
                    clearDepthToZero();
                    glClear(GL_DEPTH_BUFFER_BIT);
                    depthTarget->second.needsClear = false;
                }
            }
            glViewport(0, 0, k.targetWidth, k.targetHeight);
            glEnable(GL_SCISSOR_TEST);
            glScissor(k.scissorX0, k.scissorY0, k.scissorX1 - k.scissorX0 + 1, k.scissorY1 - k.scissorY0 + 1);

            GLuint texture = 0;
            if (k.source == SourceCache)
            {
                const auto found = m_textures.find(k.texture);
                if (found != m_textures.end())
                    texture = found->second.texture;
            }
            else if (k.source == SourceTarget || k.source == SourceTargetCopy)
            {
                const auto found = m_targets.find(k.texture);
                if (found != m_targets.end())
                {
                    texture = found->second.texture;
                    if (k.source == SourceTargetCopy)
                    {
                        // GL does not define sampling the texture a draw writes: copy it first.
                        const uint32_t w = found->second.width;
                        const uint32_t h = found->second.height;
                        if (!m_copyTexture || m_copyWidth != w || m_copyHeight != h)
                        {
                            if (m_copyTexture)
                                glDeleteTextures(1, &m_copyTexture);
                            m_copyTexture = newTexture(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, w, h, nullptr);
                            m_copyWidth = w;
                            m_copyHeight = h;
                        }
                        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebufferFor(k.texture, 0u));
                        glBindTexture(GL_TEXTURE_2D, m_copyTexture);
                        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h));
                        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
                        texture = m_copyTexture;
                    }
                }
            }
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);

            glUseProgram(m_drawProgram);
            glUniform2f(m_du.targetSize, static_cast<float>(k.targetWidth), static_cast<float>(k.targetHeight));
            glUniform1i(m_du.tex, 0);
            glUniform4i(m_du.mode, k.tme, k.tfx, k.tcc, k.linear);
            glUniform4i(m_du.wrap, k.wms, k.wmt, k.texWidth, k.texHeight);
            glUniform4i(m_du.region, k.minU, k.maxU, k.minV, k.maxV);
            glUniform4i(m_du.store, k.storeWidth, k.storeHeight, k.texaMode, k.frame16);
            glUniform4f(m_du.texa, k.ta0, k.ta1, k.aem, k.alphaScale);
            glUniform4f(m_du.fog, k.fogR, k.fogG, k.fogB, k.aref);
            const int alphaOut = k.forceAlpha ? 1 : (k.keepBlendAlpha ? 2 : 0);

            if (k.blend)
            {
                glEnable(GL_BLEND);
                glBlendEquationSeparate(k.blendEquation, GL_FUNC_ADD);
                glBlendFuncSeparate(k.blendSrc, k.blendDst, GL_ONE, GL_ZERO);
                const float fix = std::min(1.0f, static_cast<float>(k.blendFix) / 128.0f);
                glBlendColor(fix, fix, fix, fix);
            }
            else
                glDisable(GL_BLEND);

            const auto setDepth = [&](bool write)
            {
                if (k.depthFunc == 0u)
                {
                    glDisable(GL_DEPTH_TEST);
                    return;
                }
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(k.depthFunc == 1u ? GL_ALWAYS : k.depthFunc == 2u ? GL_GEQUAL : GL_GREATER);
                glDepthMask(write ? GL_TRUE : GL_FALSE);
            };
            const auto setColorMask = [&](uint8_t mask)
            {
                glColorMask((mask & 1u) != 0u, (mask & 2u) != 0u, (mask & 4u) != 0u, (mask & 8u) != 0u);
            };

            glBindVertexArray(m_vao);
            // Pass 1: the fragments that pass the alpha test (all of them without one).
            glUniform4i(m_du.test, k.fge, k.atst, 0, alphaOut);
            setDepth(k.depthWrite != 0u);
            setColorMask(k.colorMask);
            glDrawArrays(GL_TRIANGLES, static_cast<GLint>(command.first), static_cast<GLsizei>(command.count));
            // Pass 2: the failing ones, with what TEST.AFAIL still writes.
            if (k.atst >= 0 && k.afail != 0u)
            {
                uint8_t mask = k.colorMask;
                bool depthWrite = false;
                if (k.afail == 2u)
                {
                    mask = 0u;
                    depthWrite = k.depthWrite != 0u;
                }
                else if (k.afail == 3u)
                    mask &= 7u;
                if (mask != 0u || depthWrite)
                {
                    glUniform4i(m_du.test, k.fge, k.atst, 1, alphaOut);
                    setDepth(depthWrite);
                    setColorMask(mask);
                    glDrawArrays(GL_TRIANGLES, static_cast<GLint>(command.first), static_cast<GLsizei>(command.count));
                }
            }
            setColorMask(0xFu);
            glDepthMask(GL_TRUE);
        }

        void Renderer::readback(ReadbackJob &job)
        {
            const auto found = m_targets.find(job.target);
            job.pixels.assign(static_cast<size_t>(job.width) * job.height, 0u);
            if (found == m_targets.end() || job.pixels.empty())
                return;
            glBindFramebuffer(GL_READ_FRAMEBUFFER, framebufferFor(job.target, 0u));
            glReadPixels(0, 0, static_cast<GLsizei>(job.width), static_cast<GLsizei>(job.height), GL_RGBA,
                         GL_UNSIGNED_BYTE, job.pixels.data());
        }

        void Renderer::present(const PresentParams &params)
        {
            int slotIndex = 0;
            {
                std::lock_guard<std::mutex> lock(g_displayMutex);
                while (slotIndex == g_latestSlot || slotIndex == g_shownSlot)
                    ++slotIndex;
            }
            DisplaySlot &slot = g_slots[static_cast<size_t>(slotIndex)];
            if (!slot.texture || slot.width != params.outWidth || slot.height != params.outHeight)
            {
                if (slot.texture)
                    glDeleteTextures(1, &slot.texture);
                slot.texture = newTexture(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, params.outWidth, params.outHeight, nullptr);
                // The main thread sets the filter it shows the picture with.
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                if (!slot.framebuffer)
                    glGenFramebuffers(1, &slot.framebuffer);
                glBindFramebuffer(GL_FRAMEBUFFER, slot.framebuffer);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, slot.texture, 0);
                slot.width = params.outWidth;
                slot.height = params.outHeight;
            }
            glBindFramebuffer(GL_FRAMEBUFFER, slot.framebuffer);
            glViewport(0, 0, static_cast<GLsizei>(slot.width), static_cast<GLsizei>(slot.height));
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glUseProgram(m_presentProgram);
            int store[4] = {0, 0, 0, 0};
            for (int i = 0; i < 2; ++i)
            {
                const auto found = m_targets.find(params.target[i]);
                glActiveTexture(GL_TEXTURE0 + i);
                glBindTexture(GL_TEXTURE_2D, found != m_targets.end() ? found->second.texture : 0u);
                if (found != m_targets.end())
                {
                    store[i * 2] = static_cast<int>(found->second.width);
                    store[i * 2 + 1] = static_cast<int>(found->second.height);
                }
            }
            glUniform2f(m_pu.outSize, static_cast<float>(slot.width), static_cast<float>(slot.height));
            glUniform1i(m_pu.crt1, 0);
            glUniform1i(m_pu.crt2, 1);
            glUniform4i(m_pu.origin, params.originX[0], params.originY[0], params.originX[1], params.originY[1]);
            glUniform4i(m_pu.size, static_cast<int>(params.width[0]), static_cast<int>(params.height[0]),
                        static_cast<int>(params.width[1]), static_cast<int>(params.height[1]));
            glUniform4i(m_pu.store, store[0], store[1], store[2], store[3]);
            glUniform4i(m_pu.enable, params.enable[0] ? 1 : 0, params.enable[1] ? 1 : 0, params.slbg ? 1 : 0,
                        params.mmod ? 1 : 0);
            glUniform4f(m_pu.bg, static_cast<float>(params.bgcolor & 0xFFu) / 255.0f,
                        static_cast<float>((params.bgcolor >> 8) & 0xFFu) / 255.0f,
                        static_cast<float>((params.bgcolor >> 16) & 0xFFu) / 255.0f, static_cast<float>(params.alp) / 255.0f);
            glUniform2i(m_pu.field, params.field ? 1 : 0, params.oddField ? 1 : 0);
            glBindVertexArray(m_emptyVao);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, 0u);
            glActiveTexture(GL_TEXTURE0);
            // The main thread samples the picture in its own context: it has to be complete.
            const GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
            glDeleteSync(fence);
            std::lock_guard<std::mutex> lock(g_displayMutex);
            g_latestSlot = slotIndex;
        }

        void Renderer::execute(Batch &batch)
        {
            if (!batch.vertices.empty())
            {
                glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
                glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(batch.vertices.size() * sizeof(Vertex)),
                             batch.vertices.data(), GL_STREAM_DRAW);
            }
            for (Command &command : batch.commands)
            {
                switch (command.kind)
                {
                case Command::Kind::Draw:
                    draw(command);
                    break;
                case Command::Kind::CreateTarget:
                {
                    Target &target = m_targets[command.id];
                    if (target.texture)
                    {
                        forgetFramebuffers(command.id);
                        glDeleteTextures(1, &target.texture);
                    }
                    target.width = command.width;
                    target.height = command.height;
                    target.depth = command.depth;
                    target.needsClear = command.depth;
                    target.texture = command.depth
                                         ? newTexture(GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT,
                                                      command.width, command.height, nullptr)
                                         : newTexture(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, command.width, command.height,
                                                      nullptr);
                    break;
                }
                case Command::Kind::DeleteTarget:
                {
                    const auto found = m_targets.find(command.id);
                    if (found != m_targets.end())
                    {
                        forgetFramebuffers(command.id);
                        glDeleteTextures(1, &found->second.texture);
                        m_targets.erase(found);
                    }
                    break;
                }
                case Command::Kind::UploadTarget:
                {
                    const auto found = m_targets.find(command.id);
                    if (found == m_targets.end())
                        break;
                    glBindTexture(GL_TEXTURE_2D, found->second.texture);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(command.x), static_cast<GLint>(command.y),
                                    static_cast<GLsizei>(command.width), static_cast<GLsizei>(command.height), GL_RGBA,
                                    GL_UNSIGNED_BYTE, batch.blob.data() + command.blob);
                    break;
                }
                case Command::Kind::UploadTexture:
                {
                    Texture &texture = m_textures[command.id];
                    if (texture.texture && texture.width == command.width && texture.height == command.height)
                    {
                        glBindTexture(GL_TEXTURE_2D, texture.texture);
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(command.width),
                                        static_cast<GLsizei>(command.height), GL_RGBA, GL_UNSIGNED_BYTE,
                                        batch.blob.data() + command.blob);
                        break;
                    }
                    if (texture.texture)
                        glDeleteTextures(1, &texture.texture);
                    texture.texture = newTexture(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, command.width, command.height,
                                                 batch.blob.data() + command.blob);
                    texture.width = command.width;
                    texture.height = command.height;
                    break;
                }
                case Command::Kind::DeleteTexture:
                {
                    const auto found = m_textures.find(command.id);
                    if (found != m_textures.end())
                    {
                        glDeleteTextures(1, &found->second.texture);
                        m_textures.erase(found);
                    }
                    break;
                }
                case Command::Kind::Readback:
                {
                    readback(*command.job);
                    std::lock_guard<std::mutex> lock(channel().mutex);
                    command.job->done = true;
                    channel().doneCv.notify_all();
                    break;
                }
                case Command::Kind::Present:
                    present(command.present);
                    break;
                }
            }
        }

        void glThreadMain()
        {
            Channel &ch = channel();
            Renderer renderer;
            const bool ok = makeContextCurrent() && renderer.init();
            {
                std::lock_guard<std::mutex> lock(ch.mutex);
                ch.running = ok;
                if (!ok)
                    ch.stop = true;
            }
            ch.doneCv.notify_all();
            if (!ok)
            {
                std::fprintf(stderr, "[gles] the OpenGL renderer could not start its context\n");
                return;
            }
            for (;;)
            {
                Batch batch;
                {
                    std::unique_lock<std::mutex> lock(ch.mutex);
                    ch.workCv.wait(lock, [&]() { return ch.stop || !ch.queue.empty(); });
                    if (ch.queue.empty())
                        break;
                    batch = std::move(ch.queue.front());
                    ch.queue.pop_front();
                }
                renderer.execute(batch);
                {
                    std::lock_guard<std::mutex> lock(ch.mutex);
                    ch.queuedPresents -= std::min(ch.queuedPresents, batch.presents);
                }
                ch.doneCv.notify_all();
            }
        }

        // ------------------------------------------------------------------ GS memory layout

        struct PageRange
        {
            uint32_t first = 1;
            uint32_t last = 0;
            bool empty() const { return last < first; }
            bool overlaps(const PageRange &other) const
            {
                return !empty() && !other.empty() && first <= other.last && other.first <= last;
            }
        };

        uint32_t pageWidthOf(uint8_t psm)
        {
            switch (psm)
            {
            case GS_PSM_T8:
            case GS_PSM_T4:
                return 128u;
            default:
                return 64u;
            }
        }

        uint32_t pageHeightOf(uint8_t psm)
        {
            switch (psm)
            {
            case GS_PSM_CT16:
            case GS_PSM_CT16S:
            case GS_PSM_Z16:
            case GS_PSM_Z16S:
            case GS_PSM_T8:
                return 64u;
            case GS_PSM_T4:
                return 128u;
            default:
                return 32u;
            }
        }

        // The pages a w x h rectangle at a buffer's origin covers: whole rows of pages, which is
        // more than a narrow rectangle touches, never less.
        PageRange bufferPages(uint32_t blockBase, uint32_t bw, uint8_t psm, uint32_t width, uint32_t height)
        {
            PageRange range;
            if (width == 0u || height == 0u)
                return range;
            const uint32_t pw = pageWidthOf(psm);
            const uint32_t ph = pageHeightOf(psm);
            const uint32_t pagesPerRow = std::max<uint32_t>(1u, (std::max<uint32_t>(1u, bw) * 64u + pw - 1u) / pw);
            const uint32_t widthPages = (width + pw - 1u) / pw;
            range.first = blockBase / 32u;
            range.last = range.first + ((height - 1u) / ph) * pagesPerRow +
                         std::max(widthPages, pagesPerRow) - 1u + ((blockBase % 32u) != 0u ? 1u : 0u);
            return range;
        }

        bool is16Bit(uint8_t psm)
        {
            return psm == GS_PSM_CT16 || psm == GS_PSM_CT16S || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S;
        }

        // Stored target pixels (RGBA8, alpha in 1/128 units) from local memory values.
        uint32_t toStored(uint8_t psm, uint32_t raw)
        {
            uint32_t rgba = raw;
            uint32_t a = 0x80u;
            if (is16Bit(psm))
            {
                const uint32_t r = (raw & 0x1Fu) << 3;
                const uint32_t g = ((raw >> 5) & 0x1Fu) << 3;
                const uint32_t b = ((raw >> 10) & 0x1Fu) << 3;
                rgba = r | (g << 8) | (b << 16);
                a = (raw & 0x8000u) ? 0x80u : 0u;
            }
            else if (psm != GS_PSM_CT24 && psm != GS_PSM_Z24)
                a = raw >> 24;
            const uint32_t stored = (std::min<uint32_t>(a, 128u) * 255u + 64u) / 128u;
            return (rgba & 0x00FFFFFFu) | (stored << 24);
        }

        uint32_t fromStored(uint8_t psm, uint32_t stored)
        {
            const uint32_t a = ((stored >> 24) * 128u + 127u) / 255u;
            if (is16Bit(psm))
            {
                const uint32_t r = (stored & 0xFFu) >> 3;
                const uint32_t g = ((stored >> 8) & 0xFFu) >> 3;
                const uint32_t b = ((stored >> 16) & 0xFFu) >> 3;
                return r | (g << 5) | (b << 10) | ((a >> 7) << 15);
            }
            if (psm == GS_PSM_CT24 || psm == GS_PSM_Z24)
                return stored & 0x00FFFFFFu;
            return (stored & 0x00FFFFFFu) | (a << 24);
        }

        double depthRange(uint8_t zpsm)
        {
            switch (zpsm)
            {
            case GS_PSM_Z24:
                return 16777216.0;
            case GS_PSM_Z16:
            case GS_PSM_Z16S:
                return 65536.0;
            default:
                return 4294967296.0;
            }
        }

        struct BlendSetup
        {
            bool enable = false;
            uint16_t equation = GL_FUNC_ADD;
            uint16_t src = GL_ONE;
            uint16_t dst = GL_ZERO;
            bool usesSourceAlpha = false;
        };

        // GS blending, (A - B) * C + D with A, B, D in {Cs, Cd, 0} and C in {As, Ad, FIX} (0x80
        // is 1.0), as a GL blend equation. The few forms GL cannot express keep their main term.
        BlendSetup mapBlend(uint64_t alphaReg)
        {
            uint32_t a = alphaReg & 3u;
            uint32_t b = (alphaReg >> 2) & 3u;
            uint32_t c = (alphaReg >> 4) & 3u;
            uint32_t d = (alphaReg >> 6) & 3u;
            a = std::min(a, 2u);
            b = std::min(b, 2u);
            c = std::min(c, 2u);
            d = std::min(d, 2u);
            const uint16_t f = c == 0u ? GL_SRC_ALPHA : c == 1u ? GL_DST_ALPHA : GL_CONSTANT_ALPHA;
            const uint16_t invF = c == 0u ? GL_ONE_MINUS_SRC_ALPHA : c == 1u ? GL_ONE_MINUS_DST_ALPHA
                                                                         : GL_ONE_MINUS_CONSTANT_ALPHA;
            BlendSetup s;
            s.enable = true;
            s.usesSourceAlpha = c == 0u;
            const auto set = [&](uint16_t equation, uint16_t src, uint16_t dst)
            {
                s.equation = equation;
                s.src = src;
                s.dst = dst;
            };
            const auto approximate = [&](uint16_t equation, uint16_t src, uint16_t dst)
            {
                noteOnce("blend " + std::to_string(alphaReg & 0xFFu) + " approximated");
                set(equation, src, dst);
            };
            constexpr uint32_t Cs = 0u, Cd = 1u, Z = 2u;
            if (a == b)
            {
                s.usesSourceAlpha = false;
                if (d == Cs)
                    s.enable = false;
                else if (d == Cd)
                    set(GL_FUNC_ADD, GL_ZERO, GL_ONE);
                else
                    set(GL_FUNC_ADD, GL_ZERO, GL_ZERO);
                return s;
            }
            if (a == Cs && b == Cd)
            {
                if (d == Cd)
                    set(GL_FUNC_ADD, f, invF);
                else if (d == Z)
                    set(GL_FUNC_SUBTRACT, f, f);
                else
                    approximate(GL_FUNC_SUBTRACT, GL_ONE, f);
            }
            else if (a == Cs && b == Z)
            {
                if (d == Cd)
                    set(GL_FUNC_ADD, f, GL_ONE);
                else if (d == Z)
                    set(GL_FUNC_ADD, f, GL_ZERO);
                else
                    approximate(GL_FUNC_ADD, GL_ONE, GL_ZERO);
            }
            else if (a == Cd && b == Cs)
            {
                if (d == Cs)
                    set(GL_FUNC_ADD, invF, f);
                else if (d == Z)
                    set(GL_FUNC_REVERSE_SUBTRACT, f, f);
                else
                    approximate(GL_FUNC_REVERSE_SUBTRACT, f, GL_ONE);
            }
            else if (a == Cd && b == Z)
            {
                if (d == Cs)
                    set(GL_FUNC_ADD, GL_ONE, f);
                else if (d == Z)
                    set(GL_FUNC_ADD, GL_ZERO, f);
                else
                    approximate(GL_FUNC_ADD, GL_ZERO, GL_ONE);
            }
            else if (a == Z && b == Cs)
            {
                if (d == Cs)
                    set(GL_FUNC_ADD, invF, GL_ZERO);
                else if (d == Cd)
                    set(GL_FUNC_REVERSE_SUBTRACT, f, GL_ONE);
                else
                    set(GL_FUNC_ADD, GL_ZERO, GL_ZERO);
            }
            else // a == Z && b == Cd
            {
                if (d == Cs)
                    set(GL_FUNC_SUBTRACT, GL_ONE, f);
                else if (d == Cd)
                    set(GL_FUNC_ADD, GL_ZERO, invF);
                else
                    set(GL_FUNC_ADD, GL_ZERO, GL_ZERO);
            }
            return s;
        }

        // ------------------------------------------------------------------ the backend

        class GlesBackend final : public GSRasterBackend
        {
        public:
            GlesBackend() = default;
            ~GlesBackend() override = default;

            void Initialize(uint8_t *vram, uint32_t vramSize) override;
            void Reset() override;
            void Submit(const GSPrimitiveBatch &batch) override;
            void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
            void BeginTransfer(const GSTransferCommand &command) override;
            void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;
            void Flush() override;
            void TextureFlush() override;
            void Sync(GSSyncReason reason) override;
            PresentationFrame Present(const GSPresentationRequest &request) override;
            bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
            uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;
            uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
            void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
            void SnapshotVram(std::vector<uint8_t> &out) const override;
            GSTransferSnapshot GetTransferSnapshot() const override;

        private:
            // A frame or depth buffer on the GPU.
            struct Target
            {
                uint32_t id = 0;
                uint32_t fbp = 0;  // pages
                uint32_t fbw = 0;
                uint8_t psm = 0;
                bool depth = false;
                uint32_t width = 0;
                uint32_t height = 0;  // allocated rows
                uint32_t rows = 0;    // rows drawn or shown: the memory the target stands for
                bool gpuDirty = false;
                uint64_t syncEpoch = 0;
                uint64_t checkedWrites = 0;
                PageRange pages() const
                {
                    return bufferPages(fbp * 32u, fbw, psm, width, std::max<uint32_t>(rows, 1u));
                }
            };
            // A texture decoded from local memory.
            struct CachedTexture
            {
                uint32_t id = 0;
                uint32_t tbp0 = 0;
                uint8_t tbw = 0;
                uint8_t psm = 0;
                uint32_t width = 0;
                uint32_t height = 0;
                uint8_t cpsm = 0, csm = 0, csa = 0;
                uint8_t ta0 = 0, ta1 = 0;
                bool aem = false;
                uint64_t clutVersion = 0;
                uint64_t epoch = 0;
                uint64_t checkedWrites = 0;
                uint64_t lastUse = 0;
            };

            void handOver(bool wait);
            Command &push(Command::Kind kind);
            Target *findTarget(uint32_t fbp, uint32_t fbw, uint8_t psm, bool depth);
            Target *targetById(uint32_t id);
            Target &colorTarget(const GSFrameReg &frame, uint32_t rowsNeeded);
            Target &depthTarget(const GSZbufReg &zbuf, uint32_t fbw, const Target &color, uint32_t rowsNeeded);
            void growRows(Target &target, uint32_t rows);
            void removeOverlapping(const PageRange &pages, uint32_t keepId);
            void deleteTarget(size_t index);
            bool cpuWroteSince(Target &target);
            void reloadTarget(Target &target);
            void writeBack(Target &target);
            // why: what reads or writes that memory (PS2X_GLES_TRACE names it once per kind).
            void writeBackOverlapping(const PageRange &pages, const std::string &why);
            void writeBackAll();
            bool bindTexture(const GSDrawState &state, const Target &drawTarget, DrawKey &key);
            void noteCpuWrite() { ++m_cpuWrites; }
            void trace();

            mutable std::recursive_mutex m_mutex;
            mutable GSCpuBackend m_cpu;
            Batch m_batch;
            int m_openDraw = -1;
            std::vector<Target> m_targets;
            std::vector<CachedTexture> m_textures;
            uint32_t m_nextId = 1;
            uint64_t m_cpuWrites = 1;
            uint64_t m_useCounter = 0;
            std::vector<uint32_t> m_scratch;
            std::vector<uint32_t> m_decoded;
            struct Stats
            {
                uint64_t draws = 0, vertices = 0, uploads = 0, uploadTexels = 0, writeBacks = 0, reloads = 0,
                         targetsCreated = 0, presents = 0, writeBackNs = 0, directUploads = 0;
            } m_stats;
            // A host-to-local transfer into a rectangle of a GPU frame buffer of its format: the
            // pixels go to the GPU too when it ends, instead of copying the buffer back first.
            struct DirectUpload
            {
                bool active = false;
                uint32_t target = 0;
                uint32_t x = 0, y = 0, width = 0, height = 0;
            } m_directUpload;
            void finishDirectUpload();
            // A palette load from memory the GPU drew: TEX0 writes load palettes the next draws
            // often never use, so the copy back waits for an indexed draw (the load is replayed then).
            struct PendingClut
            {
                bool active = false;
                GSTex0Reg tex0{};
                GSTexClutReg texclut{};
            } m_pendingClut;
            static PageRange clutPages(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
            {
                return bufferPages(tex0.cbp, std::max<uint32_t>(texclut.cbw, 1u), tex0.cpsm, 16u, 16u);
            }
        };

        constexpr size_t kMaxTextures = 512u;
        // Batches go to the GL thread at the end of a GIF packet once this many vertices are queued.
        constexpr size_t kHandOverVertices = 3000u;

        void GlesBackend::Initialize(uint8_t *vram, uint32_t vramSize)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            m_cpu.SetSynchronous();
            m_cpu.Initialize(vram, vramSize);
        }

        void GlesBackend::Reset()
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            m_cpu.Reset();
            while (!m_targets.empty())
                deleteTarget(m_targets.size() - 1u);
            for (const CachedTexture &texture : m_textures)
                push(Command::Kind::DeleteTexture).id = texture.id;
            m_textures.clear();
            noteCpuWrite();
            handOver(false);
        }

        Command &GlesBackend::push(Command::Kind kind)
        {
            m_batch.commands.emplace_back();
            m_batch.commands.back().kind = kind;
            m_openDraw = -1;
            return m_batch.commands.back();
        }

        // Queues the batch for the GL thread; with wait, returns once the thread has run it.
        void GlesBackend::handOver(bool wait)
        {
            if (m_batch.empty())
                return;
            Channel &ch = channel();
            ReadbackJob *waitFor = nullptr;
            if (wait && !m_batch.commands.empty() && m_batch.commands.back().kind == Command::Kind::Readback)
                waitFor = m_batch.commands.back().job;
            std::unique_lock<std::mutex> lock(ch.mutex);
            if (ch.stop)
            {
                // No GL thread (it failed, or the runtime is closing): nothing will run.
                m_batch = {};
                m_openDraw = -1;
                if (waitFor)
                    waitFor->pixels.assign(static_cast<size_t>(waitFor->width) * waitFor->height, 0u);
                return;
            }
            ch.doneCv.wait(lock, [&]() { return ch.stop || ch.queuedPresents < kMaxQueuedPresents || m_batch.presents == 0u; });
            ch.queuedPresents += m_batch.presents;
            ch.queue.push_back(std::move(m_batch));
            m_batch = {};
            m_openDraw = -1;
            ch.workCv.notify_one();
            if (waitFor)
            {
                ch.doneCv.wait(lock, [&]() { return ch.stop || waitFor->done; });
                if (!waitFor->done)
                    waitFor->pixels.assign(static_cast<size_t>(waitFor->width) * waitFor->height, 0u);
            }
        }

        GlesBackend::Target *GlesBackend::findTarget(uint32_t fbp, uint32_t fbw, uint8_t psm, bool depth)
        {
            for (Target &target : m_targets)
                if (target.depth == depth && target.fbp == fbp && target.fbw == fbw && is16Bit(target.psm) == is16Bit(psm))
                    return &target;
            return nullptr;
        }

        GlesBackend::Target *GlesBackend::targetById(uint32_t id)
        {
            for (Target &target : m_targets)
                if (target.id == id)
                    return &target;
            return nullptr;
        }

        void GlesBackend::deleteTarget(size_t index)
        {
            push(Command::Kind::DeleteTarget).id = m_targets[index].id;
            m_targets.erase(m_targets.begin() + static_cast<std::ptrdiff_t>(index));
        }

        // A buffer reuses memory other targets stood for: those go (copied back first when the
        // GPU drew them), and memory they leave is local memory again.
        void GlesBackend::removeOverlapping(const PageRange &pages, uint32_t keepId)
        {
            for (size_t i = m_targets.size(); i-- > 0;)
            {
                Target &other = m_targets[i];
                if (other.id == keepId || !other.pages().overlaps(pages))
                    continue;
                if (other.gpuDirty && !other.depth)
                    writeBack(other);
                deleteTarget(i);
            }
        }

        void GlesBackend::growRows(Target &target, uint32_t rows)
        {
            rows = std::min(rows, target.height);
            if (rows <= target.rows)
                return;
            const uint32_t before = target.rows;
            target.rows = rows;
            const PageRange grown = target.pages();
            if (before == 0u || grown.last != bufferPages(target.fbp * 32u, target.fbw, target.psm, target.width, before).last)
                removeOverlapping(grown, target.id);
        }

        bool GlesBackend::cpuWroteSince(Target &target)
        {
            if (target.checkedWrites == m_cpuWrites)
                return false;
            target.checkedWrites = m_cpuWrites;
            GSTex0Reg region{};
            region.tbp0 = target.fbp * 32u;
            region.tbw = static_cast<uint8_t>(target.fbw);
            region.psm = target.psm;
            return m_cpu.TextureWrittenSince(region, target.width, std::max<uint32_t>(target.rows, 1u), target.syncEpoch);
        }

        // The target's whole allocation from local memory.
        void GlesBackend::reloadTarget(Target &target)
        {
            m_cpu.ReadRectRaw(target.psm, target.fbp * 32u, target.fbw, 0u, 0u, target.width, target.height, m_scratch);
            Command &upload = push(Command::Kind::UploadTarget);
            upload.id = target.id;
            upload.width = target.width;
            upload.height = target.height;
            upload.blob = m_batch.blob.size();
            m_batch.blob.reserve(m_batch.blob.size() + m_scratch.size());
            for (const uint32_t raw : m_scratch)
                m_batch.blob.push_back(toStored(target.psm, raw));
            target.syncEpoch = m_cpu.NewTextureEpoch();
            target.checkedWrites = m_cpuWrites;
            target.gpuDirty = false;
            ++m_stats.reloads;
        }

        void GlesBackend::writeBack(Target &target)
        {
            if (!target.gpuDirty || target.depth)
                return;
            const auto started = std::chrono::steady_clock::now();
            ReadbackJob job;
            job.target = target.id;
            job.width = target.width;
            job.height = std::max<uint32_t>(target.rows, 1u);
            push(Command::Kind::Readback).job = &job;
            handOver(true);
            m_scratch.resize(job.pixels.size());
            for (size_t i = 0; i < job.pixels.size(); ++i)
                m_scratch[i] = fromStored(target.psm, job.pixels[i]);
            m_cpu.WriteRectRaw(target.psm, target.fbp * 32u, target.fbw, job.width, job.height, m_scratch.data());
            noteCpuWrite();
            target.gpuDirty = false;
            target.syncEpoch = m_cpu.NewTextureEpoch();
            target.checkedWrites = m_cpuWrites;
            ++m_stats.writeBacks;
            m_stats.writeBackNs += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        }

        void GlesBackend::writeBackOverlapping(const PageRange &pages, const std::string &why)
        {
            for (Target &target : m_targets)
                if (target.gpuDirty && !target.depth && target.pages().overlaps(pages))
                {
                    if (traceEnabled())
                        noteOnce("copy back: " + why + " overlaps target fbp " + std::to_string(target.fbp) + " fbw " +
                                 std::to_string(target.fbw) + " psm " + std::to_string(target.psm) + " rows " +
                                 std::to_string(target.rows));
                    writeBack(target);
                }
        }

        void GlesBackend::writeBackAll()
        {
            for (Target &target : m_targets)
                writeBack(target);
        }

        GlesBackend::Target &GlesBackend::colorTarget(const GSFrameReg &frame, uint32_t rowsNeeded)
        {
            const uint32_t fbw = std::max<uint32_t>(frame.fbw, 1u);
            if (Target *found = findTarget(frame.fbp, fbw, frame.psm, false))
            {
                found->psm = frame.psm;
                if (!found->gpuDirty && cpuWroteSince(*found))
                    reloadTarget(*found);
                // Growing can drop other targets, and move this one in the list.
                const uint32_t id = found->id;
                growRows(*found, rowsNeeded);
                return *targetById(id);
            }
            Target target;
            target.id = m_nextId++;
            target.fbp = frame.fbp;
            target.fbw = fbw;
            target.psm = frame.psm;
            target.width = std::min<uint32_t>(fbw * 64u, 2048u);
            // As many rows as fit before the end of local memory, at most 1024.
            const uint32_t pageRows = frame.fbp < 512u ? (512u - frame.fbp) / std::max<uint32_t>(1u, (fbw * 64u + 63u) / 64u) : 0u;
            target.height = std::clamp<uint32_t>(pageRows * pageHeightOf(frame.psm), 1u, 1024u);
            target.rows = std::min(std::max<uint32_t>(rowsNeeded, 1u), target.height);
            removeOverlapping(target.pages(), 0u);
            Command &create = push(Command::Kind::CreateTarget);
            create.id = target.id;
            create.width = target.width;
            create.height = target.height;
            m_targets.push_back(target);
            Target &added = m_targets.back();
            reloadTarget(added);
            ++m_stats.targetsCreated;
            return added;
        }

        GlesBackend::Target &GlesBackend::depthTarget(const GSZbufReg &zbuf, uint32_t fbw, const Target &color,
                                                      uint32_t rowsNeeded)
        {
            for (Target &target : m_targets)
                if (target.depth && target.fbp == zbuf.zbp && target.fbw == fbw && target.width == color.width &&
                    target.height == color.height)
                {
                    target.psm = zbuf.psm;
                    const uint32_t id = target.id;
                    growRows(target, rowsNeeded);
                    return *targetById(id);
                }
            Target target;
            target.id = m_nextId++;
            target.fbp = zbuf.zbp;
            target.fbw = fbw;
            target.psm = zbuf.psm;
            target.depth = true;
            target.width = color.width;
            target.height = color.height;
            target.rows = std::min(std::max<uint32_t>(rowsNeeded, 1u), target.height);
            const uint32_t colorId = color.id;
            removeOverlapping(target.pages(), colorId);
            Command &create = push(Command::Kind::CreateTarget);
            create.id = target.id;
            create.width = target.width;
            create.height = target.height;
            create.depth = true;
            m_targets.push_back(target);
            ++m_stats.targetsCreated;
            return m_targets.back();
        }

        // Fills the key's texture fields: a GPU target when the texture is one, else a texture
        // decoded from local memory (after copying back what the GPU drew over it).
        bool GlesBackend::bindTexture(const GSDrawState &state, const Target &drawTarget, DrawKey &key)
        {
            const GSTex0Reg &tex = state.context.tex0;
            const uint32_t width = state.textureWidth;
            const uint32_t height = state.textureHeight;
            const uint64_t clamp = state.context.clamp;
            key.tme = 1u;
            key.tfx = tex.tfx;
            key.tcc = tex.tcc;
            key.linear = state.linearFilter ? 1u : 0u;
            key.texWidth = static_cast<uint16_t>(width);
            key.texHeight = static_cast<uint16_t>(height);
            key.wms = static_cast<uint8_t>(clamp & 3u);
            key.wmt = static_cast<uint8_t>((clamp >> 2) & 3u);
            key.minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
            key.maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
            key.minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
            key.maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);
            key.ta0 = state.texa.ta0;
            key.ta1 = state.texa.ta1;
            key.aem = state.texa.aem ? 1u : 0u;

            const PageRange pages = bufferPages(tex.tbp0, tex.tbw, tex.psm, std::max<uint32_t>(width, tex.tbw * 64u), height);
            // A frame buffer the GPU holds, read in a format of the same layout.
            for (Target &target : m_targets)
            {
                if (target.depth || target.fbp * 32u != tex.tbp0 || target.fbw != tex.tbw)
                    continue;
                const bool same32 = !is16Bit(target.psm) && (tex.psm == GS_PSM_CT32 || tex.psm == GS_PSM_CT24);
                const bool same16 = is16Bit(target.psm) && (tex.psm == GS_PSM_CT16 || tex.psm == GS_PSM_CT16S);
                if (!same32 && !same16)
                    continue;
                if (!target.gpuDirty && cpuWroteSince(target))
                    reloadTarget(target);
                key.source = target.id == drawTarget.id ? SourceTargetCopy : SourceTarget;
                key.texture = target.id;
                key.storeWidth = static_cast<uint16_t>(target.width);
                key.storeHeight = static_cast<uint16_t>(target.height);
                key.alphaScale = 128u;
                key.texaMode = tex.psm == GS_PSM_CT24 ? 1u : same16 ? 2u : 0u;
                if (target.psm == GS_PSM_CT24 && tex.psm == GS_PSM_CT32)
                    key.texaMode = 0u;
                // Last: growing can drop other targets (the loop does not go on after it).
                growRows(target, height);
                return true;
            }
            writeBackOverlapping(pages, traceEnabled() ? "texture tbp " + std::to_string(tex.tbp0) + " tbw " +
                                                             std::to_string(tex.tbw) + " psm " + std::to_string(tex.psm) +
                                                             " " + std::to_string(width) + "x" + std::to_string(height)
                                                       : std::string());
            for (const Target &target : m_targets)
                if (target.depth && target.gpuDirty && target.pages().overlaps(pages))
                    noteOnce("a texture reads a depth buffer the GPU drew (local memory has none of it)");

            const bool indexed = tex.psm == GS_PSM_T8 || tex.psm == GS_PSM_T4 || tex.psm == GS_PSM_T8H ||
                                 tex.psm == GS_PSM_T4HL || tex.psm == GS_PSM_T4HH;
            if (indexed && m_pendingClut.active)
            {
                const PendingClut pending = m_pendingClut;
                m_pendingClut = {};
                writeBackOverlapping(clutPages(pending.tex0, pending.texclut),
                                     "palette cbp " + std::to_string(pending.tex0.cbp));
                GSTex0Reg reload = pending.tex0;
                reload.cld = 1u;
                m_cpu.LoadClut(reload, pending.texclut);
            }
            const bool usesTexa = tex.psm != GS_PSM_CT32 &&
                                  (!indexed || tex.cpsm == GS_PSM_CT16 || tex.cpsm == GS_PSM_CT16S || tex.cpsm == GS_PSM_CT24);
            const uint64_t clutVersion = indexed ? m_cpu.ClutVersion() : 0u;
            CachedTexture *entry = nullptr;
            for (CachedTexture &candidate : m_textures)
            {
                if (candidate.tbp0 == tex.tbp0 && candidate.tbw == tex.tbw && candidate.psm == tex.psm &&
                    candidate.width == width && candidate.height == height &&
                    (!usesTexa || (candidate.ta0 == state.texa.ta0 && candidate.ta1 == state.texa.ta1 &&
                                   candidate.aem == state.texa.aem)) &&
                    (!indexed || (candidate.cpsm == tex.cpsm && candidate.csm == tex.csm && candidate.csa == tex.csa &&
                                  candidate.clutVersion == clutVersion)))
                {
                    entry = &candidate;
                    break;
                }
            }
            bool decode = entry == nullptr;
            if (entry && entry->checkedWrites != m_cpuWrites)
            {
                entry->checkedWrites = m_cpuWrites;
                decode = m_cpu.TextureWrittenSince(tex, std::max<uint32_t>(width, tex.tbw * 64u), height, entry->epoch);
            }
            if (!entry)
            {
                if (m_textures.size() < kMaxTextures)
                {
                    m_textures.emplace_back();
                    entry = &m_textures.back();
                    entry->id = m_nextId++;
                }
                else
                {
                    entry = &*std::min_element(m_textures.begin(), m_textures.end(),
                                               [](const CachedTexture &l, const CachedTexture &r) { return l.lastUse < r.lastUse; });
                }
                entry->tbp0 = tex.tbp0;
                entry->tbw = tex.tbw;
                entry->psm = tex.psm;
                entry->width = width;
                entry->height = height;
                entry->cpsm = tex.cpsm;
                entry->csm = tex.csm;
                entry->csa = tex.csa;
                entry->ta0 = state.texa.ta0;
                entry->ta1 = state.texa.ta1;
                entry->aem = state.texa.aem;
                entry->clutVersion = clutVersion;
            }
            if (decode)
            {
                entry->epoch = m_cpu.NewTextureEpoch();
                entry->checkedWrites = m_cpuWrites;
                m_cpu.DecodeTextureRgba(state, width, height, m_decoded);
                Command &upload = push(Command::Kind::UploadTexture);
                upload.id = entry->id;
                upload.width = width;
                upload.height = height;
                upload.blob = m_batch.blob.size();
                m_batch.blob.insert(m_batch.blob.end(), m_decoded.begin(), m_decoded.end());
                ++m_stats.uploads;
                m_stats.uploadTexels += m_decoded.size();
            }
            entry->lastUse = ++m_useCounter;
            key.source = SourceCache;
            key.texture = entry->id;
            key.storeWidth = static_cast<uint16_t>(width);
            key.storeHeight = static_cast<uint16_t>(height);
            key.alphaScale = 255u;
            key.texaMode = 0u;
            return true;
        }

        void GlesBackend::Submit(const GSPrimitiveBatch &batch)
        {
            if (batch.vertexCount == 0u)
                return;
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            const GSDrawState &state = batch.state;
            const GSContext &ctx = state.context;
            const uint64_t test = ctx.test;
            const bool zte = ((test >> 16) & 1u) != 0u;
            const uint32_t ztst = static_cast<uint32_t>((test >> 17) & 3u);
            if (zte && ztst == 0u)
                return;
            const bool ate = (test & 1u) != 0u;
            const uint32_t atst = static_cast<uint32_t>((test >> 1) & 7u);
            const uint32_t afail = static_cast<uint32_t>((test >> 12) & 3u);
            if (ate && atst == 0u && afail == 0u)
                return;
            if (((test >> 14) & 1u) != 0u)
                noteOnce("DATE (destination alpha test) not drawn");

            const int ofx = ctx.xyoffset.ofx >> 4;
            const int ofy = ctx.xyoffset.ofy >> 4;
            const bool depthUsed = zte && (ztst >= 2u || !ctx.zbuf.zmask);
            const double zRange = depthRange(ctx.zbuf.psm);
            const auto depthOf = [&](double z) -> float
            {
                if (!depthUsed)
                    return 0.0f;
                return static_cast<float>(std::clamp(z, 0.0, zRange - 1.0) / zRange);
            };
            const bool tme = state.prim.tme;
            const float texW = static_cast<float>(state.textureWidth);
            const float texH = static_cast<float>(state.textureHeight);

            // Converted vertices; every primitive becomes triangles.
            Vertex out[6];
            uint32_t count = 0;
            const auto put = [&](float x, float y, float z, float s, float t, float q, const GSVertex &color, float fog)
            {
                out[count++] = {x, y, z, s, t, q, color.r, color.g, color.b, color.a, fog};
            };
            const auto texOf = [&](const GSVertex &v, float &s, float &t, float &q)
            {
                if (!tme)
                {
                    s = t = 0.0f;
                    q = 1.0f;
                }
                else if (state.prim.fst)
                {
                    s = static_cast<float>(v.u) / 16.0f;
                    t = static_cast<float>(v.v) / 16.0f;
                    q = 1.0f;
                }
                else
                {
                    s = v.s * texW;
                    t = v.t * texH;
                    q = v.q;
                }
            };
            // A rectangle as two triangles, with texture coordinates at its corners.
            const auto quad = [&](float x0, float y0, float x1, float y1, float u0, float v0, float u1, float v1,
                                  float z, const GSVertex &color, float fog)
            {
                put(x0, y0, z, u0, v0, 1.0f, color, fog);
                put(x1, y0, z, u1, v0, 1.0f, color, fog);
                put(x0, y1, z, u0, v1, 1.0f, color, fog);
                put(x1, y0, z, u1, v0, 1.0f, color, fog);
                put(x1, y1, z, u1, v1, 1.0f, color, fog);
                put(x0, y1, z, u0, v1, 1.0f, color, fog);
            };

            float minX = 0.0f, minY = 0.0f, maxX = 0.0f, maxY = 0.0f;
            switch (state.prim.type)
            {
            case GS_PRIM_SPRITE:
            {
                if (batch.vertexCount < 2u)
                    return;
                const GSVertex &v0 = batch.vertices[0];
                const GSVertex &v1 = batch.vertices[1];
                // As the CPU backend draws them: corners truncated to pixels, the texture's
                // first coordinate at the left and top edges.
                int x0 = static_cast<int>(v0.x) - ofx;
                int y0 = static_cast<int>(v0.y) - ofy;
                int x1 = static_cast<int>(v1.x) - ofx;
                int y1 = static_cast<int>(v1.y) - ofy;
                if (x0 > x1)
                    std::swap(x0, x1);
                if (y0 > y1)
                    std::swap(y0, y1);
                x1 = x0 + std::max(1, x1 - x0);
                y1 = y0 + std::max(1, y1 - y0);
                float u0 = 0.0f, tv0 = 0.0f, u1 = 0.0f, tv1 = 0.0f;
                if (tme)
                {
                    if (state.prim.fst)
                    {
                        u0 = static_cast<float>(v0.u >> 4);
                        tv0 = static_cast<float>(v0.v >> 4);
                        u1 = static_cast<float>(v1.u >> 4);
                        tv1 = static_cast<float>(v1.v >> 4);
                    }
                    else
                    {
                        const float q0 = std::fabs(v0.q) > 1.0e-8f ? v0.q : 1.0f;
                        const float q1 = std::fabs(v1.q) > 1.0e-8f ? v1.q : 1.0f;
                        u0 = v0.s / q0 * texW;
                        tv0 = v0.t / q0 * texH;
                        u1 = v1.s / q1 * texW;
                        tv1 = v1.t / q1 * texH;
                    }
                }
                quad(static_cast<float>(x0), static_cast<float>(y0), static_cast<float>(x1), static_cast<float>(y1),
                     u0, tv0, u1, tv1, depthOf(v1.z), v1, static_cast<float>(v1.fog) / 255.0f);
                minX = static_cast<float>(x0);
                minY = static_cast<float>(y0);
                maxX = static_cast<float>(x1 - 1);
                maxY = static_cast<float>(y1 - 1);
                break;
            }
            case GS_PRIM_TRIANGLE:
            case GS_PRIM_TRISTRIP:
            case GS_PRIM_TRIFAN:
            {
                if (batch.vertexCount < 3u)
                    return;
                const GSVertex &provoking = batch.vertices[2];
                minX = minY = 1.0e9f;
                maxX = maxY = -1.0e9f;
                for (uint32_t i = 0; i < 3u; ++i)
                {
                    const GSVertex &v = batch.vertices[i];
                    float s, t, q;
                    texOf(v, s, t, q);
                    const float x = v.x - static_cast<float>(ofx);
                    const float y = v.y - static_cast<float>(ofy);
                    put(x, y, depthOf(v.z), s, t, q, state.prim.iip ? v : provoking, static_cast<float>(v.fog) / 255.0f);
                    minX = std::min(minX, x);
                    minY = std::min(minY, y);
                    maxX = std::max(maxX, x);
                    maxY = std::max(maxY, y);
                }
                break;
            }
            case GS_PRIM_LINE:
            case GS_PRIM_LINESTRIP:
            {
                if (batch.vertexCount < 2u)
                    return;
                const GSVertex &a = batch.vertices[0];
                const GSVertex &b = batch.vertices[1];
                const float ax = a.x - static_cast<float>(ofx), ay = a.y - static_cast<float>(ofy);
                const float bx = b.x - static_cast<float>(ofx), by = b.y - static_cast<float>(ofy);
                // One pixel thick across the major axis.
                const bool horizontal = std::fabs(bx - ax) >= std::fabs(by - ay);
                const float ox = horizontal ? 0.0f : 0.5f, oy = horizontal ? 0.5f : 0.0f;
                float as, at, aq, bs, bt, bq;
                texOf(a, as, at, aq);
                texOf(b, bs, bt, bq);
                const GSVertex &ca = state.prim.iip ? a : b;
                const float fa = static_cast<float>(a.fog) / 255.0f, fb = static_cast<float>(b.fog) / 255.0f;
                put(ax - ox + 0.5f, ay - oy + 0.5f, depthOf(a.z), as, at, aq, ca, fa);
                put(bx - ox + 0.5f, by - oy + 0.5f, depthOf(b.z), bs, bt, bq, b, fb);
                put(ax + ox + 0.5f, ay + oy + 0.5f, depthOf(a.z), as, at, aq, ca, fa);
                put(bx - ox + 0.5f, by - oy + 0.5f, depthOf(b.z), bs, bt, bq, b, fb);
                put(bx + ox + 0.5f, by + oy + 0.5f, depthOf(b.z), bs, bt, bq, b, fb);
                put(ax + ox + 0.5f, ay + oy + 0.5f, depthOf(a.z), as, at, aq, ca, fa);
                minX = std::min(ax, bx);
                minY = std::min(ay, by);
                maxX = std::max(ax, bx);
                maxY = std::max(ay, by);
                break;
            }
            case GS_PRIM_POINT:
            default:
            {
                const GSVertex &v = batch.vertices[0];
                const float x = static_cast<float>(static_cast<int>(v.x) - ofx);
                const float y = static_cast<float>(static_cast<int>(v.y) - ofy);
                float s, t, q;
                texOf(v, s, t, q);
                const float qq = std::fabs(q) > 1.0e-8f ? q : 1.0f;
                quad(x, y, x + 1.0f, y + 1.0f, s / qq, t / qq, s / qq, t / qq, depthOf(v.z), v,
                     static_cast<float>(v.fog) / 255.0f);
                minX = maxX = x;
                minY = maxY = y;
                break;
            }
            }

            // Nothing inside the scissor: nothing to draw.
            const GSScissorReg &sc = ctx.scissor;
            if (maxX < static_cast<float>(sc.x0) || minX > static_cast<float>(sc.x1) + 1.0f ||
                maxY < static_cast<float>(sc.y0) || minY > static_cast<float>(sc.y1) + 1.0f)
                return;
            const uint32_t rowsTouched = static_cast<uint32_t>(
                std::clamp(std::min(std::ceil(maxY), static_cast<float>(sc.y1)) + 1.0f, 1.0f, 2048.0f));

            Target &target = colorTarget(ctx.frame, rowsTouched);
            const uint32_t targetId = target.id;
            Target *depth = depthUsed ? &depthTarget(ctx.zbuf, target.fbw, target, rowsTouched) : nullptr;
            const uint32_t depthId = depth ? depth->id : 0u;
            if (depth)
                depth->gpuDirty = depth->gpuDirty || !ctx.zbuf.zmask;

            DrawKey key;
            std::memset(&key, 0, sizeof(key));
            key.atst = -1;
            if (tme)
            {
                // Copying a target back hands the batch over: look the targets up again after.
                Target *self = nullptr;
                for (Target &t : m_targets)
                    if (t.id == targetId)
                        self = &t;
                if (self)
                    bindTexture(state, *self, key);
            }
            Target *drawTarget = nullptr;
            for (Target &t : m_targets)
                if (t.id == targetId)
                    drawTarget = &t;
            if (!drawTarget)
                return;
            key.target = targetId;
            key.depth = depthId;
            bool depthStillThere = depthId == 0u;
            for (const Target &t : m_targets)
                if (t.id == depthId)
                    depthStillThere = true;
            if (!depthStillThere)
                key.depth = 0u;
            key.targetWidth = static_cast<uint16_t>(drawTarget->width);
            key.targetHeight = static_cast<uint16_t>(drawTarget->height);
            key.scissorX0 = static_cast<uint16_t>(std::min<uint32_t>(sc.x0, drawTarget->width - 1u));
            key.scissorY0 = static_cast<uint16_t>(std::min<uint32_t>(sc.y0, drawTarget->height - 1u));
            key.scissorX1 = static_cast<uint16_t>(std::clamp<uint32_t>(sc.x1, key.scissorX0, drawTarget->width - 1u));
            key.scissorY1 = static_cast<uint16_t>(std::clamp<uint32_t>(sc.y1, key.scissorY0, drawTarget->height - 1u));
            key.fge = state.prim.fge ? 1u : 0u;
            key.fogR = state.fogR;
            key.fogG = state.fogG;
            key.fogB = state.fogB;
            if (ate && atst != 1u)
            {
                key.atst = static_cast<int8_t>(atst);
                key.aref = static_cast<uint8_t>((test >> 4) & 0xFFu);
                key.afail = static_cast<uint8_t>(afail);
                if (afail == 3u && ctx.frame.psm != GS_PSM_CT32)
                    key.afail = 1u;
            }
            key.frame16 = is16Bit(ctx.frame.psm) ? 1u : 0u;
            key.depthFunc = !zte ? 0u : static_cast<uint8_t>(ztst);
            key.depthWrite = (zte && !ctx.zbuf.zmask && key.depth != 0u) ? 1u : 0u;
            if (key.depth == 0u)
                key.depthFunc = 0u;

            if (state.prim.abe)
            {
                const BlendSetup blend = mapBlend(ctx.alpha);
                key.blend = blend.enable ? 1u : 0u;
                key.blendEquation = blend.equation;
                key.blendSrc = blend.src;
                key.blendDst = blend.dst;
                key.blendFix = static_cast<uint8_t>((ctx.alpha >> 32) & 0xFFu);
                key.keepBlendAlpha = (blend.enable && blend.usesSourceAlpha) ? 1u : 0u;
                if (state.pabe)
                    noteOnce("PABE (per-pixel blend enable) drawn as plain blending");
                if (((ctx.alpha >> 32) & 0xFFu) > 0x80u && ((ctx.alpha >> 4) & 3u) == 2u)
                    noteOnce("blend FIX above 0x80 clamped to 1.0");
            }
            const bool fba = (ctx.fba & 1u) != 0u && ctx.frame.psm != GS_PSM_CT24;
            key.forceAlpha = (fba && !key.keepBlendAlpha) ? 1u : 0u;
            if (fba && key.keepBlendAlpha)
                noteOnce("FBA with source alpha blending: alpha MSB not forced");

            // FBMSK, a channel at a time.
            uint8_t mask = 0xFu;
            const uint32_t fbmsk = ctx.frame.fbmsk;
            const bool sixteen = key.frame16 != 0u;
            const uint32_t channelBits[4] = {sixteen ? 0xF8u : 0xFFu, sixteen ? 0xF800u : 0xFF00u,
                                             sixteen ? 0xF80000u : 0xFF0000u, sixteen ? 0x80000000u : 0xFF000000u};
            for (uint32_t i = 0; i < 4u; ++i)
            {
                const uint32_t bits = fbmsk & channelBits[i];
                if (bits == channelBits[i])
                    mask &= static_cast<uint8_t>(~(1u << i));
                else if (bits != 0u)
                    noteOnce("partial FBMSK drawn unmasked");
            }
            if (ctx.frame.psm == GS_PSM_CT24)
                mask &= 7u;
            key.colorMask = mask;
            if (mask != 0u)
                drawTarget->gpuDirty = true;

            const uint32_t first = static_cast<uint32_t>(m_batch.vertices.size());
            m_batch.vertices.insert(m_batch.vertices.end(), out, out + count);
            ++m_stats.vertices;
            if (m_openDraw >= 0 && static_cast<size_t>(m_openDraw) + 1u == m_batch.commands.size() &&
                std::memcmp(&m_batch.commands[static_cast<size_t>(m_openDraw)].key, &key, sizeof(key)) == 0)
            {
                m_batch.commands[static_cast<size_t>(m_openDraw)].count += count;
                return;
            }
            if (frameLogging())
                std::fprintf(stderr,
                             "[framelog] draw prim %u fbp %u fbw %u psm %u msk %08x zbp %u ztst %u tme %u tbp %u tbw %u tpsm %u %ux%u src %u "
                             "abe %u alpha %llx test %llx scissor %u,%u-%u,%u bounds %.0f,%.0f-%.0f,%.0f\n",
                             state.prim.type, ctx.frame.fbp, ctx.frame.fbw, ctx.frame.psm, ctx.frame.fbmsk, ctx.zbuf.zbp,
                             zte ? ztst : 9u, tme ? 1u : 0u, ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, state.textureWidth,
                             state.textureHeight, key.source, state.prim.abe ? 1u : 0u,
                             static_cast<unsigned long long>(ctx.alpha), static_cast<unsigned long long>(test), sc.x0, sc.y0,
                             sc.x1, sc.y1, minX, minY, maxX, maxY);
            Command &command = push(Command::Kind::Draw);
            command.key = key;
            command.first = first;
            command.count = count;
            m_openDraw = static_cast<int>(m_batch.commands.size() - 1u);
            ++m_stats.draws;
        }

        void GlesBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            if (frameLogging())
                std::fprintf(stderr, "[framelog] clut cbp %u cpsm %u csm %u csa %u cld %u tex psm %u tbp %u\n", tex0.cbp,
                             tex0.cpsm, tex0.csm, tex0.csa, tex0.cld, tex0.psm, tex0.tbp0);
            const bool indexed = tex0.psm == GS_PSM_T8 || tex0.psm == GS_PSM_T4 || tex0.psm == GS_PSM_T8H ||
                                 tex0.psm == GS_PSM_T4HL || tex0.psm == GS_PSM_T4HH;
            if (tex0.cld != 0u && indexed)
            {
                m_pendingClut = {};
                const PageRange pages = clutPages(tex0, texclut);
                for (const Target &target : m_targets)
                    if (target.gpuDirty && !target.depth && target.pages().overlaps(pages))
                        m_pendingClut = {true, tex0, texclut};
            }
            m_cpu.LoadClut(tex0, texclut);
        }

        void GlesBackend::BeginTransfer(const GSTransferCommand &command)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            if (frameLogging())
                std::fprintf(stderr, "[framelog] transfer dir %u sbp %u sbw %u spsm %u -> dbp %u dbw %u dpsm %u src %u,%u dst %u,%u %ux%u\n",
                             command.direction, command.bitbltbuf.sbp, command.bitbltbuf.sbw, command.bitbltbuf.spsm,
                             command.bitbltbuf.dbp, command.bitbltbuf.dbw, command.bitbltbuf.dpsm, command.trxpos.ssax,
                             command.trxpos.ssay, command.trxpos.dsax, command.trxpos.dsay, command.trxreg.rrw,
                             command.trxreg.rrh);
            const GSBitBltBuf &buf = command.bitbltbuf;
            const GSTrxPos &pos = command.trxpos;
            const GSTrxReg &reg = command.trxreg;
            finishDirectUpload();
            if (command.direction == 0u && reg.rrw != 0u && reg.rrh != 0u)
            {
                for (const Target &target : m_targets)
                {
                    if (target.depth || target.fbp * 32u != buf.dbp || target.fbw != buf.dbw || target.psm != buf.dpsm ||
                        pos.dsax + reg.rrw > target.width || pos.dsay + reg.rrh > target.height)
                        continue;
                    // Other targets over that memory are still copied back below; this one stays.
                    m_directUpload = {true, target.id, pos.dsax, pos.dsay, reg.rrw, reg.rrh};
                    break;
                }
            }
            // Local memory the transfer reads or writes holds what the GPU drew there first.
            if (command.direction == 0u || command.direction == 2u)
            {
                const PageRange dst = bufferPages(buf.dbp, buf.dbw, buf.dpsm, pos.dsax + reg.rrw, pos.dsay + reg.rrh);
                if (m_directUpload.active)
                {
                    for (Target &target : m_targets)
                        if (target.id != m_directUpload.target && target.gpuDirty && !target.depth &&
                            target.pages().overlaps(dst))
                            writeBack(target);
                    noteCpuWrite();
                    m_cpu.BeginTransfer(command);
                    return;
                }
                writeBackOverlapping(dst, traceEnabled() ? "transfer " + std::to_string(command.direction) + " to dbp " +
                                                                std::to_string(buf.dbp) + " dbw " + std::to_string(buf.dbw) +
                                                                " dpsm " + std::to_string(buf.dpsm) + " at " +
                                                                std::to_string(pos.dsax) + "," + std::to_string(pos.dsay) +
                                                                " " + std::to_string(reg.rrw) + "x" + std::to_string(reg.rrh)
                                                          : std::string());
                noteCpuWrite();
            }
            if (command.direction == 1u || command.direction == 2u)
                writeBackOverlapping(bufferPages(buf.sbp, buf.sbw, buf.spsm, pos.ssax + reg.rrw, pos.ssay + reg.rrh),
                                     "transfer " + std::to_string(command.direction) + " from sbp " + std::to_string(buf.sbp));
            m_cpu.BeginTransfer(command);
            if (command.direction == 2u)
                noteCpuWrite();
        }

        void GlesBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            m_cpu.UploadImage(data, sizeBytes);
            noteCpuWrite();
            if (m_directUpload.active)
            {
                const GSTransferSnapshot transfer = m_cpu.GetTransferSnapshot();
                if (transfer.copiedPixels >= transfer.totalPixels)
                    finishDirectUpload();
            }
        }

        // The transferred rectangle, from local memory to the GPU frame buffer.
        void GlesBackend::finishDirectUpload()
        {
            if (!m_directUpload.active)
                return;
            const DirectUpload upload = m_directUpload;
            m_directUpload = {};
            Target *target = targetById(upload.target);
            if (!target)
                return;
            m_cpu.ReadRectRaw(target->psm, target->fbp * 32u, target->fbw, upload.x, upload.y, upload.width,
                              upload.height, m_scratch);
            Command &command = push(Command::Kind::UploadTarget);
            command.id = target->id;
            command.x = upload.x;
            command.y = upload.y;
            command.width = upload.width;
            command.height = upload.height;
            command.blob = m_batch.blob.size();
            for (const uint32_t raw : m_scratch)
                m_batch.blob.push_back(toStored(target->psm, raw));
            // A target the GPU had not drawn since its last sync matches local memory again.
            if (!target->gpuDirty)
            {
                target->syncEpoch = m_cpu.NewTextureEpoch();
                target->checkedWrites = m_cpuWrites;
            }
            ++m_stats.directUploads;
        }

        void GlesBackend::Flush()
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            if (m_batch.vertices.size() >= kHandOverVertices || m_batch.blob.size() >= (1u << 20))
                handOver(false);
        }

        void GlesBackend::TextureFlush()
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            m_cpu.TextureFlush();
        }

        void GlesBackend::Sync(GSSyncReason reason)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            if (reason == GSSyncReason::DebugReadback || reason == GSSyncReason::LocalToHost)
                writeBackAll();
            m_cpu.Sync(reason);
        }

        PresentationFrame GlesBackend::Present(const GSPresentationRequest &request)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            PresentParams params{};
            const uint64_t pmode = request.pmode;
            const uint64_t dispfb[2] = {request.dispfb1, request.dispfb2};
            const uint64_t display[2] = {request.display1, request.display2};
            const bool enabled[2] = {(pmode & 1u) != 0u, (pmode & 2u) != 0u};
            for (int i = 0; i < 2; ++i)
            {
                GSFrameReg frame{};
                frame.fbp = static_cast<uint32_t>(dispfb[i] & 0x1FFu);
                frame.fbw = static_cast<uint32_t>((dispfb[i] >> 9) & 0x3Fu);
                frame.psm = static_cast<uint8_t>((dispfb[i] >> 15) & 0x1Fu);
                const uint32_t dw = static_cast<uint32_t>((display[i] >> 32) & 0x0FFFu);
                const uint32_t dh = static_cast<uint32_t>((display[i] >> 44) & 0x07FFu);
                const uint32_t magh = static_cast<uint32_t>((display[i] >> 23) & 0x0Fu);
                if (!enabled[i] || (frame.fbw == 0u && dw == 0u && dh == 0u && magh == 0u))
                    continue;
                // Sized as the CPU backend sizes it.
                uint32_t width = (dw + 1u) / (magh + 1u);
                uint32_t height = dh + 1u;
                if (width < 64u || height < 64u)
                {
                    width = 640u;
                    height = 448u;
                }
                width = std::min<uint32_t>(width, 640u);
                height = std::min<uint32_t>(height, 512u);
                params.enable[i] = true;
                params.originX[i] = static_cast<int>((dispfb[i] >> 32) & 0x7FFu);
                params.originY[i] = static_cast<int>((dispfb[i] >> 43) & 0x7FFu);
                params.width[i] = width;
                params.height[i] = height;
                Target &target = colorTarget(frame, static_cast<uint32_t>(params.originY[i]) + height);
                params.target[i] = target.id;
            }
            if (!params.enable[0] && !params.enable[1])
                return {};
            // colorTarget() may have dropped the first circuit's target for the second's.
            for (int i = 0; i < 2; ++i)
            {
                bool found = !params.enable[i];
                for (const Target &target : m_targets)
                    if (target.id == params.target[i])
                        found = true;
                if (!found)
                    params.enable[i] = false;
            }
            params.slbg = ((pmode >> 7) & 1u) != 0u;
            params.mmod = ((pmode >> 5) & 1u) != 0u;
            params.alp = static_cast<uint8_t>((pmode >> 8) & 0xFFu);
            params.bgcolor = static_cast<uint32_t>(request.bgcolor);
            const bool interlaced = (request.smode2 & 1u) != 0u;
            const bool frameMode = ((request.smode2 >> 1) & 1u) != 0u;
            params.field = interlaced && !frameMode;
            params.oddField = (request.vsyncTick & 1u) != 0u;
            params.outWidth = std::max(params.enable[0] ? params.width[0] : 0u, params.enable[1] ? params.width[1] : 0u);
            params.outHeight = std::max(params.enable[0] ? params.height[0] : 0u, params.enable[1] ? params.height[1] : 0u);
            if (params.outWidth == 0u || params.outHeight == 0u)
                return {};
            push(Command::Kind::Present).present = params;
            if (frameLogging())
                std::fprintf(stderr, "[framelog] present fbp %u/%u\n", params.target[0], params.target[1]);
            g_swaps.fetch_add(1u, std::memory_order_relaxed);
            ++m_batch.presents;
            ++m_stats.presents;
            handOver(false);
            trace();
            return {};
        }

        void GlesBackend::trace()
        {
            if (!traceEnabled() || m_stats.presents % 300u != 0u)
                return;
            std::fprintf(stderr,
                         "[gles] %llu swaps: %llu draw calls, %llu primitives, %llu texture uploads (%.1f Mtexels), "
                         "%llu copies back (%.1f ms), %llu direct uploads, %llu reloads, %llu targets made, %zu live, "
                         "%zu textures\n",
                         static_cast<unsigned long long>(m_stats.presents), static_cast<unsigned long long>(m_stats.draws),
                         static_cast<unsigned long long>(m_stats.vertices), static_cast<unsigned long long>(m_stats.uploads),
                         static_cast<double>(m_stats.uploadTexels) / 1.0e6, static_cast<unsigned long long>(m_stats.writeBacks),
                         static_cast<double>(m_stats.writeBackNs) / 1.0e6,
                         static_cast<unsigned long long>(m_stats.directUploads),
                         static_cast<unsigned long long>(m_stats.reloads),
                         static_cast<unsigned long long>(m_stats.targetsCreated), m_targets.size(), m_textures.size());
            const uint64_t presents = m_stats.presents;
            m_stats = {};
            m_stats.presents = presents;
        }

        bool GlesBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
        {
            if (context.frame.fbw == 0u)
                return false;
            // A sprite over the scissor, the way the CPU backend fills it.
            GSPrimitiveBatch clear{};
            clear.vertexCount = 2u;
            clear.state.context = context;
            clear.state.context.test = 0u;
            clear.state.prim.type = GS_PRIM_SPRITE;
            const float ofx = static_cast<float>(context.xyoffset.ofx >> 4);
            const float ofy = static_cast<float>(context.xyoffset.ofy >> 4);
            for (int i = 0; i < 2; ++i)
            {
                GSVertex &v = clear.vertices[static_cast<size_t>(i)];
                v.x = (i == 0 ? static_cast<float>(context.scissor.x0) : static_cast<float>(context.scissor.x1) + 1.0f) + ofx;
                v.y = (i == 0 ? static_cast<float>(context.scissor.y0) : static_cast<float>(context.scissor.y1) + 1.0f) + ofy;
                v.r = static_cast<uint8_t>(rgba);
                v.g = static_cast<uint8_t>(rgba >> 8);
                v.b = static_cast<uint8_t>(rgba >> 16);
                v.a = static_cast<uint8_t>(rgba >> 24);
            }
            Submit(clear);
            return true;
        }

        uint32_t GlesBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            return m_cpu.ConsumeLocalToHostBytes(dst, maxBytes);
        }

        uint32_t GlesBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto *self = const_cast<GlesBackend *>(this);
            self->writeBackOverlapping(bufferPages(base, bw, static_cast<uint8_t>(psm), x + 1u, y + 1u), "ReadVram");
            return m_cpu.ReadVram(psm, base, bw, x, y);
        }

        void GlesBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            writeBackOverlapping(bufferPages(base, bw, static_cast<uint8_t>(psm), x + 1u, y + 1u), "WriteVram");
            m_cpu.WriteVram(psm, base, bw, x, y, value);
            noteCpuWrite();
        }

        void GlesBackend::SnapshotVram(std::vector<uint8_t> &out) const
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            const_cast<GlesBackend *>(this)->writeBackAll();
            m_cpu.SnapshotVram(out);
        }

        GSTransferSnapshot GlesBackend::GetTransferSnapshot() const
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            return m_cpu.GetTransferSnapshot();
        }
    }

    bool requested()
    {
        static const bool s_requested = []()
        {
            const char *value = std::getenv("PS2X_RENDERER");
            return value && std::strcmp(value, "opengl") == 0;
        }();
        return s_requested;
    }

    std::unique_ptr<GSRasterBackend> createBackend()
    {
        return std::make_unique<GlesBackend>();
    }

    bool attachToWindow()
    {
        Channel &ch = channel();
        if (g_active.load())
            return true;
        if (!createContext())
        {
            std::fprintf(stderr, "[gles] no GL context could be shared with the window\n");
            std::lock_guard<std::mutex> lock(ch.mutex);
            ch.stop = true;
            return false;
        }
        ch.thread = std::thread(glThreadMain);
        std::unique_lock<std::mutex> lock(ch.mutex);
        ch.doneCv.wait(lock, [&]() { return ch.running || ch.stop; });
        if (!ch.running)
        {
            lock.unlock();
            ch.thread.join();
            return false;
        }
        g_active.store(true);
        std::fprintf(stderr, "[gles] OpenGL renderer started\n");
        return true;
    }

    bool active()
    {
        return g_active.load();
    }

    bool displayTexture(unsigned &textureId, uint32_t &width, uint32_t &height)
    {
        std::lock_guard<std::mutex> lock(g_displayMutex);
        if (g_latestSlot >= 0)
        {
            g_shownSlot = g_latestSlot;
            g_latestSlot = -1;
        }
        if (g_shownSlot < 0)
            return false;
        const DisplaySlot &slot = g_slots[static_cast<size_t>(g_shownSlot)];
        textureId = slot.texture;
        width = slot.width;
        height = slot.height;
        return slot.texture != 0u;
    }

    void shutdown()
    {
        Channel &ch = channel();
        {
            std::lock_guard<std::mutex> lock(ch.mutex);
            ch.stop = true;
        }
        ch.workCv.notify_all();
        ch.doneCv.notify_all();
        if (ch.thread.joinable())
            ch.thread.join();
        g_active.store(false);
    }
}

#endif
