// The GS frontend counts swapped frames for the FPS counter in ps2_settings.cpp, which needs
// raylib; the tests only need the call to link, so this keeps the same counter here.
#include "runtime/ps2_settings.h"

#include <atomic>

namespace ps2x::settings
{
    namespace
    {
        std::atomic<uint64_t> g_gameFrames{0};
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
