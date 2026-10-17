// Faster exception unwinding on Android.
//
// The EE scheduler leaves guest code by throwing (EeDispatcherTransfer), and the NDK's libunwind
// finds each frame's unwind tables with dl_iterate_phdr: the loader walks every library of the
// process (a hundred or more in an app, most preloaded by the zygote), under its lock, and the
// unwinder scans each one's program headers for the frame's address. In a Devil Kings battle on
// an S21 FE that walk was ~6% of the game thread.
//
// This definition shadows bionic's for this library, which links libunwind statically. It walks
// a copy of the loader's list taken when the list last changed, with this library first, so a
// guest frame is found at the first entry. The loader's load and unload counters (API 30) tell
// when to take a new copy; without them the real walk is used.

#include <dlfcn.h>
#include <link.h>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

namespace
{
    using Callback = int (*)(dl_phdr_info *, size_t, void *);
    using Iterate = int (*)(Callback, void *);

    Iterate realIterate()
    {
        static const Iterate s_real = reinterpret_cast<Iterate>(dlsym(RTLD_NEXT, "dl_iterate_phdr"));
        return s_real;
    }

    struct Counters
    {
        bool known = false;
        unsigned long long adds = 0;
        unsigned long long subs = 0;
    };

    // The counters come with every entry; the first one is enough.
    int readCounters(dl_phdr_info *info, size_t size, void *data)
    {
        auto *counters = static_cast<Counters *>(data);
        if (size >= offsetof(dl_phdr_info, dlpi_subs) + sizeof(info->dlpi_subs))
            *counters = {true, info->dlpi_adds, info->dlpi_subs};
        return 1;
    }

    struct Snapshot
    {
        Counters counters;
        size_t size = sizeof(dl_phdr_info);
        std::vector<dl_phdr_info> objects;
    };

    struct Collector
    {
        Snapshot *snapshot;
    };

    int collect(dl_phdr_info *info, size_t size, void *data)
    {
        Snapshot *snapshot = static_cast<Collector *>(data)->snapshot;
        dl_phdr_info copy{};
        const size_t bytes = size < sizeof(copy) ? size : sizeof(copy);
        __builtin_memcpy(&copy, info, bytes);
        snapshot->size = bytes;
        snapshot->objects.push_back(copy);
        return 0;
    }

    // Published snapshots are never changed or freed (the list changes a few times at startup).
    std::atomic<const Snapshot *> g_snapshot{nullptr};
    std::mutex g_refreshMutex;

    const Snapshot *refresh(Iterate real, const Counters &counters)
    {
        std::lock_guard<std::mutex> lock(g_refreshMutex);
        const Snapshot *current = g_snapshot.load(std::memory_order_acquire);
        if (current && current->counters.adds == counters.adds && current->counters.subs == counters.subs)
            return current;
        auto *snapshot = new Snapshot;
        snapshot->counters = counters;
        Collector collector{snapshot};
        real(collect, &collector);
        // This library first: nearly every frame thrown through is guest code compiled into it.
        Dl_info self{};
        if (dladdr(reinterpret_cast<void *>(&refresh), &self) && self.dli_fbase)
        {
            const auto base = reinterpret_cast<ElfW(Addr)>(self.dli_fbase);
            for (size_t index = 0; index < snapshot->objects.size(); ++index)
                if (snapshot->objects[index].dlpi_addr == base && index != 0)
                {
                    const dl_phdr_info mine = snapshot->objects[index];
                    snapshot->objects.erase(snapshot->objects.begin() + static_cast<std::ptrdiff_t>(index));
                    snapshot->objects.insert(snapshot->objects.begin(), mine);
                    break;
                }
        }
        g_snapshot.store(snapshot, std::memory_order_release);
        return snapshot;
    }
}

extern "C" int dl_iterate_phdr(Callback callback, void *data)
{
    const Iterate real = realIterate();
    if (!real)
        return 0;
    Counters counters;
    real(readCounters, &counters);
    if (!counters.known)
        return real(callback, data);
    const Snapshot *snapshot = g_snapshot.load(std::memory_order_acquire);
    if (!snapshot || snapshot->counters.adds != counters.adds || snapshot->counters.subs != counters.subs)
        snapshot = refresh(real, counters);
    for (const dl_phdr_info &entry : snapshot->objects)
    {
        dl_phdr_info info = entry;
        if (const int result = callback(&info, snapshot->size, data))
            return result;
    }
    return 0;
}
