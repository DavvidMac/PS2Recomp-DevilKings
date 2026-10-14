// Replays a GS backend recording made with PS2X_GS_RECORD through GSCpuBackend.
//
// usage: ps2_gs_replay <recording> [--hash] [--repeat <n>]
//   --hash      print a hash of VRAM after every presentation (waits for the
//               rasteriser at each one, so leave it off when timing)
//   --repeat n  replay n times and report every run's time
//
// PS2X_GS_ASYNC=1 and PS2X_GS_THREADS=<n> select the threaded rasteriser, as in
// the runner. The final VRAM hash is printed in every mode, so two builds of the
// rasteriser can be compared on the same recording.

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/ps2_profiler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace
{
    enum class RecordKind : uint8_t
    {
        Submit = 1,
        LoadClut = 2,
        TextureFlush = 3,
        BeginTransfer = 4,
        UploadImage = 5,
        Present = 6,
        ClearFramebuffer = 7,
        WriteVram = 8,
        ConsumeLocalToHost = 9,
        Reset = 10,
    };

    // CPU time of the whole process in milliseconds (0 where unsupported). Unlike wall
    // time it hardly moves when other programs compete for the cores.
    double processCpuMs()
    {
#if defined(_WIN32)
        FILETIME creation, exit, kernel, user;
        if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user))
            return 0.0;
        const auto toTicks = [](const FILETIME &time)
        { return (static_cast<uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime; };
        return static_cast<double>(toTicks(kernel) + toTicks(user)) / 10000.0;
#else
        return 0.0;
#endif
    }

    uint64_t fnv1a(const uint8_t *data, size_t size)
    {
        uint64_t hash = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < size; ++i)
            hash = (hash ^ data[i]) * 0x100000001b3ull;
        return hash;
    }

    struct Reader
    {
        const std::vector<uint8_t> &bytes;
        size_t pos = 0;

        bool take(void *out, size_t size)
        {
            if (pos + size > bytes.size())
                return false;
            std::memcpy(out, bytes.data() + pos, size);
            pos += size;
            return true;
        }
    };

    bool readFile(const char *path, std::vector<uint8_t> &out)
    {
        std::FILE *file = std::fopen(path, "rb");
        if (!file)
            return false;
        std::fseek(file, 0, SEEK_END);
        const long size = std::ftell(file);
        std::fseek(file, 0, SEEK_SET);
        out.resize(size > 0 ? static_cast<size_t>(size) : 0u);
        const bool ok = out.empty() || std::fread(out.data(), 1u, out.size(), file) == out.size();
        std::fclose(file);
        return ok;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: ps2_gs_replay <recording> [--hash] [--repeat <n>]\n");
        return 2;
    }
    bool hashEveryPresent = false;
    int repeat = 1;
    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--hash") == 0)
            hashEveryPresent = true;
        else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc)
            repeat = std::max(1, std::atoi(argv[++i]));
    }

    // PS2X_PROFILE works here as in the runner.
    ps2StartSamplingProfilerFromEnv();

    std::vector<uint8_t> bytes;
    if (!readFile(argv[1], bytes))
    {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }

    Reader header{bytes};
    char magic[8] = {};
    uint32_t fields[8] = {};
    if (!header.take(magic, sizeof(magic)) || std::memcmp(magic, "PS2XGSR1", 8) != 0 ||
        !header.take(fields, sizeof(fields)) || fields[0] != 1u)
    {
        std::fprintf(stderr, "%s is not a PS2X_GS_RECORD recording\n", argv[1]);
        return 1;
    }
    const uint32_t expected[6] = {
        static_cast<uint32_t>(sizeof(GSPrimitiveBatch)),
        static_cast<uint32_t>(sizeof(GSTex0Reg)),
        static_cast<uint32_t>(sizeof(GSTexClutReg)),
        static_cast<uint32_t>(sizeof(GSTransferCommand)),
        static_cast<uint32_t>(sizeof(GSPresentationRequest)),
        static_cast<uint32_t>(sizeof(GSContext)),
    };
    if (std::memcmp(fields + 2, expected, sizeof(expected)) != 0)
    {
        std::fprintf(stderr, "recording was made with different GS structure layouts\n");
        return 1;
    }
    const uint32_t vramSize = fields[1];
    std::vector<uint8_t> initialVram(vramSize);
    std::array<uint16_t, 512> clut{};
    std::array<uint32_t, 2> clutCbp{};
    if (!header.take(initialVram.data(), vramSize) || !header.take(clut.data(), sizeof(clut)) ||
        !header.take(clutCbp.data(), sizeof(clutCbp)))
    {
        std::fprintf(stderr, "truncated recording header\n");
        return 1;
    }
    const size_t firstRecord = header.pos;

    // One backend for every run, so its worker and band threads persist (and show
    // up as single threads in a PS2X_PROFILE report); Reset clears its caches.
    std::vector<uint8_t> vram = initialVram;
    auto backend = std::make_unique<GSCpuBackend>();
    backend->Initialize(vram.data(), vramSize);
    for (int run = 0; run < repeat; ++run)
    {
        backend->Reset();
        std::memcpy(vram.data(), initialVram.data(), vramSize);
        backend->RestoreClutForReplay(clut, clutCbp);

        Reader reader{bytes, firstRecord};
        uint64_t draws = 0;
        uint32_t presents = 0;
        std::vector<uint8_t> scratch;
        std::vector<uint8_t> snapshot;
        const auto start = std::chrono::steady_clock::now();
        const double cpuStart = processCpuMs();
        bool ok = true;
        while (ok && reader.pos < bytes.size())
        {
            uint8_t tag = 0;
            reader.take(&tag, 1u);
            switch (static_cast<RecordKind>(tag))
            {
            case RecordKind::Submit:
            {
                GSPrimitiveBatch batch{};
                ok = reader.take(&batch, sizeof(batch));
                if (ok)
                {
                    backend->Submit(batch);
                    ++draws;
                }
                break;
            }
            case RecordKind::LoadClut:
            {
                GSTex0Reg tex0{};
                GSTexClutReg texclut{};
                ok = reader.take(&tex0, sizeof(tex0)) && reader.take(&texclut, sizeof(texclut));
                if (ok)
                    backend->LoadClut(tex0, texclut);
                break;
            }
            case RecordKind::TextureFlush:
                backend->TextureFlush();
                break;
            case RecordKind::BeginTransfer:
            {
                GSTransferCommand command{};
                ok = reader.take(&command, sizeof(command));
                if (ok)
                    backend->BeginTransfer(command);
                break;
            }
            case RecordKind::UploadImage:
            {
                uint32_t size = 0;
                ok = reader.take(&size, sizeof(size));
                scratch.resize(size);
                ok = ok && reader.take(scratch.data(), size);
                if (ok)
                    backend->UploadImage(size != 0u ? scratch.data() : nullptr, size);
                break;
            }
            case RecordKind::Present:
            {
                GSPresentationRequest request{};
                ok = reader.take(&request, sizeof(request));
                if (!ok)
                    break;
                backend->QueuePresent(request, [](PresentationFrame &&) {});
                ++presents;
                if (hashEveryPresent)
                {
                    backend->SnapshotVram(snapshot);
                    std::printf("present %u vram %016llx\n", presents,
                                static_cast<unsigned long long>(fnv1a(snapshot.data(), snapshot.size())));
                }
                break;
            }
            case RecordKind::ClearFramebuffer:
            {
                GSContext context{};
                uint32_t rgba = 0;
                ok = reader.take(&context, sizeof(context)) && reader.take(&rgba, sizeof(rgba));
                if (ok)
                    backend->ClearFramebuffer(context, rgba);
                break;
            }
            case RecordKind::WriteVram:
            {
                uint32_t args[6] = {};
                ok = reader.take(args, sizeof(args));
                if (ok)
                    backend->WriteVram(args[0], args[1], args[2], args[3], args[4], args[5]);
                break;
            }
            case RecordKind::ConsumeLocalToHost:
            {
                uint32_t size = 0;
                ok = reader.take(&size, sizeof(size));
                scratch.resize(size);
                if (ok)
                    backend->ConsumeLocalToHostBytes(size != 0u ? scratch.data() : nullptr, size);
                break;
            }
            case RecordKind::Reset:
                backend->Reset();
                break;
            default:
                std::fprintf(stderr, "unknown record %u at offset %zu\n", tag, reader.pos - 1u);
                ok = false;
                break;
            }
        }
        backend->SnapshotVram(snapshot);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const double cpuMs = processCpuMs() - cpuStart;
        std::printf("run %d: %llu draws, %u presents, %.1f ms (%.2f ms/present), cpu %.0f ms, final vram %016llx%s\n",
                    run + 1, static_cast<unsigned long long>(draws), presents, ms, presents ? ms / presents : 0.0, cpuMs,
                    static_cast<unsigned long long>(fnv1a(snapshot.data(), snapshot.size())),
                    ok ? "" : " (stopped at a bad record)");
        if (!ok)
            return 1;
    }
    return 0;
}
