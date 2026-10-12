#include "emulator/core/iop_cpu.h"
#include "emulator/core/iop_kernel.h"
#include "emulator/core/iop_memory.h"
#include "emulator/imports/iop_cdvd.h"
#include "emulator/imports/iop_imports.h"
#include "emulator/imports/iop_loadcore.h"
#include "emulator/imports/iop_timrman.h"
#include "emulator/services/iop_rpc.h"
#include "ps2x/iop/iop_host.h"

#include <cstdint>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <string_view>

namespace
{
    using namespace ps2x::iop;
    using namespace ps2x::iop::detail;

    constexpr uint32_t kExportMagic = 0x41C00000u;
    constexpr int32_t kLibraryNotFound = -213;
    constexpr int32_t kIllegalLibrary = -214;

    class NullHost : public IopHost
    {
    public:
        bool readGuest(uint32_t, void *, size_t) const override { return false; }
        bool writeGuest(uint32_t, const void *, size_t) override { return false; }
        bool zeroGuest(uint32_t, size_t) override { return false; }
        bool normalizeGuestAddress(uint32_t, uint32_t &) const override { return false; }
        uint32_t allocateIopHandle(IopHandleKind) override { return 1u; }
        uint32_t allocateGuest(uint32_t, uint32_t) override { return 0u; }
        void freeGuest(uint32_t) override {}
        void audioCommand(uint32_t, uint32_t, GuestBuffer, GuestBuffer) override {}
        std::string hostPath(HostPathKind) const override { return {}; }
        std::string translateGuestPath(std::string_view path) const override { return std::string(path); }
        uint64_t openHostFile(std::string_view) override { return 0u; }
        bool hostFileSize(uint64_t, uint64_t &) const override { return false; }
        bool readHostFile(uint64_t, uint64_t, void *, size_t, size_t &) override { return false; }
        void closeHostFile(uint64_t) override {}
        int32_t memoryCard(const MemoryCardRequest &) override { return 0; }
        bool hasGuestFunction(uint32_t) const override { return false; }
        bool invokeGuestFunction(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t *) override { return false; }
        void log(LogLevel, std::string_view) override {}
    };

    class CdRootHost final : public NullHost
    {
    public:
        explicit CdRootHost(std::filesystem::path rootPath)
            : root(std::move(rootPath))
        {
        }

        std::string hostPath(HostPathKind kind) const override
        {
            return kind == HostPathKind::CdRoot ? root.string() : std::string{};
        }

    private:
        std::filesystem::path root;
    };

    class RecordingExecutor final : public IopGuestExecutor
    {
    public:
        uint32_t executeGuestFunction(uint32_t address,
                                      uint32_t a0,
                                      uint32_t,
                                      uint32_t,
                                      uint32_t,
                                      uint32_t gp) override
        {
            ++calls;
            lastAddress = address;
            lastArgument = a0;
            lastGp = gp;
            return callbackResult;
        }

        uint32_t callbackResult = 0u;
        uint32_t calls = 0u;
        uint32_t lastAddress = 0u;
        uint32_t lastArgument = 0u;
        uint32_t lastGp = 0u;
    };

    bool expect(bool condition, std::string_view message)
    {
        if (condition)
            return true;
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    bool testLoadcoreRebootLibraryMode()
    {
        IopMemory memory;
        IopImportRegistry imports(memory);
        IopLoadcore loadcore(memory, imports);

        IopCpuState cpu{};
        cpu.gpr[4] = 0u;
        cpu.gpr[5] = 2u;
        if (!expect(loadcore.dispatchImport(27u, cpu), "loadcore:27 was not handled") ||
            !expect(static_cast<int32_t>(cpu.gpr[2]) == kIllegalLibrary,
                    "loadcore:27 did not reject a null export table"))
            return false;

        constexpr uint32_t table = 0x1000u;
        memory.write32(table, kExportMagic);
        memory.write16(table + 8u, 0x0101u);
        memory.write16(table + 10u, 0x1234u);
        const char name[8] = {'t', 'e', 's', 't', 'l', 'i', 'b', '\0'};
        (void)memory.writeRam(table + 12u, name, sizeof(name));
        memory.write32(table + 20u, 0u);

        cpu = {};
        cpu.gpr[4] = table;
        cpu.gpr[5] = 2u;
        if (!expect(loadcore.dispatchImport(27u, cpu), "loadcore:27 rejected a valid export table") ||
            !expect(cpu.gpr[2] == 0u, "loadcore:27 returned an error for a valid export table") ||
            !expect(memory.read16(table + 10u) == 0x1232u,
                    "loadcore:27 did not replace only export mode bits 1 and 2"))
            return false;

        constexpr uint32_t invalidTable = 0x1100u;
        memory.write32(invalidTable, 0xDEADBEEFu);
        cpu = {};
        cpu.gpr[4] = invalidTable;
        cpu.gpr[5] = 6u;
        if (!expect(loadcore.dispatchImport(27u, cpu), "loadcore:27 did not consume an invalid-table call") ||
            !expect(static_cast<int32_t>(cpu.gpr[2]) == kLibraryNotFound,
                    "loadcore:27 returned the wrong invalid-table error"))
            return false;

        if (!expect(imports.registerExportTable(table), "test export table did not register"))
            return false;
        memory.write32(table, 0u);
        cpu = {};
        cpu.gpr[4] = table;
        cpu.gpr[5] = 6u;
        return expect(loadcore.dispatchImport(27u, cpu), "loadcore:27 rejected a registered table") &&
               expect(cpu.gpr[2] == 0u, "loadcore:27 returned an error for a registered table") &&
               expect(memory.read16(table + 10u) == 0x1236u,
                      "loadcore:27 did not update a registered table's mode");
    }

    bool pollEvent(IopKernel &kernel, int eventId, uint32_t bits, uint32_t resultAddress, int32_t expected)
    {
        IopCpuState cpu{};
        cpu.gpr[4] = static_cast<uint32_t>(eventId);
        cpu.gpr[5] = bits;
        cpu.gpr[6] = 0u; // WEF_AND
        cpu.gpr[7] = resultAddress;
        return expect(kernel.dispatchEventImport(11u, cpu), "PollEventFlag was not handled") &&
               expect(static_cast<int32_t>(cpu.gpr[2]) == expected, "PollEventFlag returned an unexpected result");
    }

    int createMailbox(IopKernel &kernel, IopMemory &memory, uint32_t attr)
    {
        constexpr uint32_t descriptor = 0x4000u;
        memory.write32(descriptor + 0u, attr);
        memory.write32(descriptor + 4u, 0u);
        IopCpuState cpu{};
        cpu.gpr[4] = descriptor;
        if (!kernel.dispatchMailboxImport(4u, cpu))
            return -1;
        return static_cast<int>(cpu.gpr[2]);
    }

    bool mailboxCall(IopKernel &kernel, uint16_t ordinal, uint32_t a0, uint32_t a1, int32_t expected, const char *what)
    {
        IopCpuState cpu{};
        cpu.gpr[4] = a0;
        cpu.gpr[5] = a1;
        return expect(kernel.dispatchMailboxImport(ordinal, cpu), what) &&
               expect(static_cast<int32_t>(cpu.gpr[2]) == expected, what);
    }

    bool testMailboxes()
    {
        IopMemory memory;
        IopKernel kernel(memory);
        kernel.reset();

        constexpr uint32_t result = 0x4100u;
        constexpr uint32_t msgA = 0x4200u;
        constexpr uint32_t msgB = 0x4210u;
        constexpr uint32_t msgC = 0x4220u;

        // FIFO mailbox: PollMbx on empty, then messages come back in send order.
        const int fifo = createMailbox(kernel, memory, 0u);
        if (!expect(fifo > 0, "CreateMbx failed") ||
            !mailboxCall(kernel, 9u, result, static_cast<uint32_t>(fifo), -424, "PollMbx on empty mailbox") ||
            !mailboxCall(kernel, 6u, static_cast<uint32_t>(fifo), msgA, 0, "SendMbx A") ||
            !mailboxCall(kernel, 6u, static_cast<uint32_t>(fifo), msgB, 0, "SendMbx B") ||
            !expect(memory.read32(msgA) == msgB, "SendMbx did not link messages") ||
            !mailboxCall(kernel, 9u, result, static_cast<uint32_t>(fifo), 0, "PollMbx A") ||
            !expect(memory.read32(result) == msgA, "FIFO mailbox returned the wrong first message") ||
            !mailboxCall(kernel, 8u, result, static_cast<uint32_t>(fifo), 0, "ReceiveMbx B") ||
            !expect(memory.read32(result) == msgB, "FIFO mailbox returned the wrong second message"))
            return false;

        // MBA_MSPRI: lower priority value is delivered first.
        const int prio = createMailbox(kernel, memory, 0x4u);
        memory.write8(msgA + 4u, 5u);
        memory.write8(msgB + 4u, 1u);
        memory.write8(msgC + 4u, 5u);
        if (!mailboxCall(kernel, 6u, static_cast<uint32_t>(prio), msgA, 0, "SendMbx prio A") ||
            !mailboxCall(kernel, 6u, static_cast<uint32_t>(prio), msgB, 0, "SendMbx prio B") ||
            !mailboxCall(kernel, 6u, static_cast<uint32_t>(prio), msgC, 0, "SendMbx prio C") ||
            !mailboxCall(kernel, 9u, result, static_cast<uint32_t>(prio), 0, "PollMbx prio 1") ||
            !expect(memory.read32(result) == msgB, "priority mailbox did not deliver the highest priority first") ||
            !mailboxCall(kernel, 9u, result, static_cast<uint32_t>(prio), 0, "PollMbx prio 2") ||
            !expect(memory.read32(result) == msgA, "priority mailbox broke FIFO order for equal priorities"))
            return false;

        // A thread blocked in ReceiveMbx is handed the next message directly.
        constexpr uint32_t threadDescriptor = 0x4300u;
        memory.write32(threadDescriptor + 0u, 0u);
        memory.write32(threadDescriptor + 4u, 0u);
        memory.write32(threadDescriptor + 8u, 0x1000u);
        memory.write32(threadDescriptor + 12u, 0x400u);
        memory.write32(threadDescriptor + 16u, 10u);
        IopCpuState cpu{};
        cpu.gpr[4] = threadDescriptor;
        if (!expect(kernel.dispatchThreadImport(4u, cpu, 0u), "CreateThread was not handled"))
            return false;
        const uint32_t threadId = cpu.gpr[2];
        cpu = {};
        cpu.gpr[4] = threadId;
        if (!expect(kernel.dispatchThreadImport(6u, cpu, 0u), "StartThread was not handled"))
            return false;
        IopThread *thread = kernel.beginNextReady(0u);
        if (!expect(thread != nullptr, "started thread was not scheduled"))
            return false;

        thread->cpu.gpr[4] = result;
        thread->cpu.gpr[5] = static_cast<uint32_t>(fifo);
        if (!expect(kernel.dispatchMailboxImport(8u, thread->cpu), "blocking ReceiveMbx was not handled") ||
            !expect(thread->cpu.yielded, "ReceiveMbx on empty mailbox did not block") ||
            !expect(thread->state == IopThreadState::Mailbox, "blocked thread is not waiting on the mailbox"))
            return false;
        kernel.endTimeslice(*thread, 0xFFFFFFFFu);

        memory.write32(result, 0u);
        return mailboxCall(kernel, 7u, static_cast<uint32_t>(fifo), msgC, 0, "iSendMbx to waiter") &&
               expect(memory.read32(result) == msgC, "waiting thread did not receive the message") &&
               expect(thread->state == IopThreadState::Ready, "waiting thread was not made ready") &&
               mailboxCall(kernel, 9u, result, static_cast<uint32_t>(fifo), -424, "message was also queued");
    }

    bool testCdvdSpecialControl()
    {
        NullHost host;
        IopMemory memory;
        IopKernel kernel(memory);
        kernel.reset();
        IopCdvd cdvd(host, memory, kernel);
        cdvd.reset();

        constexpr uint32_t param = 0x2000u;
        constexpr uint32_t eventResult = 0x2010u;
        IopCpuState cpu{};
        cpu.gpr[4] = static_cast<uint32_t>(-11); // sceCdSC: return cdvdman interrupt event flag
        cpu.gpr[5] = param;
        if (!expect(cdvd.dispatchImport(50u, cpu), "cdvdman:50 was not handled") ||
            !expect(static_cast<int32_t>(cpu.gpr[2]) > 0, "sceCdSC(-11) did not return a valid event flag"))
            return false;
        const int eventId = static_cast<int>(cpu.gpr[2]);

        if (!pollEvent(kernel, eventId, 0x29u, eventResult, 0) ||
            !expect(memory.read32(eventResult) == 0x29u, "cdvdman event flag did not start with bits 0x29"))
            return false;

        IopCpuState clear{};
        clear.gpr[4] = static_cast<uint32_t>(eventId);
        clear.gpr[5] = ~0x29u;
        if (!expect(kernel.dispatchEventImport(8u, clear), "ClearEventFlag was not handled") ||
            !pollEvent(kernel, eventId, 0x29u, eventResult, -418))
            return false;

        cpu = {};
        cpu.gpr[4] = 0x12345u;
        if (!expect(cdvd.dispatchImport(7u, cpu), "sceCdSeek was not handled") ||
            !pollEvent(kernel, eventId, 0x29u, eventResult, 0))
            return false;

        memory.write8(param, 0x30u);
        cpu = {};
        cpu.gpr[4] = static_cast<uint32_t>(-2);
        cpu.gpr[5] = param;
        if (!expect(cdvd.dispatchImport(50u, cpu), "sceCdSC(-2) was not handled") ||
            !expect(cpu.gpr[2] == 0x30u, "sceCdSC(-2) did not store the low-byte error"))
            return false;

        memory.write32(param, 0u);
        cpu = {};
        cpu.gpr[4] = static_cast<uint32_t>(-1);
        cpu.gpr[5] = param;
        if (!expect(cdvd.dispatchImport(50u, cpu), "sceCdSC(-1) was not handled") ||
            !expect(cpu.gpr[2] == 0u, "sceCdSC(-1) returned the wrong initial stream state") ||
            !expect(memory.read32(param) == 0x30u, "sceCdSC(-1) did not publish the last error"))
            return false;

        cpu = {};
        cpu.gpr[4] = 2u;
        cpu.gpr[5] = param;
        if (!expect(cdvd.dispatchImport(50u, cpu), "sceCdSC(2) was not handled") ||
            !expect(cpu.gpr[2] == 2u, "sceCdSC(2) did not update the stream state"))
            return false;

        cpu = {};
        cpu.gpr[4] = static_cast<uint32_t>(-1);
        cpu.gpr[5] = param;
        return expect(cdvd.dispatchImport(50u, cpu), "second sceCdSC(-1) was not handled") &&
               expect(cpu.gpr[2] == 2u, "sceCdSC(-1) did not preserve the stream state");
    }

    bool testCdvdSearchFile()
    {
        const auto suffix = std::to_string(
            static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count()));
        const std::filesystem::path root =
            std::filesystem::temp_directory_path() / ("ps2x-iop-cdvd-search-" + suffix);
        const std::filesystem::path movieDirectory = root / "MOVIE";
        const std::filesystem::path moviePath = movieDirectory / "OPENING.PSS";
        std::error_code error;
        std::filesystem::create_directories(movieDirectory, error);
        if (!expect(!error, "could not create the temporary CD root"))
            return false;
        {
            std::ofstream movie(moviePath, std::ios::binary);
            movie.write("PSS!", 4);
        }

        CdRootHost host(root);
        IopMemory memory;
        IopKernel kernel(memory);
        kernel.reset();
        IopCdvd cdvd(host, memory, kernel);
        cdvd.reset();

        constexpr uint32_t resultAddress = 0x2400u;
        constexpr uint32_t pathAddress = 0x2480u;
        const char path[] = "cdrom0:\\movie\\opening.pss;1";
        (void)memory.writeRam(pathAddress, path, sizeof(path));

        IopCpuState cpu{};
        cpu.gpr[4] = resultAddress;
        cpu.gpr[5] = pathAddress;
        const bool handled = cdvd.dispatchImport(10u, cpu);
        const bool passed =
            expect(handled, "cdvdman:10 was not handled") &&
            expect(cpu.gpr[2] == 1u, "sceCdSearchFile did not find a case-insensitive ISO path") &&
            expect(memory.read32(resultAddress) >= 20u, "sceCdSearchFile returned an invalid LSN") &&
            expect(memory.read32(resultAddress + 4u) == 4u, "sceCdSearchFile returned the wrong size") &&
            expect(memory.readString(resultAddress + 8u, 16u) == "OPENING.PSS",
                   "sceCdSearchFile returned the wrong file name");

        std::filesystem::remove_all(root, error);
        return passed;
    }

    bool testTimrmanPeriodicCallback()
    {
        IopTimrman timrman;
        timrman.reset();
        IopCpuState cpu{};

        cpu.gpr[4] = 1u; // SYSCLK
        cpu.gpr[5] = 32u;
        cpu.gpr[6] = 1u;
        if (!expect(timrman.dispatchImport(4u, cpu, 100u), "AllocHardTimer was not handled") ||
            !expect(static_cast<int32_t>(cpu.gpr[2]) > 0, "AllocHardTimer did not allocate a 32-bit timer"))
            return false;
        const uint32_t timerId = cpu.gpr[2];

        cpu = {};
        cpu.gpr[4] = timerId;
        cpu.gpr[5] = 100u;
        cpu.gpr[6] = 0x12340u;
        cpu.gpr[7] = 0x45670u;
        cpu.gpr[28] = 0x89AB0u;
        if (!expect(timrman.dispatchImport(20u, cpu, 100u), "SetTimerHandler was not handled") ||
            !expect(cpu.gpr[2] == 0u, "SetTimerHandler failed"))
            return false;

        cpu = {};
        cpu.gpr[4] = timerId;
        cpu.gpr[5] = 1u;
        cpu.gpr[6] = 0u;
        cpu.gpr[7] = 1u;
        if (!expect(timrman.dispatchImport(22u, cpu, 100u), "SetupHardTimer was not handled") ||
            !expect(cpu.gpr[2] == 0u, "SetupHardTimer failed"))
            return false;

        cpu = {};
        cpu.gpr[4] = timerId;
        if (!expect(timrman.dispatchImport(23u, cpu, 100u), "StartHardTimer was not handled") ||
            !expect(cpu.gpr[2] == 0u, "StartHardTimer failed") ||
            !expect(timrman.nextEventCycle(1000u) == 200u, "timer compare was scheduled at the wrong cycle"))
            return false;

        RecordingExecutor executor;
        executor.callbackResult = 100u;
        timrman.serviceDue(199u, executor);
        if (!expect(executor.calls == 0u, "timer callback ran too early"))
            return false;
        timrman.serviceDue(200u, executor);
        return expect(executor.calls == 1u, "timer callback did not run") &&
               expect(executor.lastAddress == 0x12340u, "timer called the wrong handler") &&
               expect(executor.lastArgument == 0x45670u, "timer passed the wrong common argument") &&
               expect(executor.lastGp == 0x89AB0u, "timer callback lost the registering module GP") &&
               expect(timrman.nextEventCycle(1000u) == 300u, "timer callback return did not rearm compare");
    }

    bool testSpu2VoiceAndAutoDma()
    {
        {
            IopMemory memory;
            uint64_t cycles = 0u;
            memory.setCycleCounter(&cycles);
            // One looping ADPCM block (shift 0, filter 0, nibbles 7) at halfword 0x2000.
            memory.write16(0x1F9001A8u, 0u);
            memory.write16(0x1F9001AAu, 0x2000u);
            memory.write16(0x1F9001ACu, 0x0600u);
            for (int i = 0; i < 7; ++i)
                memory.write16(0x1F9001ACu, 0x7777u);
            memory.write16(0x1F9001C0u, 0u);
            memory.write16(0x1F9001C2u, 0x2000u);
            memory.write16(0x1F900004u, 0x1000u); // pitch: 48 kHz
            memory.write16(0x1F900006u, 0x000Fu); // fastest attack
            memory.write16(0x1F900008u, 0u);
            memory.write16(0x1F900000u, 0x3FFFu);
            memory.write16(0x1F900002u, 0x3FFFu);
            memory.write16(0x1F900188u, 1u);
            memory.write16(0x1F900190u, 1u);
            memory.write16(0x1F9001A0u, 1u); // key on voice 0
            cycles = 768u * 64u;
            memory.spu2().advanceTo(cycles);
            int16_t frames[64 * 2] = {};
            const size_t count = memory.spu2().readOutput(frames, 64u);
            bool audible = false;
            for (size_t i = 0; i < count; ++i)
                audible = audible || frames[i * 2u] > 1000;
            if (count != 64u || !audible || memory.read16(0x1F90000Au) == 0u)
            {
                std::cerr << "SPU2 voice should play a keyed-on ADPCM block\n";
                return false;
            }
        }
        {
            IopMemory memory;
            uint64_t cycles = 0u;
            memory.setCycleCounter(&cycles);
            memory.spu2().advanceTo(0u);
            int16_t block[512] = {};
            for (int i = 0; i < 256; ++i)
            {
                block[i] = 1000;
                block[256 + i] = -1000;
            }
            memory.spu2().dmaWrite(0u, reinterpret_cast<const uint8_t *>(block), sizeof(block), true);
            cycles = 768u * 4u;
            memory.spu2().advanceTo(cycles);
            int16_t frames[8] = {};
            if (memory.spu2().readOutput(frames, 4u) != 4u || frames[0] < 900 || frames[1] > -900)
            {
                std::cerr << "SPU2 AutoDMA input should reach the output\n";
                return false;
            }
        }
        return true;
    }
}

int main()
{
    if (!testLoadcoreRebootLibraryMode() || !testCdvdSpecialControl() || !testCdvdSearchFile() ||
        !testTimrmanPeriodicCallback() || !testMailboxes() || !testSpu2VoiceAndAutoDma())
        return 1;
    std::cout << "ps2xIOP import tests passed\n";
    return 0;
}
