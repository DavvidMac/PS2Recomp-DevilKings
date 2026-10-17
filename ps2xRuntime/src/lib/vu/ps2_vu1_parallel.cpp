#include "ps2_vu1_parallel.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// PS2X_DMA_TRACE timeline (ps2_memory.cpp).
bool dmaTraceActive();
void dmaTrace(const char *format, ...);

namespace
{
    // PS2X_VU1_THREADS=<n>: worker threads for translated VU1 runs (0: none). Unset: 3 with 8
    // or more logical CPUs, 2 with 6 or 7 (the GS bands take most of the rest), else none; none
    // on ARM, where the workers were no faster on a phone and 2 of them hung a run (the queue
    // was only proven on x86's stronger memory ordering).
    // Read when the runtime asks for the queue (once, at start-up).
    uint32_t vu1ThreadCount()
    {
        if (const char *value = std::getenv("PS2X_VU1_THREADS"))
            return static_cast<uint32_t>(std::clamp(std::atoi(value), 0, 16));
#if defined(__aarch64__) || defined(__arm__)
        return 0u;
#endif
        const unsigned int cpus = std::thread::hardware_concurrency();
        return cpus >= 8u ? 3u : cpus >= 6u ? 2u : 0u;
    }

    // PS2X_VU1_DEPTH=<n>: runs that may be started and not yet in effect (default 10, the best
    // measured with 3 threads in Devil Kings battles).
    uint32_t vu1Depth(uint32_t threads)
    {
        const char *value = std::getenv("PS2X_VU1_DEPTH");
        const int depth = value ? std::atoi(value) : std::max(10, static_cast<int>(threads) + 2);
        return static_cast<uint32_t>(std::clamp(depth, 2, 64));
    }

    bool envSet(const char *name)
    {
        const char *value = std::getenv(name);
        return value && value[0] != '\0' && value[0] != '0';
    }

    // The diagnostics that need every run on the VIF1 thread.
    bool parallelAllowed()
    {
        static const bool s_allowed = []()
        {
            const char *compiled = std::getenv("PS2X_VU1_COMPILED");
            const bool compiledOff = compiled && compiled[0] == '0';
            return !compiledOff && !std::getenv("PS2X_VU1_VERIFY");
        }();
        return s_allowed;
    }

    uint32_t floatBits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    inline uint32_t lowestBit(uint64_t bits)
    {
#if defined(_MSC_VER) && !defined(__clang__)
        unsigned long index = 0;
        _BitScanForward64(&index, bits);
        return static_cast<uint32_t>(index);
#else
        return static_cast<uint32_t>(__builtin_ctzll(bits));
#endif
    }

    inline void pause()
    {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }
}

VU1Interpreter::Parallel::Parallel(VU1Interpreter &owner, uint32_t threadCount, uint32_t depth)
    : vu(owner)
{
    slots.reserve(depth);
    for (uint32_t index = 0; index < depth; ++index)
    {
        auto slot = std::make_unique<Slot>();
        slot->memory.resize(kQwords * 16u);
        slot->xgBuffer.resize(ps2x::vu1c::kXgkickBufferSize);
        slots.push_back(std::move(slot));
    }
    writtenSets.resize(static_cast<size_t>(depth) * 2u);
    committedMemory.resize(kQwords * 16u);
    scratch.resize(kQwords * 16u);
    checkEnabled = envSet("PS2X_VU1_PARALLEL_CHECK");
    noClip = std::getenv("PS2X_VU1_NOCLIP") != nullptr;
    for (uint32_t index = 0; index < threadCount; ++index)
        threads.emplace_back([this]() { workerMain(); });
    std::fprintf(stderr, "[vu1-par] %u worker threads, up to %u runs ahead%s\n", threadCount, depth,
                 checkEnabled ? ", every run checked against a serial run" : "");
}

VU1Interpreter::Parallel::~Parallel()
{
    stop.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(sleepMutex);
    }
    sleepCv.notify_all();
    for (std::thread &thread : threads)
        thread.join();
}

// ---------------------------------------------------------------- worker threads

void VU1Interpreter::Parallel::runSlot(Slot &slot)
{
    const bool traced = dmaTraceActive();
    if (traced)
        dmaTrace("vu1 run #%llu%s begins", static_cast<unsigned long long>(slot.sequence), slot.exact ? " (again)" : "");
    const unsigned int savedCsr = _mm_getcsr();
    _mm_setcsr(slot.csr);
    slot.program->run(slot.exec, slot.entryPc);
    _mm_setcsr(savedCsr);
    if (traced)
        dmaTrace("vu1 run #%llu ends, %llu cycles", static_cast<unsigned long long>(slot.sequence),
                 static_cast<unsigned long long>(slot.exec.cycle - slot.startCycle));
    slot.done.store(true, std::memory_order_release);
}

bool VU1Interpreter::Parallel::takeAndRun()
{
    uint64_t next = taken.load(std::memory_order_acquire);
    if (next >= published.load(std::memory_order_acquire))
        return false;
    if (taken.compare_exchange_weak(next, next + 1u, std::memory_order_acq_rel))
        runSlot(slotFor(next));
    return true;
}

void VU1Interpreter::Parallel::workerMain()
{
    uint32_t idle = 0;
    while (!stop.load(std::memory_order_acquire))
    {
        if (takeRedo() || takeAndRun())
        {
            idle = 0;
            continue;
        }
        // Runs come in bursts (one per MSCAL/MSCNT of a chain): spin a little before sleeping.
        if (++idle < 4096u)
        {
            pause();
            continue;
        }
        std::unique_lock<std::mutex> lock(sleepMutex);
        sleepers.fetch_add(1u, std::memory_order_seq_cst);
        sleepCv.wait_for(lock, std::chrono::milliseconds(2),
                         [this]() { return stop.load(std::memory_order_acquire) || workAvailable(); });
        sleepers.fetch_sub(1u, std::memory_order_seq_cst);
        idle = 0;
    }
}

bool VU1Interpreter::Parallel::workAvailable() const
{
    return redoTaken.load(std::memory_order_acquire) < redoPublished.load(std::memory_order_acquire) ||
           taken.load(std::memory_order_acquire) < published.load(std::memory_order_acquire);
}

bool VU1Interpreter::Parallel::takeRedo()
{
    uint64_t next = redoTaken.load(std::memory_order_acquire);
    if (next >= redoPublished.load(std::memory_order_acquire))
        return false;
    if (redoTaken.compare_exchange_weak(next, next + 1u, std::memory_order_acq_rel))
        runSlot(*redoQueue[next % redoQueue.size()]);
    return true;
}

void VU1Interpreter::Parallel::pushRedo(Slot &slot)
{
    const uint64_t index = redoPublished.load(std::memory_order_relaxed);
    redoQueue[index % redoQueue.size()] = &slot;
    redoPublished.store(index + 1u, std::memory_order_seq_cst);
    if (sleepers.load(std::memory_order_seq_cst) != 0u)
    {
        {
            std::lock_guard<std::mutex> lock(sleepMutex);
        }
        sleepCv.notify_one();
    }
}

void VU1Interpreter::Parallel::publish()
{
    published.store(dispatched, std::memory_order_seq_cst);
    if (sleepers.load(std::memory_order_seq_cst) != 0u)
    {
        {
            std::lock_guard<std::mutex> lock(sleepMutex);
        }
        sleepCv.notify_one();
    }
}

// ---------------------------------------------------------------- the VIF1 side

VU1Interpreter::Parallel::EntryStats &VU1Interpreter::Parallel::entryStats(const ps2x::vu1c::Program *program,
                                                                          uint32_t entryPc)
{
    for (EntryStats &entry : entries)
    {
        if (entry.program == program && entry.entryPc == entryPc)
            return entry;
    }
    entries.push_back({});
    entries.back().program = program;
    entries.back().entryPc = entryPc;
    return entries.back();
}

void VU1Interpreter::Parallel::noteDataWrite(uint32_t qword, uint32_t count)
{
    if (items.empty())
        return;
    for (uint32_t index = 0; index < count; ++index)
    {
        const uint32_t q = (qword + index) & (kQwords - 1u);
        vifWriteRuns[q] = dispatched;
        LogEntry entry;
        entry.qword = q;
        std::memcpy(entry.data, live + q * 16u, 16u);
        log.push_back(entry);
    }
}

void VU1Interpreter::Parallel::queuePath2Packet(const uint8_t *data, uint32_t size, bool directHl)
{
    if (items.empty())
    {
        memory->submitGifPacket(GifPathId::Path2, data, size, true, directHl);
        return;
    }
    Item item;
    item.run = false;
    item.packet.assign(data, data + size);
    item.directHl = directHl;
    item.logPosition = log.size();
    items.push_back(std::move(item));
}

void VU1Interpreter::Parallel::drain()
{
    if (items.empty())
        return;
    if (dmaTraceActive())
        dmaTrace("vu1 drain, %zu items", items.size());
    while (!items.empty())
        commitFront(true);
    if (dmaTraceActive())
        dmaTrace("vu1 drained");
}

void VU1Interpreter::Parallel::applyLog(size_t from, size_t to)
{
    for (size_t index = from; index < to; ++index)
        std::memcpy(committedMemory.data() + log[index].qword * 16u, log[index].data, 16u);
}

bool VU1Interpreter::Parallel::commitFront(bool wait)
{
    Item &item = items.front();
    if (item.run)
    {
        Slot &slot = slotFor(item.sequence);
        for (;;)
        {
            if (!slot.done.load(std::memory_order_acquire))
            {
                if (!wait)
                    return false;
                // Do work nobody has picked up yet (runs done again first) instead of idling.
                const auto start = std::chrono::steady_clock::now();
                while (!slot.done.load(std::memory_order_acquire))
                {
                    if (takeRedo())
                        continue;
                    uint64_t next = taken.load(std::memory_order_acquire);
                    if (next < published.load(std::memory_order_acquire))
                    {
                        if (taken.compare_exchange_weak(next, next + 1u, std::memory_order_acq_rel))
                        {
                            runSlot(slotFor(next));
                            ++stats.stolen;
                        }
                        continue;
                    }
                    pause();
                }
                const auto waited = std::chrono::steady_clock::now() - start;
                stats.waited += waited;
                if (dmaTraceActive())
                    dmaTrace("vu1 vif1 thread waited %.1f us for #%llu",
                             std::chrono::duration<double, std::micro>(waited).count(),
                             static_cast<unsigned long long>(slot.sequence));
            }
            if (commitRun(slot))
                break;
            if (!wait)
                return false;
        }
        ++committedRuns;
    }
    else
    {
        memory->submitGifPacket(GifPathId::Path2, item.packet.data(), static_cast<uint32_t>(item.packet.size()), true,
                                item.directHl);
    }
    const size_t from = item.logPosition;
    const size_t to = items.size() > 1u ? items[1].logPosition : log.size();
    applyLog(from, to);
    items.pop_front();
    if (items.empty())
    {
        log.clear();
        if (checkEnabled && std::memcmp(committedMemory.data(), live, dataSize) != 0)
        {
            ++stats.checkMismatches;
            std::fprintf(stderr, "[vu1-par] CHECK: VU1 data differs from the in-order result after the last run\n");
        }
    }
    return true;
}

bool VU1Interpreter::Parallel::inputsMatch(const Slot &slot) const
{
    const ps2x::vu1c::EntryInputs &in = *slot.inputs;
    const VU1State &now = vu.m_state;
    const VU1State &then = slot.input;
    for (uint32_t reg = 1; reg < 16u; ++reg)
    {
        if ((in.vi >> reg) & 1u && now.vi[reg] != then.vi[reg])
            return false;
    }
    for (uint32_t reg = 1; reg < 32u; ++reg)
    {
        if (in.vf[reg] == 0u)
            continue;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((in.vf[reg] & (8u >> component)) != 0u &&
                floatBits(now.vf[reg][component]) != floatBits(then.vf[reg][component]))
                return false;
        }
    }
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((in.acc & (8u >> component)) != 0u && floatBits(now.acc[component]) != floatBits(then.acc[component]))
            return false;
    }
    if ((in.other & ps2x::vu1c::kInputQ) != 0u && floatBits(now.q) != floatBits(then.q))
        return false;
    if ((in.other & ps2x::vu1c::kInputI) != 0u && floatBits(now.i) != floatBits(then.i))
        return false;
    if ((in.other & ps2x::vu1c::kInputMac) != 0u && now.mac != then.mac)
        return false;
    if ((in.other & ps2x::vu1c::kInputStatus) != 0u && now.status != then.status)
        return false;
    return ((now.clip ^ then.clip) & in.clip) == 0u;
}

bool VU1Interpreter::Parallel::readsAhead(const Slot &slot) const
{
    QwordSet ahead{};
    for (uint64_t run = slot.firstAhead; run < slot.sequence; ++run)
    {
        const QwordSet &written = writtenSets[run % writtenSets.size()];
        for (size_t word = 0; word < ahead.size(); ++word)
            ahead[word] |= written[word];
    }
    const ps2x::vu1c::Exec &x = slot.exec;
    for (size_t word = 0; word < ahead.size(); ++word)
    {
        if (ahead[word] == 0u)
            continue;
        // Read before written in full: the value went into the run.
        uint64_t clash = ahead[word] & x.memRead[word];
        // Written in some lanes only (and not read): the other lanes pass through, which only
        // matters where they differ from what the runs ahead left (committedMemory is exactly
        // what this run would have found).
        uint64_t partial = ahead[word] & x.memWritten[word] & ~x.memRead[word];
        while (partial != 0u)
        {
            const uint32_t bit = lowestBit(partial);
            partial &= partial - 1u;
            const uint32_t q = static_cast<uint32_t>(word) * 64u + bit;
            const uint8_t lanes = x.memLanes[q];
            for (uint32_t component = 0; component < 4u && lanes != 0xFu; ++component)
            {
                if ((lanes & (8u >> component)) == 0u &&
                    std::memcmp(slot.memory.data() + q * 16u + component * 4u,
                                committedMemory.data() + q * 16u + component * 4u, 4u) != 0)
                {
                    clash |= 1ull << bit;
                    break;
                }
            }
        }
        if (clash == 0u)
            continue;
        static const bool s_log = std::getenv("PS2X_VU1_STATS") != nullptr;
        static uint32_t s_lines = 0;
        static uint32_t s_seen = 0;
        if (s_log && (++s_seen % 512u) == 0u && s_lines < 80u)
        {
            ++s_lines;
            const uint32_t q = static_cast<uint32_t>(word) * 64u + lowestBit(clash);
            const bool read = ((x.memRead[word] >> (q % 64u)) & 1u) != 0u;
            uint64_t writer = slot.sequence;
            for (uint64_t run = slot.firstAhead; run < slot.sequence; ++run)
                if ((writtenSets[run % writtenSets.size()][word] >> (q % 64u)) & 1u)
                    writer = run;
            std::fprintf(stderr, "[vu1-par] %s entry %04x top %03x reads qword %03x (%s) written by the run %llu back\n",
                         slot.program->name, slot.entryPc, slot.top, q, read ? "read" : "lanes kept",
                         static_cast<unsigned long long>(slot.sequence - writer));
        }
        return true;
    }
    return false;
}

void VU1Interpreter::Parallel::submitPath1(const Slot &slot)
{
    size_t offset = 0;
    for (uint32_t size : slot.gifSizes)
    {
        if (memory)
            memory->submitGifPacket(GifPathId::Path1, slot.gif.data() + offset, size);
        else if (gs)
            gs->processGIFPacket(slot.gif.data() + offset, size);
        offset += size;
    }
}

bool VU1Interpreter::Parallel::commitRun(Slot &slot)
{
    VU1Interpreter &v = vu;
    const ps2x::vu1c::Exec &x = slot.exec;
    EntryStats &entry = entryStats(slot.program, slot.entryPc);

    // The state the runs ahead left must be the one this run would have started from.
    const bool clean = !v.pipelinesPending() && !v.m_state.branchPending && !v.m_state.ebit &&
                       !v.m_state.haltAfterDelaySlot && v.m_workingClip == v.m_state.clip &&
                       (slot.mscal || !v.m_viBranchBackupValid);
    const bool ended = x.exit == ps2x::vu1c::Exit::Ended && slot.errors == 0u && !x.viBackupValid;
    bool ok = false;
    bool redo = false;
    if (slot.exact)
    {
        ok = ended;
        if (!ok)
            ++stats.rerunExit;
    }
    else
    {
        ++entry.attempts;
        if (!clean)
            ++stats.rerunState;
        else if (!slot.mscal && v.m_state.pc != slot.entryPc)
            ++stats.rerunPc;
        else if (!ended)
            ++stats.rerunExit;
        else if (!inputsMatch(slot))
        {
            ++stats.rerunInputs;
            ++entry.inputFailures;
        }
        else if (readsAhead(slot))
        {
            ++stats.rerunMemory;
            ++entry.memoryFailures;
        }
        else
            ok = true;
        if (!ok)
        {
            ++entry.failures;
            // Stale: a worker does it again from the exact state (committedMemory is what it
            // would have found), unless only the ordinary path can (an unfinished state).
            redo = clean;
        }
    }

    if (redo)
    {
        prepare(slot, slot.mscal, slot.mscal ? slot.entryPc : (v.m_state.pc & v.microAddressMask()), slot.top,
                slot.itop, slot.maxCycles, committedMemory.data(), true);
        ++stats.redone;
        if (dmaTraceActive())
            dmaTrace("vu1 commit #%llu stale, again on a worker", static_cast<unsigned long long>(slot.sequence));
        pushRedo(slot);
        return false;
    }

    QwordSet &written = writtenSets[slot.sequence % writtenSets.size()];
    if (ok)
    {
        if (checkEnabled)
            check(slot);
        for (size_t word = 0; word < written.size(); ++word)
            written[word] = x.memWritten[word];
        merge(slot);
        ++stats.merged;
        if (dmaTraceActive())
            dmaTrace("vu1 commit #%llu", static_cast<unsigned long long>(slot.sequence));
    }
    else
    {
        if (dmaTraceActive())
            dmaTrace("vu1 commit #%llu stale, again in place", static_cast<unsigned long long>(slot.sequence));
        rerun(slot, written);
    }
    slot.exact = false;
    if (!v.m_state.ebit && !v.m_state.branchPending && !v.pipelinesPending())
        entry.exitPc = v.m_state.pc;
    else
        entry.exitPc = kUnknownPc;

    // A run done again costs about what running it in place does, plus the copies: only a
    // start point that nearly always fails goes in place for a while (longer each time).
    if (entry.attempts >= 64u)
    {
        if (entry.failures * 10u > entry.attempts * 9u)
        {
            entry.penalty = std::min<uint32_t>(entry.penalty == 0u ? 256u : entry.penalty * 2u, 8192u);
            entry.backoff = entry.penalty;
        }
        else
        {
            entry.penalty = 0u;
        }
        entry.attempts = 0u;
        entry.failures = 0u;
    }
    reportStats();
    return true;
}

void VU1Interpreter::Parallel::merge(Slot &slot)
{
    VU1Interpreter &v = vu;
    const ps2x::vu1c::Exec &x = slot.exec;
    VU1State &s = v.m_state;
    const uint64_t start = slot.startCycle;
    const uint64_t base = v.m_cycle;
    if (slot.exact)
    {
        // Started from this very state: its whole result stands, as after a run in place.
        v.importCompiledState(x);
        s.top = slot.top;
        s.itop = slot.itop;
        s.stoppedByD = false;
        s.stoppedByT = false;
    }
    else
    {
        if (slot.mscal)
        {
            // execute() clears the pipelines and ready cycles before the run.
            v.m_vfReady = {};
            v.m_viReady = {};
            v.m_accReady = {};
        }
        // Registers the run wrote (their ready cycles moved past the start) take its values.
        for (uint32_t reg = 1; reg < 32u; ++reg)
        {
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if (x.vfReady[reg][component] > start)
                {
                    s.vf[reg][component] = x.vf[reg][component];
                    v.m_vfReady[reg][component] = x.vfReady[reg][component] - start + base;
                }
            }
        }
        for (uint32_t reg = 1; reg < 16u; ++reg)
        {
            if (x.viReady[reg] > start)
            {
                s.vi[reg] = x.vi[reg];
                v.m_viReady[reg] = x.viReady[reg] - start + base;
            }
        }
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if (x.accReady[component] > start)
            {
                s.acc[component] = x.acc[component];
                v.m_accReady[component] = x.accReady[component] - start + base;
            }
        }
        if ((x.effects & ps2x::vu1c::kEffectDivide) != 0u)
            s.q = x.q;
        if ((x.effects & ps2x::vu1c::kEffectI) != 0u)
            s.i = x.i;
        if ((x.effects & ps2x::vu1c::kEffectMac) != 0u)
            s.mac = x.mac;
        if ((slot.inputs->other & ps2x::vu1c::kInputStatus) != 0u)
        {
            s.status = x.status; // started from these very bits
        }
        else
        {
            // The run started with a clear status: its sticky bits add to the ones before.
            uint32_t status = (s.status | x.status) & 0xFC0u;
            status |= ((x.effects & ps2x::vu1c::kEffectStatus) != 0u ? x.status : s.status) & 0x00Fu;
            status |= ((x.effects & ps2x::vu1c::kEffectDivide) != 0u ? x.status : s.status) & 0x030u;
            s.status = status;
        }
        // Each CLIP shifts six bits in: fewer than four leave some of the earlier flags.
        const auto clipAfter = [&](uint32_t own, uint32_t before) -> uint32_t
        {
            if ((x.effects & ps2x::vu1c::kEffectClipSet) != 0u || x.clipWrites >= 4u)
                return own & 0xFFFFFFu;
            const uint32_t bits = 6u * x.clipWrites;
            return ((before << bits) | (own & ((1u << bits) - 1u))) & 0xFFFFFFu;
        };
        const uint32_t clipBefore = s.clip;
        s.clip = clipAfter(x.clip, clipBefore);
        v.m_workingClip = clipAfter(x.workingClip, clipBefore);

        s.pc = x.exitPc & v.microAddressMask();
        s.top = slot.top;
        s.itop = slot.itop;
        s.ebit = false;
        s.haltAfterDelaySlot = false;
        s.branchPending = false;
        s.branchTarget = 0;
        s.branchDelay = 0;
        s.stoppedByD = false;
        s.stoppedByT = false;
        s.vf[0][0] = 0.0f;
        s.vf[0][1] = 0.0f;
        s.vf[0][2] = 0.0f;
        s.vf[0][3] = 1.0f;
        v.m_cycle = base + (x.cycle - start);
        s.cycles = v.m_cycle;
        v.m_viBranchBackupValid = false;
        v.m_viBranchBackupReg = 0;
        v.m_viBranchBackupValue = 0;
    }

    // Memory: the qwords it wrote, except where the VIF wrote again after the run started.
    for (uint32_t word = 0; word < kQwords / 64u; ++word)
    {
        uint64_t bits = x.memWritten[word];
        while (bits != 0u)
        {
            const uint32_t q = word * 64u + lowestBit(bits);
            bits &= bits - 1u;
            const uint8_t *data = slot.memory.data() + q * 16u;
            std::memcpy(committedMemory.data() + q * 16u, data, 16u);
            if (vifWriteRuns[q] <= slot.sequence)
                std::memcpy(live + q * 16u, data, 16u);
        }
    }
    submitPath1(slot);
}

void VU1Interpreter::Parallel::rerun(Slot &slot, QwordSet &written)
{
    // committedMemory holds exactly what the run would have found: run it there, the
    // ordinary way, as execute()/resume() would have.
    std::memcpy(scratch.data(), committedMemory.data(), dataSize);
    inCommit = true;
    if (slot.mscal)
        vu.execute(vuCode, codeSize, committedMemory.data(), dataSize, *gs, memory, slot.entryPc, slot.top, slot.itop,
                   slot.maxCycles);
    else
        vu.resume(vuCode, codeSize, committedMemory.data(), dataSize, *gs, memory, slot.top, slot.itop, slot.maxCycles);
    inCommit = false;
    written = {};
    for (uint32_t q = 0; q < kQwords; ++q)
    {
        const uint8_t *data = committedMemory.data() + q * 16u;
        if (std::memcmp(data, scratch.data() + q * 16u, 16u) == 0)
            continue;
        written[q / 64u] |= 1ull << (q % 64u);
        if (vifWriteRuns[q] <= slot.sequence)
            std::memcpy(live + q * 16u, data, 16u);
    }
}

// PS2X_VU1_PARALLEL_CHECK: runs the program again on the exact state and memory, as it
// would have run in order, and compares with what the merge is about to put into effect.
void VU1Interpreter::Parallel::check(Slot &slot)
{
    VU1Interpreter &v = vu;
    ++stats.checked;
    const VU1State savedState = v.m_state;
    const uint64_t savedCycle = v.m_cycle;
    const uint32_t savedClip = v.m_workingClip;
    const auto savedVf = v.m_vfReady;
    const auto savedVi = v.m_viReady;
    const auto savedAcc = v.m_accReady;

    // The serial run.
    if (slot.mscal)
    {
        v.resetScheduler();
        v.m_state.pc = slot.entryPc;
        v.m_state.vf[0][0] = 0.0f;
        v.m_state.vf[0][1] = 0.0f;
        v.m_state.vf[0][2] = 0.0f;
        v.m_state.vf[0][3] = 1.0f;
    }
    v.m_state.top = slot.top;
    v.m_state.itop = slot.itop;
    std::vector<uint8_t> serialMemory = committedMemory;
    std::vector<uint8_t> serialPackets;
    const unsigned int savedCsr = _mm_getcsr();
    _mm_setcsr(slot.csr & ~0x0040u); // runCompiled adds DAZ itself
    v.runCompiled(*slot.program, serialMemory.data(), dataSize, v.m_cycle + slot.maxCycles, &serialPackets);
    _mm_setcsr(savedCsr);
    const bool serialEnded = v.m_compiledExec->exit == ps2x::vu1c::Exit::Ended;
    const VU1State serial = v.m_state;
    const uint64_t serialCycle = v.m_cycle;
    const uint32_t serialClip = v.m_workingClip;

    // The merge, on a copy.
    v.m_state = savedState;
    v.m_cycle = savedCycle;
    v.resetScheduler();
    v.m_workingClip = savedClip;
    v.m_vfReady = savedVf;
    v.m_viReady = savedVi;
    v.m_accReady = savedAcc;
    std::vector<uint8_t> savedCommitted = committedMemory;
    std::vector<uint8_t> savedLive(live, live + dataSize);
    GS *savedGs = gs;
    PS2Memory *savedMemory = memory;
    gs = nullptr;
    memory = nullptr; // no packets out
    merge(slot);
    gs = savedGs;
    memory = savedMemory;

    char what[160] = "";
    if (!serialEnded)
        std::snprintf(what, sizeof(what), "the serial run did not end");
    for (uint32_t reg = 0; reg < 32u && !what[0]; ++reg)
        for (uint32_t c = 0; c < 4u && !what[0]; ++c)
            if (floatBits(serial.vf[reg][c]) != floatBits(v.m_state.vf[reg][c]))
                std::snprintf(what, sizeof(what), "vf%u[%u] serial=%08x merged=%08x", reg, c,
                              floatBits(serial.vf[reg][c]), floatBits(v.m_state.vf[reg][c]));
    for (uint32_t reg = 0; reg < 16u && !what[0]; ++reg)
        if (serial.vi[reg] != v.m_state.vi[reg])
            std::snprintf(what, sizeof(what), "vi%u serial=%d merged=%d", reg, serial.vi[reg], v.m_state.vi[reg]);
    if (!what[0] && std::memcmp(serial.acc, v.m_state.acc, sizeof(serial.acc)) != 0)
        std::snprintf(what, sizeof(what), "acc");
    if (!what[0] && floatBits(serial.q) != floatBits(v.m_state.q))
        std::snprintf(what, sizeof(what), "q");
    if (!what[0] && floatBits(serial.i) != floatBits(v.m_state.i))
        std::snprintf(what, sizeof(what), "i");
    if (!what[0] && serial.mac != v.m_state.mac)
        std::snprintf(what, sizeof(what), "mac serial=%04x merged=%04x", serial.mac, v.m_state.mac);
    if (!what[0] && serial.status != v.m_state.status)
        std::snprintf(what, sizeof(what), "status serial=%03x merged=%03x", serial.status, v.m_state.status);
    if (!what[0] && serial.clip != v.m_state.clip)
        std::snprintf(what, sizeof(what), "clip serial=%06x merged=%06x", serial.clip, v.m_state.clip);
    if (!what[0] && serialClip != v.m_workingClip)
        std::snprintf(what, sizeof(what), "working clip");
    if (!what[0] && serial.pc != v.m_state.pc)
        std::snprintf(what, sizeof(what), "pc serial=%04x merged=%04x", serial.pc, v.m_state.pc);
    if (!what[0] && serialCycle != v.m_cycle)
    {
        uint32_t differing = 0;
        for (uint32_t q = 0; q < kQwords; ++q)
            differing += std::memcmp(serialMemory.data() + q * 16u, committedMemory.data() + q * 16u, 16u) != 0;
        std::snprintf(what, sizeof(what),
                      "cycles serial=%llu merged=%llu (%u qwords and %s packets differ; clip now %06x, at start %06x)",
                      static_cast<unsigned long long>(serialCycle - savedCycle),
                      static_cast<unsigned long long>(v.m_cycle - savedCycle), differing,
                      serialPackets != slot.gif ? "the" : "no", savedState.clip, slot.input.clip);
    }
    if (!what[0] && std::memcmp(serialMemory.data(), committedMemory.data(), dataSize) != 0)
    {
        uint32_t offset = 0;
        while (serialMemory[offset] == committedMemory[offset])
            ++offset;
        std::snprintf(what, sizeof(what), "vu memory at 0x%04x", offset);
    }
    if (!what[0] && serialPackets != slot.gif)
        std::snprintf(what, sizeof(what), "gif packets serial=%zu bytes merged=%zu bytes", serialPackets.size(),
                      slot.gif.size());
    if (what[0])
    {
        ++stats.checkMismatches;
        if (stats.checkMismatches <= 100u)
            std::fprintf(stderr, "[vu1-par] CHECK MISMATCH %s entry=%04x run=%llu ahead=%llu: %s\n", slot.program->name,
                         slot.entryPc, static_cast<unsigned long long>(slot.sequence),
                         static_cast<unsigned long long>(slot.sequence - slot.firstAhead), what);
    }

    // Back to the state before the merge; commitRun merges for real.
    v.m_state = savedState;
    v.m_cycle = savedCycle;
    v.m_workingClip = savedClip;
    v.m_vfReady = savedVf;
    v.m_viReady = savedVi;
    v.m_accReady = savedAcc;
    committedMemory = savedCommitted;
    std::memcpy(live, savedLive.data(), dataSize);
}

void VU1Interpreter::Parallel::reportStats()
{
    static const bool s_enabled = std::getenv("PS2X_VU1_STATS") != nullptr;
    if (!s_enabled)
        return;
    const auto now = std::chrono::steady_clock::now();
    const double window = std::chrono::duration<double>(now - stats.windowStart).count();
    if (window < 1.0)
        return;
    std::fprintf(stderr,
                 "[vu1-par] per s: started %.0f, merged %.0f, run again: pc %.0f inputs %.0f memory %.0f exit %.0f "
                 "state %.0f (%.0f of them on a worker); in place %.0f; done by the VIF1 thread %.0f; VIF1 thread "
                 "waited %.1f ms%s\n",
                 stats.started / window, stats.merged / window, stats.rerunPc / window, stats.rerunInputs / window,
                 stats.rerunMemory / window, stats.rerunExit / window, stats.rerunState / window,
                 stats.redone / window, stats.inline_ / window,
                 stats.stolen / window, std::chrono::duration<double, std::milli>(stats.waited).count() / window,
                 checkEnabled ? "" : "");
    std::fprintf(stderr,
                 "[vu1-par] in place because: D/T %.0f, exit unknown %.0f, state %.0f, not translated %.0f, "
                 "no inputs %.0f, backing off %.0f\n",
                 stats.refusals[1] / window, stats.refusals[2] / window, stats.refusals[3] / window,
                 stats.refusals[4] / window, stats.refusals[5] / window, stats.refusals[6] / window);
    for (EntryStats &entry : entries)
    {
        std::fprintf(stderr,
                     "[vu1-par]   %s entry %04x: exit %04x, %u of %u failed, backoff %u; per s: inputs %.0f, memory %.0f\n",
                     entry.program->name, entry.entryPc, entry.exitPc, entry.failures, entry.attempts, entry.backoff,
                     entry.inputFailures / window, entry.memoryFailures / window);
        entry.inputFailures = 0u;
        entry.memoryFailures = 0u;
    }
    if (checkEnabled)
        std::fprintf(stderr, "[vu1-par] checked %llu runs, %llu mismatches\n",
                     static_cast<unsigned long long>(stats.checked),
                     static_cast<unsigned long long>(stats.checkMismatches));
    const uint64_t checked = stats.checked;
    const uint64_t mismatches = stats.checkMismatches;
    stats = Stats{};
    stats.checked = checked;
    stats.checkMismatches = mismatches;
    stats.windowStart = now;
}

void VU1Interpreter::Parallel::prepare(Slot &slot, bool mscal, uint32_t entryPc, uint32_t top, uint32_t itop,
                                      uint32_t maxCycles, const uint8_t *source, bool exact)
{
    VU1Interpreter &v = vu;
    slot.mscal = mscal;
    slot.entryPc = entryPc;
    slot.top = top;
    slot.itop = itop;
    slot.maxCycles = maxCycles;
    slot.exact = exact;
    slot.startCycle = v.m_cycle;
    slot.csr = (_mm_getcsr() & ~0x6000u) | 0x6000u | 0x0040u; // round to zero, as run() sets, and DAZ

    VU1State &in = slot.input;
    in = v.m_state;
    in.pc = entryPc;
    in.ebit = false;
    in.haltAfterDelaySlot = false;
    in.stoppedByD = false;
    in.stoppedByT = false;
    in.top = top;
    in.itop = itop;
    in.branchPending = false;
    in.branchTarget = 0;
    in.branchDelay = 0;
    if (mscal)
    {
        in.vf[0][0] = 0.0f;
        in.vf[0][1] = 0.0f;
        in.vf[0][2] = 0.0f;
        in.vf[0][3] = 1.0f;
    }
    if (!exact && (slot.inputs->other & ps2x::vu1c::kInputStatus) == 0u)
        in.status = 0u; // the run's own sticky bits, for the merge

    std::memcpy(slot.memory.data(), source, dataSize);
    ps2x::vu1c::Exec &x = slot.exec;
    std::memcpy(x.vf, in.vf, sizeof(x.vf));
    std::memcpy(x.vi, in.vi, sizeof(x.vi));
    std::memcpy(x.acc, in.acc, sizeof(x.acc));
    x.q = in.q;
    x.i = in.i;
    x.mac = in.mac;
    x.status = in.status;
    x.clip = in.clip;
    x.top = in.top;
    x.itop = in.itop;
    x.cycle = v.m_cycle;
    x.budgetEnd = v.m_cycle + maxCycles;
    x.lastReady = v.m_cycle;
    for (uint32_t reg = 0; reg < 32u; ++reg)
        for (uint32_t component = 0; component < 4u; ++component)
            x.vfReady[reg][component] = mscal ? 0u : v.m_vfReady[reg][component];
    for (uint32_t reg = 0; reg < 16u; ++reg)
        x.viReady[reg] = mscal ? 0u : v.m_viReady[reg];
    for (uint32_t component = 0; component < 4u; ++component)
        x.accReady[component] = mscal ? 0u : v.m_accReady[component];
    x.flagHead = 0u;
    x.flagCount = 0u;
    x.workingClip = in.clip; // for an MSCNT, the committed state has them equal
    x.liteSticky = 0u;
    x.fdivPending = false;
    x.fdivReady = 0u;
    x.fdivValue = 0.0f;
    x.fdivStatusDi = 0u;
    x.nextCommit = ps2x::vu1c::kNever;
    x.viBackupValid = false;
    x.viBackupReg = 0u;
    x.viBackupValue = 0;
    x.xgActive = false;
    x.xgSource = 0u;
    x.xgCopied = 0u;
    x.xgTagEnd = 0u;
    x.xgTotal = 0u;
    x.xgTagEop = false;
    x.xgIssue = 0u;
    x.xgPacket = slot.xgBuffer.data();
    x.mem = slot.memory.data();
    x.memSize = dataSize;
    x.noClip = noClip;
    x.submitGif = [](void *user, const uint8_t *data, uint32_t size)
    {
        auto *target = static_cast<Parallel::Slot *>(user);
        target->gif.insert(target->gif.end(), data, data + size);
        target->gifSizes.push_back(size);
    };
    x.gifUser = &slot;
    x.reportError = [](void *user, uint32_t) { ++static_cast<Parallel::Slot *>(user)->errors; };
    x.errorUser = &slot;
    x.exit = ps2x::vu1c::Exit::Refused;
    x.exitPc = entryPc;
    x.effects = 0u;
    x.clipWrites = 0u;
    x.clearMemSets();
    slot.gif.clear();
    slot.gifSizes.clear();
    slot.errors = 0u;
    slot.done.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------- VU1Interpreter

Vu1RunQueue *VU1Interpreter::runQueue()
{
    if (!m_parallel && m_unit == Unit::VU1 && parallelAllowed())
    {
        if (const uint32_t threads = vu1ThreadCount())
            m_parallel = std::make_unique<Parallel>(*this, threads, vu1Depth(threads));
    }
    return m_parallel.get();
}

bool VU1Interpreter::runInParallel(bool mscal, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData,
                                   uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t startPC, uint32_t top,
                                   uint32_t itop, uint32_t maxCycles)
{
    Parallel &p = *m_parallel;
    if (p.inCommit)
        return false; // a run being redone at commit time goes the ordinary way
    p.vuCode = vuCode;
    p.codeSize = codeSize;
    p.live = vuData;
    p.dataSize = dataSize;
    p.gs = &gs;
    p.memory = memory;

    // Put finished runs into effect (the fewer ahead, the fresher the state a new run
    // starts from) and make room for one more.
    while (!p.items.empty() && p.commitFront(false))
    {
    }
    while (p.dispatched - p.committedRuns >= p.slots.size())
        p.commitFront(true);

    // MSCNT continues where the run ahead ended; for one not in effect yet, where it
    // ended last time (checked when this run takes effect).
    const bool ahead = p.runsAhead();
    const uint32_t entryPc = (mscal ? startPC : (ahead ? p.predictedPc : m_state.pc)) & microAddressMask();
    const ps2x::vu1c::Program *program = nullptr;
    const ps2x::vu1c::EntryInputs *inputs = nullptr;
    int refusal = 0;
    if (!memory || dataSize != Parallel::kQwords * 16u || m_state.dBitEnabled || m_state.tBitEnabled)
        refusal = 1;
    else if (!mscal && ahead && p.predictedPc == Parallel::kUnknownPc)
        refusal = 2;
    else if (!mscal && !ahead &&
             (pipelinesPending() || m_state.branchPending || m_state.ebit || m_state.haltAfterDelaySlot ||
              m_workingClip != m_state.clip || m_viBranchBackupValid))
        refusal = 3; // execute() resets all of this for an MSCAL
    if (refusal == 0)
    {
        program = compiledProgramFor(vuCode, codeSize, memory);
        inputs = program ? program->inputsAt(entryPc) : nullptr;
        if (!program)
            refusal = 4;
        else if (!inputs)
            refusal = 5;
    }
    if (refusal == 0)
    {
        Parallel::EntryStats &entry = p.entryStats(program, entryPc);
        if (entry.backoff != 0u)
        {
            --entry.backoff;
            refusal = 6;
        }
    }
    if (refusal != 0)
    {
        p.drain();
        ++p.stats.inline_;
        ++p.stats.refusals[refusal];
        if (dmaTraceActive())
            dmaTrace("vu1 in place (reason %d) %s %04x", refusal, program ? program->name + 8 : "-", entryPc);
        return false;
    }

    const uint64_t sequence = p.dispatched;
    Parallel::Slot &slot = p.slotFor(sequence);
    slot.program = program;
    slot.inputs = inputs;
    slot.sequence = sequence;
    slot.firstAhead = p.committedRuns;
    p.prepare(slot, mscal, entryPc, top, itop, maxCycles, vuData, false);

    if (p.items.empty())
    {
        std::memcpy(p.committedMemory.data(), vuData, dataSize);
        p.log.clear();
    }
    Parallel::Item item;
    item.run = true;
    item.sequence = sequence;
    item.logPosition = p.log.size();
    p.items.push_back(std::move(item));
    p.predictedPc = p.entryStats(program, entryPc).exitPc;
    ++p.dispatched;
    ++p.stats.started;
    if (dmaTraceActive())
        dmaTrace("vu1 start #%llu %s %04x", static_cast<unsigned long long>(sequence), program->name + 8, entryPc);
    p.publish();
    return true;
}
