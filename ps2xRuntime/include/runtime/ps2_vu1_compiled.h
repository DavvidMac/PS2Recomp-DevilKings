#ifndef PS2_VU1_COMPILED_H
#define PS2_VU1_COMPILED_H

// Runtime support for VU1 micro programs translated ahead of time to C++ by
// ps2xRuntime/tools/vu1rec.py.
//
// Translated programs must produce exactly what VU1Interpreter produces: the same
// arithmetic, the same stall timing, the same flag/Q latencies and the same
// interleaving of XGKICK transfers with stores. The model differs from the
// interpreter in one way that does not change results: VF/VI/ACC writes land in
// the register file immediately and only their ready cycles are tracked. The VU
// interlocks on those registers, so no instruction can observe the old value
// once the write has issued. MAC/status/clip flags and Q are not interlocked,
// so they keep explicit pending queues that commit exactly when the interpreter
// commits them.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

// Hot helpers are forced inline into the (very large) translated run functions,
// where the inliner would otherwise give up; their rare slow paths stay out of line.
#if defined(_MSC_VER) && !defined(__clang__)
#define PS2X_VU1C_INLINE __forceinline
#define PS2X_VU1C_NOINLINE __declspec(noinline)
#else
#define PS2X_VU1C_INLINE inline __attribute__((always_inline))
#define PS2X_VU1C_NOINLINE __attribute__((noinline))
#endif

namespace ps2x::vu1c
{
    constexpr uint64_t kNever = ~0ull;
    constexpr uint32_t kXgkickBufferSize = 0x10000u;

    // Bit for component c (x=0 .. w=3) in a DEST mask (x=8, y=4, z=2, w=1).
    constexpr uint8_t lane(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    inline uint32_t floatBits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    inline float bitsFloat(uint32_t bits)
    {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    // Same as VU1Interpreter::normalizeOperand: denormals read as signed zero,
    // Inf/NaN as the signed largest finite value.
    inline float norm(float value)
    {
        uint32_t bits = floatBits(value);
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent == 0u)
            bits &= 0x80000000u;
        else if (exponent == 0xFFu)
            bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
        return bitsFloat(bits);
    }

    // Same as VU1Interpreter::normalizeResult.
    inline float normResult(float value, uint32_t &laneFlags)
    {
        uint32_t bits = floatBits(value);
        const uint32_t sign = bits & 0x80000000u;
        const uint32_t magnitude = bits & 0x7FFFFFFFu;
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        laneFlags = sign != 0u ? 0x2u : 0u;
        if (magnitude == 0u)
            laneFlags |= 0x1u;
        else if (exponent == 0u)
        {
            laneFlags |= 0x5u;
            bits = sign;
        }
        else if (exponent == 0xFFu)
        {
            laneFlags |= 0x8u;
            bits = sign | 0x7F7FFFFFu;
        }
        return bitsFloat(bits);
    }

    // Same as VU1Interpreter::normalizeFmacExactResult.
    inline uint8_t exactFlags(float &value, long double exact)
    {
#if defined(__clang__) || defined(__GNUC__)
        // std::signbit(long double) is an out-of-line CRT call (_ldsign) on MSVC targets.
        const bool negative = __builtin_signbit(exact);
#else
        const bool negative = std::signbit(exact);
#endif
        const long double magnitude = std::fabs(exact);
        uint8_t flags = negative ? 0x2u : 0u;
        uint32_t bits = negative ? 0x80000000u : 0u;
        if (magnitude == 0.0L)
        {
            flags |= 0x1u;
            value = bitsFloat(bits);
        }
        else if (magnitude > static_cast<long double>(std::numeric_limits<float>::max()))
        {
            flags |= 0x8u;
            value = bitsFloat(bits | 0x7F7FFFFFu);
        }
        else if (magnitude < static_cast<long double>(std::numeric_limits<float>::min()))
        {
            flags |= 0x5u;
            value = bitsFloat(bits);
        }
        return flags;
    }

    // One lane of VU1Interpreter::normalizeFmacResult for an instruction whose exact
    // result is `exact()`: the fast path is the interpreter's, the slow path recomputes.
    template <typename Exact>
    inline uint8_t fmacLane(float &value, bool productSum, float accRaw, Exact exact)
    {
        const uint32_t bits = floatBits(value);
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent >= 2u && exponent <= 253u &&
            (!productSum || ((floatBits(accRaw) >> 23) & 0xFFu) <= exponent + 16u))
        {
            return (bits & 0x80000000u) != 0u ? 0x2u : 0u;
        }
        return exactFlags(value, exact());
    }

    // One lane of VU1Interpreter::calculateFmacProductSticky.
    inline uint32_t productSticky(float left, float right)
    {
        float product = left * right;
        const uint32_t bits = floatBits(product);
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent >= 2u && exponent <= 253u)
            return (bits & 0x80000000u) != 0u ? 0x2u : 0u;
        return exactFlags(product, static_cast<long double>(left) * static_cast<long double>(right)) & 0xFu;
    }

    // Same as the FTOI helper in ps2_vu1_upper.cpp.
    inline int32_t floatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }

    // CLIP judgement of one value against +/- the W limit, as in execUpper.
    inline bool exceedsClipPlane(float value, uint32_t signMask, int32_t limit)
    {
        int32_t ordered = 0;
        const uint32_t bits = floatBits(value) ^ signMask;
        std::memcpy(&ordered, &bits, sizeof(ordered));
        return ordered > limit;
    }

    inline uint32_t clipFlags(const float *xyz, float w)
    {
        const uint32_t wBits = floatBits(w);
        const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
        uint32_t flags = 0u;
        if (exceedsClipPlane(xyz[0], 0x00000000u, limit))
            flags |= 0x01u;
        if (exceedsClipPlane(xyz[0], 0x80000000u, limit))
            flags |= 0x02u;
        if (exceedsClipPlane(xyz[1], 0x00000000u, limit))
            flags |= 0x04u;
        if (exceedsClipPlane(xyz[1], 0x80000000u, limit))
            flags |= 0x08u;
        if (exceedsClipPlane(xyz[2], 0x00000000u, limit))
            flags |= 0x10u;
        if (exceedsClipPlane(xyz[2], 0x80000000u, limit))
            flags |= 0x20u;
        return flags;
    }

    struct FlagEntry
    {
        uint64_t issue = 0;
        uint64_t ready = 0;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t extraSticky = 0;
        uint32_t clip = 0;
        bool writesMac = false;
        bool writesStatus = false;
        bool writesSticky = false;
        bool writesClip = false;
    };

    enum class Exit : uint8_t
    {
        Ended,   // E-bit program end; pipelines flushed like VU1Interpreter::flushPipelines.
        Handoff, // Stopped at exitPc; the interpreter continues from the exported state.
        Refused  // Entry PC not handled by this program; nothing was executed.
    };

    struct Exec
    {
        // Architectural state.
        float vf[32][4];
        int32_t vi[16];
        float acc[4];
        float q;
        float i;
        uint32_t mac;
        uint32_t status;
        uint32_t clip;
        uint32_t top;
        uint32_t itop;

        // Timing (absolute VU cycles, same clock as VU1Interpreter::m_cycle).
        uint64_t cycle;
        uint64_t budgetEnd;
        uint64_t lastReady;
        uint64_t vfReady[32][4];
        uint64_t viReady[16];
        uint64_t accReady[4];

        // MAC/status/clip writes in issue order (their ready cycles never decrease).
        FlagEntry flags[16];
        uint32_t flagHead;
        uint32_t flagCount;
        uint32_t workingClip;

        // FDIV unit (Q).
        bool fdivPending;
        uint64_t fdivReady;
        float fdivValue;
        uint32_t fdivStatusDi;

        // No queued flag write or FDIV result lands before this cycle (kNever when
        // nothing is queued); lets commitTo return without looking at the queues.
        uint64_t nextCommit;

        // A branch right after an integer op reads that op's old VI value.
        bool viBackupValid;
        uint8_t viBackupReg;
        int32_t viBackupValue;

        // PATH1 transfer started by XGKICK: qword k is copied at cycle xgIssue + 1 + 2k.
        bool xgActive;
        uint32_t xgSource;
        uint32_t xgCopied;
        uint32_t xgTagEnd;
        uint32_t xgTotal;
        bool xgTagEop;
        uint64_t xgIssue;
        uint8_t *xgPacket;

        uint8_t *mem;
        uint32_t memSize;
        bool noClip;

        void (*submitGif)(void *user, const uint8_t *data, uint32_t size);
        void *gifUser;
        void (*reportError)(void *user, uint32_t code);
        void *errorUser;

        Exit exit;
        uint32_t exitPc;

        // ------------------------------------------------------------ timing

        void noteReady(uint64_t ready)
        {
            if (ready > lastReady)
                lastReady = ready;
        }

        uint64_t vfStall(uint64_t t, uint32_t reg, uint8_t lanes) const
        {
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((lanes & lane(component)) != 0u)
                    t = std::max(t, vfReady[reg][component]);
            }
            return t;
        }

        void vfWritten(uint32_t reg, uint8_t lanes, uint64_t ready)
        {
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((lanes & lane(component)) != 0u)
                    vfReady[reg][component] = ready;
            }
            noteReady(ready);
        }

        uint64_t accStall(uint64_t t, uint8_t lanes) const
        {
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((lanes & lane(component)) != 0u)
                    t = std::max(t, accReady[component]);
            }
            return t;
        }

        void accWritten(uint8_t lanes, uint64_t ready)
        {
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((lanes & lane(component)) != 0u)
                    accReady[component] = ready;
            }
            noteReady(ready);
        }

        // ------------------------------------------------------------ flags and Q

        // Applies every MAC/status/clip and Q write that the interpreter would have
        // committed by cycle t, in its order: flags first, then FDIV, per cycle.
        PS2X_VU1C_INLINE void commitTo(uint64_t t)
        {
            if (t >= nextCommit)
                commitDue(t);
        }

        PS2X_VU1C_NOINLINE void commitDue(uint64_t t)
        {
            for (;;)
            {
                const uint64_t nextFlag = flagCount != 0u ? flags[flagHead].ready : kNever;
                const uint64_t nextDiv = fdivPending ? fdivReady : kNever;
                const uint64_t next = std::min(nextFlag, nextDiv);
                if (next > t)
                {
                    nextCommit = next;
                    return;
                }
                while (flagCount != 0u && flags[flagHead].ready <= next)
                {
                    const FlagEntry &entry = flags[flagHead];
                    if (entry.writesMac)
                        mac = entry.mac;
                    if (entry.writesStatus)
                    {
                        const uint32_t current = entry.status & 0xFu;
                        status = (status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
                    }
                    if (entry.writesSticky)
                        status = (status & 0x03Fu) | (entry.status & 0xFC0u);
                    if (entry.writesClip)
                        clip = entry.clip;
                    flagHead = (flagHead + 1u) & 15u;
                    --flagCount;
                }
                if (fdivPending && fdivReady <= next)
                {
                    q = fdivValue;
                    const uint32_t currentDi = fdivStatusDi & 0x30u;
                    status = (status & 0xFCFu) | currentDi | (currentDi << 6);
                    fdivPending = false;
                }
            }
        }

        PS2X_VU1C_INLINE FlagEntry *pushFlag(uint64_t t)
        {
            // Entries due by now have left the interpreter's slots before this issue.
            commitTo(t);
            if (flagCount >= 8u)
            {
                // The interpreter has eight flag slots and stops the program when they overflow.
                fail(0xFFFFFFFFu);
                return nullptr;
            }
            FlagEntry &entry = flags[(flagHead + flagCount) & 15u];
            ++flagCount;
            entry = {};
            entry.issue = t;
            entry.ready = t + 4u;
            // Ready cycles never decrease, so this only matters when the queue was empty.
            if (entry.ready < nextCommit)
                nextCommit = entry.ready;
            noteReady(entry.ready);
            return &entry;
        }

        PS2X_VU1C_INLINE void fmacFlags(uint64_t t, const uint8_t laneFlags[4], uint8_t dest, uint32_t extraSticky)
        {
            if (dest == 0u)
                return;
            uint32_t newMac = 0u;
            uint32_t newStatus = 0u;
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((dest & lane(component)) == 0u)
                    continue;
                // Z, S, U and O of a lane land in MAC bits 0-3, 4-7, 8-11 and 12-15 (x highest).
                const uint32_t flagsOfLane = laneFlags[component];
                const uint32_t spread = (flagsOfLane & 0x1u) | ((flagsOfLane & 0x2u) << 3) |
                                        ((flagsOfLane & 0x4u) << 6) | ((flagsOfLane & 0x8u) << 9);
                newMac |= spread << (3u - component);
                newStatus |= flagsOfLane;
            }
            if (FlagEntry *entry = pushFlag(t))
            {
                entry->mac = newMac;
                entry->status = newStatus;
                entry->extraSticky = extraSticky;
                entry->writesMac = true;
                entry->writesStatus = true;
            }
        }

        void clipFlag(uint64_t t, uint32_t newClip)
        {
            workingClip = ((workingClip << 6) | (newClip & 0x3Fu)) & 0xFFFFFFu;
            if (FlagEntry *entry = pushFlag(t))
            {
                entry->clip = workingClip;
                entry->writesClip = true;
            }
        }

        void fsset(uint64_t t, uint32_t immediate)
        {
            for (uint32_t index = 0; index < flagCount; ++index)
            {
                FlagEntry &entry = flags[(flagHead + index) & 15u];
                if (entry.issue == t)
                    entry.writesStatus = false;
            }
            if (FlagEntry *entry = pushFlag(t))
            {
                entry->status = immediate & 0xFC0u;
                entry->writesSticky = true;
            }
        }

        void fcset(uint64_t t, uint32_t newClip)
        {
            workingClip = newClip & 0xFFFFFFu;
            for (uint32_t index = 0; index < flagCount; ++index)
            {
                FlagEntry &entry = flags[(flagHead + index) & 15u];
                if (entry.issue == t)
                    entry.writesClip = false;
            }
            if (FlagEntry *entry = pushFlag(t))
            {
                entry->clip = workingClip;
                entry->writesClip = true;
            }
        }

        void queueQ(uint64_t t, float value, uint32_t latency, uint32_t statusDi)
        {
            uint32_t ignored = 0u;
            fdivPending = true;
            fdivReady = t + latency;
            fdivValue = normResult(value, ignored);
            fdivStatusDi = statusDi & 0x30u;
            if (fdivReady < nextCommit)
                nextCommit = fdivReady;
            noteReady(fdivReady);
        }

        // ------------------------------------------------------------ XGKICK and memory

        void fail(uint32_t code)
        {
            if (reportError)
                reportError(errorUser, code);
        }

        // Copies the next qword of the active transfer (VU1Interpreter::progressXgkick body).
        void xgCopyQword()
        {
            if (xgCopied > kXgkickBufferSize - 16u)
            {
                fail(0xFFFFFFFBu);
                xgActive = false;
                return;
            }
            const uint32_t qwordOffset = xgCopied;
            const uint32_t sourceOffset = (xgSource + xgCopied) % memSize;
            if (sourceOffset + 16u <= memSize)
            {
                std::memcpy(xgPacket + xgCopied, mem + sourceOffset, 16u);
            }
            else
            {
                for (uint32_t index = 0; index < 16u; ++index)
                    xgPacket[xgCopied + index] = mem[(xgSource + xgCopied + index) % memSize];
            }
            xgCopied += 16u;

            if (xgTagEnd == 0u)
            {
                uint64_t tagLo = 0;
                std::memcpy(&tagLo, xgPacket + qwordOffset, sizeof(tagLo));
                const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
                const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
                uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
                if (nreg == 0u)
                    nreg = 16u;
                uint64_t tagBytes = 16u;
                if (format == 0u)
                    tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
                else if (format == 1u)
                    tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
                else if (format == 2u)
                    tagBytes += static_cast<uint64_t>(nloop) * 16u;
                else
                {
                    fail(0xFFFFFFF8u);
                    xgActive = false;
                    return;
                }
                if (tagBytes > kXgkickBufferSize - qwordOffset)
                {
                    fail(0xFFFFFFFBu);
                    xgActive = false;
                    return;
                }
                xgTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
                xgTagEop = ((tagLo >> 15) & 1u) != 0u;
                if (xgTagEop)
                    xgTotal = xgTagEnd;
            }

            if (xgCopied >= xgTagEnd)
            {
                if (xgTagEop)
                {
                    if (submitGif)
                        submitGif(gifUser, xgPacket, xgTotal);
                    xgActive = false;
                }
                else
                {
                    xgTagEnd = 0u;
                    xgTagEop = false;
                }
            }
        }

        // Performs every qword copy scheduled at or before cycle `boundary`.
        void xgProgressTo(uint64_t boundary)
        {
            while (xgActive && xgIssue + 1u + 2u * static_cast<uint64_t>(xgCopied / 16u) <= boundary)
                xgCopyQword();
        }

        // Finishes the transfer and returns the cycle at which it became idle.
        uint64_t xgFinish()
        {
            uint64_t idle = 0;
            while (xgActive)
            {
                idle = xgIssue + 1u + 2u * static_cast<uint64_t>(xgCopied / 16u);
                xgCopyQword();
            }
            return idle;
        }

        void xgkick(uint64_t t, int32_t qwordAddress)
        {
            xgActive = true;
            xgSource = (static_cast<uint32_t>(static_cast<uint16_t>(qwordAddress)) * 16u) % memSize;
            xgCopied = 0u;
            xgTagEnd = 0u;
            xgTotal = 0u;
            xgTagEop = false;
            xgIssue = t;
        }

        // Stores land at cycle t + 1, after the PATH1 copies of cycles <= t.
        void store(uint64_t t, uint32_t address, const uint32_t words[4], uint8_t dest)
        {
            if (xgActive)
                xgProgressTo(t);
            uint32_t current[4];
            std::memcpy(current, mem + address, sizeof(current));
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((dest & lane(component)) != 0u)
                    current[component] = words[component];
            }
            std::memcpy(mem + address, current, sizeof(current));
            noteReady(t + 1u);
        }

        void storeVf(uint64_t t, uint32_t address, uint32_t reg, uint8_t dest)
        {
            uint32_t words[4];
            std::memcpy(words, vf[reg], sizeof(words));
            store(t, address, words, dest);
        }

        void storeVi(uint64_t t, uint32_t address, int32_t value, uint8_t dest)
        {
            const uint32_t word = static_cast<uint32_t>(static_cast<uint16_t>(value & 0xFFFF));
            const uint32_t words[4] = {word, word, word, word};
            store(t, address, words, dest);
        }

        void load(uint32_t address, float out[4]) const
        {
            std::memcpy(out, mem + address, 16u);
        }

        int32_t loadVi(uint32_t address, uint8_t dest) const
        {
            uint32_t component = 3u;
            if ((dest & 0x8u) != 0u)
                component = 0u;
            else if ((dest & 0x4u) != 0u)
                component = 1u;
            else if ((dest & 0x2u) != 0u)
                component = 2u;
            uint32_t word = 0;
            std::memcpy(&word, mem + address + component * 4u, sizeof(word));
            return static_cast<int16_t>(word & 0xFFFFu);
        }

        uint32_t address(int32_t base, int32_t offset) const
        {
            return ((static_cast<uint32_t>(base + offset)) * 16u) & (memSize - 1u);
        }

        uint32_t addressU16(int32_t base) const
        {
            return (static_cast<uint32_t>(static_cast<uint16_t>(base)) * 16u) & (memSize - 1u);
        }

        // ------------------------------------------------------------ program exits

        // E-bit end: flushPipelines() advances until every pending write has landed.
        void end(uint64_t lastPairCycle, uint32_t resumePc)
        {
            uint64_t finalCycle = std::max(lastPairCycle + 1u, lastReady);
            if (xgActive)
                finalCycle = std::max(finalCycle, xgFinish());
            commitTo(finalCycle);
            cycle = finalCycle;
            exitPc = resumePc;
            exit = Exit::Ended;
        }

        void handoff(uint32_t pc)
        {
            exitPc = pc;
            exit = Exit::Handoff;
        }
    };

    using ProgramFn = void (*)(Exec &x, uint32_t entryPc);

    struct Program
    {
        const char *name;
        const uint8_t *code; // micro memory image the program was translated from
        uint32_t codeBytes;  // bytes of `code` that must match (up to the last reachable pair)
        ProgramFn run;
    };

    // Registers a translated program; called from static initialisers in generated code.
    void registerProgram(const Program *program);

    // Translated program whose image matches micro memory, or nullptr.
    const Program *findProgram(const uint8_t *microMemory, uint32_t size);

    // Number of registered programs (for diagnostics).
    uint32_t programCount();
}

#endif
