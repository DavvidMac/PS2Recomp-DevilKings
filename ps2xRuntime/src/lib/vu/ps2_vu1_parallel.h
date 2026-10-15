#ifndef PS2_VU1_PARALLEL_H
#define PS2_VU1_PARALLEL_H

#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu1_compiled.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Translated VU1 runs on worker threads (PS2X_VU1_THREADS). A run from an entry point whose
// inputs are known (ps2x::vu1c::EntryInputs) starts on a copy of VU1 data memory as soon
// as the VIF reaches its MSCAL/MSCNT, from the state of the runs already in effect, while
// the runs ahead of it may still be going. Runs take effect in program order on the thread
// that started them (the VIF1 thread): one whose entry point, inputs or memory reads turn
// out to differ from what the runs ahead of it left is run again there on the exact state,
// so the result is always that of running them one after the other.
struct VU1Interpreter::Parallel final : public Vu1RunQueue
{
    static constexpr uint32_t kQwords = 1024u;
    static constexpr uint32_t kUnknownPc = ~0u;
    using QwordSet = std::array<uint64_t, kQwords / 64u>;

    struct Slot
    {
        // Set by the starting thread before the run is published.
        const ps2x::vu1c::Program *program = nullptr;
        const ps2x::vu1c::EntryInputs *inputs = nullptr;
        bool mscal = false;
        uint32_t entryPc = 0;
        uint32_t top = 0;
        uint32_t itop = 0;
        uint32_t maxCycles = 0;
        bool exact = false; // started from the state the runs ahead left: no checks needed
        uint64_t sequence = 0;
        uint64_t firstAhead = 0; // oldest run not in effect when this one started
        uint64_t startCycle = 0;
        unsigned int csr = 0;
        VU1State input{};            // the state the run started from
        std::vector<uint8_t> memory; // VU1 data at the start, the run's result after
        ps2x::vu1c::Exec exec{};
        std::vector<uint8_t> xgBuffer;
        // Set by the thread running it.
        std::vector<uint8_t> gif; // PATH1 packets back to back
        std::vector<uint32_t> gifSizes;
        uint32_t errors = 0;
        std::atomic<bool> done{false};
    };

    // Something that takes effect in program order: a run, or a PATH2 packet the VIF sent
    // after the runs before it. logPosition: VIF data writes logged before it.
    struct Item
    {
        bool run = false;
        uint64_t sequence = 0;
        std::vector<uint8_t> packet;
        bool directHl = false;
        size_t logPosition = 0;
    };

    struct LogEntry
    {
        uint32_t qword;
        uint8_t data[16];
    };

    struct EntryStats
    {
        const ps2x::vu1c::Program *program = nullptr;
        uint32_t entryPc = 0;
        uint32_t exitPc = kUnknownPc; // where the last run from here ended
        uint32_t attempts = 0;
        uint32_t failures = 0;
        uint32_t backoff = 0;
        uint32_t penalty = 0;
    };

    struct Stats
    {
        uint64_t started = 0;
        uint64_t merged = 0;
        uint64_t rerunPc = 0;
        uint64_t rerunInputs = 0;
        uint64_t rerunMemory = 0;
        uint64_t rerunExit = 0;
        uint64_t rerunState = 0;
        uint64_t redone = 0;
        uint64_t inline_ = 0;
        // In place because: 1 D/T or no memory, 2 exit unknown, 3 state, 4 not translated,
        // 5 no inputs for the entry, 6 backing off.
        uint64_t refusals[7] = {};
        uint64_t stolen = 0;
        uint64_t checked = 0;
        uint64_t checkMismatches = 0;
        std::chrono::steady_clock::duration waited{};
        std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
    };

    Parallel(VU1Interpreter &owner, uint32_t threadCount, uint32_t depth);
    ~Parallel() override;

    // Vu1RunQueue (the VIF1 thread).
    bool pending() const override { return !items.empty(); }
    void drain() override;
    void noteDataWrite(uint32_t qword, uint32_t count) override;
    void queuePath2Packet(const uint8_t *data, uint32_t size, bool directHl) override;

    Slot &slotFor(uint64_t sequence) { return *slots[sequence % slots.size()]; }
    bool runsAhead() const { return dispatched != committedRuns; }
    EntryStats &entryStats(const ps2x::vu1c::Program *program, uint32_t entryPc);
    // Puts the front item into effect; with wait, waits for (or does) its run. False when
    // the front run has not finished and wait is false.
    bool commitFront(bool wait);
    // Puts the run into effect, or, when it turns out stale, starts it again on a worker
    // from the exact state and returns false.
    bool commitRun(Slot &slot);
    // Fills the slot for a run from `entryPc` on a copy of `source`, from vu's current state.
    void prepare(Slot &slot, bool mscal, uint32_t entryPc, uint32_t top, uint32_t itop, uint32_t maxCycles,
                 const uint8_t *source, bool exact);
    void pushRedo(Slot &slot);
    bool takeRedo();
    bool workAvailable() const;
    bool inputsMatch(const Slot &slot) const;
    bool readsAhead(const Slot &slot) const;
    void merge(Slot &slot);
    void rerun(Slot &slot, QwordSet &written);
    void check(Slot &slot);
    void submitPath1(const Slot &slot);
    void applyLog(size_t from, size_t to);
    void publish();
    bool takeAndRun();
    static void runSlot(Slot &slot);
    void workerMain();
    void reportStats();

    VU1Interpreter &vu;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<std::thread> threads;
    std::atomic<uint64_t> published{0}; // runs handed to the workers
    std::atomic<uint64_t> taken{0};     // runs a thread has picked up
    std::atomic<bool> stop{false};
    std::mutex sleepMutex;
    std::condition_variable sleepCv;
    std::atomic<uint32_t> sleepers{0};
    // Runs done again from the exact state: before anything else, they hold up the commits.
    std::array<Slot *, 64> redoQueue{};
    std::atomic<uint64_t> redoPublished{0};
    std::atomic<uint64_t> redoTaken{0};

    // Starting thread only.
    uint64_t dispatched = 0;    // runs started
    uint64_t committedRuns = 0; // runs in effect
    uint32_t predictedPc = kUnknownPc;
    bool inCommit = false;
    std::deque<Item> items;
    std::vector<LogEntry> log;
    std::vector<uint8_t> committedMemory; // VU1 data with the items before items.front() in effect
    std::vector<uint8_t> scratch;
    std::array<uint64_t, kQwords> vifWriteRuns{}; // runs started when the VIF last wrote each qword
    std::vector<QwordSet> writtenSets;            // per run, by sequence
    std::vector<EntryStats> entries;
    Stats stats;
    bool checkEnabled = false;
    bool noClip = false;

    // The VU1 memory and GIF the runs belong to (the last start's).
    uint8_t *vuCode = nullptr;
    uint32_t codeSize = 0;
    uint8_t *live = nullptr;
    uint32_t dataSize = 0;
    GS *gs = nullptr;
    PS2Memory *memory = nullptr;
};

#endif
