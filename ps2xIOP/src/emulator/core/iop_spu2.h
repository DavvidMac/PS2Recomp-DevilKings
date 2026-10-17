#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace ps2x::iop::detail
{
    // SPU2 sound processor: two cores of 24 ADPCM voices with ADSR envelopes,
    // AutoDMA PCM input and a 48 kHz stereo output that the host drains.
    // Register storage stays with IopMemory; this class mirrors every 16-bit
    // write and overrides the registers the hardware updates by itself.
    class Spu2
    {
    public:
        static constexpr uint32_t kBase = 0x1F900000u;
        static constexpr uint32_t kEnd = 0x1F900800u;
        static constexpr uint32_t kIopCyclesPerSample = 768u;

        Spu2();

        void reset();

        [[nodiscard]] static bool isRegister(uint32_t phys) noexcept { return phys >= kBase && phys < kEnd; }

        void write16(uint32_t phys, uint16_t value);
        // Returns true and sets value when the register is produced by the SPU.
        [[nodiscard]] bool read16(uint32_t phys, uint16_t &value) const;

        // DMA from IOP RAM: AutoDMA PCM (ADMA) or a block into SPU RAM at TSA.
        void dmaWrite(uint32_t core, const uint8_t *data, uint32_t bytes, bool autoDma);

        // Generate output samples up to the given IOP cycle.
        void advanceTo(uint64_t cycle);

        // Host side (any thread): copy up to frames stereo frames, returns count.
        size_t readOutput(int16_t *destination, size_t frames);

    private:
        struct Voice
        {
            bool active = false;
            uint32_t startAddress = 0;   // SSA, halfwords
            uint32_t loopAddress = 0;    // LSAX, halfwords
            uint32_t nextAddress = 0;    // NAX, halfwords (current block)
            bool loopWritten = false;
            uint32_t counter = 0;        // 12-bit pitch fraction
            std::array<int16_t, 28> block{};
            uint32_t blockIndex = 28;
            int32_t history1 = 0;
            int32_t history2 = 0;
            int16_t previous = 0;
            int16_t current = 0;
            bool blockEnd = false;
            bool blockRepeat = false;
            enum class Phase : uint8_t
            {
                Off,
                Attack,
                Decay,
                Sustain,
                Release
            } phase = Phase::Off;
            int32_t level = 0;           // ENVX 0..0x7FFF
            uint32_t envelopeWait = 0;
            int16_t volumeLeft = 0;
            int16_t volumeRight = 0;
        };

        struct Core
        {
            std::array<Voice, 24> voices{};
            uint32_t endx = 0;
            std::deque<int16_t> admaLeft;
            std::deque<int16_t> admaRight;
            bool masterVolumeWritten = false;
            uint32_t keyOnCount = 0;
        };

        [[nodiscard]] uint16_t reg(uint32_t offset) const noexcept { return m_regs[(offset >> 1u) & 0x3FFu]; }
        [[nodiscard]] uint32_t coreReg(uint32_t core, uint32_t local) const noexcept { return reg(core * 0x400u + local); }
        [[nodiscard]] uint32_t voiceAddress(uint32_t core, uint32_t voice, uint32_t field) const noexcept;
        [[nodiscard]] uint32_t transferAddress(uint32_t core) const noexcept;
        void setTransferAddress(uint32_t core, uint32_t halfword);

        void keyOn(uint32_t core, uint32_t voice);
        void keyOff(uint32_t core, uint32_t voice);
        void decodeBlock(Voice &voice);
        void advanceSample(uint32_t core, uint32_t voiceIndex, Voice &voice);
        void stepEnvelope(uint32_t core, uint32_t voiceIndex, Voice &voice);
        void mixSample();

        std::array<uint16_t, 0x400> m_regs{};
        std::vector<uint16_t> m_ram;
        std::array<Core, 2> m_cores{};
        uint64_t m_nextSampleCycle = 0;
        bool m_clockStarted = false;

        std::mutex m_outputMutex;
        std::vector<int16_t> m_output;
        size_t m_outputRead = 0;
        size_t m_outputCount = 0;
        // PS2X_AUDIO_TRACE (or PS2X_FPS_TRACE): frames asked for, played as silence (not mixed
        // yet) and dropped (mixed too far ahead), and the buffer level seen by the host.
        struct OutputTrace
        {
            uint64_t requested = 0;
            uint64_t silent = 0;
            uint64_t dropped = 0;
            size_t minLevel = SIZE_MAX;
            size_t maxLevel = 0;
        };
        OutputTrace m_outputTrace;
    };
}
