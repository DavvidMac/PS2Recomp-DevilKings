#include "iop_spu2.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kRamHalfwords = 1024u * 1024u; // 2 MB
        constexpr size_t kOutputCapacityFrames = 48000u / 4u; // 250 ms
        constexpr uint32_t kVoiceAddressBase = 0x1C0u;
        constexpr uint32_t kVoiceAddressStride = 0xCu;
        constexpr uint32_t kKeyOn = 0x1A0u;
        constexpr uint32_t kKeyOff = 0x1A4u;
        constexpr uint32_t kTransferAddress = 0x1A8u;
        constexpr uint32_t kDataPort = 0x1ACu;
        constexpr uint32_t kVoiceMixLeft = 0x188u;
        constexpr uint32_t kVoiceMixRight = 0x190u;
        constexpr uint32_t kEndx = 0x340u;
        constexpr uint32_t kCoreVolumeBase = 0x760u;
        constexpr uint32_t kCoreVolumeStride = 0x28u;

        constexpr int32_t kFilterPositive[5] = {0, 60, 115, 98, 122};
        constexpr int32_t kFilterNegative[5] = {0, 0, -52, -55, -60};

        int16_t clamp16(int32_t value)
        {
            return static_cast<int16_t>(std::clamp(value, -32768, 32767));
        }

        // Fixed volume registers hold a 15-bit signed value; sweep mode (bit 15)
        // is not modelled and keeps the previous level.
        int16_t decodeVolume(uint16_t value, int16_t previous)
        {
            if ((value & 0x8000u) != 0u)
                return previous;
            return static_cast<int16_t>(static_cast<uint16_t>(value << 1u));
        }
    }

    Spu2::Spu2()
        : m_ram(kRamHalfwords, 0u),
          m_output(kOutputCapacityFrames * 2u, 0)
    {
    }

    void Spu2::reset()
    {
        m_regs.fill(0u);
        std::fill(m_ram.begin(), m_ram.end(), uint16_t{0});
        for (Core &core : m_cores)
            core = Core{};
        m_clockStarted = false;
        m_nextSampleCycle = 0u;
        std::lock_guard<std::mutex> lock(m_outputMutex);
        m_outputRead = 0u;
        m_outputCount = 0u;
    }

    uint32_t Spu2::voiceAddress(uint32_t core, uint32_t voice, uint32_t field) const noexcept
    {
        const uint32_t local = kVoiceAddressBase + voice * kVoiceAddressStride + field;
        return ((coreReg(core, local) & 0xFu) << 16u) | coreReg(core, local + 2u);
    }

    uint32_t Spu2::transferAddress(uint32_t core) const noexcept
    {
        return ((coreReg(core, kTransferAddress) & 0xFu) << 16u) | coreReg(core, kTransferAddress + 2u);
    }

    void Spu2::setTransferAddress(uint32_t core, uint32_t halfword)
    {
        halfword &= kRamHalfwords - 1u;
        m_regs[(core * 0x400u + kTransferAddress) >> 1u] = static_cast<uint16_t>(halfword >> 16u);
        m_regs[(core * 0x400u + kTransferAddress + 2u) >> 1u] = static_cast<uint16_t>(halfword);
    }

    void Spu2::write16(uint32_t phys, uint16_t value)
    {
        const uint32_t offset = phys - kBase;
        m_regs[(offset >> 1u) & 0x3FFu] = value;

        if (offset >= kCoreVolumeBase)
        {
            const uint32_t core = (offset - kCoreVolumeBase) / kCoreVolumeStride;
            const uint32_t local = (offset - kCoreVolumeBase) % kCoreVolumeStride;
            if (core < 2u && local < 4u)
                m_cores[core].masterVolumeWritten = true;
            return;
        }

        const uint32_t core = offset >> 10u;
        const uint32_t local = offset & 0x3FFu;
        if (local >= kVoiceAddressBase && local < kVoiceAddressBase + 24u * kVoiceAddressStride)
        {
            const uint32_t voice = (local - kVoiceAddressBase) / kVoiceAddressStride;
            const uint32_t field = (local - kVoiceAddressBase) % kVoiceAddressStride;
            if (field == 4u || field == 6u)
            {
                Voice &target = m_cores[core].voices[voice];
                target.loopAddress = voiceAddress(core, voice, 4u);
                target.loopWritten = true;
            }
            return;
        }

        switch (local)
        {
        case kKeyOn:
        case kKeyOn + 2u:
        {
            const uint32_t first = local == kKeyOn ? 0u : 16u;
            for (uint32_t bit = 0; bit < 16u && first + bit < 24u; ++bit)
                if ((value >> bit) & 1u)
                    keyOn(core, first + bit);
            break;
        }
        case kKeyOff:
        case kKeyOff + 2u:
        {
            const uint32_t first = local == kKeyOff ? 0u : 16u;
            for (uint32_t bit = 0; bit < 16u && first + bit < 24u; ++bit)
                if ((value >> bit) & 1u)
                    keyOff(core, first + bit);
            break;
        }
        case kDataPort:
        {
            const uint32_t address = transferAddress(core);
            m_ram[address & (kRamHalfwords - 1u)] = value;
            setTransferAddress(core, address + 1u);
            break;
        }
        default:
            break;
        }
    }

    bool Spu2::read16(uint32_t phys, uint16_t &value) const
    {
        const uint32_t offset = phys - kBase;
        if (offset >= kCoreVolumeBase)
            return false;
        const uint32_t core = offset >> 10u;
        const uint32_t local = offset & 0x3FFu;

        if (local < 24u * 0x10u)
        {
            const Voice &voice = m_cores[core].voices[local >> 4u];
            switch (local & 0xFu)
            {
            case 0xAu:
                value = static_cast<uint16_t>(voice.level);
                return true;
            case 0xCu:
                value = static_cast<uint16_t>(voice.volumeLeft);
                return true;
            case 0xEu:
                value = static_cast<uint16_t>(voice.volumeRight);
                return true;
            default:
                return false;
            }
        }
        if (local >= kVoiceAddressBase && local < kVoiceAddressBase + 24u * kVoiceAddressStride)
        {
            const Voice &voice = m_cores[core].voices[(local - kVoiceAddressBase) / kVoiceAddressStride];
            const uint32_t field = (local - kVoiceAddressBase) % kVoiceAddressStride;
            if (field == 8u)
            {
                value = static_cast<uint16_t>((voice.nextAddress >> 16u) & 0xFu);
                return true;
            }
            if (field == 0xAu)
            {
                value = static_cast<uint16_t>(voice.nextAddress);
                return true;
            }
            return false;
        }
        if (local == kEndx)
        {
            value = static_cast<uint16_t>(m_cores[core].endx);
            return true;
        }
        if (local == kEndx + 2u)
        {
            value = static_cast<uint16_t>(m_cores[core].endx >> 16u);
            return true;
        }
        return false;
    }

    void Spu2::dmaWrite(uint32_t core, const uint8_t *data, uint32_t bytes, bool autoDma)
    {
        if (core > 1u || data == nullptr)
            return;
        if (autoDma)
        {
            // AutoDMA blocks are 512 bytes of left samples then 512 of right.
            Core &target = m_cores[core];
            for (uint32_t block = 0; block + 1024u <= bytes; block += 1024u)
            {
                for (uint32_t i = 0; i < 256u; ++i)
                {
                    int16_t left;
                    int16_t right;
                    std::memcpy(&left, data + block + i * 2u, sizeof(left));
                    std::memcpy(&right, data + block + 512u + i * 2u, sizeof(right));
                    target.admaLeft.push_back(left);
                    target.admaRight.push_back(right);
                }
            }
            // Never let a stalled consumer grow the queue without bound.
            while (target.admaLeft.size() > 48000u)
            {
                target.admaLeft.pop_front();
                target.admaRight.pop_front();
            }
            return;
        }

        uint32_t address = transferAddress(core);
        for (uint32_t i = 0; i + 1u < bytes; i += 2u)
        {
            uint16_t halfword;
            std::memcpy(&halfword, data + i, sizeof(halfword));
            m_ram[address & (kRamHalfwords - 1u)] = halfword;
            ++address;
        }
        setTransferAddress(core, address);
    }

    void Spu2::keyOn(uint32_t core, uint32_t voiceIndex)
    {
        ++m_cores[core].keyOnCount;
        Voice &voice = m_cores[core].voices[voiceIndex];
        voice = Voice{};
        voice.active = true;
        voice.startAddress = voiceAddress(core, voiceIndex, 0u);
        voice.nextAddress = voice.startAddress;
        voice.loopAddress = voiceAddress(core, voiceIndex, 4u);
        voice.phase = Voice::Phase::Attack;
        voice.blockIndex = 28u;
        m_cores[core].endx &= ~(1u << voiceIndex);
        decodeBlock(voice);
    }

    void Spu2::keyOff(uint32_t core, uint32_t voiceIndex)
    {
        Voice &voice = m_cores[core].voices[voiceIndex];
        if (voice.active && voice.phase != Voice::Phase::Off)
        {
            voice.phase = Voice::Phase::Release;
            voice.envelopeWait = 0u;
        }
    }

    void Spu2::decodeBlock(Voice &voice)
    {
        const uint32_t address = voice.nextAddress & (kRamHalfwords - 1u);
        const uint16_t header = m_ram[address];
        uint32_t shift = header & 0xFu;
        if (shift > 12u)
            shift = 9u;
        const uint32_t filter = std::min<uint32_t>((header >> 4u) & 0x7u, 4u);
        const uint32_t flags = header >> 8u;

        if ((flags & 0x4u) != 0u && !voice.loopWritten)
            voice.loopAddress = voice.nextAddress;
        voice.blockEnd = (flags & 0x1u) != 0u;
        voice.blockRepeat = (flags & 0x2u) != 0u;

        for (uint32_t i = 0; i < 28u; ++i)
        {
            const uint16_t word = m_ram[(address + 1u + i / 4u) & (kRamHalfwords - 1u)];
            const int32_t nibble = (word >> ((i % 4u) * 4u)) & 0xF;
            int32_t sample = static_cast<int16_t>(static_cast<uint16_t>(nibble << 12u)) >> shift;
            sample += (voice.history1 * kFilterPositive[filter] + voice.history2 * kFilterNegative[filter] + 32) >> 6;
            sample = clamp16(sample);
            voice.history2 = voice.history1;
            voice.history1 = sample;
            voice.block[i] = static_cast<int16_t>(sample);
        }
        voice.blockIndex = 0u;
    }

    void Spu2::advanceSample(uint32_t core, uint32_t voiceIndex, Voice &voice)
    {
        const uint32_t pitch = std::min<uint32_t>(coreReg(core, voiceIndex * 0x10u + 4u), 0x3FFFu);
        voice.counter += pitch;
        while (voice.counter >= 0x1000u && voice.active)
        {
            voice.counter -= 0x1000u;
            if (voice.blockIndex >= 28u)
            {
                if (voice.blockEnd)
                {
                    m_cores[core].endx |= 1u << voiceIndex;
                    voice.nextAddress = voice.loopAddress;
                    if (!voice.blockRepeat)
                    {
                        voice.active = false;
                        voice.phase = Voice::Phase::Off;
                        voice.level = 0;
                        break;
                    }
                }
                else
                {
                    voice.nextAddress += 8u;
                }
                decodeBlock(voice);
            }
            voice.previous = voice.current;
            voice.current = voice.block[voice.blockIndex++];
        }
    }

    void Spu2::stepEnvelope(uint32_t core, uint32_t voiceIndex, Voice &voice)
    {
        if (voice.envelopeWait > 0u)
        {
            --voice.envelopeWait;
            return;
        }
        const uint16_t adsr1 = static_cast<uint16_t>(coreReg(core, voiceIndex * 0x10u + 6u));
        const uint16_t adsr2 = static_cast<uint16_t>(coreReg(core, voiceIndex * 0x10u + 8u));
        const int32_t sustainLevel = std::min<int32_t>(((adsr1 & 0xF) + 1) * 0x800, 0x7FFF);

        bool exponential = false;
        bool decrease = false;
        int32_t shift = 0;
        int32_t step = 0;
        switch (voice.phase)
        {
        case Voice::Phase::Attack:
            exponential = (adsr1 & 0x8000u) != 0u;
            shift = (adsr1 >> 10u) & 0x1F;
            step = 7 - ((adsr1 >> 8u) & 0x3);
            break;
        case Voice::Phase::Decay:
            exponential = true;
            decrease = true;
            shift = ((adsr1 >> 4u) & 0xF) << 2;
            step = -8;
            break;
        case Voice::Phase::Sustain:
            exponential = (adsr2 & 0x8000u) != 0u;
            decrease = (adsr2 & 0x4000u) != 0u;
            shift = (adsr2 >> 8u) & 0x1F;
            step = decrease ? -8 + ((adsr2 >> 6u) & 0x3) : 7 - ((adsr2 >> 6u) & 0x3);
            break;
        case Voice::Phase::Release:
            exponential = (adsr2 & 0x20u) != 0u;
            decrease = true;
            shift = (adsr2 & 0x1F) << 2;
            step = -8;
            break;
        case Voice::Phase::Off:
            return;
        }

        uint32_t cycles = 1u << std::max(0, shift - 11);
        int32_t delta = step << std::max(0, 11 - shift);
        if (exponential && !decrease && voice.level > 0x6000)
            cycles *= 4u;
        if (exponential && decrease)
            delta = (delta * voice.level) >> 15;
        voice.level = std::clamp(voice.level + delta, 0, 0x7FFF);
        voice.envelopeWait = cycles - 1u;

        switch (voice.phase)
        {
        case Voice::Phase::Attack:
            if (voice.level >= 0x7FFF)
            {
                voice.phase = Voice::Phase::Decay;
                voice.envelopeWait = 0u;
            }
            break;
        case Voice::Phase::Decay:
            if (voice.level <= sustainLevel)
            {
                voice.phase = Voice::Phase::Sustain;
                voice.envelopeWait = 0u;
            }
            break;
        case Voice::Phase::Release:
            if (voice.level <= 0)
            {
                voice.active = false;
                voice.phase = Voice::Phase::Off;
            }
            break;
        default:
            break;
        }
    }

    void Spu2::mixSample()
    {
        int32_t inputLeft = 0;
        int32_t inputRight = 0;
        for (uint32_t core = 0; core < 2u; ++core)
        {
            Core &state = m_cores[core];
            const uint32_t mixLeft = coreReg(core, kVoiceMixLeft) | (coreReg(core, kVoiceMixLeft + 2u) << 16u);
            const uint32_t mixRight = coreReg(core, kVoiceMixRight) | (coreReg(core, kVoiceMixRight + 2u) << 16u);
            int32_t left = 0;
            int32_t right = 0;
            for (uint32_t v = 0; v < 24u; ++v)
            {
                Voice &voice = state.voices[v];
                if (!voice.active)
                    continue;
                stepEnvelope(core, v, voice);
                advanceSample(core, v, voice);
                if (!voice.active)
                    continue;
                const int32_t interpolated =
                    voice.previous + (((static_cast<int32_t>(voice.current) - voice.previous) * static_cast<int32_t>(voice.counter)) >> 12);
                const int32_t sample = (interpolated * voice.level) >> 15;
                voice.volumeLeft = decodeVolume(static_cast<uint16_t>(coreReg(core, v * 0x10u)), voice.volumeLeft);
                voice.volumeRight = decodeVolume(static_cast<uint16_t>(coreReg(core, v * 0x10u + 2u)), voice.volumeRight);
                if ((mixLeft >> v) & 1u)
                    left += (sample * voice.volumeLeft) >> 15;
                if ((mixRight >> v) & 1u)
                    right += (sample * voice.volumeRight) >> 15;
            }
            if (!state.admaLeft.empty())
            {
                left += state.admaLeft.front();
                right += state.admaRight.front();
                state.admaLeft.pop_front();
                state.admaRight.pop_front();
            }
            // Core 0 output feeds core 1's input.
            left += inputLeft;
            right += inputRight;

            int32_t masterLeft = 0x7FFF;
            int32_t masterRight = 0x7FFF;
            if (state.masterVolumeWritten)
            {
                const uint32_t base = kCoreVolumeBase + core * kCoreVolumeStride;
                masterLeft = decodeVolume(reg(base), 0x7FFF);
                masterRight = decodeVolume(reg(base + 2u), 0x7FFF);
            }
            inputLeft = clamp16((left * masterLeft) >> 15);
            inputRight = clamp16((right * masterRight) >> 15);
        }

        std::lock_guard<std::mutex> lock(m_outputMutex);
        const size_t capacity = m_output.size() / 2u;
        if (m_outputCount == capacity)
        {
            m_outputRead = (m_outputRead + 1u) % capacity;
            --m_outputCount;
            ++m_outputTrace.dropped;
        }
        const size_t write = (m_outputRead + m_outputCount) % capacity;
        m_output[write * 2u] = static_cast<int16_t>(inputLeft);
        m_output[write * 2u + 1u] = static_cast<int16_t>(inputRight);
        ++m_outputCount;
    }

    void Spu2::advanceTo(uint64_t cycle)
    {
        static const bool trace = std::getenv("PS2X_SPU2_TRACE") != nullptr;
        if (trace && m_clockStarted && cycle / (kIopCyclesPerSample * 48000u * 2u) != m_nextSampleCycle / (kIopCyclesPerSample * 48000u * 2u))
        {
            for (uint32_t c = 0; c < 2u; ++c)
            {
                uint32_t active = 0u;
                for (const Voice &v : m_cores[c].voices)
                    active += v.active ? 1u : 0u;
                const uint32_t vb = kCoreVolumeBase + c * kCoreVolumeStride;
                std::fprintf(stderr, "[spu2] core%u active=%u keyons=%u mvol=%04x/%04x evol=%04x/%04x avol=%04x/%04x bvol=%04x/%04x mmix=%04x vmix=%04x%04x/%04x%04x attr=%04x admaq=%zu\n",
                             c, active, m_cores[c].keyOnCount, reg(vb), reg(vb + 2u), reg(vb + 4u), reg(vb + 6u), reg(vb + 8u), reg(vb + 10u),
                             reg(vb + 12u), reg(vb + 14u), coreReg(c, 0x198u), coreReg(c, kVoiceMixLeft + 2u), coreReg(c, kVoiceMixLeft),
                             coreReg(c, kVoiceMixRight + 2u), coreReg(c, kVoiceMixRight), coreReg(c, 0x19Au), m_cores[c].admaLeft.size());
            }
        }
        if (!m_clockStarted)
        {
            m_clockStarted = true;
            m_nextSampleCycle = cycle;
            return;
        }
        uint32_t produced = 0u;
        while (m_nextSampleCycle + kIopCyclesPerSample <= cycle)
        {
            mixSample();
            m_nextSampleCycle += kIopCyclesPerSample;
            if (++produced >= 48000u)
            {
                // Over a second behind (host stall): resynchronise.
                m_nextSampleCycle = cycle;
                break;
            }
        }
    }

    size_t Spu2::readOutput(int16_t *destination, size_t frames)
    {
        static const bool s_trace = std::getenv("PS2X_AUDIO_TRACE") != nullptr || std::getenv("PS2X_FPS_TRACE") != nullptr;
        OutputTrace report;
        size_t count = 0u;
        {
            std::lock_guard<std::mutex> lock(m_outputMutex);
            const size_t capacity = m_output.size() / 2u;
            count = std::min(frames, m_outputCount);
            if (s_trace)
            {
                OutputTrace &trace = m_outputTrace;
                trace.requested += frames;
                trace.silent += frames - count;
                trace.minLevel = std::min(trace.minLevel, m_outputCount);
                trace.maxLevel = std::max(trace.maxLevel, m_outputCount);
                if (trace.requested >= 48000u * 5u)
                {
                    report = trace;
                    trace = {};
                }
            }
            for (size_t i = 0; i < count; ++i)
            {
                destination[i * 2u] = m_output[m_outputRead * 2u];
                destination[i * 2u + 1u] = m_output[m_outputRead * 2u + 1u];
                m_outputRead = (m_outputRead + 1u) % capacity;
            }
            m_outputCount -= count;
        }
        if (report.requested != 0u)
        {
            const double requested = static_cast<double>(report.requested);
            std::fprintf(stderr, "[audio] last %.1f s: %.2f%% silence (mixing behind), %.2f%% dropped (mixing ahead), buffer %.0f-%.0f ms\n",
                         requested / 48000.0, 100.0 * static_cast<double>(report.silent) / requested,
                         100.0 * static_cast<double>(report.dropped) / requested,
                         static_cast<double>(report.minLevel) / 48.0, static_cast<double>(report.maxLevel) / 48.0);
        }
        return count;
    }
}
