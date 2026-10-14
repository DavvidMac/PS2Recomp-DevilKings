#pragma once

#include "iop_spu2.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ps2x::iop::detail
{
    class IopMemory
    {
    public:
        static constexpr uint32_t RamSize = 2u * 1024u * 1024u;
        static constexpr uint32_t ScratchBase = 0x1F800000u;
        static constexpr uint32_t ScratchSize = 0x400u;
        static constexpr uint32_t HardwareBase = 0x1F801000u;
        static constexpr uint32_t HardwareEnd = 0x1F900000u;
        static constexpr uint32_t Spu2Base = 0x1F900000u;
        static constexpr uint32_t Spu2End = 0x1FA00000u;
        static constexpr uint32_t SifBase = 0x1D000000u;
        static constexpr uint32_t SifEnd = 0x1D001000u;
        static constexpr uint32_t HeapBase = 0x00120000u;
        static constexpr uint32_t HeapLimit = 0x001F0000u;

        struct Allocation
        {
            uint32_t address = 0;
            uint32_t size = 0;
        };

        struct DmaStart
        {
            int irq = 0;
            uint64_t delayCycles = 0;
        };

        IopMemory();

        void reset();

        // RAM accesses take the inline paths; everything else (scratchpad, hardware,
        // SPU2, unaligned or wrapping accesses) goes through the *Slow functions.
        [[nodiscard]] uint8_t read8(uint32_t address) const
        {
            const uint32_t phys = physicalAddress(address);
            if (phys < RamSize)
                return m_ram[phys];
            return read8Slow(address);
        }

        [[nodiscard]] uint16_t read16(uint32_t address) const
        {
            const uint32_t phys = physicalAddress(address);
            if (phys + 1u < RamSize)
            {
                uint16_t value;
                std::memcpy(&value, m_ram.data() + phys, sizeof(value));
                return value;
            }
            return read16Slow(address);
        }

        [[nodiscard]] uint32_t read32(uint32_t address) const
        {
            const uint32_t phys = physicalAddress(address);
            if ((phys & 3u) == 0u && phys + 3u < RamSize)
            {
                uint32_t value;
                std::memcpy(&value, m_ram.data() + phys, sizeof(value));
                return value;
            }
            return read32Slow(address);
        }

        void write8(uint32_t address, uint8_t value)
        {
            const uint32_t phys = physicalAddress(address);
            if (phys < RamSize)
            {
                m_ram[phys] = value;
                m_owned[phys] = 1u;
                return;
            }
            write8Slow(address, value);
        }

        void write16(uint32_t address, uint16_t value)
        {
            const uint32_t phys = physicalAddress(address);
            if (phys + 1u < RamSize)
            {
                std::memcpy(m_ram.data() + phys, &value, sizeof(value));
                std::memset(m_owned.data() + phys, 1, sizeof(value));
                return;
            }
            write16Slow(address, value);
        }

        void write32(uint32_t address, uint32_t value)
        {
            const uint32_t phys = physicalAddress(address);
            if ((phys & 3u) == 0u && phys + 3u < RamSize)
            {
                std::memcpy(m_ram.data() + phys, &value, sizeof(value));
                std::memset(m_owned.data() + phys, 1, sizeof(value));
                return;
            }
            write32Slow(address, value);
        }

        [[nodiscard]] bool readRam(uint32_t address, void *destination, size_t size) const;
        [[nodiscard]] bool writeRam(uint32_t address, const void *source, size_t size);
        [[nodiscard]] bool zeroRam(uint32_t address, size_t size);
        [[nodiscard]] bool ownsRamRange(uint32_t address, size_t size) const;
        [[nodiscard]] bool isHardwareAddress(uint32_t address) const;
        [[nodiscard]] std::string readString(uint32_t address, size_t limit = 1024u) const;

        [[nodiscard]] uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt);
        [[nodiscard]] bool freeAllocation(uint32_t address);
        [[nodiscard]] uint32_t maxFreeMemory() const;
        [[nodiscard]] std::optional<Allocation> allocationContaining(uint32_t address) const;

        [[nodiscard]] uint32_t interruptStatus() const noexcept { return m_interruptStatus; }
        [[nodiscard]] uint32_t interruptMask() const noexcept { return m_interruptMask; }
        [[nodiscard]] uint32_t interruptControl() const noexcept { return m_interruptControl; }
        void setInterruptStatus(uint32_t value) noexcept { m_interruptStatus = value; }
        void setInterruptMask(uint32_t value) noexcept { m_interruptMask = value; }
        void setInterruptControl(uint32_t value) noexcept { m_interruptControl = value & 1u; }

        [[nodiscard]] bool hasDmaStart() const noexcept { return m_dmaStart.has_value(); }
        [[nodiscard]] std::optional<DmaStart> takeDmaStart() noexcept;
        [[nodiscard]] std::span<const uint8_t> ram() const noexcept { return m_ram; }

        [[nodiscard]] static uint32_t physicalAddress(uint32_t address) noexcept
        {
            return address & 0x1FFFFFFFu;
        }

        // IOP cycle counter used to pace SPU2 AutoDMA transfers in real time.
        void setCycleCounter(const uint64_t *counter) noexcept { m_cycleCounter = counter; }

        [[nodiscard]] Spu2 &spu2() noexcept { return m_spu2; }

    private:
        [[nodiscard]] uint8_t read8Slow(uint32_t address) const;
        [[nodiscard]] uint16_t read16Slow(uint32_t address) const;
        [[nodiscard]] uint32_t read32Slow(uint32_t address) const;
        void write8Slow(uint32_t address, uint8_t value);
        void write16Slow(uint32_t address, uint16_t value);
        void write32Slow(uint32_t address, uint32_t value);
        [[nodiscard]] uint32_t readHardware32(uint32_t address) const;
        void writeHardware32(uint32_t address, uint32_t value);
        void markOwned(uint32_t address, size_t size);

        std::vector<uint8_t> m_ram;
        std::vector<uint8_t> m_owned;
        std::vector<uint8_t> m_scratch;
        std::unordered_map<uint32_t, uint32_t> m_hardware;
        std::vector<Allocation> m_allocations;
        uint32_t m_heapCursor = HeapBase;
        uint32_t m_interruptStatus = 0;
        uint32_t m_interruptMask = 0;
        uint32_t m_interruptControl = 1;
        std::optional<DmaStart> m_dmaStart;
        const uint64_t *m_cycleCounter = nullptr;
        struct SpuAutoDma
        {
            bool active = false;
            uint32_t madr = 0;
            uint64_t startCycle = 0;
            uint64_t bytes = 0;
        };
        std::array<SpuAutoDma, 2> m_spuAutoDma{};
        Spu2 m_spu2;
    };
}
