#pragma once

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace ps2x::iop::detail
{
    // Formats an IOP printf-style string. Variadic arguments follow the o32 ABI:
    // the registers after the format (starting at `firstArgReg`), then the stack
    // from sp+16 + 4 * register-slot.
    inline std::string formatGuestString(const IopMemory &memory, const IopCpuState &cpu, uint32_t formatAddress,
                                         int firstArgReg)
    {
        const std::string format = memory.readString(formatAddress, 2048u);
        int slot = firstArgReg - 4;
        const auto nextArg = [&]() -> uint32_t
        {
            const int current = slot++;
            if (current < 4)
                return cpu.gpr[4 + current];
            return memory.read32(cpu.gpr[29] + 16u + 4u * static_cast<uint32_t>(current - 4));
        };

        std::string out;
        out.reserve(format.size() + 32u);
        for (size_t i = 0; i < format.size(); ++i)
        {
            if (format[i] != '%')
            {
                out += format[i];
                continue;
            }

            std::string spec = "%";
            size_t j = i + 1;
            while (j < format.size() && std::string_view("-+ #0123456789.lhz").find(format[j]) != std::string_view::npos)
            {
                if (format[j] != 'l' && format[j] != 'h' && format[j] != 'z')
                    spec += format[j];
                ++j;
            }
            if (j >= format.size())
            {
                out += format.substr(i);
                break;
            }

            const char conversion = format[j];
            char buffer[128];
            buffer[0] = '\0';
            switch (conversion)
            {
            case '%':
                out += '%';
                break;
            case 'd':
            case 'i':
                std::snprintf(buffer, sizeof(buffer), (spec + 'd').c_str(), static_cast<int32_t>(nextArg()));
                break;
            case 'u':
            case 'x':
            case 'X':
            case 'o':
            case 'c':
                std::snprintf(buffer, sizeof(buffer), (spec + conversion).c_str(), nextArg());
                break;
            case 'p':
                std::snprintf(buffer, sizeof(buffer), "0x%08x", nextArg());
                break;
            case 's':
            {
                const std::string text = memory.readString(nextArg(), 512u);
                std::snprintf(buffer, sizeof(buffer), (spec + 's').c_str(), text.c_str());
                break;
            }
            default:
                out += format.substr(i, j - i + 1);
                break;
            }
            out += buffer;
            i = j;
        }
        return out;
    }
}
