#include "runtime/ps2_vu1_compiled.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace ps2x::vu1c
{
    namespace
    {
        std::vector<const Program *> &registry()
        {
            static std::vector<const Program *> s_programs;
            return s_programs;
        }
    }

    void registerProgram(const Program *program)
    {
        if (program)
            registry().push_back(program);
    }

    const Program *findProgram(const uint8_t *microMemory, uint32_t size)
    {
        std::vector<const Program *> &programs = registry();
        for (size_t index = 0; index < programs.size(); ++index)
        {
            const Program *program = programs[index];
            if (program->codeBytes > size || std::memcmp(program->code, microMemory, program->codeBytes) != 0)
                continue;
            // Games swap between a few programs; keep the latest match first.
            if (index != 0u)
                std::rotate(programs.begin(), programs.begin() + static_cast<std::ptrdiff_t>(index),
                            programs.begin() + static_cast<std::ptrdiff_t>(index) + 1);
            return program;
        }
        return nullptr;
    }

    uint32_t programCount()
    {
        return static_cast<uint32_t>(registry().size());
    }
}
