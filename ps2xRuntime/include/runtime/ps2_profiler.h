#ifndef PS2_PROFILER_H
#define PS2_PROFILER_H

// Diagnostics: PS2X_PROFILE=<delaySeconds>:<seconds>:<path> starts a sampling profiler that,
// <delaySeconds> after this call, records where every thread of the process is executing about
// once per millisecond for <seconds>, then writes the busiest functions per thread to <path>.
// Does nothing when the variable is unset or on platforms without a sampler.
void ps2StartSamplingProfilerFromEnv();

#endif
