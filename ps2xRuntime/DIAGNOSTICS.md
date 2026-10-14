# Runtime diagnostics

These debugging switches are read once at startup from the environment (some are initialized on first use).

## CPU/memory

| Variable | Syntax | What it does |
|---|---|---|
| `PS2X_WATCH` | `<addr>[:<bytes>[:<vsync>]]` | Reports writes overlapping the watched EE RAM range from the selected VSync. Address uses C-style base detection; omitted byte length defaults to 4 and omitted VSync to 0. Prints `[watch]` with registers and a stack scan; capped at 4,000 reports. |
| `PS2X_RDRAM_DUMP` | `<vsync>:<path>` | Writes 32 MiB of EE RAM to the file at that VSync; prints `[rdram-dump]`. |
| `PS2X_PROFILE` | `<delaySeconds>:<seconds>:<path>` | Windows only. A sampling profiler starts `<delaySeconds>` after start-up, records the instruction pointer of every thread about once per millisecond for `<seconds>`, then writes the 40 busiest functions of each thread to `<path>`; prints `[profile]`. |
| `PS2X_PEEK` | `<hex-addr>[:<words>][,...]` | At each thread dump, prints memory words for each address. Word count defaults to 1 and is capped at 64 per address. |
| `PS2X_THREAD_DUMP` | `<vsync-interval>` | Prints guest thread state every N VSyncs as `[thread-dump]`; 0 or omitted disables it. |
| `PS2X_WAKE_TRACE` | `<tid>` | Logs wakeups directed at the thread ID as `[wake]`; omitted disables tracing. |

## DMA/VIF/VU1

| Variable | Syntax | What it does |
|---|---|---|
| `PS2X_DMA_WATCH` | `<lo>:<hi>[:<vsync>]` | Watches DMA tag payloads overlapping the half-open guest address range `[lo,hi)` for 60 VSyncs from the optional start (default 0). Prints `[dma]`; capped at 20,000 reports. |
| `PS2X_VIF1_WATCH` | `<qword>` | Index (0-1023) of a VU1 data-memory qword. Logs every UNPACK that writes it (with the first four payload words, plus eight qwords of a V4-32 payload), every UNPACK dropped for lack of data, and every MSCAL/MSCNT/FLUSH, all as `[vif1]`. Capped at 20,000 lines. |
| `PS2X_VU1_TRACE` | `<pc>:<pairs>[:<skip>[:<lower>]]` | Once VU1 reaches micro address `<pc>` (hex) for the `<skip>`+1-th time (decimal, default 0), logs the next `<pairs>` (decimal) executed instruction pairs as `[vu1]` with CLIP, Q, VI and VF writes; `<lower>` (hex) only matches where the lower instruction word equals it. Fires once. |
| `PS2X_VU1_NOCLIP` | Any value | Forces VU1 clip flags to inside; enabled by presence, including an empty value. No output. |
| `PS2X_VU1_DUMP` | `<vsync>:<path>` | Writes VU1 data memory to `<path>` and micro memory to `<path>.code` at that VSync; prints `[vu1-dump]`. |
| `PS2X_VU1_STATS` | Any value | About once a second prints `[vu1-stats]`: VU1 micro program runs, instruction pairs and cycles per second, the share of wall time spent in VU1 and host nanoseconds per pair. |
| `PS2X_VU1_CAPTURE` | `<dir>` | Identifies the VU1 micro memory image behind every MSCAL by content hash, saves each new image once as `<dir>/vu1code_<hash>.bin` (input for `ps2xRuntime/tools/vu1rec.py`) and prints `[vu1-capture]` call counts per image and start PC every few seconds. |
| `PS2X_VU1_COMPILED` | `0` | Disables the micro programs translated to C++ by `tools/vu1rec.py`, so every VU1 run uses the interpreter. Translated programs are used by default when micro memory matches their image. |
| `PS2X_VU1_VERIFY` | Any value | Runs every translated micro program and then the interpreter on the same input, keeps the interpreter's result and reports differences in registers, flags, cycles, VU memory or GIF output as `[vu1-verify] MISMATCH ...`, plus a running total every 4096 runs. |

## GS/video

| Variable | Syntax | What it does |
|---|---|---|
| `PS2X_GS_DRAW_TRACE` | `<vsync>[:<count>]` | Traces submitted GS primitives and image transfers starting at the VSync, for 2 VSyncs by default (explicit count replaces 2). Prints `[gs:draw]` and `[gs:verts]` details. |
| `PS2X_GS_SKIP_TBP` | `<tbp>[,<tbp>...]` | Skips textured primitives sampling listed TBP0 texture bases; values use C-style base detection. |
| `PS2X_FPS_TRACE` | Any value | Prints the guest frame rate about once a second as `[fps] game frames/s=... vsync=...`; the scripts use the `vsync=` field to follow progress. Enabled by presence. |
| `PS2X_FRAME_DUMP_DIR` | `<directory>` | Saves presented frames as PNG files in the directory, starting at the configured frame interval. |
| `PS2X_FRAME_DUMP_INTERVAL` | `<vsyncs>` | Sets frame dump spacing; default 120, minimum 1. |
| `PS2X_FRAME_DUMP_FROM` | `<vsync>` | Sets first eligible frame dump VSync; default 0. |
| `PS2X_DEINTERLACE` | `0` or other | Disables deinterlacing only when the first character is `0`; otherwise enabled by default. |
| `PS2X_GS_ASYNC` | `1` or other | Enables asynchronous GS execution only when the first character is `1`; default off. |
| `PS2X_GS_THREADS` | `<count>` | Sets GS worker band count, clamped to 1-16; default is half the hardware concurrency, at least 1. |
| `PS2X_SEPARATE_ZBUFFER` | `1` or other | Enables a separate depth plane only when the first character is `1`; default off. |
| `PS2X_GS_STATS` | Any value | About once a second prints `[gs-stats]`: queued draws, worker batches, rasterised lists (`flushes`, of which `inline` ran on the worker alone) and why each list was cut, texture tiles decoded ahead (`predecode`) or on first use, time waiting on the decode lock, busy shares of the worker and band threads, and how long the EE waited for a full queue, a pending presentation or a drain. |
| `PS2X_GS_RECORD` | `<path>[:<vsync>[:<count>]]` | Records every call the GS frontend makes on the CPU backend, from the first presentation at or after VSync `<vsync>` (default 0) for `<count>` presentations (default 30), after a snapshot of VRAM and the palette; prints `[gs-record]`. `ps2_gs_replay <path> [--hash] [--repeat <n>]` (built with the GS cache tests) replays it, timing the rasteriser and hashing VRAM to compare builds. |

## IOP/audio/CD

| Variable | Syntax | What it does |
|---|---|---|
| `PS2X_IOP_HOST_SYNC` | `0` or other | Disables IOP host synchronization when first character is `0`; enabled by default. |
| `PS2X_IOP_MODULE_ARGS` | `<path-fragment>:<arg>[,<arg>...][;...]` | Appends arguments to matching IOP module loads; path fragment matching is case-insensitive. |
| `PS2X_IOPHEAP_TRACE` | Any value | Logs IOP heap allocation and free operations with `[iopheap]`; enabled by presence. |
| `PS2X_SIF_TRACE` | `<level>` | Logs SIF transfers as `[sif:dma]`; level 2 also prints the first 64 payload bytes. Default 0. |
| `PS2X_AUDIO_DUMP` | `<file>` | Writes played audio as 48 kHz stereo signed 16-bit PCM to the file (opened for overwrite). |
| `PS2X_CD_TRACE` | Any value | Logs every CD read, including VSync, with `[cd:read]`; enabled by presence. |
| `PS2X_MPEG_TRACE` | Any value | Logs MPEG calls and waits with `[mpeg:call]` and `[mpeg:wait]`; enabled by presence. |
| `PS2X_MPEG_MB_RASTER` | `1` or other | Uses raster macroblock decode order when the first character is `1`; otherwise uses the default ordering. |

## Input/automation

| Variable | Syntax | What it does |
|---|---|---|
| `PS2X_PAD_SCRIPT` | `<from>-<to>:<BUTTON>[+<BUTTON>...][;...]` | Holds named pad buttons on port 0 for inclusive VSync ranges. Names include directions, face buttons, START/SELECT, and L1/R1/L2/R2/L3/R3. |

## Examples

```bash
PS2X_WATCH=0x1759b78 ./ps2EntryRunner.exe SLUS_212.97
PS2X_GS_DRAW_TRACE=120:4 ./ps2EntryRunner.exe SLUS_212.97
PS2X_DMA_WATCH=0x100000:0x110000:120 ./ps2EntryRunner.exe SLUS_212.97
PS2X_VU1_TRACE=0x1234:32:0 ./ps2EntryRunner.exe SLUS_212.97
PS2X_RDRAM_DUMP=240:/tmp/ee-ram.bin ./ps2EntryRunner.exe SLUS_212.97
PS2X_PAD_SCRIPT='100-140:CROSS+START' ./ps2EntryRunner.exe SLUS_212.97
```
