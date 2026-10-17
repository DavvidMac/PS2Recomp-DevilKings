# Runtime diagnostics

These debugging switches are read once at startup from the environment (some are initialized on first use).

## CPU/memory

| Variable | Syntax | What it does |
|---|---|---|
| `PS2X_WATCH` | `<addr>[:<bytes>[:<vsync>]]` | Reports writes overlapping the watched EE RAM range from the selected VSync. Address uses C-style base detection; omitted byte length defaults to 4 and omitted VSync to 0. Prints `[watch]` with registers and a stack scan; capped at 4,000 reports. |
| `PS2X_RDRAM_DUMP` | `<vsync>:<path>` | Writes 32 MiB of EE RAM to the file at that VSync; prints `[rdram-dump]`. |
| `PS2X_PROFILE` | `<delaySeconds>:<seconds>:<path>` | Windows only. A sampling profiler starts `<delaySeconds>` after start-up, records the instruction pointer of every thread about once per millisecond for `<seconds>`, then writes the 40 busiest functions of each thread, and the busiest source lines of its top four functions, to `<path>`; prints `[profile]`. |
| `PS2X_PROFILE_LINES` | `<count>` | Number of source lines `PS2X_PROFILE` lists for each of the busiest functions; default 12. |
| `PS2X_PEEK` | `<hex-addr>[:<words>][,...]` | At each thread dump, prints memory words for each address. Word count defaults to 1 and is capped at 64 per address. |
| `PS2X_THREAD_DUMP` | `<vsync-interval>` | Prints guest thread state every N VSyncs as `[thread-dump]`; 0 or omitted disables it. |
| `PS2X_WAKE_TRACE` | `<tid>` | Logs wakeups directed at the thread ID as `[wake]`; omitted disables tracing. |

## DMA/VIF/VU1

| Variable | Syntax | What it does |
|---|---|---|
| `PS2X_DMA_WATCH` | `<lo>:<hi>[:<vsync>]` | Watches DMA tag payloads overlapping the half-open guest address range `[lo,hi)` for 60 VSyncs from the optional start (default 0). Prints `[dma]`; capped at 20,000 reports. |
| `PS2X_DMA_TIMING` | Any value | About every five seconds prints `[dma-timing]`: GIF, VIF0 and VIF1 DMA kicks per second, their data per second and the host time per second the EE thread spent starting and processing them (for VIF1 that includes the VU1 programs the transfer starts and the GIF packets they send), and with `PS2X_VIF1_THREAD` how busy the VIF1 thread was and how long the EE waited for it. |
| `PS2X_VIF1_THREAD` | `0` or `1` | The game runtime processes VIF1 DMA transfers on a thread of their own (VIF commands, the VU1 programs they start and their PATH1/PATH2 output) while the EE goes on, as on the console; `0` processes them on the EE thread instead. A bare `PS2Memory`, as in the tests, only uses the thread with `1`. The transfer is copied when it starts; every EE access to the GIF, VIF1, VIF1/GIF DMA channel, DMA controller or GS privileged registers (CSR aside), to VU1 memory and the GS helpers first waits for the queued transfers and completes them. |
| `PS2X_DMA_TRACE` | `<vsync>[:<vsyncs>]` | From VSync `<vsync>` for `<vsyncs>` VSyncs (default 4) prints a timeline as `[dma-trace]` (microseconds, VSync): `sceDmaSend`/`sceDmaSync` calls with channel and caller, reads and writes of the GIF/VIF and DMA controller registers, the end of every DMA kick with its size, GS privileged register writes and `sceGsSyncV` waits. |
| `PS2X_VIF1_WATCH` | `<qword>` | Index (0-1023) of a VU1 data-memory qword. Logs every UNPACK that writes it (with the first four payload words, plus eight qwords of a V4-32 payload), every UNPACK dropped for lack of data, and every MSCAL/MSCNT/FLUSH, all as `[vif1]`. Capped at 20,000 lines. |
| `PS2X_VIF_UNPACK_GENERIC` | `1` | Makes every VIF1 UNPACK take the generic per-vector loop instead of the direct per-format copy used for unmasked mode-0 UNPACKs with CL >= WL, to compare the two. |
| `PS2X_VU1_TRACE` | `<pc>:<pairs>[:<skip>[:<lower>]]` | Once VU1 reaches micro address `<pc>` (hex) for the `<skip>`+1-th time (decimal, default 0), logs the next `<pairs>` (decimal) executed instruction pairs as `[vu1]` with CLIP, Q, VI and VF writes; `<lower>` (hex) only matches where the lower instruction word equals it. Fires once. |
| `PS2X_VU1_NOCLIP` | Any value | Forces VU1 clip flags to inside; enabled by presence, including an empty value. No output. |
| `PS2X_VU1_DUMP` | `<vsync>:<path>` | Writes VU1 data memory to `<path>` and micro memory to `<path>.code` at that VSync; prints `[vu1-dump]`. |
| `PS2X_VU1_STATS` | Any value | About once a second prints `[vu1-stats]`: VU1 micro program runs, instruction pairs and cycles per second, the share of wall time spent in VU1, host nanoseconds per pair (interpreted runs only) and per VU1 cycle (all runs, worker threads aside). With `PS2X_VU1_THREADS` also `[vu1-par]`: runs started on the workers, put into effect as they were and run again (by reason: the PC, inputs or VU memory left by the runs ahead differed, an early exit, an unfinished state), runs kept on the VIF1 thread (by reason), the VIF1 thread's wait for the workers, per entry point how often its runs had to be run again, and a sample of the VU memory reads that made runs stale (`reads qword ... written by the run N back`). |
| `PS2X_VU1_THREADS` | `<n>` | Starts translated VU1 runs on `<n>` worker threads (up to 16; `0`: none; unset: 3 with 8 or more logical CPUs, 2 with 6 or 7, else none) as soon as the VIF reaches their MSCAL/MSCNT, on a copy of VU1 data memory, from entry points whose inputs `tools/vu1rec.py` lists. They take effect in program order on the VIF1 thread; a run whose entry PC, inputs (registers and flags it reads before writing them) or VU memory reads differ from what the runs ahead of it left is run again from the exact state (on a worker when it can), so the result is the in-order one. About +10% in Devil Kings battles with `3` and depth 10. PATH2 packets and VIF writes to VU1 memory keep their order with the runs. Off with `PS2X_VU1_VERIFY`. |
| `PS2X_VU1_DEPTH` | `<n>` | With `PS2X_VU1_THREADS`, how many runs may be started and not yet in effect (2 to 64; default 10). |
| `PS2X_VU1_PARALLEL_CHECK` | `1` | With `PS2X_VU1_THREADS`, runs every worker run again in order on the VIF1 thread before it takes effect and reports differences in registers, flags, cycles, VU memory or GIF output as `[vu1-par] CHECK MISMATCH ...`; the counts go with `PS2X_VU1_STATS`. Slow. |
| `PS2X_VU1_CAPTURE` | `<dir>` | Identifies the VU1 micro memory image behind every MSCAL by content hash and saves each new image that no translated program covers once as `<dir>/vu1code_<hash>.bin`; prints `[vu1-capture]` call counts per image and start PC every few seconds. Translate the images with `python -I ps2xRuntime/tools/vu1rec.py --prune ps2xRuntime/src/runner/vu1 <dir>/*.bin` (one file per distinct program; `--prune` drops programs no image produces, so pass them all). |
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
| `PS2X_GS_THREADS` | `<count>` | Sets the number of GS band threads, clamped to 1-16; default is three quarters of the hardware concurrency, at least 1. With 1, the worker draws everything itself. |
| `PS2X_GS_NO_DRAW` | `1` | The CPU backend draws nothing: the frame stays blank, but the game runs, to measure what everything other than drawing costs. |
| `PS2X_STDOUT` | `1` | Android: keeps stdout (where `RUNTIME_LOG` writes) going to logcat; by default only stderr goes there and to `ultima-sessao.log` next to the ELF. On Android the runtime also reads `ps2x.env` next to the ELF (`NAME=value` lines) before anything else, since an app has no command line. |
| `PS2X_RENDERER` | `vulkan` | Shows the paraLLEl-GS scanout (the GS emulated in Vulkan compute) instead of the CPU backend's frame, in builds configured with `-DPS2X_PGS_DIR=<paraLLEl-GS checkout>`; the settings menu sets it with Video > Renderer. The GS frontend keeps parsing the stream but the CPU backend draws nothing (a GS->EE transfer copies paraLLEl-GS's VRAM into it first). Falls back to the software renderer if Vulkan does not start. Read at start-up. |
| `PS2X_PGS_DUAL` | `1` | With the Vulkan renderer, keeps the CPU backend drawing as well (to compare the two, or to rule paraLLEl-GS out of a bug). |
| `PS2X_PGS_SCALE` | `1`, `1aa`, `2`, `2aa`, `4` | Internal resolution of the Vulkan renderer: the scanout is 1, 2 or 4 times the native size (1, 4 or 16 samples per pixel); `1aa` and `2aa` draw 4 and 8 samples per pixel at 1x and 2x, which smooths edges. Video > Internal resolution in the settings menu. Read at start-up. |
| `PS2X_PGS_TRACE` | any value | Logs the display registers the Vulkan renderer scans out and the scanout size every 300 swaps. |
| `PS2X_SEPARATE_ZBUFFER` | `1` or other | Enables a separate depth plane only when the first character is `1`; default off. |
| `PS2X_GS_STATS` | Any value | About once a second prints `[gs-stats]`: queued draws, worker batches, rasterised lists (`flushes`, of which `inline` ran on the worker alone) and why each list was cut, texture tiles decoded ahead (`predecode`) or on first use, time waiting on the decode lock, busy shares of the worker and band threads, and how long the EE waited for a full queue, a pending presentation or a drain. |
| `PS2X_GS_RECORD` | `<path>[:<vsync>[:<count>]]` | Records every call the GS frontend makes on the CPU backend, from the first presentation at or after VSync `<vsync>` (default 0) for `<count>` presentations (default 30), after a snapshot of VRAM and the palette; prints `[gs-record]`. `ps2_gs_replay <path> [--hash] [--repeat <n>]` (built with the GS cache tests) replays it, timing the rasteriser and hashing VRAM to compare builds. |

## IOP/audio/CD

| Variable | Syntax | What it does |
|---|---|---|
| `PS2X_IOP_HOST_SYNC` | `0` or other | Disables IOP host synchronization when first character is `0`; enabled by default. |
| `PS2X_IOP_MODULE_ARGS` | `<path-fragment>:<arg>[,<arg>...][;...]` | Appends arguments to matching IOP module loads; path fragment matching is case-insensitive. |
| `PS2X_IOP_NATIVE` | `0` | Keeps every IOP routine interpreted. By default, hot routines of known IOP libraries whose code matches word for word run natively (the IOP log says `[IOP] native <module>+<offset>`); first one: the CRI ADX mixer's clamp to 16 bits, about 30% of the IOP's interpreted instructions in a Devil Kings battle. |
| `PS2X_IOP_PC_STATS` | Any value | Samples the IOP program counter every 1024 instructions and about every five seconds prints `[iop-pc]` with the IOP instruction rate and the 16 busiest addresses, each with the module (IRX) and offset holding it. |
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
| `PS2X_PAD_SCRIPT_CLOCK` | `frames` | Times `PS2X_PAD_SCRIPT` in game frames (display swaps) instead of VSyncs, so a script recorded on one host still lands on the same screens on a slower one, where loading takes more VSyncs. |
| `PS2X_POKE` | `<addr>[.b/.h]=<value>[,...]` (hex) | Writes each value to EE RAM on every VSync, like a CodeBreaker constant-write code: 32 bits, or 8/16 bits with `.b`/`.h`; prints each as `[poke]` at startup. |
| `PS2X_WIDESCREEN` | `16:9`, `21:9` or a factor | Asks the game override for a wider 3D view (Devil Kings: scales X of the view-to-screen and camera clip matrices after each projection set-up); the settings menu sets it with Video > Aspect 16:9 / 21:9, which also shows the picture at that shape. Read at start-up. Experimental: in Devil Kings battles the main projection stays 4:3 and only the VU1 clipper's fans take the scale, leaving holes in the ground and a patched sky. |
| `PS2X_WIDESCREEN_HUD` | unset (centred), `edges` or `stretch` | Where the 2D HUD goes when a game override widened the view (Video > HUD no widescreen): centred keeps the 4:3 HUD in the middle; `edges` moves each piece of HUD (the vertices between two PRIM writes) whole to the nearer side of the screen, and centres everything for a few vsyncs after a piece as wide as a menu panel; `stretch` leaves it as drawn. The override says what is HUD (Devil Kings: PATH2, Z 0xFFFF, textures from TBP0 8192, which leaves out its blur pass over the frame); the X of those vertices is rewritten on a copy of the GIF packet, and pieces spanning the whole frame (fades, bands) are left alone. Only frames with a 3D scene (PATH1 or XGKICK drawing into that frame buffer in the last 4 vsyncs) are touched and shown wide; a 2D screen (menu, map, movie) is left as drawn and shown at 4:3 with bars, so its pieces stay lined up with its background. Read at start-up. |
| `PS2X_WIDESCREEN_HUD_TRACE` | `<first vsync>:<count>[,...]` | One `[hud]` line per GIF packet with primitives in those vsync windows: each run (the vertices between two PRIM writes) as `H` (HUD) or `-`, its vertex count, screen X range, frame buffer and texture base (`t-1` = untextured). With any value, also a `[hud] t=N shown at wide/4:3` line when the shown frame changes between a 3D scene and a 2D screen. For tuning a game's HUD rule. |
| `PS2X_SKIP_INTRO` | `1` | Asks the game override to skip the start-up logos (Devil Kings: straight from the memory card notice to the opening). Read at start-up. |
| `PS2X_SETTINGS` | `<path>` or `off` | Player settings file read at start-up (default `settings.ini` next to the ELF) and written by the in-game menu (F1; the debug panel moved to F2). Settings with a `PS2X_*` variable reach the runtime through it, and a variable already set in the environment wins over the file; `off` ignores the file (use it in test scripts). Cheat groups: each `[section]` of `cheats/*.pnach` next to the ELF (PCSX2 `patch=` lines) is a switch in the menu, written on every VSync while on. |

## Examples

```bash
PS2X_WATCH=0x1759b78 ./ps2EntryRunner.exe SLUS_212.97
PS2X_GS_DRAW_TRACE=120:4 ./ps2EntryRunner.exe SLUS_212.97
PS2X_DMA_WATCH=0x100000:0x110000:120 ./ps2EntryRunner.exe SLUS_212.97
PS2X_VU1_TRACE=0x1234:32:0 ./ps2EntryRunner.exe SLUS_212.97
PS2X_RDRAM_DUMP=240:/tmp/ee-ram.bin ./ps2EntryRunner.exe SLUS_212.97
PS2X_PAD_SCRIPT='100-140:CROSS+START' ./ps2EntryRunner.exe SLUS_212.97
```
