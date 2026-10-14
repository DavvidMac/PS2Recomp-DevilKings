"""Translate a VU1 micro program image to C++ (ahead-of-time VU1 recompilation).

usage: python -I vu1rec.py [--prune] <out_dir> <image.bin> [<image.bin> ...]

Each 16 KB micro memory image (as written by PS2X_VU1_CAPTURE) becomes
<out_dir>/vu1prog_<name>.cpp, which registers itself with the runtime
(ps2_vu1_compiled.h). At MSCAL/MSCNT time VU1Interpreter runs the translated
code when micro memory matches the image up to the end of the translated code,
and the interpreter otherwise. <name> is a hash of those bytes, so images that
only differ after the code (leftovers of earlier uploads) share one file.
--prune deletes the other vu1prog_*.cpp files in <out_dir>, so pass every image.

The generated code reproduces VU1Interpreter exactly: instruction usage
(stall sources and latencies) is decoded here the same way as
decodeUpperUsage/decodeLowerUsage, and every instruction's arithmetic mirrors
ps2_vu1_upper.cpp / ps2_vu1_lower.cpp. Programs using anything not handled
here (EFU, R register, D/T breakpoints, branches in delay slots) are rejected
and stay on the interpreter.
"""
import os
import sys

FMAC_LATENCY = 4
ACC_FORWARD_LATENCY = 1


class Reject(Exception):
    pass


def lane(component):
    return 1 << (3 - component)


def DEST(i):
    return (i >> 21) & 0xF


def FT(i):
    return (i >> 16) & 0x1F


def FS(i):
    return (i >> 11) & 0x1F


def FD(i):
    return (i >> 6) & 0x1F


def VIT(i):
    return (i >> 16) & 0xF


def VIS(i):
    return (i >> 11) & 0xF


def VID(i):
    return (i >> 6) & 0xF


def IMM11(i):
    value = i & 0x7FF
    return value - 0x800 if value & 0x400 else value


class Usage:
    def __init__(self):
        self.vf_read = []          # [(reg, lanes)]
        self.vf_write = (0, 0)     # (reg, lanes)
        self.vi_read = 0
        self.vi_write = 0
        self.acc_read = 0
        self.acc_write = 0
        self.latency = 0
        self.vf_latency = 0
        self.vi_latency = 0
        self.pipeline = 'none'
        self.wait_q = False
        self.wait_p = False
        self.reads_clip = False
        self.writes_clip = False
        self.delays_branch = False
        self.reserved = False

    def add_vf_read(self, reg, lanes):
        if lanes == 0:
            return
        for index, (existing, existing_lanes) in enumerate(self.vf_read):
            if existing == reg:
                self.vf_read[index] = (reg, existing_lanes | lanes)
                return
        if len(self.vf_read) < 2:
            self.vf_read.append((reg, lanes))

    def add_vf_write(self, reg, lanes):
        if reg == 0 or lanes == 0:
            return
        current, current_lanes = self.vf_write
        if current == 0:
            self.vf_write = (reg, lanes)
        elif current == reg:
            self.vf_write = (reg, current_lanes | lanes)

    def read_vi(self, reg):
        if reg:
            self.vi_read |= 1 << reg

    def write_vi(self, reg):
        if reg:
            self.vi_write |= 1 << reg

    def vf_read_lanes(self, reg):
        for existing, lanes in self.vf_read:
            if existing == reg:
                return lanes
        return 0


PRODUCT_SUM_OPS = {0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x21, 0x23, 0x25, 0x27, 0x29, 0x2D, 0x2E}
PRODUCT_SUM_SPECIAL = {0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x21, 0x23, 0x25, 0x27, 0x29, 0x2D}


def upper_special(upper):
    return (upper & 3) | ((upper >> 4) & 0x7C)


def decode_upper(upper):
    usage = Usage()
    usage.pipeline = 'fmac'
    usage.latency = FMAC_LATENCY
    op = upper & 0x3F
    dest, fs, ft, fd = DEST(upper), FS(upper), FT(upper), FD(upper)
    if op <= 0x2F:
        usage.add_vf_read(fs, dest)
        usage.add_vf_write(fd, dest)
        if op <= 0x1B:
            usage.add_vf_read(ft, lane(op & 3))
        elif op >= 0x28:
            usage.add_vf_read(ft, 0xE if op == 0x2E else dest)
        if op in PRODUCT_SUM_OPS:
            usage.acc_read = dest
        return usage
    if op >= 0x3C:
        special = upper_special(upper)
        writes_acc = (special <= 0x0F or 0x18 <= special <= 0x1C or special == 0x1E or
                      0x20 <= special <= 0x2A or 0x2C <= special <= 0x2E)
        if writes_acc:
            usage.add_vf_read(fs, dest)
            if special <= 0x1B:
                usage.add_vf_read(ft, lane(special & 3))
            elif 0x28 <= special <= 0x2E:
                usage.add_vf_read(ft, 0xE if special == 0x2E else dest)
            usage.acc_write = dest
            if special in PRODUCT_SUM_SPECIAL:
                usage.acc_read = dest
        elif 0x10 <= special <= 0x17 or special == 0x1D:
            usage.add_vf_read(fs, dest)
            usage.add_vf_write(ft, dest)
        elif special == 0x1F:
            usage.add_vf_read(fs, 0xE)
            usage.add_vf_read(ft, 0x1)
            usage.writes_clip = True
        elif special not in (0x2F, 0x30):
            usage.reserved = True
        return usage
    usage.reserved = True
    return usage


def lower_special(lower):
    return (lower & 3) | ((lower >> 4) & 0x7C)


def decode_lower(lower):
    usage = Usage()
    if lower in (0, 0x8000033C):
        return usage
    op_hi = (lower >> 25) & 0x7F
    vf_t, vf_s = FT(lower), FS(lower)
    vi_t, vi_s, vi_d = VIT(lower), VIS(lower), VID(lower)
    dest = DEST(lower)
    if op_hi == 0x00:
        usage.pipeline, usage.latency = 'lsu', 4
        usage.read_vi(vi_s)
        usage.add_vf_write(vf_t, dest)
    elif op_hi == 0x01:
        usage.pipeline, usage.latency = 'lsu', 1
        usage.read_vi(vi_t)
        usage.add_vf_read(vf_s, dest)
    elif op_hi == 0x04:
        usage.pipeline, usage.latency = 'lsu', 4
        usage.read_vi(vi_s)
        usage.write_vi(vi_t)
    elif op_hi == 0x05:
        usage.pipeline, usage.latency = 'lsu', 1
        usage.read_vi(vi_s)
        usage.read_vi(vi_t)
    elif op_hi in (0x08, 0x09):
        usage.pipeline, usage.latency = 'ialu', 1
        usage.delays_branch = True
        usage.read_vi(vi_s)
        usage.write_vi(vi_t)
    elif op_hi in (0x10, 0x12, 0x13):
        usage.pipeline, usage.latency = 'ialu', 1
        usage.reads_clip = True
        usage.write_vi(1)
    elif op_hi == 0x11:
        usage.pipeline, usage.latency = 'fmac', FMAC_LATENCY
        usage.writes_clip = True
    elif op_hi in (0x14, 0x16, 0x17):
        usage.pipeline, usage.latency = 'ialu', 1
        usage.write_vi(vi_t)
    elif op_hi == 0x15:
        usage.pipeline, usage.latency = 'fmac', FMAC_LATENCY
    elif op_hi in (0x18, 0x1A, 0x1B):
        usage.pipeline, usage.latency = 'ialu', 1
        usage.read_vi(vi_s)
        usage.write_vi(vi_t)
    elif op_hi == 0x1C:
        usage.pipeline, usage.latency = 'ialu', 1
        usage.reads_clip = True
        usage.write_vi(vi_t)
    elif op_hi == 0x20:
        usage.pipeline = 'branch'
    elif op_hi == 0x21:
        usage.pipeline, usage.latency = 'branch', 1
        usage.write_vi(vi_t)
    elif op_hi == 0x24:
        usage.pipeline = 'branch'
        usage.read_vi(vi_s)
    elif op_hi == 0x25:
        usage.pipeline, usage.latency = 'branch', 1
        usage.read_vi(vi_s)
        usage.write_vi(vi_t)
    elif op_hi in (0x28, 0x29):
        usage.pipeline = 'branch'
        usage.read_vi(vi_s)
        usage.read_vi(vi_t)
    elif op_hi in (0x2C, 0x2D, 0x2E, 0x2F):
        usage.pipeline = 'branch'
        usage.read_vi(vi_s)
    elif op_hi == 0x40:
        direct = lower & 0x3F
        if direct in (0x30, 0x31, 0x34, 0x35):
            usage.pipeline, usage.latency = 'ialu', 1
            usage.delays_branch = True
            usage.read_vi(vi_s)
            usage.read_vi(vi_t)
            usage.write_vi(vi_d)
        elif direct == 0x32:
            usage.pipeline, usage.latency = 'ialu', 1
            usage.delays_branch = True
            usage.read_vi(vi_s)
            usage.write_vi(vi_t)
        elif direct < 0x3C:
            usage.reserved = True
        else:
            special = lower_special(lower)
            if special in (0x30, 0x31):
                usage.pipeline, usage.latency = 'fmac', 4
                usage.add_vf_read(vf_s, 0xF if special == 0x31 else dest)
                usage.add_vf_write(vf_t, dest)
            elif special in (0x34, 0x36):
                usage.pipeline, usage.latency, usage.vi_latency = 'lsu', 4, 1
                usage.delays_branch = True
                usage.read_vi(vi_s)
                usage.write_vi(vi_s)
                usage.add_vf_write(vf_t, dest)
            elif special in (0x35, 0x37):
                usage.pipeline, usage.latency = 'lsu', 1
                usage.delays_branch = True
                usage.read_vi(vi_t)
                usage.write_vi(vi_t)
                usage.add_vf_read(vf_s, dest)
            elif special == 0x38:
                usage.pipeline, usage.latency = 'fdiv', 7
                usage.add_vf_read(vf_s, lane((lower >> 21) & 3))
                usage.add_vf_read(vf_t, lane((lower >> 23) & 3))
            elif special == 0x39:
                usage.pipeline, usage.latency = 'fdiv', 7
                usage.add_vf_read(vf_t, lane((lower >> 23) & 3))
            elif special == 0x3A:
                usage.pipeline, usage.latency = 'fdiv', 13
                usage.add_vf_read(vf_s, lane((lower >> 21) & 3))
                usage.add_vf_read(vf_t, lane((lower >> 23) & 3))
            elif special == 0x3B:
                usage.pipeline = 'fdiv'
                usage.wait_q = True
            elif special == 0x3C:
                usage.pipeline, usage.latency = 'ialu', 1
                usage.delays_branch = True
                usage.add_vf_read(vf_s, lane((lower >> 21) & 3))
                usage.write_vi(vi_t)
            elif special == 0x3D:
                usage.pipeline, usage.latency = 'fmac', 4
                usage.read_vi(vi_s)
                usage.add_vf_write(vf_t, dest)
            elif special == 0x3E:
                usage.pipeline, usage.latency = 'lsu', 4
                usage.read_vi(vi_s)
                usage.write_vi(vi_t)
            elif special == 0x3F:
                usage.pipeline, usage.latency = 'lsu', 1
                usage.read_vi(vi_s)
                usage.read_vi(vi_t)
            elif special in (0x68, 0x69):
                usage.pipeline, usage.latency = 'ialu', 1
                usage.write_vi(vi_t)
            elif special == 0x6C:
                usage.pipeline, usage.latency = 'xgkick', 2
                usage.read_vi(vi_s)
            else:
                # R register, MFP, EFU and WAITP are not translated.
                usage.reserved = True
    else:
        usage.reserved = True
    return usage


BRANCH_OPS = {0x20, 0x21, 0x24, 0x25, 0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F}
MAC_READS = {0x18, 0x1A, 0x1B}            # FMEQ / FMAND / FMOR
STATUS_USES = {0x14, 0x15, 0x16, 0x17}    # FSEQ / FSSET / FSAND / FSOR

# FMAC operations (upper opcode, or upper special for the ACC forms): (operation, operand).
FMAC_TABLE = {}
for _base, _kind in ((0x00, 'add'), (0x04, 'sub'), (0x08, 'madd'), (0x0C, 'msub'), (0x18, 'mul')):
    for _bc in range(4):
        FMAC_TABLE[_base + _bc] = (_kind, 'bc')
FMAC_TABLE.update({0x1C: ('mul', 'q'), 0x1E: ('mul', 'i'), 0x20: ('add', 'q'), 0x21: ('madd', 'q'),
                   0x22: ('add', 'i'), 0x23: ('madd', 'i'), 0x24: ('sub', 'q'), 0x25: ('msub', 'q'),
                   0x26: ('sub', 'i'), 0x27: ('msub', 'i'), 0x28: ('add', 'v'), 0x29: ('madd', 'v'),
                   0x2A: ('mul', 'v'), 0x2C: ('sub', 'v'), 0x2D: ('msub', 'v'), 0x2E: ('op', 'v')})


def flag_fmac(upper):
    """True when the upper word is an FMAC operation that writes MAC and status (Exec::fmacVec)."""
    if DEST(upper) == 0:
        return False
    op = upper & 0x3F
    if op >= 0x3C:
        special = upper_special(upper)
        if special in (0x1D, 0x1F, 0x2F, 0x30) or 0x10 <= special <= 0x17:
            return False
        return special in FMAC_TABLE
    if 0x10 <= op <= 0x17 or op in (0x1D, 0x1F, 0x2B, 0x2F):
        return False
    return op in FMAC_TABLE


class Pair:
    def __init__(self, image, pc):
        self.pc = pc
        self.lower = int.from_bytes(image[pc:pc + 4], 'little')
        self.upper = int.from_bytes(image[pc + 4:pc + 8], 'little')
        self.i_bit = bool(self.upper & 0x80000000)
        self.e_bit = bool(self.upper & 0x40000000)
        self.d_bit = bool(self.upper & 0x10000000)
        self.t_bit = bool(self.upper & 0x08000000)
        self.up = decode_upper(self.upper)
        self.lo = decode_lower(self.lower) if not self.i_bit else Usage()
        up_reg = self.up.vf_write[0]
        self.shadow = 0
        self.suppressed = 0
        if up_reg and (self.lo.vf_read_lanes(up_reg) or self.lo.vf_write[0] == up_reg):
            self.shadow = up_reg
            if self.lo.vf_write[0] == up_reg:
                self.suppressed = up_reg
        self.op_hi = (self.lower >> 25) & 0x7F
        self.is_branch = not self.i_bit and self.op_hi in BRANCH_OPS

    def branch_target(self):
        return (self.pc + 8 + IMM11(self.lower) * 8) & 0x3FFF


def analyse(image):
    """Reachable pairs from PC 0, MSCNT resume points and BAL/JALR return points."""
    size = len(image)
    pairs = {}

    def pair(pc):
        if pc not in pairs:
            pairs[pc] = Pair(image, pc)
        return pairs[pc]

    def check_slot(slot):
        if slot.is_branch or slot.e_bit:
            raise Reject(f'branch or E bit in a delay slot at {slot.pc:04x}')
        if slot.d_bit or slot.t_bit:
            raise Reject(f'D/T bit at {slot.pc:04x}')
        if slot.up.reserved or slot.lo.reserved:
            raise Reject(f'unsupported instruction at {slot.pc:04x}: {slot.upper:08x} {slot.lower:08x}')

    roots = [0]
    returns = set()
    seen = set()
    while roots:
        start = roots.pop()
        work = [start]
        while work:
            pc = work.pop()
            if pc in seen or pc + 8 > size:
                continue
            seen.add(pc)
            p = pair(pc)
            if p.d_bit or p.t_bit:
                raise Reject(f'D/T bit at {pc:04x}')
            if p.up.reserved or p.lo.reserved:
                raise Reject(f'unsupported instruction at {pc:04x}: {p.upper:08x} {p.lower:08x}')
            if p.e_bit:
                if p.is_branch:
                    raise Reject(f'branch with E bit at {pc:04x}')
                slot = pair(pc + 8)
                check_slot(slot)
                roots.append((pc + 16) & 0x3FFF)
                continue
            if p.is_branch:
                slot = pair(pc + 8)
                check_slot(slot)
                if p.op_hi in (0x21, 0x25):
                    ret = (pc + 16) & 0x3FFF
                    if ret not in returns:
                        returns.add(ret)
                        roots.append(ret)
                if p.op_hi in (0x24, 0x25):
                    continue  # targets are the BAL/JALR return points
                work.append(p.branch_target())
                if p.op_hi not in (0x20, 0x21):
                    work.append((pc + 16) & 0x3FFF)
                continue
            work.append((pc + 8) & 0x3FFF)
    labels = {pc: pairs[pc] for pc in sorted(seen)}
    return labels, pairs


# ----------------------------------------------------------------------------- code emission

def f(x):
    return f'x.vf[{x}]'


def comps(mask):
    return [c for c in range(4) if mask & lane(c)]


class RegCache:
    """VF registers and ACC held in __m128 locals inside a fast block. Writes go through to
    Exec as well, so code outside the cache (lower instructions, helpers, exits) always reads
    current values from memory. `norm` holds, per register, the DEST lanes known to be
    normalised (written by an FMAC in this block); ACC always is."""

    def __init__(self):
        self.cached = set()
        self.declared = set()
        self.norm = {}
        self.acc_cached = False
        self.acc_declared = False
        self.vec_written = set()

    def get(self, reg, lines):
        name = f'v{reg}'
        self.declared.add(reg)
        if reg not in self.cached:
            lines.append(f'{name} = _mm_loadu_ps(x.vf[{reg}]);')
            self.cached.add(reg)
        return name

    def get_acc(self, lines):
        self.acc_declared = True
        if not self.acc_cached:
            lines.append('aacc = _mm_loadu_ps(x.acc);')
            self.acc_cached = True
        return 'aacc'

    def invalidate(self, reg):
        self.cached.discard(reg)
        self.norm.pop(reg, None)

    def declarations(self):
        names = [f'v{reg}' for reg in sorted(self.declared)]
        if self.acc_declared:
            names.append('aacc')
        return [f'__m128 {", ".join(names)};'] if names else []


class Emitter:
    def __init__(self, name, pairs, all_pairs):
        self.name = name
        self.pairs = pairs
        self.all_pairs = all_pairs
        self.out = []
        # Without FSSET an FDIV result may land ahead of queued flag writes (Exec::commitQ).
        self.has_fsset = any(not q.i_bit and q.op_hi == 0x15 for q in all_pairs.values())
        self.fmacs = {pc for pc, q in all_pairs.items() if flag_fmac(q.upper)}
        self.lite = self.lite_fmacs()
        # Hot state is written back (and lite sticky bits folded into status) before every exit.
        self.writeback = 'x.cycle = cyc; x.viBackupValid = vbValid; x.viBackupReg = vbReg; x.viBackupValue = vbValue;'
        if self.lite:
            self.writeback += ' x.foldLite(liteSign, liteMinAbs);'
        self.heads = self.block_heads()

    def block_heads(self):
        """Pairs where control can arrive other than by falling through from the pair before:
        each starts a block with a fast path (fast_block). The program entry, MSCNT resume
        points, BAL/JALR return points and branch targets; a JR can reach any pair, so its
        targets that are not heads take the per-pair code."""
        labels = self.pairs
        heads = set()
        for pc, p in labels.items():
            before = (pc - 8) & 0x3FFF
            q = labels.get(before)
            if q is None or q.is_branch or q.e_bit:
                heads.add(pc)
            if p.e_bit:
                heads.add((pc + 16) & 0x3FFF)
            elif p.is_branch:
                if p.op_hi not in (0x24, 0x25):
                    heads.add(p.branch_target())
                if p.op_hi not in (0x20,):
                    heads.add((pc + 16) & 0x3FFF)
        return {pc for pc in heads if pc in labels}

    def label(self, pc):
        return f'F{pc:04x}' if pc in self.heads else f'P{pc:04x}'

    def block_pcs(self, head):
        """The pairs of the block starting at head (branch and E-bit delay slots excluded)."""
        pcs = [head]
        pc = head
        while True:
            p = self.pairs[pc]
            if p.is_branch or p.e_bit:
                return pcs
            nxt = (pc + 8) & 0x3FFF
            if nxt not in self.pairs or nxt in self.heads:
                return pcs
            pcs.append(nxt)
            pc = nxt

    def fast_pair(self, p, k, base, in_delay_slot, prod, liveins, out):
        """Code for pair p as the k-th pair of a block (see fast_block). Appends the stall
        computation of t<k> to out and returns (the pair's body, its branch). Ready cycles of
        registers written earlier in the block come from the locals t<j> + latency, the rest
        from Exec (loaded once per block); every write still stores its ready cycle, so the
        state is exact at every exit."""
        cache = self.cache
        cache.vec_written = set()
        up_compute, up_write = self.upper_op(p, [], cache)
        lo_compute, lo_write, branch = self.lower_op(p)
        # Registers written outside the cache (lower instructions, the other upper ones) are
        # read from memory again by later pairs.
        up_reg = p.up.vf_write[0]
        if up_reg and up_reg not in cache.vec_written:
            cache.invalidate(up_reg)
        lo_reg = p.lo.vf_write[0]
        if lo_reg and p.suppressed != lo_reg:
            cache.invalidate(lo_reg)
        tk = f't{k}'
        terms = []

        def need(key, load):
            if key in prod:
                j, lat = prod[key]
                term = f't{j} + {lat}u'
            else:
                term = 'li_' + '_'.join(str(part) for part in key)
                if term not in liveins:
                    liveins.add(term)
                    out.append(f'const uint64_t {term} = {load};')
            if term not in terms:
                terms.append(term)

        for reg, lanes in p.up.vf_read + p.lo.vf_read:
            if reg:
                for c in comps(lanes):
                    need(('vf', reg, c), f'x.vfReady[{reg}][{c}]')
        vi_reads = (p.up.vi_read | p.lo.vi_read) & 0xFFFE
        for reg in range(1, 16):
            if vi_reads & (1 << reg):
                need(('vi', reg), f'x.viReady[{reg}]')
        for c in comps(p.up.acc_read | p.lo.acc_read):
            need(('acc', c), f'x.accReady[{c}]')
        out.append(f'uint64_t {tk} = {base};')
        for term in terms:
            out.append(f'{tk} = std::max({tk}, {term});')
        if p.lo.pipeline == 'fdiv' or p.lo.wait_q:
            out.append(f'if (x.fdivPending) {tk} = std::max({tk}, x.fdivReady);')
        if p.lo.pipeline == 'xgkick':
            out.append(f'if (x.xgActive) {tk} = std::max({tk}, x.xgFinish());')
        if not in_delay_slot:
            out.append(f'if ({tk} >= budget) {{ cyc = {base}; x.noteReady(lrl); {self.writeback} '
                       f'x.handoff(0x{p.pc:04x}u); return; }}')

        body = [f't = {tk};']
        body.extend(self.commit_lines(p))
        body.extend(up_compute)
        body.extend(lo_compute)
        if p.i_bit:
            body.append(f'x.i = norm(bitsFloat(0x{p.lower:08x}u));')
        body.extend(up_write)
        body.extend(lo_write)

        writes = []
        lo_reg, lo_lanes = p.lo.vf_write
        if lo_reg and p.suppressed != lo_reg:
            lat = p.lo.vf_latency or p.lo.latency
            for c in comps(lo_lanes):
                body.append(f'x.vfReady[{lo_reg}][{c}] = t + {lat}u;')
                writes.append((('vf', lo_reg, c), lat))
        up_reg, up_lanes = p.up.vf_write
        if up_reg:
            lat = p.up.vf_latency or p.up.latency
            for c in comps(up_lanes):
                body.append(f'x.vfReady[{up_reg}][{c}] = t + {lat}u;')
                writes.append((('vf', up_reg, c), lat))
        vi_writes = p.lo.vi_write & 0xFFFE
        written_vi = 0
        for reg in range(1, 16):
            if vi_writes & (1 << reg):
                lat = p.lo.vi_latency or p.lo.latency
                body.append(f'x.viReady[{reg}] = t + {lat}u;')
                writes.append((('vi', reg), lat))
                if not written_vi:
                    written_vi = reg
        if p.up.acc_write:
            for c in comps(p.up.acc_write):
                body.append(f'x.accReady[{c}] = t + {ACC_FORWARD_LATENCY}u;')
                writes.append((('acc', c), ACC_FORWARD_LATENCY))
        if writes:
            body.append(f'lrl = std::max(lrl, t + {max(lat for _, lat in writes)}u);')
        if written_vi and p.lo.delays_branch:
            body.insert(self.find_first_write(body), f'const int32_t loldvi = x.vi[{written_vi}];')
            body.append(f'vbValid = true; vbReg = {written_vi}; vbValue = loldvi;')
        else:
            body.append('vbValid = false;')
        for key, lat in writes:
            prod[key] = (k, lat)
        return body, branch

    def fast_block(self, head):
        """Code entered at block head `head` (label F<head>): the block's pairs with their
        stall timing in locals, then the branch, E-bit end or fall-through to the next head."""
        out = [f'F{head:04x}:', '    {', '        uint64_t lrl = 0u;']
        self.cache = RegCache()
        prod = {}
        liveins = set()
        k = 0
        base = 'cyc'

        def add(lines, indent='        '):
            out.extend(indent + line for line in lines)

        pcs = self.block_pcs(head)
        for pc in pcs:
            p = self.pairs[pc]
            out.append(f'        // 0x{pc:04x} up={p.upper:08x} lo={p.lower:08x}')
            pre = []
            body, branch = self.fast_pair(p, k, base, False, prod, liveins, pre)
            add(pre)
            pair_k = k
            k += 1
            base = f't{pair_k} + 1u'
            if p.e_bit or branch:
                kind, target = branch if branch else ('end', None)
                if kind == 'cond':
                    out.append('        bool taken = false;')
                    body.append('taken = ltaken;')
                elif kind == 'jump':
                    out.append('        uint32_t jumpTarget = 0u;')
                    body.append('jumpTarget = ljump;')
                out.append('        {')
                add(body, '            ')
                out.append('        }')
                slot = self.all_pairs[(pc + 8) & 0x3FFF]
                out.append(f'        // delay slot 0x{slot.pc:04x}')
                pre = []
                slot_body, _ = self.fast_pair(slot, k, base, True, prod, liveins, pre)
                add(pre)
                out.append('        {')
                add(slot_body, '            ')
                out.append('        }')
                out.append(f'        cyc = t{k} + 1u;')
                out.append('        x.noteReady(lrl);')
                if p.e_bit:
                    out.append(f'        {self.writeback}')
                    out.append(f'        x.end(cyc - 1u, 0x{(pc + 16) & 0x3FFF:04x}u);')
                    out.append('        return;')
                elif kind == 'always':
                    out.append(f'        goto {self.label(target)};')
                elif kind == 'cond':
                    out.append(f'        if (taken) goto {self.label(target)};')
                    out.append(f'        goto {self.label((pc + 16) & 0x3FFF)};')
                else:
                    out.append('        switch (jumpTarget)')
                    out.append('        {')
                    for other in self.pairs:
                        out.append(f'        case 0x{other:04x}: goto {self.label(other)};')
                    out.append(f'        default: {self.writeback} x.handoff(jumpTarget); return;')
                    out.append('        }')
                out.append('    }')
                return self.with_declarations(out)
            out.append('        {')
            add(body, '            ')
            out.append('        }')
        nxt = (pcs[-1] + 8) & 0x3FFF
        out.append(f'        cyc = t{k - 1} + 1u;')
        out.append('        x.noteReady(lrl);')
        if nxt in self.pairs:
            out.append(f'        goto {self.label(nxt)};')
        else:
            out.append(f'        {self.writeback}')
            out.append(f'        x.handoff(0x{nxt:04x}u);')
            out.append('        return;')
        out.append('    }')
        return self.with_declarations(out)

    def with_declarations(self, out):
        """Puts the block's vector locals after its opening lines (label, brace, lrl)."""
        return out[:3] + ['        ' + line for line in self.cache.declarations()] + out[3:]

    def lite_fmacs(self):
        """FMAC pairs whose MAC and status writes no flag read can observe (Exec::fmacLite).

        A MAC read at pair r sees the last FMAC that has landed by its issue cycle. An FMAC
        one to three pairs before r on some path may or may not have landed (stalls), so it
        stays exact; the first FMAC four or more pairs back has landed (every pair takes at
        least a cycle) and hides everything before it on that path. The program end flushes
        the pipelines, so it sees the last FMAC on every path. Leaving the translated code
        (a JR to a pair it does not have, running off its end) counts as a MAC read by the
        next pair. A budget handoff needs nothing: the interpreter goes on with this code.
        Programs that read status or use FSSET get no lite FMACs, so the sticky bits that
        foldLite adds at exits are never observed late or out of order.
        """
        nodes = self.all_pairs
        if any(not q.i_bit and q.op_hi in STATUS_USES for q in nodes.values()):
            return set()
        labels = self.pairs
        succ = {}

        def edge(a, b):
            succ.setdefault(a, set()).add(b)

        def go(pc):
            return pc if pc in labels else 'exit'

        for pc, p in labels.items():
            slot = (pc + 8) & 0x3FFF
            if p.e_bit:
                edge(pc, slot)
                edge(slot, 'end')
            elif p.is_branch:
                edge(pc, slot)
                if p.op_hi in (0x24, 0x25):
                    for other in labels:
                        edge(slot, other)
                    edge(slot, 'exit')
                else:
                    edge(slot, go(p.branch_target()))
                    if p.op_hi not in (0x20, 0x21):
                        edge(slot, go((pc + 16) & 0x3FFF))
            else:
                edge(pc, go((pc + 8) & 0x3FFF))
        preds = {}
        for a, targets in succ.items():
            for b in targets:
                preds.setdefault(b, set()).add(a)

        exact = set()

        def walk_read(start):
            work = [(q, 1) for q in preds.get(start, ())]
            seen = set()
            while work:
                q, distance = work.pop()
                key = (q, min(distance, 4))
                if key in seen:
                    continue
                seen.add(key)
                if q in self.fmacs:
                    exact.add(q)
                    if distance >= 4:
                        continue
                work.extend((before, distance + 1) for before in preds.get(q, ()))

        def walk_end(start):
            work = list(preds.get(start, ()))
            seen = set()
            while work:
                q = work.pop()
                if q in seen:
                    continue
                seen.add(q)
                if q in self.fmacs:
                    exact.add(q)
                    continue
                work.extend(preds.get(q, ()))

        for pc, q in nodes.items():
            if not q.i_bit and q.op_hi in MAC_READS:
                walk_read(pc)
        walk_read('exit')
        walk_end('end')
        return self.fmacs - exact

    def line(self, text=''):
        self.out.append(text)

    # ---------------------------------------------------------------- upper

    def upper_op(self, p, lines, cache=None):
        """Returns (code computing temporaries, code writing results) for the upper word."""
        u = p.upper
        op = u & 0x3F
        dest, fs, ft, fd = DEST(u), FS(u), FT(u), FD(u)
        compute, write = [], []
        lanes = comps(dest)
        if op >= 0x3C:
            special = upper_special(u)
            if special in (0x2F, 0x30):
                return compute, write
            if special == 0x1F:  # CLIP
                compute.append(f'const uint32_t clipNew = x.noClip ? 0u : clipFlags(x.vf[{fs}], x.vf[{ft}][3]);')
                compute.append('x.clipFlag(t, clipNew);')
                return compute, write
            if 0x10 <= special <= 0x13:  # ITOF
                scale = ['', ' / 16.0f', ' / 4096.0f', ' / 32768.0f'][special & 3]
                for c in lanes:
                    compute.append(f'int32_t iv{c}; std::memcpy(&iv{c}, &x.vf[{fs}][{c}], 4); '
                                   f'const float ur{c} = static_cast<float>(iv{c}){scale};')
                if ft:
                    for c in lanes:
                        write.append(f'x.vf[{ft}][{c}] = ur{c};')
                return compute, write
            if 0x14 <= special <= 0x17:  # FTOI
                scale = ['1.0f', '16.0f', '4096.0f', '32768.0f'][special & 3]
                for c in lanes:
                    compute.append(f'const float ur{c} = bitsFloat(static_cast<uint32_t>(floatToInt(norm(x.vf[{fs}][{c}]), {scale})));')
                if ft:
                    for c in lanes:
                        write.append(f'x.vf[{ft}][{c}] = ur{c};')
                return compute, write
            if special == 0x1D:  # ABS
                for c in lanes:
                    compute.append(f'const float ur{c} = std::fabs(norm(x.vf[{fs}][{c}]));')
                if ft:
                    for c in lanes:
                        write.append(f'x.vf[{ft}][{c}] = ur{c};')
                return compute, write
            kind, product_sum = self.fmac_kind(special, True)
            return self.fmac(p.pc, u, kind, product_sum, fs, ft, dest, 'acc', special, cache)
        if 0x10 <= op <= 0x17 or op in (0x1D, 0x1F, 0x2B, 0x2F):  # MAX / MINI
            is_max = op in (0x10, 0x11, 0x12, 0x13, 0x1D, 0x2B)
            cmp = '>' if is_max else '<'
            for c in lanes:
                if op <= 0x17:
                    other = f'norm(x.vf[{ft}][{op & 3}])'
                elif op in (0x1D, 0x1F):
                    other = 'norm(x.i)'
                else:
                    other = f'norm(x.vf[{ft}][{c}])'
                compute.append(f'const float us{c} = norm(x.vf[{fs}][{c}]); const float uo{c} = {other}; '
                               f'const float ur{c} = (us{c} {cmp} uo{c}) ? us{c} : uo{c};')
            if fd:
                for c in lanes:
                    write.append(f'x.vf[{fd}][{c}] = ur{c};')
            return compute, write
        kind, product_sum = self.fmac_kind(op, False)
        return self.fmac(p.pc, u, kind, product_sum, fs, ft, dest, 'vf', op, cache)

    def fmac_kind(self, op, special):
        if op not in FMAC_TABLE:
            raise Reject(f'unhandled FMAC op {op:02x}')
        kind = FMAC_TABLE[op]
        product_sum = (op in PRODUCT_SUM_SPECIAL) if special else (op in PRODUCT_SUM_OPS)
        if special and op == 0x2E:
            kind = ('opmula', 'v')
        return kind, product_sum

    def fmac(self, pc, u, kind, product_sum, fs, ft, dest, target, op, cache=None):
        (operation, source) = kind
        compute, write = [], []
        kinds = {'add': 'kFmacAdd', 'sub': 'kFmacSub', 'mul': 'kFmacMul', 'madd': 'kFmacMadd',
                 'msub': 'kFmacMsub', 'op': 'kFmacOpmsub', 'opmula': 'kFmacOpmula'}
        # Operands are normalised on the lanes that count; broadcasts and the cross product's
        # lane order (left y z x w, right z x y w) are applied before the operation.
        cross = operation in ('op', 'opmula')
        lanes_needed = 0xE if cross else dest
        left = self.vf_operand(fs, lanes_needed, cache, compute)
        if source == 'bc':
            right = self.bc_operand(ft, op & 3, cache, compute)
        elif source == 'q':
            right = '_mm_set1_ps(norm(x.q))'
        elif source == 'i':
            right = '_mm_set1_ps(norm(x.i))'
        else:
            right = self.vf_operand(ft, lanes_needed, cache, compute)
        if cross:
            compute.append(f'const __m128 uls = {left}; const __m128 urs = {right};')
            left = '_mm_shuffle_ps(uls, uls, _MM_SHUFFLE(3, 0, 2, 1))'
            right = '_mm_shuffle_ps(urs, urs, _MM_SHUFFLE(3, 1, 0, 2))'
        if operation in ('madd', 'msub', 'op'):
            accv = cache.get_acc(compute) if cache else '_mm_loadu_ps(x.acc)'
        else:
            accv = '_mm_setzero_ps()'
        ps = 'true' if product_sum else 'false'
        if pc in self.lite:
            compute.append(f'const __m128 urv = x.fmacLite<{kinds[operation]}, {ps}>'
                           f'({dest}u, {left}, {right}, {accv}, liteSign, liteMinAbs);')
        else:
            compute.append(f'const __m128 urv = x.fmacVec<{kinds[operation]}, {ps}>'
                           f'(t, {dest}u, {left}, {right}, {accv});')
        if target == 'acc':
            write.extend(self.vec_write(None, dest, cache))
        else:
            fd = FD(u)
            if fd:
                write.extend(self.vec_write(fd, dest, cache))
        return compute, write

    def vf_operand(self, reg, lanes, cache, lines):
        """VF[reg] as an FMAC operand whose DEST `lanes` must be normalised."""
        if cache is None:
            return f'norm4(_mm_loadu_ps(x.vf[{reg}]))'
        name = cache.get(reg, lines)
        if reg == 0 or (cache.norm.get(reg, 0) & lanes) == lanes:
            return name
        return f'norm4({name})'

    def bc_operand(self, reg, comp, cache, lines):
        """Component comp of VF[reg] broadcast to four lanes, normalised."""
        if cache is None:
            return f'_mm_set1_ps(norm(x.vf[{reg}][{comp}]))'
        name = cache.get(reg, lines)
        splat = f'_mm_shuffle_ps({name}, {name}, _MM_SHUFFLE({comp}, {comp}, {comp}, {comp}))'
        if reg == 0 or (cache.norm.get(reg, 0) & lane(comp)):
            return splat
        return f'norm4({splat})'

    def vec_write(self, reg, dest, cache):
        """Writes the DEST lanes of urv to VF[reg] (ACC when reg is None)."""
        mask = sum(1 << c for c in comps(dest))
        mem = 'x.acc' if reg is None else f'x.vf[{reg}]'
        if cache is None:
            if dest == 0xF:
                return [f'_mm_storeu_ps({mem}, urv);']
            return [f'_mm_storeu_ps({mem}, _mm_blend_ps(_mm_loadu_ps({mem}), urv, 0x{mask:x}));']
        lines = []
        if reg is None:
            if dest == 0xF:
                cache.acc_declared = True
                cache.acc_cached = True
                name = 'aacc'
                lines.append('aacc = urv;')
            else:
                name = cache.get_acc(lines)
                lines.append(f'aacc = _mm_blend_ps(aacc, urv, 0x{mask:x});')
        else:
            if dest == 0xF:
                cache.declared.add(reg)
                cache.cached.add(reg)
                name = f'v{reg}'
                lines.append(f'{name} = urv;')
            else:
                name = cache.get(reg, lines)
                lines.append(f'{name} = _mm_blend_ps({name}, urv, 0x{mask:x});')
            cache.norm[reg] = cache.norm.get(reg, 0) | dest
            cache.vec_written.add(reg)
        lines.append(f'_mm_storeu_ps({mem}, {name});')
        return lines

    # ---------------------------------------------------------------- lower

    def lower_op(self, p):
        """Returns (compute, write, branch) code; branch describes control flow."""
        l = p.lower
        compute, write = [], []
        branch = None
        if p.i_bit or l in (0, 0x8000033C):
            return compute, write, branch
        op_hi = p.op_hi
        vf_t, vf_s = FT(l), FS(l)
        vi_t, vi_s, vi_d = VIT(l), VIS(l), VID(l)
        dest = DEST(l)
        lanes = comps(dest)
        imm = IMM11(l)

        def vi_result(reg, expr):
            if reg:
                compute.append(f'const int32_t lvi = {expr};')
                write.append(f'x.vi[{reg}] = static_cast<int16_t>(lvi);')

        def vf_result(reg, values):
            for c in lanes:
                compute.append(f'const float lr{c} = {values[c]};')
            if reg and p.suppressed != reg:
                for c in lanes:
                    write.append(f'x.vf[{reg}][{c}] = lr{c};')

        def branch_vi(reg):
            if reg == 0:
                return '0'
            return f'(vbValid && vbReg == {reg} ? vbValue : x.vi[{reg}])'

        if op_hi == 0x00:  # LQ
            compute.append(f'float lm[4]; x.load(x.address(x.vi[{vi_s}], {imm}), lm);')
            vf_result(vf_t, [f'lm[{c}]' for c in range(4)])
        elif op_hi == 0x01:  # SQ
            compute.append(f'x.storeVf(t, x.address(x.vi[{vi_t}], {imm}), {vf_s}, {dest}u);')
        elif op_hi == 0x04:  # ILW
            vi_result(vi_t, f'x.loadVi(x.address(x.vi[{vi_s}], {imm}), {dest}u)')
        elif op_hi == 0x05:  # ISW
            compute.append(f'x.storeVi(t, x.address(x.vi[{vi_s}], {imm}), x.vi[{vi_t}], {dest}u);')
        elif op_hi in (0x08, 0x09):  # IADDIU / ISUBIU
            imm15 = (l & 0x7FF) | ((l >> 10) & 0x7800)
            sign = '+' if op_hi == 0x08 else '-'
            vi_result(vi_t, f'static_cast<int16_t>(x.vi[{vi_s}] {sign} {imm15})')
        elif op_hi in (0x10, 0x12, 0x13):  # FCEQ / FCAND / FCOR
            imm24 = l & 0xFFFFFF
            expr = {0x10: f'((x.clip & 0xFFFFFFu) == 0x{imm24:06x}u) ? 1 : 0',
                    0x12: f'((x.clip & 0x{imm24:06x}u) != 0u) ? 1 : 0',
                    0x13: f'((x.clip | 0x{imm24:06x}u) == 0xFFFFFFu) ? 1 : 0'}[op_hi]
            vi_result(1, expr)
        elif op_hi == 0x11:  # FCSET
            compute.append(f'x.fcset(t, 0x{l & 0xFFFFFF:06x}u);')
        elif op_hi in (0x14, 0x15, 0x16, 0x17):  # FSEQ / FSSET / FSAND / FSOR
            imm12 = (((l >> 21) & 1) << 11) | (l & 0x7FF)
            if op_hi == 0x15:
                compute.append(f'x.fsset(t, 0x{imm12:03x}u);')
            else:
                expr = {0x14: f'((x.status & 0xFFFu) == 0x{imm12:03x}u) ? 1 : 0',
                        0x16: f'static_cast<int32_t>((x.status & 0xFFFu) & 0x{imm12:03x}u)',
                        0x17: f'static_cast<int32_t>((x.status & 0xFFFu) | 0x{imm12:03x}u)'}[op_hi]
                vi_result(vi_t, expr)
        elif op_hi in (0x18, 0x1A, 0x1B):  # FMEQ / FMAND / FMOR
            src = f'static_cast<uint32_t>(static_cast<uint16_t>(x.vi[{vi_s}]))'
            expr = {0x18: f'((x.mac & 0xFFFFu) == {src}) ? 1 : 0',
                    0x1A: f'static_cast<int32_t>(x.mac & {src})',
                    0x1B: f'static_cast<int32_t>(x.mac | {src})'}[op_hi]
            vi_result(vi_t, expr)
        elif op_hi == 0x1C:  # FCGET
            vi_result(vi_t, 'static_cast<int32_t>(x.clip & 0x0FFFu)')
        elif op_hi in (0x20, 0x21):  # B / BAL
            if op_hi == 0x21:
                vi_result(vi_t, str((p.pc + 16) // 8))
            branch = ('always', p.branch_target())
        elif op_hi in (0x24, 0x25):  # JR / JALR
            compute.append(f'const uint32_t ljump = (static_cast<uint32_t>(static_cast<uint16_t>({branch_vi(vi_s)})) * 8u) & 0x3FFFu;')
            if op_hi == 0x25:
                vi_result(vi_t, str((p.pc + 16) // 8))
            branch = ('jump', 'ljump')
        elif op_hi in (0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F):
            a = f'static_cast<int16_t>({branch_vi(vi_s)})'
            b = f'static_cast<int16_t>({branch_vi(vi_t)})'
            cond = {0x28: f'{a} == {b}', 0x29: f'{a} != {b}', 0x2C: f'{a} < 0', 0x2D: f'{a} > 0',
                    0x2E: f'{a} <= 0', 0x2F: f'{a} >= 0'}[op_hi]
            compute.append(f'const bool ltaken = {cond};')
            branch = ('cond', p.branch_target())
        elif op_hi == 0x40:
            direct = l & 0x3F
            if direct == 0x30:
                vi_result(vi_d, f'static_cast<int16_t>(x.vi[{vi_s}] + x.vi[{vi_t}])')
            elif direct == 0x31:
                vi_result(vi_d, f'static_cast<int16_t>(x.vi[{vi_s}] - x.vi[{vi_t}])')
            elif direct == 0x32:
                imm5 = (l >> 6) & 0x1F
                imm5 = imm5 - 32 if imm5 & 0x10 else imm5
                vi_result(vi_t, f'static_cast<int16_t>(x.vi[{vi_s}] + ({imm5}))')
            elif direct == 0x34:
                vi_result(vi_d, f'x.vi[{vi_s}] & x.vi[{vi_t}]')
            elif direct == 0x35:
                vi_result(vi_d, f'x.vi[{vi_s}] | x.vi[{vi_t}]')
            else:
                special = lower_special(l)
                if special == 0x30:  # MOVE
                    vf_result(vf_t, [f'x.vf[{vf_s}][{c}]' for c in range(4)])
                elif special == 0x31:  # MR32
                    vf_result(vf_t, [f'x.vf[{vf_s}][{(c + 1) % 4}]' for c in range(4)])
                elif special in (0x34, 0x36):  # LQI / LQD
                    if special == 0x34:
                        compute.append(f'float lm[4]; x.load(x.addressU16(x.vi[{vi_s}]), lm);')
                        new_vi = f'static_cast<int16_t>(x.vi[{vi_s}] + 1)'
                    else:
                        compute.append(f'const int32_t ldec = static_cast<int16_t>(x.vi[{vi_s}] - 1);')
                        compute.append(f'float lm[4]; x.load(x.addressU16({"ldec" if vi_s else "x.vi[0]"}), lm);')
                        new_vi = 'ldec'
                    vf_result(vf_t, [f'lm[{c}]' for c in range(4)])
                    if vi_s:
                        vi_result(vi_s, new_vi)
                elif special in (0x35, 0x37):  # SQI / SQD
                    if special == 0x35:
                        compute.append(f'x.storeVf(t, x.addressU16(x.vi[{vi_t}]), {vf_s}, {dest}u);')
                        new_vi = f'static_cast<int16_t>(x.vi[{vi_t}] + 1)'
                    else:
                        compute.append(f'const int32_t ldec = static_cast<int16_t>(x.vi[{vi_t}] - 1);')
                        compute.append(f'x.storeVf(t, x.addressU16({"ldec" if vi_t else "x.vi[0]"}), {vf_s}, {dest}u);')
                        new_vi = 'ldec'
                    if vi_t:
                        vi_result(vi_t, new_vi)
                elif special == 0x38:  # DIV
                    fsf, ftf = (l >> 21) & 3, (l >> 23) & 3
                    compute.append(f'const float lnum = norm(x.vf[{vf_s}][{fsf}]); const float lden = norm(x.vf[{vf_t}][{ftf}]);')
                    compute.append('uint32_t ldi = 0u; float lq = 0.0f;')
                    compute.append('if (lden == 0.0f) { ldi = lnum == 0.0f ? 0x10u : 0x20u; '
                                   'lq = std::signbit(lnum) != std::signbit(lden) ? -std::numeric_limits<float>::max() '
                                   ': std::numeric_limits<float>::max(); } else { lq = lnum / lden; }')
                    compute.append('uint32_t lign = 0u; lq = normResult(lq, lign);')
                    compute.append('x.queueQ(t, lq, 7u, ldi);')
                elif special == 0x39:  # SQRT
                    ftf = (l >> 23) & 3
                    compute.append(f'const float lval = norm(x.vf[{vf_t}][{ftf}]);')
                    compute.append('x.queueQ(t, std::sqrt(std::fabs(lval)), 7u, lval < 0.0f ? 0x10u : 0u);')
                elif special == 0x3A:  # RSQRT
                    fsf, ftf = (l >> 21) & 3, (l >> 23) & 3
                    compute.append(f'const float lnum = norm(x.vf[{vf_s}][{fsf}]); const float lrad = norm(x.vf[{vf_t}][{ftf}]);')
                    compute.append('const float lden = std::sqrt(std::fabs(lrad)); uint32_t ldi = lrad < 0.0f ? 0x10u : 0u; float lq = 0.0f;')
                    compute.append('if (lden != 0.0f) lq = lnum / lden; else { ldi = lnum == 0.0f ? 0x10u : 0x20u; '
                                   'lq = std::signbit(lnum) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max(); }')
                    compute.append('uint32_t lign = 0u; lq = normResult(lq, lign);')
                    compute.append('x.queueQ(t, lq, 13u, ldi);')
                elif special == 0x3B:  # WAITQ
                    pass
                elif special == 0x3C:  # MTIR
                    comp = (l >> 21) & 3
                    vi_result(vi_t, f'static_cast<int16_t>(floatBits(x.vf[{vf_s}][{comp}]) & 0xFFFFu)')
                elif special == 0x3D:  # MFIR
                    compute.append(f'const float lfi = bitsFloat(static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(x.vi[{vi_s}] & 0xFFFF))));')
                    vf_result(vf_t, ['lfi'] * 4)
                elif special == 0x3E:  # ILWR
                    vi_result(vi_t, f'x.loadVi(x.addressU16(x.vi[{vi_s}]), {dest}u)')
                elif special == 0x3F:  # ISWR
                    compute.append(f'x.storeVi(t, x.addressU16(x.vi[{vi_s}]), x.vi[{vi_t}], {dest}u);')
                elif special == 0x68:  # XTOP
                    vi_result(vi_t, 'static_cast<int32_t>(x.top & 0x3FFu)')
                elif special == 0x69:  # XITOP
                    vi_result(vi_t, 'static_cast<int32_t>(x.itop & 0x3FFu)')
                elif special == 0x6C:  # XGKICK
                    compute.append(f'x.xgkick(t, x.vi[{vi_s}]);')
                else:
                    raise Reject(f'lower special {special:02x} at {p.pc:04x}')
        else:
            raise Reject(f'lower op {op_hi:02x} at {p.pc:04x}')
        return compute, write, branch

    # ---------------------------------------------------------------- one pair

    def pair_code(self, p, in_delay_slot):
        """C++ for one instruction pair, ending with cyc = t + 1."""
        out = []
        up_compute, up_write = self.upper_op(p, out)
        lo_compute, lo_write, branch = self.lower_op(p)

        out.append('t = cyc;')
        for reg, lanes in p.up.vf_read + p.lo.vf_read:
            if reg:
                out.append(f't = x.vfStall(t, {reg}, {lanes}u);')
        vi_reads = (p.up.vi_read | p.lo.vi_read) & 0xFFFE
        for reg in range(1, 16):
            if vi_reads & (1 << reg):
                out.append(f't = std::max(t, x.viReady[{reg}]);')
        acc_read = p.up.acc_read | p.lo.acc_read
        if acc_read:
            out.append(f't = x.accStall(t, {acc_read}u);')
        if p.lo.pipeline == 'fdiv' or p.lo.wait_q:
            out.append('if (x.fdivPending) t = std::max(t, x.fdivReady);')
        if p.lo.pipeline == 'xgkick':
            out.append('if (x.xgActive) t = std::max(t, x.xgFinish());')
        if not in_delay_slot:
            out.append(f'if (t >= budget) {{ {self.writeback} x.handoff(0x{p.pc:04x}u); return; }}')

        out.extend(self.commit_lines(p))

        out.extend(up_compute)
        out.extend(lo_compute)
        if p.i_bit:
            out.append(f'x.i = norm(bitsFloat(0x{p.lower:08x}u));')
        out.extend(up_write)
        out.extend(lo_write)

        # Ready cycles, as markPairWrites.
        lo_reg, lo_lanes = p.lo.vf_write
        if lo_reg and p.suppressed != lo_reg:
            lat = p.lo.vf_latency or p.lo.latency
            out.append(f'x.vfWritten({lo_reg}, {lo_lanes}u, t + {lat}u);')
        up_reg, up_lanes = p.up.vf_write
        if up_reg:
            lat = p.up.vf_latency or p.up.latency
            out.append(f'x.vfWritten({up_reg}, {up_lanes}u, t + {lat}u);')
        vi_writes = p.lo.vi_write & 0xFFFE
        written_vi = 0
        for reg in range(1, 16):
            if vi_writes & (1 << reg):
                lat = p.lo.vi_latency or p.lo.latency
                out.append(f'x.viReady[{reg}] = t + {lat}u; x.noteReady(t + {lat}u);')
                if not written_vi:
                    written_vi = reg
        if p.up.acc_write:
            out.append(f'x.accWritten({p.up.acc_write}u, t + {ACC_FORWARD_LATENCY}u);')

        # Branch-read quirk: the next pair's branch sees the old value of this pair's VI write.
        if written_vi and p.lo.delays_branch:
            # The old value is the one before this pair's write; recompute from the write list.
            out.insert(self.find_first_write(out), f'const int32_t loldvi = x.vi[{written_vi}];')
            out.append(f'vbValid = true; vbReg = {written_vi}; vbValue = loldvi;')
        else:
            out.append('vbValid = false;')
        out.append('cyc = t + 1u;')
        return out, branch

    def commit_lines(self, p):
        """Flag and Q commits a pair needs before it runs (reads of flags or Q, FDIV issue)."""
        u = p.upper
        op = u & 0x3F
        q_ops = {0x1C, 0x20, 0x21, 0x24, 0x25}
        q_special = {0x1C, 0x20, 0x21, 0x24, 0x25}
        if op >= 0x3C:
            reads_q = upper_special(u) in q_special
        else:
            reads_q = op in q_ops
        lower_reads_flags = (not p.i_bit) and p.op_hi in (0x10, 0x12, 0x13, 0x14, 0x16, 0x17, 0x18, 0x1A, 0x1B, 0x1C)
        if lower_reads_flags or ((reads_q or p.lo.pipeline == 'fdiv') and self.has_fsset):
            return ['x.commitTo(t);']
        if reads_q or p.lo.pipeline == 'fdiv':
            return ['x.commitQ(t);']
        return []

    @staticmethod
    def find_first_write(lines):
        # Insert the old-VI capture before the first register write of the pair.
        for index, text in enumerate(lines):
            if text.startswith('x.vi[') and '=' in text and 'Ready' not in text:
                return index
        return len(lines)

    # ---------------------------------------------------------------- whole program

    def emit(self, image_name, image_bytes, code_bytes):
        name = self.name
        L = self.line
        L('// Generated by ps2xRuntime/tools/vu1rec.py; do not edit.')
        L(f'// Source image: {image_name}')
        L('#include "runtime/ps2_vu1_compiled.h"')
        L('')
        L(f'namespace vu1prog_{name}')
        L('{')
        L('using namespace ps2x::vu1c;')
        L('')
        L(f'const uint8_t kImage[{code_bytes}] = {{')
        for offset in range(0, code_bytes, 16):
            chunk = ', '.join(f'0x{b:02x}' for b in image_bytes[offset:min(offset + 16, code_bytes)])
            L(f'    {chunk},')
        L('};')
        L('')
        L('void run(Exec &x, uint32_t entryPc)')
        L('{')
        L('    uint64_t t = 0;')
        # Hot scalar state lives in locals the compiler can keep in registers (VU memory
        # stores could alias the Exec fields); it is written back before every exit.
        L('    uint64_t cyc = x.cycle;')
        L('    const uint64_t budget = x.budgetEnd;')
        L('    bool vbValid = x.viBackupValid;')
        L('    uint8_t vbReg = x.viBackupReg;')
        L('    int32_t vbValue = x.viBackupValue;')
        if self.lite:
            L('    // Sticky status bits of the lite FMACs (Exec::fmacLite), folded in at exits.')
            L('    __m128i liteSign = _mm_setzero_si128();')
            L('    __m128i liteMinAbs = _mm_set1_epi32(-1);')
        L('    switch (entryPc)')
        L('    {')
        for pc in self.pairs:
            L(f'    case 0x{pc:04x}: goto {self.label(pc)};')
        L('    default: return;')
        L('    }')
        for pc, p in self.pairs.items():
            if pc in self.heads:
                # Blocks are entered at their head; the per-pair code below serves entries
                # in the middle of a block (resuming after a handoff, a JR).
                for text in self.fast_block(pc):
                    L(text)
                continue
            L(f'P{pc:04x}: // up={p.upper:08x} lo={p.lower:08x}')
            L('    {')
            code, branch = self.pair_code(p, False)
            for text in code:
                L('        ' + text)
            if p.e_bit:
                slot = self.all_pairs[pc + 8]
                slot_code, _ = self.pair_code(slot, True)
                L(f'        // E-bit delay slot 0x{pc + 8:04x}')
                for text in slot_code:
                    L('        ' + text)
                L(f'        {self.writeback}')
                L(f'        x.end(cyc - 1u, 0x{(pc + 16) & 0x3FFF:04x}u);')
                L('        return;')
            elif branch:
                kind, target = branch
                if kind == 'cond':
                    L('        const bool taken = ltaken;')
                elif kind == 'jump':
                    L('        const uint32_t jumpTarget = ljump;')
                slot = self.all_pairs[pc + 8]
                slot_code, _ = self.pair_code(slot, True)
                L(f'        // delay slot 0x{pc + 8:04x}')
                L('        {')
                for text in slot_code:
                    L('            ' + text)
                L('        }')
                if kind == 'always':
                    L(f'        goto {self.label(target)};')
                elif kind == 'cond':
                    L(f'        if (taken) goto {self.label(target)};')
                    L(f'        goto {self.label((pc + 16) & 0x3FFF)};')
                else:
                    L('        switch (jumpTarget)')
                    L('        {')
                    for other in self.pairs:
                        L(f'        case 0x{other:04x}: goto {self.label(other)};')
                    L(f'        default: {self.writeback} x.handoff(jumpTarget); return;')
                    L('        }')
            else:
                nxt = (pc + 8) & 0x3FFF
                if nxt in self.pairs:
                    L(f'        goto {self.label(nxt)};')
                else:
                    L(f'        {self.writeback}')
                    L(f'        x.handoff(0x{nxt:04x}u);')
                    L('        return;')
            L('    }')
        L('}')
        L('')
        L(f'const Program kProgram{{"vu1prog_{name}", kImage, {code_bytes}u, run}};')
        L('struct Registrar')
        L('{')
        L('    Registrar() { registerProgram(&kProgram); }')
        L('} registrar;')
        L('}')
        return '\n'.join(self.out) + '\n'


image_bytes_full = b''


def fnv1a64(data):
    value = 0xcbf29ce484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return value


def translate(path, out_dir, done):
    global image_bytes_full
    image = open(path, 'rb').read()
    image_bytes_full = image
    pairs, all_pairs = analyse(image)
    base = os.path.basename(path)
    code_bytes = max(pairs) + 8
    for p in pairs.values():
        if p.is_branch or p.e_bit:
            code_bytes = max(code_bytes, p.pc + 16)
    # The runtime matches only these bytes, so they name the program.
    name = f'{fnv1a64(image[:code_bytes]):016x}'
    if name in done:
        return name, len(pairs), code_bytes, False
    done.add(name)
    emitter = Emitter(name, pairs, all_pairs)
    text = emitter.emit(base, image, code_bytes)
    translate.lite_counts[name] = (len(emitter.lite), len(emitter.fmacs))
    target = os.path.join(out_dir, f'vu1prog_{name}.cpp')
    old = open(target, encoding='utf-8').read() if os.path.exists(target) else None
    if old != text:
        with open(target, 'w', encoding='utf-8', newline='\n') as out:
            out.write(text)
    return name, len(pairs), code_bytes, True


translate.lite_counts = {}


def main():
    args = sys.argv[1:]
    prune = '--prune' in args
    args = [arg for arg in args if arg != '--prune']
    if len(args) < 2:
        raise SystemExit(__doc__)
    out_dir = args[0]
    os.makedirs(out_dir, exist_ok=True)
    done = set()
    # Sorted, so the image named in each file's header does not depend on argument order.
    for path in sorted(args[1:], key=os.path.basename):
        try:
            name, count, size, written = translate(path, out_dir, done)
            note = '' if written else ' (same code as an earlier image)'
            if written:
                lite, fmacs = translate.lite_counts[name]
                note = f', {lite} of {fmacs} FMACs without flag writes'
            print(f'{os.path.basename(path)}: {count} pairs, {size} bytes -> vu1prog_{name}.cpp{note}')
        except Reject as error:
            print(f'{os.path.basename(path)}: not translated ({error})')
    if prune:
        for entry in sorted(os.listdir(out_dir)):
            if entry.startswith('vu1prog_') and entry.endswith('.cpp') and entry[8:-4] not in done:
                os.remove(os.path.join(out_dir, entry))
                print(f'removed {entry}')


if __name__ == '__main__':
    main()
