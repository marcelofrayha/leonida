// Emulates BMI1, BMI2, MOVBE and legacy LZCNT/TZCNT on an x86-64 host that lacks them (docs/cpu_features.md).
//
// Pre-Haswell AVX1 + SSE4.2 hosts (Intel Sandy Bridge and Ivy Bridge) have 256-bit XSAVE/YMM state, POPCNT, CX16,
// AES-NI and PCLMULQDQ, but lack BMI1, BMI2, LZCNT and MOVBE:
//   - VEX-encoded BMI1 (ANDN, BEXTR, BLSI, BLSMSK, BLSR), VEX-encoded BMI2 (BZHI, MULX, PDEP, PEXT, RORX, SARX,
//     SHLX, SHRX) and legacy MOVBE raise #UD (STATUS_ILLEGAL_INSTRUCTION), so a vectored handler decodes the faulting
//     instruction, applies it to the thread's CONTEXT and steps RIP past it.
//   - Legacy LZCNT (F3 0F BD /r) and TZCNT (F3 0F BC /r) do NOT raise #UD without CPUID.80000001H:ECX.LZCNT /
//     CPUID.07H:EBX.BMI1: the CPU ignores the F3 prefix and executes BSR / BSF instead. BSR returns (width - 1 - clz)
//     instead of clz, and BSF leaves CF undefined (whereas 988 .pdata tzcnt sites immediately feed cmovb/cmovae/jae/jb)
//     and leaves the destination unmodified on zero. Before an image's sections are protected, patch_image() walks its
//     .pdata functions and 0xCC-delimited leaf functions: >=5-byte 32/64-bit register sites jump to branchless native
//     BSF/BSR + CMOV + BT stubs within 2 GiB, and 4-byte, 16-bit or memory sites become UD2 (0F 0B) backed by a
//     side-table of the original instruction bytes for the #UD handler. LEONIDA_LZCNT_PATCH=0 disables patching.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rt::bmi_emulation {

enum class Op : uint8_t {
    none = 0,
    andn,
    bextr,
    blsi,
    blsmsk,
    blsr,
    bzhi,
    mulx,
    pdep,
    pext,
    rorx,
    sarx,
    shlx,
    shrx,
    movbe_load,
    movbe_store,
    tzcnt,
    lzcnt,
};

struct Decoded {
    Op op = Op::none;
    uint8_t length = 0;  // total instruction bytes, prefixes included
    uint8_t size = 4;    // operand width in bytes: 2, 4 or 8
    uint8_t dst = 0;     // destination GPR (for mulx: high half)
    uint8_t dst2 = 0;    // second destination GPR (for mulx: low half)
    uint8_t src1 = 0;    // source or control GPR (vvvv or reg)
    uint8_t rm = 0;      // r/m GPR when !mem
    uint8_t imm = 0;     // immediate byte (rorx)
    bool mem = false;    // ModRM.mod != 3
    bool addr32 = false; // 0x67 address-size override
    uint8_t seg = 0;     // 0x64 (FS) or 0x65 (GS), else 0
    int8_t base = -1;    // 0..15 GPR, -1 none, -2 RIP-relative
    int8_t index = -1;   // 0..15 GPR, -1 none
    uint8_t scale = 1;   // 1, 2, 4 or 8
    int32_t disp = 0;
};

namespace detail {

inline bool decode_modrm(const uint8_t* code, std::size_t avail, std::size_t& i, uint8_t rex_r, uint8_t rex_x,
                         uint8_t rex_b, uint8_t& reg_out, Decoded& d) {
    if (i >= avail) return false;
    const uint8_t modrm = code[i++];
    const uint8_t mod = modrm >> 6;
    const uint8_t rm_low = modrm & 7;
    reg_out = uint8_t(((modrm >> 3) & 7) | (rex_r << 3));
    if (mod == 3) {
        d.mem = false;
        d.rm = uint8_t(rm_low | (rex_b << 3));
        return true;
    }
    d.mem = true;
    if (rm_low == 4) {
        if (i >= avail) return false;
        const uint8_t sib = code[i++];
        d.scale = uint8_t(1u << (sib >> 6));
        const uint8_t idx = uint8_t(((sib >> 3) & 7) | (rex_x << 3));
        d.index = (idx == 4) ? int8_t(-1) : int8_t(idx);  // RSP means no index; R12 (idx == 12) is valid
        const uint8_t base_low = sib & 7;
        if (mod == 0 && base_low == 5) {
            if (i + 4 > avail) return false;
            d.base = -1;
            std::memcpy(&d.disp, code + i, 4);
            i += 4;
            return true;
        }
        d.base = int8_t(base_low | (rex_b << 3));
    } else if (mod == 0 && rm_low == 5) {
        if (i + 4 > avail) return false;
        d.base = -2;  // RIP-relative (even when REX.B is set)
        std::memcpy(&d.disp, code + i, 4);
        i += 4;
        return true;
    } else {
        d.base = int8_t(rm_low | (rex_b << 3));
    }
    if (mod == 1) {
        if (i + 1 > avail) return false;
        d.disp = int8_t(code[i++]);
    } else if (mod == 2) {
        if (i + 4 > avail) return false;
        std::memcpy(&d.disp, code + i, 4);
        i += 4;
    }
    return true;
}

// Length of one x86-64 instruction at `code[0..avail)`, or 0 if truncated / invalid in 64-bit mode.
// Sets `tz_or_lz` to 1 for TZCNT (F3 0F BC /r), 2 for LZCNT (F3 0F BD /r), or 0 otherwise.
inline std::size_t insn_len64(const uint8_t* code, std::size_t avail, int* tz_or_lz = nullptr) {
    if (tz_or_lz) *tz_or_lz = 0;
    std::size_t i = 0;
    bool p66 = false, pf2 = false, pf3 = false, pf0 = false, addr32 = false, any_rex = false;
    uint8_t rex = 0;
    for (; i < avail && i < 14; ++i) {
        const uint8_t b = code[i];
        if ((b & 0xF0) == 0x40) {
            rex = b;
            any_rex = true;
            continue;
        }
        if (b == 0x66) p66 = true;
        else if (b == 0xF2) pf2 = true;
        else if (b == 0xF3) pf3 = true;
        else if (b == 0xF0) pf0 = true;
        else if (b == 0x67) addr32 = true;
        else if (b == 0x26 || b == 0x2E || b == 0x36 || b == 0x3E || b == 0x64 || b == 0x65) {}
        else break;
        rex = 0;
    }
    if (i >= avail) return 0;
    const auto modrm_tail = [&](std::size_t pos) -> std::size_t {
        if (pos >= avail) return 0;
        const uint8_t m = code[pos++];
        const uint8_t mod = m >> 6, rm = m & 7;
        if (mod != 3) {
            if (rm == 4) {
                if (pos >= avail) return 0;
                const uint8_t sib = code[pos++];
                if (mod == 0 && (sib & 7) == 5) pos += 4;
            } else if (mod == 0 && rm == 5) {
                pos += 4;
            }
            if (mod == 1) pos += 1;
            else if (mod == 2) pos += 4;
        }
        return pos <= avail ? pos : 0;
    };
    const uint8_t op = code[i++];
    if (op == 0xC4 || op == 0xC5) {
        if (p66 || pf2 || pf3 || pf0 || any_rex) return 0;
        uint8_t vmap = 1, vop = 0;
        if (op == 0xC5) {
            if (i + 2 > avail) return 0;
            vop = code[i + 1];
            i += 2;
        } else {
            if (i + 3 > avail) return 0;
            vmap = code[i] & 0x1F;
            vop = code[i + 2];
            i += 3;
        }
        if (vmap < 1 || vmap > 3) return 0;
        if (vmap == 1 && vop == 0x77) return i <= 15 ? i : 0;  // vzeroupper / vzeroall
        i = modrm_tail(i);
        if (!i) return 0;
        if (vmap == 3 || (vmap == 1 && (vop == 0x70 || vop == 0x71 || vop == 0x72 || vop == 0x73 ||
                                        vop == 0xC2 || vop == 0xC4 || vop == 0xC5 || vop == 0xC6)))
            ++i;
        return (i <= avail && i <= 15) ? i : 0;
    }
    const bool rex_w = (rex & 8) != 0;
    if (op == 0x0F) {
        if (i >= avail) return 0;
        const uint8_t op2 = code[i++];
        if (op2 == 0x38) {
            if (i >= avail) return 0;
            i = modrm_tail(i + 1);
            return (i && i <= 15) ? i : 0;
        }
        if (op2 == 0x3A) {
            if (i >= avail) return 0;
            i = modrm_tail(i + 1);
            if (!i) return 0;
            ++i;
            return (i <= avail && i <= 15) ? i : 0;
        }
        switch (op2) {
        case 0x05: case 0x06: case 0x07: case 0x08: case 0x09: case 0x0B: case 0x0E:
        case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35: case 0x37: case 0x77:
        case 0xA0: case 0xA1: case 0xA2: case 0xA8: case 0xA9: case 0xAA:
        case 0xC8: case 0xC9: case 0xCA: case 0xCB: case 0xCC: case 0xCD: case 0xCE: case 0xCF:
            return i <= 15 ? i : 0;
        default: break;
        }
        if (op2 >= 0x80 && op2 <= 0x8F) {
            i += (p66 && !rex_w) ? 2 : 4;
            return (i <= avail && i <= 15) ? i : 0;
        }
        const int kind = (pf3 && !pf2 && !pf0 && (op2 == 0xBC || op2 == 0xBD)) ? (op2 == 0xBC ? 1 : 2) : 0;
        i = modrm_tail(i);
        if (!i) return 0;
        if (op2 == 0x78 && (p66 || pf2) && !pf3) i += 2;
        else if (op2 == 0x0F || op2 == 0x70 || op2 == 0x71 || op2 == 0x72 || op2 == 0x73 || op2 == 0xA4 ||
                 op2 == 0xAC || op2 == 0xBA || op2 == 0xC2 || op2 == 0xC4 || op2 == 0xC5 || op2 == 0xC6)
            ++i;
        if (i > avail || i > 15) return 0;
        if (tz_or_lz) *tz_or_lz = kind;
        return i;
    }
    switch (op) {
    case 0x06: case 0x07: case 0x0E: case 0x16: case 0x17: case 0x1E: case 0x1F:
    case 0x27: case 0x2F: case 0x37: case 0x3F: case 0x60: case 0x61: case 0x62:
    case 0x82: case 0x9A: case 0xCE: case 0xD6: case 0xEA:
        return 0;  // invalid in 64-bit mode
    default: break;
    }
    if ((op >= 0x50 && op <= 0x5F) || (op >= 0x90 && op <= 0x9F) ||
        (op >= 0xA4 && op <= 0xA7) || (op >= 0xAA && op <= 0xAF) ||
        op == 0xC3 || op == 0xC9 || op == 0xCB || op == 0xCC || op == 0xCF || op == 0xD7 ||
        (op >= 0xEC && op <= 0xEF) || op == 0xF4 || op == 0xF5 || (op >= 0xF8 && op <= 0xFD))
        return i <= 15 ? i : 0;
    if ((op & 0xC7) == 0x04 || op == 0x6A || (op >= 0x70 && op <= 0x7F) || op == 0xA8 ||
        (op >= 0xB0 && op <= 0xB7) || op == 0xCD || op == 0xD4 || op == 0xD5 ||
        (op >= 0xE0 && op <= 0xE7) || op == 0xEB) {
        ++i;
        return (i <= avail && i <= 15) ? i : 0;
    }
    if (op == 0xC2 || op == 0xCA) {
        i += 2;
        return (i <= avail && i <= 15) ? i : 0;
    }
    if (op == 0xC8) {
        i += 3;
        return (i <= avail && i <= 15) ? i : 0;
    }
    if ((op & 0xC7) == 0x05 || op == 0x68 || op == 0xA9) {
        i += (p66 && !rex_w) ? 2 : 4;
        return (i <= avail && i <= 15) ? i : 0;
    }
    if (op == 0xE8 || op == 0xE9) {
        i += 4;
        return (i <= avail && i <= 15) ? i : 0;
    }
    if (op >= 0xB8 && op <= 0xBF) {
        i += rex_w ? 8 : (p66 ? 2 : 4);
        return (i <= avail && i <= 15) ? i : 0;
    }
    if (op >= 0xA0 && op <= 0xA3) {
        i += addr32 ? 4 : 8;
        return (i <= avail && i <= 15) ? i : 0;
    }
    if (i >= avail) return 0;
    const uint8_t reg_field = (code[i] >> 3) & 7;
    i = modrm_tail(i);
    if (!i) return 0;
    if (op == 0x80 || op == 0x83 || op == 0xC0 || op == 0xC1 || op == 0xC6 || op == 0x6B) ++i;
    else if (op == 0x81 || op == 0x69 || op == 0xC7) i += (p66 && !rex_w) ? 2 : 4;
    else if (op == 0xF6 && reg_field <= 1) ++i;
    else if (op == 0xF7 && reg_field <= 1) i += (p66 && !rex_w) ? 2 : 4;
    return (i <= avail && i <= 15) ? i : 0;
}

}  // namespace detail

inline Decoded decode(const uint8_t* code, std::size_t avail) {
    Decoded d;
    std::size_t i = 0;
    bool p66 = false, pf2 = false, pf3 = false, pf0 = false, any_rex = false;
    uint8_t rex = 0;
    for (; i < avail && i < 14; ++i) {
        const uint8_t b = code[i];
        if ((b & 0xF0) == 0x40) {
            rex = b;
            any_rex = true;
            continue;
        }
        if (b == 0x66) p66 = true;
        else if (b == 0xF2) pf2 = true;
        else if (b == 0xF3) pf3 = true;
        else if (b == 0xF0) pf0 = true;
        else if (b == 0x67) d.addr32 = true;
        else if (b == 0x64 || b == 0x65) d.seg = b;
        else if (b == 0x26 || b == 0x2E || b == 0x36 || b == 0x3E) d.seg = 0;
        else break;
        rex = 0;
    }
    if (pf0 || i >= avail) return {};

    // 3-byte VEX (0xC4): BMI1 and BMI2 GPR instructions live in maps 2 (0F38) and 3 (0F3A) with VEX.L == 0.
    if (code[i] == 0xC4) {
        if (p66 || pf2 || pf3 || any_rex || i + 4 > avail) return {};
        const uint8_t b1 = code[i + 1];
        const uint8_t b2 = code[i + 2];
        const uint8_t op = code[i + 3];
        i += 4;
        const uint8_t mmmmm = b1 & 0x1F;
        const uint8_t vex_l = (b2 >> 2) & 1;
        if (vex_l != 0 || (mmmmm != 2 && mmmmm != 3)) return {};
        const uint8_t rex_r = (~b1 >> 7) & 1;
        const uint8_t rex_x = (~b1 >> 6) & 1;
        const uint8_t rex_b = (~b1 >> 5) & 1;
        const uint8_t vex_w = (b2 >> 7) & 1;
        const uint8_t vvvv = (~b2 >> 3) & 0xF;
        const uint8_t pp = b2 & 3;
        d.size = vex_w ? 8 : 4;
        uint8_t reg = 0;
        if (!detail::decode_modrm(code, avail, i, rex_r, rex_x, rex_b, reg, d)) return {};
        if (mmmmm == 2) {
            if (op == 0xF2 && pp == 0) {  // ANDN r1, r2, r/m
                d.op = Op::andn;
                d.dst = reg;
                d.src1 = vvvv;
            } else if (op == 0xF3 && pp == 0) {  // BLSR /1, BLSMSK /2, BLSI /3 (dst = vvvv, src = r/m)
                if (reg == 1) d.op = Op::blsr;
                else if (reg == 2) d.op = Op::blsmsk;
                else if (reg == 3) d.op = Op::blsi;
                else return {};
                d.dst = vvvv;
            } else if (op == 0xF5) {
                if (pp == 0) {  // BZHI r1, r/m, r2
                    d.op = Op::bzhi;
                    d.dst = reg;
                    d.src1 = vvvv;
                } else if (pp == 3) {  // PDEP r1, r2, r/m
                    d.op = Op::pdep;
                    d.dst = reg;
                    d.src1 = vvvv;
                } else if (pp == 2) {  // PEXT r1, r2, r/m
                    d.op = Op::pext;
                    d.dst = reg;
                    d.src1 = vvvv;
                } else {
                    return {};
                }
            } else if (op == 0xF6 && pp == 3) {  // MULX r1_hi, r2_lo, r/m
                d.op = Op::mulx;
                d.dst = reg;
                d.dst2 = vvvv;
            } else if (op == 0xF7) {  // BEXTR (pp=0), SHLX (pp=1), SARX (pp=2), SHRX (pp=3): dst=reg, src=r/m, ctl=vvvv
                d.op = pp == 0 ? Op::bextr : pp == 1 ? Op::shlx : pp == 2 ? Op::sarx : Op::shrx;
                d.dst = reg;
                d.src1 = vvvv;
            } else {
                return {};
            }
        } else {  // mmmmm == 3 (0F3A)
            if (op != 0xF0 || pp != 3 || vvvv != 0 || i >= avail) return {};  // RORX r1, r/m, imm8 (vvvv must be 1111b)
            d.op = Op::rorx;
            d.dst = reg;
            d.imm = code[i++];
        }
        if (i > 15) return {};
        d.length = uint8_t(i);
        return d;
    }

    // Legacy 0F ...
    if (code[i] != 0x0F || i + 2 > avail) return {};
    const uint8_t op2 = code[i + 1];
    i += 2;
    const uint8_t rex_w = (rex >> 3) & 1;
    const uint8_t rex_r = (rex >> 2) & 1;
    const uint8_t rex_x = (rex >> 1) & 1;
    const uint8_t rex_b = rex & 1;
    d.size = rex_w ? 8 : (p66 ? 2 : 4);

    if (op2 == 0x38) {  // MOVBE: 0F 38 F0 /r (load) and 0F 38 F1 /r (store); F2 is CRC32, not MOVBE
        if (pf2 || pf3 || i >= avail) return {};
        const uint8_t op3 = code[i++];
        if (op3 != 0xF0 && op3 != 0xF1) return {};
        uint8_t reg = 0;
        if (!detail::decode_modrm(code, avail, i, rex_r, rex_x, rex_b, reg, d) || !d.mem || i > 15) return {};
        d.op = op3 == 0xF0 ? Op::movbe_load : Op::movbe_store;
        d.dst = reg;
        d.src1 = reg;
        d.length = uint8_t(i);
        return d;
    }

    if ((op2 == 0xBC || op2 == 0xBD) && pf3 && !pf2) {  // TZCNT (F3 0F BC /r), LZCNT (F3 0F BD /r)
        uint8_t reg = 0;
        if (!detail::decode_modrm(code, avail, i, rex_r, rex_x, rex_b, reg, d) || i > 15) return {};
        d.op = op2 == 0xBC ? Op::tzcnt : Op::lzcnt;
        d.dst = reg;
        d.length = uint8_t(i);
        return d;
    }

    return {};
}

inline uint64_t pdep_u64(uint64_t src, uint64_t mask) {
    uint64_t res = 0;
    for (uint64_t k = 1; mask != 0; k <<= 1) {
        const uint64_t bit = mask & (uint64_t(0) - mask);
        if (src & k) res |= bit;
        mask ^= bit;
    }
    return res;
}

inline uint64_t pext_u64(uint64_t src, uint64_t mask) {
    uint64_t res = 0;
    for (uint64_t k = 1; mask != 0; k <<= 1) {
        const uint64_t bit = mask & (uint64_t(0) - mask);
        if (src & bit) res |= k;
        mask ^= bit;
    }
    return res;
}

inline uint64_t count_tz(uint64_t v, unsigned bits) {
    const uint64_t masked = bits == 64 ? v : (v & ((uint64_t(1) << bits) - 1));
    if (!masked) return bits;
    unsigned n = 0;
    uint64_t x = masked;
    while ((x & 1u) == 0) {
        x >>= 1;
        ++n;
    }
    return n;
}

inline uint64_t count_lz(uint64_t v, unsigned bits) {
    const uint64_t masked = bits == 64 ? v : (v & ((uint64_t(1) << bits) - 1));
    if (!masked) return bits;
    unsigned n = 0;
    const uint64_t msb = uint64_t(1) << (bits - 1);
    uint64_t x = masked;
    while ((x & msb) == 0) {
        x <<= 1;
        ++n;
    }
    return n;
}

inline uint64_t bswap_val(uint64_t v, unsigned size) {
    if (size == 2) {
        const uint16_t x = uint16_t(v);
        return uint16_t((x >> 8) | (x << 8));
    }
    if (size == 4) {
        const uint32_t x = uint32_t(v);
        return ((x >> 24) & 0xFFu) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | ((x << 24) & 0xFF000000u);
    }
    return ((v >> 56) & 0xFFull) | ((v >> 40) & 0xFF00ull) | ((v >> 24) & 0xFF0000ull) | ((v >> 8) & 0xFF000000ull) |
           ((v << 8) & 0xFF00000000ull) | ((v << 24) & 0xFF0000000000ull) | ((v << 40) & 0xFF000000000000ull) |
           ((v << 56) & 0xFF00000000000000ull);
}

}  // namespace rt::bmi_emulation

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

#if defined(__clang__)
#define LEONIDA_BMI_NO_BUILTIN __attribute__((no_builtin))
#else
#define LEONIDA_BMI_NO_BUILTIN
#endif

namespace rt::bmi_emulation {

struct Host {
    bool bmi1 = false, bmi2 = false, lzcnt = false, movbe = false;
};

inline Host detect_host() {
    Host h;
    int r[4]{};
    __cpuid(r, 0);
    const int max_leaf = r[0];
    if (max_leaf >= 1) {
        __cpuid(r, 1);
        h.movbe = ((r[2] >> 22) & 1) != 0;  // CPUID.01H:ECX.MOVBE[bit 22]
    }
    if (max_leaf >= 7) {
        __cpuidex(r, 7, 0);
        h.bmi1 = ((r[1] >> 3) & 1) != 0;    // CPUID.07H:EBX.BMI1[bit 3]
        h.bmi2 = ((r[1] >> 8) & 1) != 0;    // CPUID.07H:EBX.BMI2[bit 8]
    }
    __cpuid(r, 0x80000000);
    if (unsigned(r[0]) >= 0x80000001u) {
        __cpuid(r, 0x80000001);
        h.lzcnt = ((r[2] >> 5) & 1) != 0;   // CPUID.80000001H:ECX.LZCNT[bit 5]
    }
    return h;
}

inline const char* name(Op op) {
    switch (op) {
    case Op::andn: return "ANDN (BMI1)";
    case Op::bextr: return "BEXTR (BMI1)";
    case Op::blsi: return "BLSI (BMI1)";
    case Op::blsmsk: return "BLSMSK (BMI1)";
    case Op::blsr: return "BLSR (BMI1)";
    case Op::bzhi: return "BZHI (BMI2)";
    case Op::mulx: return "MULX (BMI2)";
    case Op::pdep: return "PDEP (BMI2)";
    case Op::pext: return "PEXT (BMI2)";
    case Op::rorx: return "RORX (BMI2)";
    case Op::sarx: return "SARX (BMI2)";
    case Op::shlx: return "SHLX (BMI2)";
    case Op::shrx: return "SHRX (BMI2)";
    case Op::movbe_load: case Op::movbe_store: return "MOVBE";
    case Op::tzcnt: return "TZCNT (BMI1)";
    case Op::lzcnt: return "LZCNT";
    default: return "none";
    }
}

inline std::atomic<uint64_t>& count(Op op) {
    static std::atomic<uint64_t> counts[20];
    if (op == Op::movbe_store) op = Op::movbe_load;
    return counts[unsigned(op)];
}

namespace detail {

__declspec(noinline) LEONIDA_BMI_NO_BUILTIN inline void raw_copy(void* dst, const void* src, std::size_t bytes) {
    __movsb(static_cast<unsigned char*>(dst), static_cast<const unsigned char*>(src), bytes);
}

inline int capture_exception(EXCEPTION_POINTERS* ep, EXCEPTION_RECORD* out) {
    if (out && ep && ep->ExceptionRecord) *out = *ep->ExceptionRecord;
    return EXCEPTION_EXECUTE_HANDLER;
}

// Performs a real user-mode memory copy inside its own SEH frame (no C++ destructors in frame):
// - Valid pages cost zero syscalls.
// - First-touch xmem pages and write-watched D3D12 shadow pages fault through the front of the VEH chain, get
//   committed/reopened there, and complete transparently.
// - Genuine faults (unmapped, PAGE_NOACCESS, read-only write, user PAGE_GUARD) are caught and returned in `fault`.
inline bool safe_read(uint64_t addr, void* dst, std::size_t bytes, EXCEPTION_RECORD* fault) {
    EXCEPTION_RECORD local{};
    __try {
        raw_copy(dst, reinterpret_cast<const void*>(addr), bytes);
        return true;
    } __except (capture_exception(GetExceptionInformation(), &local)) {
        if (fault) *fault = local;
        return false;
    }
}

inline bool safe_write(uint64_t addr, const void* src, std::size_t bytes, EXCEPTION_RECORD* fault) {
    EXCEPTION_RECORD local{};
    __try {
        raw_copy(reinterpret_cast<void*>(addr), src, bytes);
        return true;
    } __except (capture_exception(GetExceptionInformation(), &local)) {
        if (fault) *fault = local;
        return false;
    }
}

inline std::size_t readable_bytes(uint64_t addr, std::size_t want) {
    std::size_t got = 0;
    while (got < want) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery(reinterpret_cast<void*>(addr + got), &mbi, sizeof mbi) || mbi.State != MEM_COMMIT ||
            (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) || !mbi.Protect)
            break;
        const uint64_t end = reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
        const std::size_t rem = want - got;
        got += std::size_t(end - (addr + got)) < rem ? std::size_t(end - (addr + got)) : rem;
    }
    return got;
}

inline uint64_t& gpr_ref(CONTEXT* c, unsigned r) {
    switch (r & 15u) {
    case 0: return c->Rax;   case 1: return c->Rcx;   case 2: return c->Rdx;   case 3: return c->Rbx;
    case 4: return c->Rsp;   case 5: return c->Rbp;   case 6: return c->Rsi;   case 7: return c->Rdi;
    case 8: return c->R8;    case 9: return c->R9;    case 10: return c->R10;  case 11: return c->R11;
    case 12: return c->R12;  case 13: return c->R13;  case 14: return c->R14;  case 15: return c->R15;
    default: return c->Rax;
    }
}

inline uint64_t gpr_get(const CONTEXT* c, unsigned r) {
    return gpr_ref(const_cast<CONTEXT*>(c), r);
}

inline void gpr_write(CONTEXT* c, unsigned r, uint64_t val, unsigned size) {
    uint64_t& dst = gpr_ref(c, r);
    if (size == 2) dst = (dst & ~uint64_t(0xFFFFu)) | (val & 0xFFFFu);
    else if (size == 4) dst = uint32_t(val);
    else dst = val;
}

inline uint64_t effective_addr(const Decoded& d, const CONTEXT* c) {
    uint64_t ea = 0;
    if (d.base == -2) ea = c->Rip + d.length;
    else if (d.base >= 0) ea = d.addr32 ? uint32_t(gpr_get(c, unsigned(d.base))) : gpr_get(c, unsigned(d.base));
    if (d.index >= 0) {
        const uint64_t idx = d.addr32 ? uint32_t(gpr_get(c, unsigned(d.index))) : gpr_get(c, unsigned(d.index));
        ea += idx * d.scale;
    }
    ea += uint64_t(int64_t(d.disp));
    if (d.addr32) ea = uint32_t(ea);
    if (d.seg == 0x65) ea += reinterpret_cast<uint64_t>(NtCurrentTeb());
    return ea;
}

// Clear CF(0), PF(2), AF(4), ZF(6), SF(7), OF(11) and set requested flags.
inline void set_flags(CONTEXT* c, bool cf, bool zf, bool sf) {
    uint32_t f = c->EFlags & ~uint32_t(0x8D5u);
    if (cf) f |= 0x01u;
    if (zf) f |= 0x40u;
    if (sf) f |= 0x80u;
    c->EFlags = f;
}

struct Ud2Entry {
    uint64_t va = 0;
    uint8_t len = 0;
    uint8_t bytes[15]{};
    bool operator<(const Ud2Entry& o) const { return va < o.va; }
};

struct Ud2Table {
    std::shared_mutex mutex;
    std::vector<Ud2Entry> entries;
};

inline Ud2Table& ud2_table() {
    static Ud2Table t;
    return t;
}

inline bool lookup_ud2(uint64_t va, uint8_t out[15], std::size_t& len_out) {
    Ud2Table& t = ud2_table();
    std::shared_lock lock(t.mutex);
    Ud2Entry key{};
    key.va = va;
    const auto it = std::lower_bound(t.entries.begin(), t.entries.end(), key);
    if (it == t.entries.end() || it->va != va) return false;
    std::memcpy(out, it->bytes, it->len);
    len_out = it->len;
    return true;
}

inline uint8_t* alloc_stub_pages(uint64_t lo, uint64_t hi, std::size_t size) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const uint64_t gran = si.dwAllocationGranularity;
    const uint64_t limit = uint64_t(1) << 31;
    for (uint64_t at = (hi + gran - 1) & ~(gran - 1); at + size < lo + limit - 0x10000; at += gran) {
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(at), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
            return static_cast<uint8_t*>(p);
    }
    for (uint64_t at = (lo > size ? (lo - size) & ~(gran - 1) : 0); at > gran && hi - at < limit - 0x10000; at -= gran) {
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(at), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
            return static_cast<uint8_t*>(p);
    }
    return nullptr;
}

// Emits a branchless native TZCNT or LZCNT stub for 32/64-bit register-to-register sites (dst != rsp, src != rsp).
inline std::vector<uint8_t> build_cnt_stub(bool is_lzcnt, bool w64, uint8_t dst, uint8_t src, uint64_t stub_va,
                                           uint64_t resume_va) {
    std::vector<uint8_t> out;
    out.reserve(32);
    // Pick a scratch register in {rcx(1), rdx(2), rax(0)} distinct from both dst and src (always < 8).
    uint8_t tmp = 1;
    if (tmp == dst || tmp == src) tmp = 2;
    if (tmp == dst || tmp == src) tmp = 0;

    const uint8_t width = w64 ? 64 : 32;
    const uint8_t bit = w64 ? 6 : 5;
    out.push_back(uint8_t(0x50 + tmp));  // push tmp

    if (!is_lzcnt) {
        // mov tmp32, width
        out.push_back(uint8_t(0xB8 + tmp));
        out.push_back(width);
        out.push_back(0);
        out.push_back(0);
        out.push_back(0);
        // bsf dst, src
        const uint8_t rex_bsf = uint8_t((w64 ? 0x48 : 0) | (dst >= 8 ? 0x44 : 0) | (src >= 8 ? 0x41 : 0));
        if (rex_bsf) out.push_back(rex_bsf);
        out.push_back(0x0F);
        out.push_back(0xBC);
        out.push_back(uint8_t(0xC0 | ((dst & 7) << 3) | (src & 7)));
        // cmove dst, tmp
        const uint8_t rex_cmov = uint8_t((w64 ? 0x48 : 0) | (dst >= 8 ? 0x44 : 0));
        if (rex_cmov) out.push_back(rex_cmov);
        out.push_back(0x0F);
        out.push_back(0x44);
        out.push_back(uint8_t(0xC0 | ((dst & 7) << 3) | tmp));
        // test dst, dst (sets ZF = (dst == 0), clears CF/OF)
        const uint8_t rex_test = uint8_t((w64 ? 0x48 : 0) | (dst >= 8 ? 0x45 : 0));
        if (rex_test) out.push_back(rex_test);
        out.push_back(0x85);
        out.push_back(uint8_t(0xC0 | ((dst & 7) << 3) | (dst & 7)));
    } else {
        // bsr tmp, src (leaves src intact even when dst == src)
        const uint8_t rex_bsr = uint8_t((w64 ? 0x48 : 0) | (src >= 8 ? 0x41 : 0));
        if (rex_bsr) out.push_back(rex_bsr);
        out.push_back(0x0F);
        out.push_back(0xBD);
        out.push_back(uint8_t(0xC0 | (tmp << 3) | (src & 7)));
        // mov dst32, 2*width - 1 (does not touch ZF from bsr; zero-extends dst to 64 bits)
        if (dst >= 8) out.push_back(0x41);
        out.push_back(uint8_t(0xB8 + (dst & 7)));
        out.push_back(uint8_t(2 * width - 1));
        out.push_back(0);
        out.push_back(0);
        out.push_back(0);
        // cmovne dst, tmp
        const uint8_t rex_cmov = uint8_t((w64 ? 0x48 : 0) | (dst >= 8 ? 0x44 : 0));
        if (rex_cmov) out.push_back(rex_cmov);
        out.push_back(0x0F);
        out.push_back(0x45);
        out.push_back(uint8_t(0xC0 | ((dst & 7) << 3) | tmp));
        // xor dst, width - 1 (yields clz in 0..width and sets ZF = (dst == 0))
        const uint8_t rex_xor = uint8_t((w64 ? 0x48 : 0) | (dst >= 8 ? 0x41 : 0));
        if (rex_xor) out.push_back(rex_xor);
        out.push_back(0x83);
        out.push_back(uint8_t(0xF0 | (dst & 7)));
        out.push_back(uint8_t(width - 1));
    }

    // bt dst, bit (sets CF = (dst == width), leaves ZF untouched)
    const uint8_t rex_bt = uint8_t((w64 ? 0x48 : 0) | (dst >= 8 ? 0x41 : 0));
    if (rex_bt) out.push_back(rex_bt);
    out.push_back(0x0F);
    out.push_back(0xBA);
    out.push_back(uint8_t(0xE0 | (dst & 7)));
    out.push_back(bit);

    out.push_back(uint8_t(0x58 + tmp));  // pop tmp
    out.push_back(0xE9);
    const int32_t rel = int32_t(int64_t(resume_va) - int64_t(stub_va + out.size() + 4));
    const auto* rb = reinterpret_cast<const uint8_t*>(&rel);
    out.insert(out.end(), rb, rb + 4);
    return out;
}

inline bool is_block_terminator(const uint8_t* ins, std::size_t len) {
    if (!len) return false;
    std::size_t p = 0;
    while (p + 1 < len) {
        const uint8_t b = ins[p];
        if ((b & 0xF0) == 0x40 || b == 0x66 || b == 0x67 || b == 0xF2 || b == 0xF3 ||
            b == 0x26 || b == 0x2E || b == 0x36 || b == 0x3E || b == 0x64 || b == 0x65)
            ++p;
        else
            break;
    }
    const uint8_t op = ins[p];
    if (op == 0xC3 || op == 0xC2 || op == 0xEB || op == 0xE9) return true;
    if (op == 0xFF && p + 1 < len && ((ins[p + 1] >> 3) & 7) == 4) return true;  // jmp r/m64
    if (op == 0x0F && p + 1 < len && ins[p + 1] == 0x0B) return true;            // ud2
    return false;
}

}  // namespace detail

inline bool apply(const Decoded& d, CONTEXT* c, EXCEPTION_RECORD* fault = nullptr) {
    if (d.op == Op::none) return false;
    const unsigned bits = unsigned(d.size) * 8u;
    const uint64_t mask_n = bits == 64 ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1u);

    if (d.op == Op::movbe_store) {
        const uint64_t swapped = bswap_val(detail::gpr_get(c, d.src1), d.size);
        const uint64_t addr = detail::effective_addr(d, c);
        if (!detail::safe_write(addr, &swapped, d.size, fault)) return false;
        c->Rip += d.length;
        return true;
    }

    uint64_t rm_val = 0;
    if (d.mem) {
        const uint64_t addr = detail::effective_addr(d, c);
        if (!detail::safe_read(addr, &rm_val, d.size, fault)) return false;
    } else {
        rm_val = detail::gpr_get(c, d.rm) & mask_n;
    }

    const uint64_t msb = uint64_t(1) << (bits - 1u);
    switch (d.op) {
    case Op::andn: {
        const uint64_t s1 = detail::gpr_get(c, d.src1) & mask_n;
        const uint64_t res = (~s1 & rm_val) & mask_n;
        detail::gpr_write(c, d.dst, res, d.size);
        detail::set_flags(c, false, res == 0, (res & msb) != 0);
        break;
    }
    case Op::bextr: {
        const uint64_t ctl = detail::gpr_get(c, d.src1);
        const unsigned start = unsigned(ctl & 0xFFu);
        const unsigned len = unsigned((ctl >> 8) & 0xFFu);
        uint64_t res = 0;
        if (start < bits && len != 0) {
            const uint64_t shifted = (rm_val & mask_n) >> start;
            res = len >= bits ? shifted : (shifted & ((uint64_t(1) << len) - 1u));
        }
        detail::gpr_write(c, d.dst, res, d.size);
        detail::set_flags(c, false, res == 0, false);
        break;
    }
    case Op::blsi: {
        const uint64_t s = rm_val & mask_n;
        const uint64_t res = ((uint64_t(0) - s) & s) & mask_n;
        detail::gpr_write(c, d.dst, res, d.size);
        detail::set_flags(c, s != 0, res == 0, (res & msb) != 0);
        break;
    }
    case Op::blsmsk: {
        const uint64_t s = rm_val & mask_n;
        const uint64_t res = ((s - 1u) ^ s) & mask_n;
        detail::gpr_write(c, d.dst, res, d.size);
        detail::set_flags(c, s == 0, res == 0, (res & msb) != 0);
        break;
    }
    case Op::blsr: {
        const uint64_t s = rm_val & mask_n;
        const uint64_t res = ((s - 1u) & s) & mask_n;
        detail::gpr_write(c, d.dst, res, d.size);
        detail::set_flags(c, s == 0, res == 0, (res & msb) != 0);
        break;
    }
    case Op::bzhi: {
        const unsigned idx = unsigned(detail::gpr_get(c, d.src1) & 0xFFu);
        const bool cf = idx >= bits;
        const uint64_t res = cf ? (rm_val & mask_n) : (rm_val & ((uint64_t(1) << idx) - 1u));
        detail::gpr_write(c, d.dst, res, d.size);
        detail::set_flags(c, cf, res == 0, (res & msb) != 0);
        break;
    }
    case Op::mulx: {
        const uint64_t edx_rdx = detail::gpr_get(c, 2) & mask_n;
        uint64_t lo = 0, hi = 0;
        if (d.size == 4) {
            const uint64_t prod = uint64_t(uint32_t(edx_rdx)) * uint64_t(uint32_t(rm_val));
            lo = uint32_t(prod);
            hi = uint32_t(prod >> 32);
        } else {
            lo = _umul128(edx_rdx, rm_val, &hi);
        }
        // Write low half first so that when both operands name the same GPR, the high half wins (Intel SDM).
        detail::gpr_write(c, d.dst2, lo, d.size);
        detail::gpr_write(c, d.dst, hi, d.size);
        break;
    }
    case Op::pdep: {
        const uint64_t s1 = detail::gpr_get(c, d.src1) & mask_n;
        detail::gpr_write(c, d.dst, pdep_u64(s1, rm_val & mask_n), d.size);
        break;
    }
    case Op::pext: {
        const uint64_t s1 = detail::gpr_get(c, d.src1) & mask_n;
        detail::gpr_write(c, d.dst, pext_u64(s1, rm_val & mask_n), d.size);
        break;
    }
    case Op::rorx: {
        const unsigned cnt = d.imm & (bits - 1u);
        const uint64_t s = rm_val & mask_n;
        const uint64_t res = cnt == 0 ? s : (((s >> cnt) | (s << (bits - cnt))) & mask_n);
        detail::gpr_write(c, d.dst, res, d.size);
        break;
    }
    case Op::sarx: {
        const unsigned cnt = unsigned(detail::gpr_get(c, d.src1)) & (bits - 1u);
        const uint64_t res = d.size == 4 ? uint32_t(int32_t(uint32_t(rm_val)) >> cnt)
                                         : uint64_t(int64_t(rm_val) >> cnt);
        detail::gpr_write(c, d.dst, res, d.size);
        break;
    }
    case Op::shlx: {
        const unsigned cnt = unsigned(detail::gpr_get(c, d.src1)) & (bits - 1u);
        detail::gpr_write(c, d.dst, ((rm_val & mask_n) << cnt) & mask_n, d.size);
        break;
    }
    case Op::shrx: {
        const unsigned cnt = unsigned(detail::gpr_get(c, d.src1)) & (bits - 1u);
        detail::gpr_write(c, d.dst, (rm_val & mask_n) >> cnt, d.size);
        break;
    }
    case Op::movbe_load:
        detail::gpr_write(c, d.dst, bswap_val(rm_val, d.size), d.size);
        break;
    case Op::tzcnt: {
        const uint64_t s = rm_val & mask_n;
        const uint64_t res = count_tz(s, bits);
        detail::gpr_write(c, d.dst, res, d.size);
        detail::set_flags(c, s == 0, res == 0, false);
        break;
    }
    case Op::lzcnt: {
        const uint64_t s = rm_val & mask_n;
        const uint64_t res = count_lz(s, bits);
        detail::gpr_write(c, d.dst, res, d.size);
        detail::set_flags(c, s == 0, res == 0, false);
        break;
    }
    default:
        return false;
    }
    c->Rip += d.length;
    return true;
}

inline bool lzcnt_patch_enabled() {
    char v[8]{};
    std::size_t n = 0;
    return !(getenv_s(&n, v, "LEONIDA_LZCNT_PATCH") == 0 && n == 2 && v[0] == '0');
}

struct PatchResult {
    std::size_t tzcnt_sites = 0;
    std::size_t lzcnt_sites = 0;
    std::size_t stub_sites = 0;
    std::size_t ud2_sites = 0;
    uint8_t* stubs = nullptr;
    std::size_t stub_bytes = 0;
};

// Patches TZCNT (when `patch_tzcnt`) and LZCNT (when `patch_lzcnt`) across `.pdata` functions and 0xCC-delimited
// leaf blocks inside `exec_sections` (pairs of section RVA and byte size within `[base, base + image_size)`).
inline PatchResult patch_image(uint8_t* base, uint32_t image_size,
                               const std::vector<std::pair<uint32_t, uint32_t>>& exec_sections,
                               const RUNTIME_FUNCTION* pdata, uint32_t pdata_count, bool patch_tzcnt,
                               bool patch_lzcnt) {
    PatchResult res;
    if (!base || !image_size || (!patch_tzcnt && !patch_lzcnt) || exec_sections.empty()) return res;

    const auto has_candidate_bytes = [&](const uint8_t* p, std::size_t len) -> bool {
        for (std::size_t k = 0; k + 2 <= len; ++k) {
            if (p[k] == 0x0F && ((patch_tzcnt && p[k + 1] == 0xBC) || (patch_lzcnt && p[k + 1] == 0xBD)))
                return true;
        }
        return false;
    };

    struct FoundSite {
        uint32_t rva;
        Decoded dec;
    };
    std::vector<FoundSite> found;

    const auto scan_block = [&](uint32_t beg_rva, uint32_t end_rva, bool require_terminator) {
        const std::size_t len = end_rva - beg_rva;
        const uint8_t* block = base + beg_rva;
        if (!has_candidate_bytes(block, len)) return;
        std::vector<FoundSite> local;
        std::size_t pos = 0;
        std::size_t last_pos = 0, last_len = 0;
        while (pos < len) {
            int kind = 0;
            const std::size_t ilen = detail::insn_len64(block + pos, len - pos, &kind);
            if (!ilen) return;
            if ((kind == 1 && patch_tzcnt) || (kind == 2 && patch_lzcnt)) {
                const Decoded d = decode(block + pos, ilen);
                if (d.op != Op::none && d.length == ilen) local.push_back({beg_rva + uint32_t(pos), d});
            }
            last_pos = pos;
            last_len = ilen;
            pos += ilen;
        }
        if (pos != len || local.empty()) return;
        if (require_terminator && !detail::is_block_terminator(block + last_pos, last_len)) return;
        found.insert(found.end(), local.begin(), local.end());
    };

    for (const auto& [sec_rva, sec_size] : exec_sections) {
        if (sec_rva >= image_size || !sec_size) continue;
        const uint32_t sec_end = uint32_t(std::min<uint64_t>(uint64_t(sec_rva) + sec_size, image_size));
        std::vector<std::pair<uint32_t, uint32_t>> funcs;
        if (pdata && pdata_count) {
            for (uint32_t i = 0; i < pdata_count; ++i) {
                const uint32_t b = pdata[i].BeginAddress, e = pdata[i].EndAddress;
                if (e > b && b >= sec_rva && e <= sec_end) funcs.push_back({b, e});
            }
            std::sort(funcs.begin(), funcs.end());
        }
        uint32_t cur = sec_rva;
        const auto scan_gap = [&](uint32_t gb, uint32_t ge) {
            const uint8_t* g = base + gb;
            const std::size_t glen = ge - gb;
            if (!has_candidate_bytes(g, glen)) return;
            std::size_t p = 0;
            while (p < glen) {
                while (p < glen && g[p] == 0xCC) ++p;
                if (p >= glen) break;
                std::size_t sub_end = p;
                while (sub_end < glen && !(sub_end + 1 < glen && g[sub_end] == 0xCC && g[sub_end + 1] == 0xCC))
                    ++sub_end;
                while (sub_end > p && g[sub_end - 1] == 0xCC) --sub_end;
                if (sub_end > p) scan_block(gb + uint32_t(p), gb + uint32_t(sub_end), true);
                p = sub_end + 1;
            }
        };
        for (const auto& [fb, fe] : funcs) {
            if (fb > cur) scan_gap(cur, fb);
            if (fe > cur) {
                scan_block(std::max(fb, cur), fe, false);
                cur = fe;
            }
        }
        if (cur < sec_end) scan_gap(cur, sec_end);
    }

    if (found.empty()) return res;

    std::size_t want_stubs = 0;
    for (const auto& s : found) {
        if (s.dec.length >= 5 && !s.dec.mem && (s.dec.size == 4 || s.dec.size == 8) && s.dec.dst != 4 && s.dec.rm != 4)
            ++want_stubs;
    }

    uint8_t* stubs = nullptr;
    std::size_t stub_cap = 0;
    std::size_t stub_used = 0;
    if (want_stubs) {
        stub_cap = (want_stubs * 32u + 0xFFFFu) & ~std::size_t(0xFFFFu);
        stubs = detail::alloc_stub_pages(reinterpret_cast<uint64_t>(base),
                                         reinterpret_cast<uint64_t>(base) + image_size, stub_cap);
    }

    std::vector<detail::Ud2Entry> new_ud2;
    for (const auto& s : found) {
        if (s.dec.op == Op::tzcnt) ++res.tzcnt_sites;
        else ++res.lzcnt_sites;

        uint8_t* site_ptr = base + s.rva;
        const uint64_t site_va = reinterpret_cast<uint64_t>(site_ptr);
        const bool can_stub = stubs && s.dec.length >= 5 && !s.dec.mem && (s.dec.size == 4 || s.dec.size == 8) &&
                              s.dec.dst != 4 && s.dec.rm != 4;
        if (can_stub) {
            const uint64_t stub_va = reinterpret_cast<uint64_t>(stubs + stub_used);
            const auto body = detail::build_cnt_stub(s.dec.op == Op::lzcnt, s.dec.size == 8, s.dec.dst, s.dec.rm,
                                                     stub_va, site_va + s.dec.length);
            const int64_t rel64 = int64_t(stub_va) - int64_t(site_va + 5);
            if (stub_used + body.size() <= stub_cap && rel64 >= INT32_MIN && rel64 <= INT32_MAX) {
                std::memcpy(stubs + stub_used, body.data(), body.size());
                stub_used += (body.size() + 15u) & ~std::size_t(15u);
                const int32_t rel32 = int32_t(rel64);
                site_ptr[0] = 0xE9;
                std::memcpy(site_ptr + 1, &rel32, 4);
                for (uint8_t k = 5; k < s.dec.length; ++k) site_ptr[k] = 0x90;
                ++res.stub_sites;
                continue;
            }
        }

        detail::Ud2Entry e{};
        e.va = site_va;
        e.len = s.dec.length;
        std::memcpy(e.bytes, site_ptr, s.dec.length);
        new_ud2.push_back(e);
        site_ptr[0] = 0x0F;
        site_ptr[1] = 0x0B;  // UD2 -> STATUS_ILLEGAL_INSTRUCTION (never collides with 0xCC STATUS_BREAKPOINT)
        ++res.ud2_sites;
    }

    if (stubs) {
        if (res.stub_sites) {
            DWORD old = 0;
            VirtualProtect(stubs, stub_cap, PAGE_EXECUTE_READ, &old);
            FlushInstructionCache(GetCurrentProcess(), stubs, stub_used);
            res.stubs = stubs;
            res.stub_bytes = stub_used;
        } else {
            VirtualFree(stubs, 0, MEM_RELEASE);
        }
    }

    if (!new_ud2.empty()) {
        detail::Ud2Table& t = detail::ud2_table();
        std::unique_lock lock(t.mutex);
        const uint64_t img_lo = reinterpret_cast<uint64_t>(base);
        const uint64_t img_hi = img_lo + image_size;
        t.entries.erase(std::remove_if(t.entries.begin(), t.entries.end(),
                                       [&](const detail::Ud2Entry& x) { return x.va >= img_lo && x.va < img_hi; }),
                        t.entries.end());
        t.entries.insert(t.entries.end(), new_ud2.begin(), new_ud2.end());
        std::sort(t.entries.begin(), t.entries.end());
    }
    return res;
}

inline LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != STATUS_ILLEGAL_INSTRUCTION) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT* c = info->ContextRecord;
    uint8_t code[15]{};
    std::size_t avail = sizeof code;
    if ((c->Rip & 0xFFFu) <= 0x1000u - sizeof code) {
        if (!detail::safe_read(c->Rip, code, sizeof code, nullptr)) return EXCEPTION_CONTINUE_SEARCH;
    } else {
        avail = detail::readable_bytes(c->Rip, sizeof code);
        if (avail < 2) return EXCEPTION_CONTINUE_SEARCH;
        std::memcpy(code, reinterpret_cast<const void*>(c->Rip), avail);
    }

    Decoded d;
    if (code[0] == 0x0F && code[1] == 0x0B) {
        uint8_t orig[15]{};
        std::size_t orig_len = 0;
        if (!detail::lookup_ud2(c->Rip, orig, orig_len)) return EXCEPTION_CONTINUE_SEARCH;
        d = decode(orig, orig_len);
    } else {
        d = decode(code, avail);
    }
    if (d.op == Op::none) return EXCEPTION_CONTINUE_SEARCH;

    const uint64_t fault_rip = c->Rip;
    EXCEPTION_RECORD fault{};
    if (!apply(d, c, &fault)) {
        EXCEPTION_RECORD* r = info->ExceptionRecord;
        r->ExceptionCode = fault.ExceptionCode;
        r->ExceptionFlags = 0;
        r->NumberParameters = fault.NumberParameters;
        for (DWORD k = 0; k < fault.NumberParameters && k < EXCEPTION_MAXIMUM_PARAMETERS; ++k)
            r->ExceptionInformation[k] = fault.ExceptionInformation[k];
        return EXCEPTION_CONTINUE_SEARCH;
    }

    if (count(d.op).fetch_add(1, std::memory_order_relaxed) == 0)
        std::fprintf(stderr, "[rt] cpu: this CPU has no %s; emulating it (first at %p)\n", name(d.op),
                     reinterpret_cast<void*>(fault_rip));
    return EXCEPTION_CONTINUE_EXECUTION;
}

inline bool install() {
    static void* const cookie = AddVectoredExceptionHandler(1, &handler);
    return cookie != nullptr;
}

}  // namespace rt::bmi_emulation
#endif
