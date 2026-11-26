// Emulates F16C, FMA3 and AVX2 instructions on an x86-64 host that has AVX1 + SSE4.2 + OSXSAVE/YMM state
// (docs/cpu_features.md).
//
// Pre-Haswell AVX1 + SSE4.2 hosts (Intel Sandy Bridge and Ivy Bridge) already provide 256-bit YMM0..YMM15 registers
// saved by Windows in XSTATE_AVX (XCR0 bits 1-2), and execute all 128-bit and 256-bit AVX1 instructions natively.
// Every F16C, FMA3 and AVX2 instruction (including register-source VBROADCASTSS/VBROADCASTSD) raises #UD
// (STATUS_ILLEGAL_INSTRUCTION) cleanly on such a host.
//
// A vectored handler decodes the VEX-encoded instruction at RIP, reads and writes full 256-bit YMM state through
// CONTEXT::FltSave.XmmRegisters (bits [127:0]) and LocateXStateFeature(XSTATE_AVX) (bits [255:128]), zeroes
// YMM[dst][255:128] on every 128-bit VEX register write as required by the x86-64 VEX specification, and advances RIP.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rt::avx2_emulation {

enum class Category : uint8_t {
    none = 0,
    f16c,
    fma3,
    avx2,
};

struct Decoded {
    Category cat = Category::none;
    uint8_t vmap = 0;      // 1 = 0F, 2 = 0F38, 3 = 0F3A
    uint8_t op = 0;        // primary opcode byte
    uint8_t pp = 0;        // 0 = none, 1 = 66, 2 = F3, 3 = F2
    uint8_t vex_l = 0;     // 0 = 128-bit, 1 = 256-bit
    uint8_t vex_w = 0;     // 0 or 1
    uint8_t reg = 0;       // ModRM.reg (0..15)
    uint8_t vvvv = 0;      // VEX.vvvv (0..15, already inverted)
    uint8_t rm = 0;        // ModRM.rm (0..15) when !mem
    uint8_t imm = 0;       // imm8 when vmap == 3 or map-1 immediate op
    uint8_t length = 0;    // total instruction length in bytes
    uint8_t mem_bytes = 0; // memory operand byte width when mem (0 for per-lane vpmaskmov*)
    bool mem = false;      // ModRM.mod != 3
    bool addr32 = false;   // 0x67 address-size override
    uint8_t seg = 0;       // 0x64 (FS) or 0x65 (GS), else 0
    int8_t base = -1;      // 0..15 GPR, -1 none, -2 RIP-relative
    int8_t index = -1;     // 0..15 GPR, -1 none
    uint8_t scale = 1;     // 1, 2, 4 or 8
    int32_t disp = 0;
};

union alignas(32) Ymm {
    uint8_t u8[32];
    int8_t i8[32];
    uint16_t u16[16];
    int16_t i16[16];
    uint32_t u32[8];
    int32_t i32[8];
    uint64_t u64[4];
    int64_t i64[4];
    float f32[8];
    double f64[4];
};

namespace detail {

inline bool decode_modrm(const uint8_t* code, std::size_t avail, std::size_t& i, uint8_t rex_r, uint8_t rex_x,
                         uint8_t rex_b, Decoded& d) {
    if (i >= avail) return false;
    const uint8_t modrm = code[i++];
    const uint8_t mod = modrm >> 6;
    const uint8_t rm_low = modrm & 7;
    d.reg = uint8_t(((modrm >> 3) & 7) | (rex_r << 3));
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
        d.index = (idx == 4) ? int8_t(-1) : int8_t(idx);
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
        d.base = -2;
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

inline int16_t sat_i16(int32_t x) { return x > 32767 ? int16_t(32767) : x < -32768 ? int16_t(-32768) : int16_t(x); }
inline int8_t sat_i8(int32_t x) { return x > 127 ? int8_t(127) : x < -128 ? int8_t(-128) : int8_t(x); }
inline uint8_t sat_u8(int32_t x) { return x > 255 ? uint8_t(255) : x < 0 ? uint8_t(0) : uint8_t(x); }
inline uint16_t sat_u16(int32_t x) { return x > 65535 ? uint16_t(65535) : x < 0 ? uint16_t(0) : uint16_t(x); }

}  // namespace detail

// IEEE-754 binary16 -> binary32 (exact, preserves signed zero, Inf and NaN payload).
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03FFu;
            bits = sign | (uint32_t(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = mant == 0 ? (sign | 0x7F800000u) : (sign | 0x7FC00000u | (mant << 13));
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

// IEEE-754 binary32 -> binary16 with rounding control `rc` (0=RN, 1=RD, 2=RU, 3=RZ).
inline uint16_t f32_to_f16(float f, unsigned rc) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof bits);
    const uint16_t sign = uint16_t((bits >> 16) & 0x8000u);
    const uint32_t raw_exp = (bits >> 23) & 0xFFu;
    const uint32_t mant = bits & 0x7FFFFFu;
    if (raw_exp == 0xFFu) {
        if (mant == 0) return uint16_t(sign | 0x7C00u);
        return uint16_t(sign | 0x7E00u | uint16_t(mant >> 13));
    }
    int exp = int(raw_exp) - 127;
    if (raw_exp == 0 && mant == 0) return sign;

    const auto overflow_val = [&]() -> uint16_t {
        switch (rc & 3u) {
        case 0: return uint16_t(sign | 0x7C00u);
        case 1: return sign ? uint16_t(sign | 0x7C00u) : uint16_t(0x7BFFu);
        case 2: return sign ? uint16_t(sign | 0x7BFFu) : uint16_t(0x7C00u);
        default: return uint16_t(sign | 0x7BFFu);
        }
    };
    if (exp > 15) return overflow_val();

    uint32_t sig = raw_exp == 0 ? mant : (0x800000u | mant);
    if (raw_exp == 0) exp = -126;

    uint32_t q = 0;
    bool round_bit = false, sticky = false;
    if (exp >= -14) {
        q = sig >> 13;
        round_bit = ((sig >> 12) & 1u) != 0;
        sticky = (sig & 0xFFFu) != 0;
    } else {
        const unsigned shift = unsigned(13 + (-14 - exp));
        if (shift >= 25) {
            q = 0;
            round_bit = false;
            sticky = sig != 0;
        } else {
            q = sig >> shift;
            round_bit = ((sig >> (shift - 1u)) & 1u) != 0;
            sticky = (sig & ((uint32_t(1) << (shift - 1u)) - 1u)) != 0;
        }
    }

    bool inc = false;
    switch (rc & 3u) {
    case 0: inc = round_bit && (sticky || (q & 1u)); break;
    case 1: inc = (sign != 0) && (round_bit || sticky); break;
    case 2: inc = (sign == 0) && (round_bit || sticky); break;
    default: inc = false; break;
    }

    uint32_t packed = (exp >= -14 ? (uint32_t(exp + 14) << 10) + q : q) + (inc ? 1u : 0u);
    if (packed >= 0x7C00u) return overflow_val();
    return uint16_t(sign | uint16_t(packed));
}

inline Decoded decode(const uint8_t* code, std::size_t avail) {
    Decoded d;
    std::size_t i = 0;
    for (; i < avail && i < 14; ++i) {
        const uint8_t b = code[i];
        if (b == 0x67) d.addr32 = true;
        else if (b == 0x64 || b == 0x65) d.seg = b;
        else if (b == 0x26 || b == 0x2E || b == 0x36 || b == 0x3E) d.seg = 0;
        else break;
    }
    if (i >= avail || (code[i] != 0xC4 && code[i] != 0xC5)) return {};

    uint8_t rex_r = 0, rex_x = 0, rex_b = 0;
    if (code[i] == 0xC5) {
        if (i + 3 > avail) return {};
        const uint8_t b1 = code[i + 1];
        rex_r = (~b1 >> 7) & 1;
        d.vvvv = (~b1 >> 3) & 0xF;
        d.vex_l = (b1 >> 2) & 1;
        d.pp = b1 & 3;
        d.vmap = 1;
        d.vex_w = 0;
        d.op = code[i + 2];
        i += 3;
    } else {
        if (i + 4 > avail) return {};
        const uint8_t b1 = code[i + 1];
        const uint8_t b2 = code[i + 2];
        rex_r = (~b1 >> 7) & 1;
        rex_x = (~b1 >> 6) & 1;
        rex_b = (~b1 >> 5) & 1;
        d.vmap = b1 & 0x1F;
        d.vex_w = (b2 >> 7) & 1;
        d.vvvv = (~b2 >> 3) & 0xF;
        d.vex_l = (b2 >> 2) & 1;
        d.pp = b2 & 3;
        d.op = code[i + 3];
        i += 4;
    }

    if (d.vmap < 1 || d.vmap > 3) return {};
    if (!detail::decode_modrm(code, avail, i, rex_r, rex_x, rex_b, d)) return {};

    // Classify and determine immediate + memory operand sizes.
    bool has_imm = false;
    if (d.vmap == 1) {
        // All 128-bit map-1 VEX instructions are AVX1 (native on Sandy Bridge); AVX2 adds VEX.L == 1 integer ops.
        if (d.vex_l != 1) return {};
        if (d.op == 0x70 && (d.pp == 1 || d.pp == 2 || d.pp == 3)) {  // vpshufd (66), vpshufhw (F3), vpshuflw (F2)
            if (d.vvvv != 0) return {};
            d.cat = Category::avx2;
            d.mem_bytes = 32;
            has_imm = true;
        } else if ((d.op == 0x71 || d.op == 0x72 || d.op == 0x73) && d.pp == 1) {
            if (d.mem) return {};
            const uint8_t sub = d.reg & 7;
            const bool ok = (d.op == 0x71 || d.op == 0x72) ? (sub == 2 || sub == 4 || sub == 6)
                                                           : (sub == 2 || sub == 3 || sub == 6 || sub == 7);
            if (!ok || (d.reg >= 8)) return {};
            d.cat = Category::avx2;
            has_imm = true;
        } else if (d.op == 0xD7 && d.pp == 1) {  // vpmovmskb r32/64, ymm
            if (d.mem || d.vvvv != 0) return {};
            d.cat = Category::avx2;
        } else if (d.pp == 1) {
            switch (d.op) {
            case 0xD1: case 0xD2: case 0xD3: case 0xE1: case 0xE2: case 0xF1: case 0xF2: case 0xF3:
                d.cat = Category::avx2;
                d.mem_bytes = 16;  // XMM/m128 shift count even for 256-bit YMM shifts
                break;
            case 0x60: case 0x61: case 0x62: case 0x63: case 0x64: case 0x65: case 0x66: case 0x67:
            case 0x68: case 0x69: case 0x6A: case 0x6B: case 0x6C: case 0x6D:
            case 0x74: case 0x75: case 0x76:
            case 0xD4: case 0xD5: case 0xD8: case 0xD9: case 0xDA: case 0xDB: case 0xDC: case 0xDD: case 0xDE: case 0xDF:
            case 0xE0: case 0xE3: case 0xE4: case 0xE5: case 0xE8: case 0xE9: case 0xEA: case 0xEB: case 0xEC: case 0xED:
            case 0xEE: case 0xEF:
            case 0xF4: case 0xF5: case 0xF6: case 0xF8: case 0xF9: case 0xFA: case 0xFB: case 0xFC: case 0xFD: case 0xFE:
                d.cat = Category::avx2;
                d.mem_bytes = 32;
                break;
            default:
                return {};
            }
        } else {
            return {};
        }
    } else if (d.vmap == 2) {
        if (d.pp != 1) return {};
        // F16C: vcvtph2ps (0F38 13, W=0, vvvv=0)
        if (d.op == 0x13) {
            if (d.vex_w != 0 || d.vvvv != 0) return {};
            d.cat = Category::f16c;
            d.mem_bytes = d.vex_l ? 16 : 8;
        }
        // FMA3: 0x96..0x9F, 0xA6..0xAF, 0xB6..0xBF
        else if (((d.op & 0xF0) == 0x90 || (d.op & 0xF0) == 0xA0 || (d.op & 0xF0) == 0xB0) && (d.op & 0x0F) >= 6) {
            const uint8_t lo = d.op & 0x0F;
            const bool scalar = (lo >= 8) && ((lo & 1u) != 0);
            if (scalar && d.vex_l != 0) return {};
            d.cat = Category::fma3;
            d.mem_bytes = scalar ? (d.vex_w ? 8 : 4) : (d.vex_l ? 32 : 16);
        }
        // Broadcasts & permutes & variable shifts & masked moves
        else if (d.op == 0x18 && d.vex_w == 0 && d.vvvv == 0) {  // vbroadcastss
            d.cat = Category::avx2;
            d.mem_bytes = 4;
        } else if (d.op == 0x19 && d.vex_w == 0 && d.vvvv == 0 && d.vex_l == 1) {  // vbroadcastsd
            d.cat = Category::avx2;
            d.mem_bytes = 8;
        } else if (d.op == 0x5A && d.vex_w == 0 && d.vvvv == 0 && d.vex_l == 1 && d.mem) {  // vbroadcasti128
            d.cat = Category::avx2;
            d.mem_bytes = 16;
        } else if ((d.op == 0x78 || d.op == 0x79 || d.op == 0x58 || d.op == 0x59) && d.vex_w == 0 && d.vvvv == 0) {
            d.cat = Category::avx2;
            d.mem_bytes = d.op == 0x78 ? 1 : d.op == 0x79 ? 2 : d.op == 0x58 ? 4 : 8;
        } else if ((d.op == 0x16 || d.op == 0x36) && d.vex_w == 0 && d.vex_l == 1) {  // vpermps, vpermd
            d.cat = Category::avx2;
            d.mem_bytes = 32;
        } else if (d.op == 0x45 || d.op == 0x47 || (d.op == 0x46 && d.vex_w == 0)) {  // vpsrlv*, vpsllv*, vpsravd
            d.cat = Category::avx2;
            d.mem_bytes = d.vex_l ? 32 : 16;
        } else if ((d.op == 0x8C || d.op == 0x8E) && d.mem) {  // vpmaskmovd / vpmaskmovq
            d.cat = Category::avx2;
            d.mem_bytes = 0;  // per-lane masked access
        } else if (d.vex_l == 1) {
            if ((d.op >= 0x20 && d.op <= 0x25) || (d.op >= 0x30 && d.op <= 0x35)) {
                if (d.vvvv != 0) return {};
                const uint8_t k = d.op & 0x0F;
                d.cat = Category::avx2;
                d.mem_bytes = (k == 0 || k == 3 || k == 5) ? 16 : (k == 1 || k == 4) ? 8 : 4;
            } else if ((d.op == 0x1C || d.op == 0x1D || d.op == 0x1E) && d.vvvv == 0) {  // vpabsb/w/d
                d.cat = Category::avx2;
                d.mem_bytes = 32;
            } else {
                switch (d.op) {
                case 0x00: case 0x01: case 0x02: case 0x03: case 0x04: case 0x05: case 0x06: case 0x07:
                case 0x08: case 0x09: case 0x0A: case 0x0B:
                case 0x28: case 0x29: case 0x2B: case 0x37:
                case 0x38: case 0x39: case 0x3A: case 0x3B: case 0x3C: case 0x3D: case 0x3E: case 0x3F:
                case 0x40:
                    d.cat = Category::avx2;
                    d.mem_bytes = 32;
                    break;
                default:
                    return {};
                }
            }
        } else {
            return {};
        }
    } else {  // d.vmap == 3 (0F3A, all have imm8)
        if (d.pp != 1) return {};
        has_imm = true;
        if (d.op == 0x1D && d.vex_w == 0 && d.vvvv == 0) {  // vcvtps2ph
            d.cat = Category::f16c;
            d.mem_bytes = d.vex_l ? 16 : 8;
        } else if (d.op == 0x02 && d.vex_w == 0) {  // vpblendd (128 & 256)
            d.cat = Category::avx2;
            d.mem_bytes = d.vex_l ? 32 : 16;
        } else if ((d.op == 0x00 || d.op == 0x01) && d.vex_w == 1 && d.vvvv == 0 && d.vex_l == 1) {  // vpermq, vpermpd
            d.cat = Category::avx2;
            d.mem_bytes = 32;
        } else if (d.op == 0x38 && d.vex_w == 0 && d.vex_l == 1) {  // vinserti128
            d.cat = Category::avx2;
            d.mem_bytes = 16;
        } else if (d.op == 0x39 && d.vex_w == 0 && d.vvvv == 0 && d.vex_l == 1) {  // vextracti128
            d.cat = Category::avx2;
            d.mem_bytes = 16;
        } else if ((d.op == 0x46 && d.vex_w == 0 && d.vex_l == 1) ||  // vperm2i128
                   (d.op == 0x0E && d.vex_l == 1) ||                  // vpblendw
                   (d.op == 0x0F && d.vex_l == 1) ||                  // vpalignr
                   (d.op == 0x4C && d.vex_w == 0 && d.vex_l == 1)) {  // vpblendvb
            d.cat = Category::avx2;
            d.mem_bytes = 32;
        } else {
            return {};
        }
    }

    if (has_imm) {
        if (i >= avail) return {};
        d.imm = code[i++];
    }
    if (i > 15) return {};
    d.length = uint8_t(i);
    return d;
}

}  // namespace rt::avx2_emulation

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <cstdio>

#if defined(__clang__)
#define LEONIDA_AVX2_NO_BUILTIN __attribute__((no_builtin))
#else
#define LEONIDA_AVX2_NO_BUILTIN
#endif

namespace rt::avx2_emulation {

struct Host {
    bool avx2 = false, fma = false, f16c = false;
};

inline Host detect_host() {
    Host h;
    int r[4]{};
    __cpuid(r, 0);
    const int max_leaf = r[0];
    if (max_leaf >= 1) {
        __cpuid(r, 1);
        h.fma = ((r[2] >> 12) & 1) != 0;   // CPUID.01H:ECX.FMA[bit 12]
        h.f16c = ((r[2] >> 29) & 1) != 0;  // CPUID.01H:ECX.F16C[bit 29]
    }
    if (max_leaf >= 7) {
        __cpuidex(r, 7, 0);
        h.avx2 = ((r[1] >> 5) & 1) != 0;   // CPUID.07H:EBX.AVX2[bit 5]
    }
    return h;
}

inline const char* category_name(Category c) {
    switch (c) {
    case Category::f16c: return "F16C";
    case Category::fma3: return "FMA3";
    case Category::avx2: return "AVX2";
    default: return "none";
    }
}

inline std::atomic<uint64_t>& count(Category c) {
    static std::atomic<uint64_t> counts[4];
    return counts[unsigned(c) & 3u];
}

namespace detail {

__declspec(noinline) LEONIDA_AVX2_NO_BUILTIN inline void raw_copy(void* dst, const void* src, std::size_t bytes) {
    __movsb(static_cast<unsigned char*>(dst), static_cast<const unsigned char*>(src), bytes);
}

inline int capture_exception(EXCEPTION_POINTERS* ep, EXCEPTION_RECORD* out) {
    if (out && ep && ep->ExceptionRecord) *out = *ep->ExceptionRecord;
    return EXCEPTION_EXECUTE_HANDLER;
}

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

inline uint64_t effective_addr(const Decoded& d, const CONTEXT* c) {
    uint64_t ea = 0;
    if (d.base == -2) ea = c->Rip + d.length;
    else if (d.base >= 0) {
        const uint64_t b = gpr_ref(const_cast<CONTEXT*>(c), unsigned(d.base));
        ea = d.addr32 ? uint32_t(b) : b;
    }
    if (d.index >= 0) {
        const uint64_t idx = gpr_ref(const_cast<CONTEXT*>(c), unsigned(d.index));
        ea += (d.addr32 ? uint32_t(idx) : idx) * d.scale;
    }
    ea += uint64_t(int64_t(d.disp));
    if (d.addr32) ea = uint32_t(ea);
    if (d.seg == 0x65) ea += reinterpret_cast<uint64_t>(NtCurrentTeb());
    return ea;
}

inline Ymm read_ymm(const CONTEXT* c, unsigned r) {
    Ymm out{};
    std::memcpy(&out.u8[0], &c->FltSave.XmmRegisters[r & 15u], 16);
    DWORD64 mask = 0;
    if (GetXStateFeaturesMask(const_cast<CONTEXT*>(c), &mask) && (mask & XSTATE_MASK_AVX)) {
        DWORD len = 0;
        const auto* hi = static_cast<const M128A*>(LocateXStateFeature(const_cast<CONTEXT*>(c), XSTATE_AVX, &len));
        if (hi && len >= 16 * sizeof(M128A)) std::memcpy(&out.u8[16], &hi[r & 15u], 16);
    }
    return out;
}

// Writes XMM[r] and updates YMM[r][255:128]:
// - For 128-bit VEX writes (`full256 == false`), zeroes YMM[r][255:128].
// - For 256-bit VEX writes (`full256 == true`), writes `v.u8[16..31]` to XSTATE_AVX, zero-initializing the other
//   15 upper lanes first if XSTATE_MASK_AVX was previously clean (0).
inline void write_vec(CONTEXT* c, unsigned r, const Ymm& v, bool full256) {
    r &= 15u;
    std::memcpy(&c->FltSave.XmmRegisters[r], &v.u8[0], 16);
    DWORD64 mask = 0;
    const bool has_xstate = GetXStateFeaturesMask(c, &mask) != FALSE;
    DWORD len = 0;
    auto* hi = has_xstate ? static_cast<M128A*>(LocateXStateFeature(c, XSTATE_AVX, &len)) : nullptr;
    if (!hi || len < 16 * sizeof(M128A)) return;
    if (!full256) {
        if (mask & XSTATE_MASK_AVX) std::memset(&hi[r], 0, 16);
        return;
    }
    if ((mask & XSTATE_MASK_AVX) == 0) {
        std::memset(hi, 0, 16 * sizeof(M128A));
        SetXStateFeaturesMask(c, mask | XSTATE_MASK_AVX);
    }
    std::memcpy(&hi[r], &v.u8[16], 16);
}

// Executes one 128-bit lane (`lane` in {0, 1}) of a lane-wise 256-bit integer op (map 1 or map 2).
inline void exec_lane128_map1(uint8_t op, uint8_t pp, uint8_t imm, const uint8_t* a8, const uint8_t* b8, uint8_t* out8) {
    Ymm a{}, b{}, r{};
    std::memcpy(a.u8, a8, 16);
    std::memcpy(b.u8, b8, 16);
    switch (op) {
    case 0x60: for (int k = 0; k < 8; ++k) { r.u8[2 * k] = a.u8[k]; r.u8[2 * k + 1] = b.u8[k]; } break;
    case 0x68: for (int k = 0; k < 8; ++k) { r.u8[2 * k] = a.u8[k + 8]; r.u8[2 * k + 1] = b.u8[k + 8]; } break;
    case 0x61: for (int k = 0; k < 4; ++k) { r.u16[2 * k] = a.u16[k]; r.u16[2 * k + 1] = b.u16[k]; } break;
    case 0x69: for (int k = 0; k < 4; ++k) { r.u16[2 * k] = a.u16[k + 4]; r.u16[2 * k + 1] = b.u16[k + 4]; } break;
    case 0x62: for (int k = 0; k < 2; ++k) { r.u32[2 * k] = a.u32[k]; r.u32[2 * k + 1] = b.u32[k]; } break;
    case 0x6A: for (int k = 0; k < 2; ++k) { r.u32[2 * k] = a.u32[k + 2]; r.u32[2 * k + 1] = b.u32[k + 2]; } break;
    case 0x6C: r.u64[0] = a.u64[0]; r.u64[1] = b.u64[0]; break;
    case 0x6D: r.u64[0] = a.u64[1]; r.u64[1] = b.u64[1]; break;
    case 0x63:
        for (int k = 0; k < 8; ++k) r.i8[k] = sat_i8(a.i16[k]);
        for (int k = 0; k < 8; ++k) r.i8[8 + k] = sat_i8(b.i16[k]);
        break;
    case 0x67:
        for (int k = 0; k < 8; ++k) r.u8[k] = sat_u8(a.i16[k]);
        for (int k = 0; k < 8; ++k) r.u8[8 + k] = sat_u8(b.i16[k]);
        break;
    case 0x6B:
        for (int k = 0; k < 4; ++k) r.i16[k] = sat_i16(a.i32[k]);
        for (int k = 0; k < 4; ++k) r.i16[4 + k] = sat_i16(b.i32[k]);
        break;
    case 0x64: for (int k = 0; k < 16; ++k) r.u8[k] = a.i8[k] > b.i8[k] ? 0xFFu : 0u; break;
    case 0x65: for (int k = 0; k < 8; ++k) r.u16[k] = a.i16[k] > b.i16[k] ? 0xFFFFu : 0u; break;
    case 0x66: for (int k = 0; k < 4; ++k) r.u32[k] = a.i32[k] > b.i32[k] ? 0xFFFFFFFFu : 0u; break;
    case 0x74: for (int k = 0; k < 16; ++k) r.u8[k] = a.u8[k] == b.u8[k] ? 0xFFu : 0u; break;
    case 0x75: for (int k = 0; k < 8; ++k) r.u16[k] = a.u16[k] == b.u16[k] ? 0xFFFFu : 0u; break;
    case 0x76: for (int k = 0; k < 4; ++k) r.u32[k] = a.u32[k] == b.u32[k] ? 0xFFFFFFFFu : 0u; break;
    case 0x70:
        if (pp == 1) {  // vpshufd: source is b
            for (int k = 0; k < 4; ++k) r.u32[k] = b.u32[(imm >> (2 * k)) & 3];
        } else if (pp == 2) {  // vpshufhw
            r.u64[0] = b.u64[0];
            for (int k = 0; k < 4; ++k) r.u16[4 + k] = b.u16[4 + ((imm >> (2 * k)) & 3)];
        } else {  // vpshuflw
            for (int k = 0; k < 4; ++k) r.u16[k] = b.u16[(imm >> (2 * k)) & 3];
            r.u64[1] = b.u64[1];
        }
        break;
    case 0xDB: for (int k = 0; k < 2; ++k) r.u64[k] = a.u64[k] & b.u64[k]; break;
    case 0xDF: for (int k = 0; k < 2; ++k) r.u64[k] = (~a.u64[k]) & b.u64[k]; break;
    case 0xEB: for (int k = 0; k < 2; ++k) r.u64[k] = a.u64[k] | b.u64[k]; break;
    case 0xEF: for (int k = 0; k < 2; ++k) r.u64[k] = a.u64[k] ^ b.u64[k]; break;
    case 0xFC: for (int k = 0; k < 16; ++k) r.u8[k] = uint8_t(a.u8[k] + b.u8[k]); break;
    case 0xFD: for (int k = 0; k < 8; ++k) r.u16[k] = uint16_t(a.u16[k] + b.u16[k]); break;
    case 0xFE: for (int k = 0; k < 4; ++k) r.u32[k] = a.u32[k] + b.u32[k]; break;
    case 0xD4: for (int k = 0; k < 2; ++k) r.u64[k] = a.u64[k] + b.u64[k]; break;
    case 0xF8: for (int k = 0; k < 16; ++k) r.u8[k] = uint8_t(a.u8[k] - b.u8[k]); break;
    case 0xF9: for (int k = 0; k < 8; ++k) r.u16[k] = uint16_t(a.u16[k] - b.u16[k]); break;
    case 0xFA: for (int k = 0; k < 4; ++k) r.u32[k] = a.u32[k] - b.u32[k]; break;
    case 0xFB: for (int k = 0; k < 2; ++k) r.u64[k] = a.u64[k] - b.u64[k]; break;
    case 0xEC: for (int k = 0; k < 16; ++k) r.i8[k] = sat_i8(int32_t(a.i8[k]) + int32_t(b.i8[k])); break;
    case 0xED: for (int k = 0; k < 8; ++k) r.i16[k] = sat_i16(int32_t(a.i16[k]) + int32_t(b.i16[k])); break;
    case 0xDC: for (int k = 0; k < 16; ++k) r.u8[k] = sat_u8(int32_t(a.u8[k]) + int32_t(b.u8[k])); break;
    case 0xDD: for (int k = 0; k < 8; ++k) r.u16[k] = sat_u16(int32_t(a.u16[k]) + int32_t(b.u16[k])); break;
    case 0xE8: for (int k = 0; k < 16; ++k) r.i8[k] = sat_i8(int32_t(a.i8[k]) - int32_t(b.i8[k])); break;
    case 0xE9: for (int k = 0; k < 8; ++k) r.i16[k] = sat_i16(int32_t(a.i16[k]) - int32_t(b.i16[k])); break;
    case 0xD8: for (int k = 0; k < 16; ++k) r.u8[k] = sat_u8(int32_t(a.u8[k]) - int32_t(b.u8[k])); break;
    case 0xD9: for (int k = 0; k < 8; ++k) r.u16[k] = sat_u16(int32_t(a.u16[k]) - int32_t(b.u16[k])); break;
    case 0xE0: for (int k = 0; k < 16; ++k) r.u8[k] = uint8_t((uint32_t(a.u8[k]) + uint32_t(b.u8[k]) + 1u) >> 1); break;
    case 0xE3: for (int k = 0; k < 8; ++k) r.u16[k] = uint16_t((uint32_t(a.u16[k]) + uint32_t(b.u16[k]) + 1u) >> 1); break;
    case 0xDA: for (int k = 0; k < 16; ++k) r.u8[k] = (std::min)(a.u8[k], b.u8[k]); break;
    case 0xDE: for (int k = 0; k < 16; ++k) r.u8[k] = (std::max)(a.u8[k], b.u8[k]); break;
    case 0xEA: for (int k = 0; k < 8; ++k) r.i16[k] = (std::min)(a.i16[k], b.i16[k]); break;
    case 0xEE: for (int k = 0; k < 8; ++k) r.i16[k] = (std::max)(a.i16[k], b.i16[k]); break;
    case 0xD5: for (int k = 0; k < 8; ++k) r.u16[k] = uint16_t(uint32_t(a.u16[k]) * uint32_t(b.u16[k])); break;
    case 0xE4: for (int k = 0; k < 8; ++k) r.u16[k] = uint16_t((uint32_t(a.u16[k]) * uint32_t(b.u16[k])) >> 16); break;
    case 0xE5: for (int k = 0; k < 8; ++k) r.i16[k] = int16_t((int32_t(a.i16[k]) * int32_t(b.i16[k])) >> 16); break;
    case 0xF4:
        r.u64[0] = uint64_t(a.u32[0]) * uint64_t(b.u32[0]);
        r.u64[1] = uint64_t(a.u32[2]) * uint64_t(b.u32[2]);
        break;
    case 0xF5:
        for (int k = 0; k < 4; ++k)
            r.i32[k] = int32_t(a.i16[2 * k]) * int32_t(b.i16[2 * k]) +
                       int32_t(a.i16[2 * k + 1]) * int32_t(b.i16[2 * k + 1]);
        break;
    case 0xF6: {
        uint64_t s0 = 0, s1 = 0;
        for (int k = 0; k < 8; ++k) {
            s0 += uint64_t(a.u8[k] > b.u8[k] ? a.u8[k] - b.u8[k] : b.u8[k] - a.u8[k]);
            s1 += uint64_t(a.u8[8 + k] > b.u8[8 + k] ? a.u8[8 + k] - b.u8[8 + k] : b.u8[8 + k] - a.u8[8 + k]);
        }
        r.u64[0] = s0;
        r.u64[1] = s1;
        break;
    }
    default: break;
    }
    std::memcpy(out8, r.u8, 16);
}

inline void exec_lane128_map2(uint8_t op, const uint8_t* a8, const uint8_t* b8, uint8_t* out8) {
    Ymm a{}, b{}, r{};
    std::memcpy(a.u8, a8, 16);
    std::memcpy(b.u8, b8, 16);
    switch (op) {
    case 0x00:
        for (int k = 0; k < 16; ++k) r.u8[k] = (b.u8[k] & 0x80u) ? 0u : a.u8[b.u8[k] & 15u];
        break;
    case 0x01:
        for (int k = 0; k < 4; ++k) r.u16[k] = uint16_t(a.u16[2 * k] + a.u16[2 * k + 1]);
        for (int k = 0; k < 4; ++k) r.u16[4 + k] = uint16_t(b.u16[2 * k] + b.u16[2 * k + 1]);
        break;
    case 0x02:
        for (int k = 0; k < 2; ++k) r.u32[k] = a.u32[2 * k] + a.u32[2 * k + 1];
        for (int k = 0; k < 2; ++k) r.u32[2 + k] = b.u32[2 * k] + b.u32[2 * k + 1];
        break;
    case 0x03:
        for (int k = 0; k < 4; ++k) r.i16[k] = sat_i16(int32_t(a.i16[2 * k]) + int32_t(a.i16[2 * k + 1]));
        for (int k = 0; k < 4; ++k) r.i16[4 + k] = sat_i16(int32_t(b.i16[2 * k]) + int32_t(b.i16[2 * k + 1]));
        break;
    case 0x04:
        for (int k = 0; k < 8; ++k)
            r.i16[k] = sat_i16(int32_t(a.u8[2 * k]) * int32_t(b.i8[2 * k]) +
                               int32_t(a.u8[2 * k + 1]) * int32_t(b.i8[2 * k + 1]));
        break;
    case 0x05:
        for (int k = 0; k < 4; ++k) r.u16[k] = uint16_t(a.u16[2 * k] - a.u16[2 * k + 1]);
        for (int k = 0; k < 4; ++k) r.u16[4 + k] = uint16_t(b.u16[2 * k] - b.u16[2 * k + 1]);
        break;
    case 0x06:
        for (int k = 0; k < 2; ++k) r.u32[k] = a.u32[2 * k] - a.u32[2 * k + 1];
        for (int k = 0; k < 2; ++k) r.u32[2 + k] = b.u32[2 * k] - b.u32[2 * k + 1];
        break;
    case 0x07:
        for (int k = 0; k < 4; ++k) r.i16[k] = sat_i16(int32_t(a.i16[2 * k]) - int32_t(a.i16[2 * k + 1]));
        for (int k = 0; k < 4; ++k) r.i16[4 + k] = sat_i16(int32_t(b.i16[2 * k]) - int32_t(b.i16[2 * k + 1]));
        break;
    case 0x08:
        for (int k = 0; k < 16; ++k) r.i8[k] = b.i8[k] < 0 ? int8_t(-int32_t(a.i8[k])) : (b.i8[k] == 0 ? 0 : a.i8[k]);
        break;
    case 0x09:
        for (int k = 0; k < 8; ++k) r.i16[k] = b.i16[k] < 0 ? int16_t(-int32_t(a.i16[k])) : (b.i16[k] == 0 ? 0 : a.i16[k]);
        break;
    case 0x0A:
        for (int k = 0; k < 4; ++k) r.i32[k] = b.i32[k] < 0 ? int32_t(0u - a.u32[k]) : (b.i32[k] == 0 ? 0 : a.i32[k]);
        break;
    case 0x0B:
        for (int k = 0; k < 8; ++k) r.i16[k] = int16_t(((int32_t(a.i16[k]) * int32_t(b.i16[k])) + 0x4000) >> 15);
        break;
    case 0x1C: for (int k = 0; k < 16; ++k) r.u8[k] = uint8_t(b.i8[k] < 0 ? -int32_t(b.i8[k]) : b.i8[k]); break;
    case 0x1D: for (int k = 0; k < 8; ++k) r.u16[k] = uint16_t(b.i16[k] < 0 ? -int32_t(b.i16[k]) : b.i16[k]); break;
    case 0x1E: for (int k = 0; k < 4; ++k) r.u32[k] = b.i32[k] < 0 ? (0u - b.u32[k]) : b.u32[k]; break;
    case 0x28:
        r.i64[0] = int64_t(a.i32[0]) * int64_t(b.i32[0]);
        r.i64[1] = int64_t(a.i32[2]) * int64_t(b.i32[2]);
        break;
    case 0x29: for (int k = 0; k < 2; ++k) r.u64[k] = a.u64[k] == b.u64[k] ? ~uint64_t(0) : 0ull; break;
    case 0x2B:
        for (int k = 0; k < 4; ++k) r.u16[k] = sat_u16(a.i32[k]);
        for (int k = 0; k < 4; ++k) r.u16[4 + k] = sat_u16(b.i32[k]);
        break;
    case 0x37: for (int k = 0; k < 2; ++k) r.u64[k] = a.i64[k] > b.i64[k] ? ~uint64_t(0) : 0ull; break;
    case 0x38: for (int k = 0; k < 16; ++k) r.i8[k] = (std::min)(a.i8[k], b.i8[k]); break;
    case 0x39: for (int k = 0; k < 4; ++k) r.i32[k] = (std::min)(a.i32[k], b.i32[k]); break;
    case 0x3A: for (int k = 0; k < 8; ++k) r.u16[k] = (std::min)(a.u16[k], b.u16[k]); break;
    case 0x3B: for (int k = 0; k < 4; ++k) r.u32[k] = (std::min)(a.u32[k], b.u32[k]); break;
    case 0x3C: for (int k = 0; k < 16; ++k) r.i8[k] = (std::max)(a.i8[k], b.i8[k]); break;
    case 0x3D: for (int k = 0; k < 4; ++k) r.i32[k] = (std::max)(a.i32[k], b.i32[k]); break;
    case 0x3E: for (int k = 0; k < 8; ++k) r.u16[k] = (std::max)(a.u16[k], b.u16[k]); break;
    case 0x3F: for (int k = 0; k < 4; ++k) r.u32[k] = (std::max)(a.u32[k], b.u32[k]); break;
    case 0x40: for (int k = 0; k < 4; ++k) r.u32[k] = a.u32[k] * b.u32[k]; break;
    default: break;
    }
    std::memcpy(out8, r.u8, 16);
}

inline void exec_shift_count(Ymm& out, const Ymm& src, uint64_t cnt, unsigned elem_bits, int mode) {
    // mode: 0 = logical right, 1 = arithmetic right, 2 = logical left
    if (elem_bits == 16) {
        for (int k = 0; k < 16; ++k) {
            if (mode == 2) out.u16[k] = cnt >= 16 ? 0u : uint16_t(src.u16[k] << cnt);
            else if (mode == 0) out.u16[k] = cnt >= 16 ? 0u : uint16_t(src.u16[k] >> cnt);
            else out.i16[k] = cnt >= 16 ? int16_t(src.i16[k] < 0 ? -1 : 0) : int16_t(src.i16[k] >> cnt);
        }
    } else if (elem_bits == 32) {
        for (int k = 0; k < 8; ++k) {
            if (mode == 2) out.u32[k] = cnt >= 32 ? 0u : (src.u32[k] << cnt);
            else if (mode == 0) out.u32[k] = cnt >= 32 ? 0u : (src.u32[k] >> cnt);
            else out.i32[k] = cnt >= 32 ? (src.i32[k] < 0 ? -1 : 0) : (src.i32[k] >> cnt);
        }
    } else {
        for (int k = 0; k < 4; ++k) {
            out.u64[k] = cnt >= 64 ? 0ull : (mode == 2 ? (src.u64[k] << cnt) : (src.u64[k] >> cnt));
        }
    }
}

}  // namespace detail

inline bool apply(const Decoded& d, CONTEXT* c, EXCEPTION_RECORD* fault = nullptr) {
    if (d.cat == Category::none) return false;
    const bool full256 = d.vex_l == 1;

    // Special memory-store instructions first (vcvtps2ph, vextracti128, vpmaskmov* store)
    if (d.vmap == 3 && d.op == 0x1D) {  // vcvtps2ph xmm/m, xmm/ymm, imm8 (src = reg, dst = rm/mem)
        const Ymm src = detail::read_ymm(c, d.reg);
        const unsigned rc = (d.imm & 4u) ? ((c->FltSave.MxCsr >> 13) & 3u) : (d.imm & 3u);
        Ymm dst{};
        const int lanes = full256 ? 8 : 4;
        for (int k = 0; k < lanes; ++k) dst.u16[k] = f32_to_f16(src.f32[k], rc);
        if (d.mem) {
            if (!detail::safe_write(detail::effective_addr(d, c), dst.u8, full256 ? 16 : 8, fault)) return false;
        } else {
            detail::write_vec(c, d.rm, dst, false);  // always an XMM destination; zeroes YMM[rm][255:128]
        }
        c->Rip += d.length;
        return true;
    }

    if (d.vmap == 3 && d.op == 0x39) {  // vextracti128 xmm/m128, ymm, imm8 (src = reg, dst = rm/mem)
        const Ymm src = detail::read_ymm(c, d.reg);
        Ymm dst{};
        std::memcpy(dst.u8, (d.imm & 1u) ? &src.u8[16] : &src.u8[0], 16);
        if (d.mem) {
            if (!detail::safe_write(detail::effective_addr(d, c), dst.u8, 16, fault)) return false;
        } else {
            detail::write_vec(c, d.rm, dst, false);
        }
        c->Rip += d.length;
        return true;
    }

    if (d.vmap == 2 && (d.op == 0x8C || d.op == 0x8E)) {  // vpmaskmovd / vpmaskmovq (faults ONLY on enabled lanes!)
        const Ymm mask = detail::read_ymm(c, d.vvvv);
        const uint64_t base_addr = detail::effective_addr(d, c);
        const bool is_q = d.vex_w != 0;
        const int lanes = full256 ? (is_q ? 4 : 8) : (is_q ? 2 : 4);
        const std::size_t elem_sz = is_q ? 8 : 4;
        if (d.op == 0x8C) {
            Ymm dst{};
            for (int k = 0; k < lanes; ++k) {
                const bool en = is_q ? (mask.i64[k] < 0) : (mask.i32[k] < 0);
                if (en && !detail::safe_read(base_addr + std::size_t(k) * elem_sz, &dst.u8[std::size_t(k) * elem_sz],
                                             elem_sz, fault))
                    return false;
            }
            detail::write_vec(c, d.reg, dst, full256);
        } else {
            const Ymm src = detail::read_ymm(c, d.reg);
            for (int k = 0; k < lanes; ++k) {
                const bool en = is_q ? (mask.i64[k] < 0) : (mask.i32[k] < 0);
                if (en && !detail::safe_write(base_addr + std::size_t(k) * elem_sz, &src.u8[std::size_t(k) * elem_sz],
                                              elem_sz, fault))
                    return false;
            }
        }
        c->Rip += d.length;
        return true;
    }

    // Load r/m operand (register or `d.mem_bytes` from memory)
    Ymm rm_vec{};
    if (d.mem) {
        if (!detail::safe_read(detail::effective_addr(d, c), rm_vec.u8, d.mem_bytes, fault)) return false;
    } else {
        rm_vec = detail::read_ymm(c, d.rm);
    }

    if (d.cat == Category::f16c) {  // vcvtph2ps xmm/ymm, xmm/m64/m128
        Ymm out{};
        const int lanes = full256 ? 8 : 4;
        for (int k = 0; k < lanes; ++k) out.f32[k] = f16_to_f32(rm_vec.u16[k]);
        detail::write_vec(c, d.reg, out, full256);
        c->Rip += d.length;
        return true;
    }

    if (d.cat == Category::fma3) {
        const Ymm r1 = detail::read_ymm(c, d.reg);
        const Ymm r2 = detail::read_ymm(c, d.vvvv);
        const Ymm& r3 = rm_vec;
        const uint8_t hi_nib = d.op & 0xF0u;
        const uint8_t lo_nib = d.op & 0x0Fu;
        const Ymm& A = (hi_nib == 0x90u) ? r1 : r2;
        const Ymm& B = (hi_nib == 0xA0u) ? r1 : r3;
        const Ymm& C = (hi_nib == 0x90u) ? r2 : (hi_nib == 0xA0u) ? r3 : r1;
        const bool scalar = (lo_nib >= 8) && ((lo_nib & 1u) != 0);
        Ymm out = scalar ? r1 : Ymm{};

        if (d.vex_w == 0) {
            const int lanes = scalar ? 1 : (full256 ? 8 : 4);
            for (int k = 0; k < lanes; ++k) {
                float a = A.f32[k], b = B.f32[k], cv = C.f32[k];
                if (lo_nib == 6) { if ((k & 1) == 0) cv = -cv; }       // fmaddsub: odd +, even -
                else if (lo_nib == 7) { if ((k & 1) != 0) cv = -cv; }  // fmsubadd: odd -, even +
                else {
                    if (lo_nib >= 0x0Cu) a = -a;                       // fnmadd / fnmsub
                    if (lo_nib == 0x0Au || lo_nib == 0x0Bu || lo_nib == 0x0Eu || lo_nib == 0x0Fu) cv = -cv;
                }
                out.f32[k] = std::fmaf(a, b, cv);
            }
        } else {
            const int lanes = scalar ? 1 : (full256 ? 4 : 2);
            for (int k = 0; k < lanes; ++k) {
                double a = A.f64[k], b = B.f64[k], cv = C.f64[k];
                if (lo_nib == 6) { if ((k & 1) == 0) cv = -cv; }
                else if (lo_nib == 7) { if ((k & 1) != 0) cv = -cv; }
                else {
                    if (lo_nib >= 0x0Cu) a = -a;
                    if (lo_nib == 0x0Au || lo_nib == 0x0Bu || lo_nib == 0x0Eu || lo_nib == 0x0Fu) cv = -cv;
                }
                out.f64[k] = std::fma(a, b, cv);
            }
        }
        detail::write_vec(c, d.reg, out, full256);
        c->Rip += d.length;
        return true;
    }

    // AVX2
    Ymm out{};
    const Ymm v1 = detail::read_ymm(c, d.vvvv);

    if (d.vmap == 1) {
        if (d.op == 0xD7) {  // vpmovmskb r32/64, ymm
            uint32_t m = 0;
            for (int k = 0; k < 32; ++k)
                if (rm_vec.u8[k] & 0x80u) m |= (uint32_t(1) << k);
            detail::gpr_ref(c, d.reg) = m;  // zero-extended to 64 bits
            c->Rip += d.length;
            return true;
        }
        if (d.op == 0x71 || d.op == 0x72 || d.op == 0x73) {  // immediate shifts: dst = vvvv, src = rm_vec
            const uint8_t sub = d.reg & 7u;
            if (d.op == 0x73 && (sub == 3 || sub == 7)) {
                const unsigned sh = d.imm;
                for (int lane = 0; lane < 2; ++lane) {
                    const uint8_t* s = &rm_vec.u8[lane * 16];
                    uint8_t* dst = &out.u8[lane * 16];
                    if (sh < 16) {
                        if (sub == 7) std::memcpy(dst + sh, s, 16 - sh);  // vpslldq
                        else std::memcpy(dst, s + sh, 16 - sh);           // vpsrldq
                    }
                }
            } else {
                const unsigned bits = d.op == 0x71 ? 16 : d.op == 0x72 ? 32 : 64;
                const int mode = sub == 2 ? 0 : sub == 4 ? 1 : 2;
                detail::exec_shift_count(out, rm_vec, d.imm, bits, mode);
            }
            detail::write_vec(c, d.vvvv, out, true);
            c->Rip += d.length;
            return true;
        }
        if (d.op == 0xD1 || d.op == 0xD2 || d.op == 0xD3 || d.op == 0xE1 || d.op == 0xE2 ||
            d.op == 0xF1 || d.op == 0xF2 || d.op == 0xF3) {
            const unsigned bits = (d.op == 0xD1 || d.op == 0xE1 || d.op == 0xF1) ? 16
                                  : (d.op == 0xD2 || d.op == 0xE2 || d.op == 0xF2) ? 32 : 64;
            const int mode = (d.op < 0xE0) ? 0 : (d.op < 0xF0) ? 1 : 2;
            detail::exec_shift_count(out, v1, rm_vec.u64[0], bits, mode);
            detail::write_vec(c, d.reg, out, true);
            c->Rip += d.length;
            return true;
        }
        detail::exec_lane128_map1(d.op, d.pp, d.imm, &v1.u8[0], &rm_vec.u8[0], &out.u8[0]);
        detail::exec_lane128_map1(d.op, d.pp, d.imm, &v1.u8[16], &rm_vec.u8[16], &out.u8[16]);
        detail::write_vec(c, d.reg, out, true);
        c->Rip += d.length;
        return true;
    }

    if (d.vmap == 2) {
        if (d.op == 0x78) {
            std::memset(out.u8, rm_vec.u8[0], full256 ? 32 : 16);
        } else if (d.op == 0x79) {
            for (int k = 0; k < (full256 ? 16 : 8); ++k) out.u16[k] = rm_vec.u16[0];
        } else if (d.op == 0x58 || d.op == 0x18) {
            for (int k = 0; k < (full256 ? 8 : 4); ++k) out.u32[k] = rm_vec.u32[0];
        } else if (d.op == 0x59 || d.op == 0x19) {
            for (int k = 0; k < (full256 ? 4 : 2); ++k) out.u64[k] = rm_vec.u64[0];
        } else if (d.op == 0x5A) {
            std::memcpy(&out.u8[0], rm_vec.u8, 16);
            std::memcpy(&out.u8[16], rm_vec.u8, 16);
        } else if (d.op == 0x16 || d.op == 0x36) {  // vpermps / vpermd: idx = v1, table = rm_vec
            for (int k = 0; k < 8; ++k) out.u32[k] = rm_vec.u32[v1.u32[k] & 7u];
        } else if (d.op == 0x45 || d.op == 0x46 || d.op == 0x47) {  // vpsrlvd/q, vpsravd, vpsllvd/q
            if (d.vex_w == 0) {
                const int lanes = full256 ? 8 : 4;
                for (int k = 0; k < lanes; ++k) {
                    const uint32_t sh = rm_vec.u32[k];
                    if (d.op == 0x47) out.u32[k] = sh >= 32 ? 0u : (v1.u32[k] << sh);
                    else if (d.op == 0x45) out.u32[k] = sh >= 32 ? 0u : (v1.u32[k] >> sh);
                    else out.i32[k] = sh >= 32 ? (v1.i32[k] < 0 ? -1 : 0) : (v1.i32[k] >> sh);
                }
            } else {
                const int lanes = full256 ? 4 : 2;
                for (int k = 0; k < lanes; ++k) {
                    const uint64_t sh = rm_vec.u64[k];
                    out.u64[k] = sh >= 64 ? 0ull : (d.op == 0x47 ? (v1.u64[k] << sh) : (v1.u64[k] >> sh));
                }
            }
        } else if (d.op >= 0x20 && d.op <= 0x25) {
            switch (d.op) {
            case 0x20: for (int k = 0; k < 16; ++k) out.i16[k] = rm_vec.i8[k]; break;
            case 0x21: for (int k = 0; k < 8; ++k) out.i32[k] = rm_vec.i8[k]; break;
            case 0x22: for (int k = 0; k < 4; ++k) out.i64[k] = rm_vec.i8[k]; break;
            case 0x23: for (int k = 0; k < 8; ++k) out.i32[k] = rm_vec.i16[k]; break;
            case 0x24: for (int k = 0; k < 4; ++k) out.i64[k] = rm_vec.i16[k]; break;
            case 0x25: for (int k = 0; k < 4; ++k) out.i64[k] = rm_vec.i32[k]; break;
            }
        } else if (d.op >= 0x30 && d.op <= 0x35) {
            switch (d.op) {
            case 0x30: for (int k = 0; k < 16; ++k) out.u16[k] = rm_vec.u8[k]; break;
            case 0x31: for (int k = 0; k < 8; ++k) out.u32[k] = rm_vec.u8[k]; break;
            case 0x32: for (int k = 0; k < 4; ++k) out.u64[k] = rm_vec.u8[k]; break;
            case 0x33: for (int k = 0; k < 8; ++k) out.u32[k] = rm_vec.u16[k]; break;
            case 0x34: for (int k = 0; k < 4; ++k) out.u64[k] = rm_vec.u16[k]; break;
            case 0x35: for (int k = 0; k < 4; ++k) out.u64[k] = rm_vec.u32[k]; break;
            }
        } else {
            detail::exec_lane128_map2(d.op, &v1.u8[0], &rm_vec.u8[0], &out.u8[0]);
            detail::exec_lane128_map2(d.op, &v1.u8[16], &rm_vec.u8[16], &out.u8[16]);
        }
        detail::write_vec(c, d.reg, out, full256);
        c->Rip += d.length;
        return true;
    }

    // d.vmap == 3
    if (d.op == 0x02) {  // vpblendd
        const int lanes = full256 ? 8 : 4;
        for (int k = 0; k < lanes; ++k) out.u32[k] = ((d.imm >> k) & 1u) ? rm_vec.u32[k] : v1.u32[k];
    } else if (d.op == 0x00 || d.op == 0x01) {  // vpermq / vpermpd
        for (int k = 0; k < 4; ++k) out.u64[k] = rm_vec.u64[(d.imm >> (2 * k)) & 3u];
    } else if (d.op == 0x38) {  // vinserti128
        out = v1;
        std::memcpy((d.imm & 1u) ? &out.u8[16] : &out.u8[0], rm_vec.u8, 16);
    } else if (d.op == 0x46) {  // vperm2i128
        const auto pick = [&](uint8_t ctl, uint8_t* dst16) {
            if (ctl & 8u) {
                std::memset(dst16, 0, 16);
            } else {
                const uint8_t* src16 = (ctl & 2u) ? &rm_vec.u8[(ctl & 1u) * 16] : &v1.u8[(ctl & 1u) * 16];
                std::memcpy(dst16, src16, 16);
            }
        };
        pick(d.imm & 0x0Fu, &out.u8[0]);
        pick((d.imm >> 4) & 0x0Fu, &out.u8[16]);
    } else if (d.op == 0x0E) {  // vpblendw (same 8-bit imm8 mask applied to both 128-bit lanes)
        for (int lane = 0; lane < 2; ++lane)
            for (int k = 0; k < 8; ++k)
                out.u16[lane * 8 + k] = ((d.imm >> k) & 1u) ? rm_vec.u16[lane * 8 + k] : v1.u16[lane * 8 + k];
    } else if (d.op == 0x0F) {  // vpalignr (per 128-bit lane)
        for (int lane = 0; lane < 2; ++lane) {
            uint8_t tmp[32]{};
            std::memcpy(tmp, &rm_vec.u8[lane * 16], 16);
            std::memcpy(tmp + 16, &v1.u8[lane * 16], 16);
            if (d.imm < 32) {
                const std::size_t n = std::min<std::size_t>(16, 32 - d.imm);
                std::memcpy(&out.u8[lane * 16], tmp + d.imm, n);
            }
        }
    } else if (d.op == 0x4C) {  // vpblendvb
        const Ymm m = detail::read_ymm(c, (d.imm >> 4) & 0xFu);
        for (int k = 0; k < 32; ++k) out.u8[k] = (m.u8[k] & 0x80u) ? rm_vec.u8[k] : v1.u8[k];
    }
    detail::write_vec(c, d.reg, out, full256);
    c->Rip += d.length;
    return true;
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
        if (avail < 3) return EXCEPTION_CONTINUE_SEARCH;
        std::memcpy(code, reinterpret_cast<const void*>(c->Rip), avail);
    }

    const Decoded d = decode(code, avail);
    if (d.cat == Category::none) return EXCEPTION_CONTINUE_SEARCH;

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

    if (count(d.cat).fetch_add(1, std::memory_order_relaxed) == 0)
        std::fprintf(stderr, "[rt] cpu: this CPU has no %s; emulating it (first at %p)\n", category_name(d.cat),
                     reinterpret_cast<void*>(fault_rip));
    return EXCEPTION_CONTINUE_EXECUTION;
}

inline bool install() {
    static void* const cookie = AddVectoredExceptionHandler(1, &handler);
    return cookie != nullptr;
}

}  // namespace rt::avx2_emulation
#endif
