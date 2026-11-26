// Tests for BMI1, BMI2, MOVBE and .pdata/leaf LZCNT/TZCNT emulation (src/bmi_emulation.h, docs/cpu_features.md):
//   1. Decoder coverage on hand-assembled VEX and legacy encodings (including negative cases such as CRC32, VEX.L=1,
//      register MOVBE, LOCK, and non-zero RORX vvvv).
//   2. `apply` on synthetic CONTEXT records (32-bit zero-extension, 16-bit upper-bit preservation, SIB / R12 index /
//      R13 disp32 / RIP-relative addressing, MULX identical destinations, flags, and read-only / unmapped memory fault
//      propagation).
//   3. Load-time TZCNT/LZCNT patcher on a synthetic PE-like executable buffer containing an immediate byte-trap
//      (`mov eax, 0xBD0F48F3`), 4-byte UD2 sites, >=5-byte branchless native stubs, memory sites, and a 0xCC-delimited
//      leaf routine outside .pdata, verifying exact `dst`, `CF` and `ZF` across incoming flag combinations.
//   4. Live machine-code execution from a VirtualAlloc executable page with the #UD handler installed (emulated on
//      pre-Haswell AVX1 hosts such as Sandy Bridge / Ivy Bridge, native on Haswell+ / Zen). No game files.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "bmi_emulation.h"

namespace be = rt::bmi_emulation;

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

be::Decoded dec(std::initializer_list<uint8_t> bytes) {
    std::vector<uint8_t> v(bytes);
    return be::decode(v.data(), v.size());
}

void test_decode() {
    // andn rax(0), r8(8), [r9 + 4*r10 + 16]
    auto d = dec({0xC4, 0x82, 0xB8, 0xF2, 0x44, 0x91, 0x10});
    check(d.op == be::Op::andn && d.length == 7 && d.size == 8 && d.dst == 0 && d.src1 == 8 && d.mem &&
              d.base == 9 && d.index == 10 && d.scale == 4 && d.disp == 16,
          "decode andn 64-bit SIB");

    // bextr eax(0), ecx(1), edx(2)
    d = dec({0xC4, 0xE2, 0x68, 0xF7, 0xC1});
    check(d.op == be::Op::bextr && d.length == 5 && d.size == 4 && d.dst == 0 && d.rm == 1 && d.src1 == 2 && !d.mem,
          "decode bextr 32-bit");

    // blsi eax(0), ecx(1); blsmsk eax(0), ecx(1); blsr rax(0), rcx(1)
    check(dec({0xC4, 0xE2, 0x78, 0xF3, 0xD9}).op == be::Op::blsi, "decode blsi");
    check(dec({0xC4, 0xE2, 0x78, 0xF3, 0xD1}).op == be::Op::blsmsk, "decode blsmsk");
    d = dec({0xC4, 0xE2, 0xF8, 0xF3, 0xC9});
    check(d.op == be::Op::blsr && d.size == 8 && d.dst == 0 && d.rm == 1, "decode blsr 64-bit");

    // bzhi, mulx, pdep, pext, rorx, sarx, shlx, shrx
    d = dec({0xC4, 0xE2, 0x68, 0xF5, 0xC1});
    check(d.op == be::Op::bzhi && d.dst == 0 && d.rm == 1 && d.src1 == 2, "decode bzhi");
    d = dec({0xC4, 0x42, 0xB3, 0xF6, 0x02});
    check(d.op == be::Op::mulx && d.size == 8 && d.dst == 8 && d.dst2 == 9 && d.mem && d.base == 10, "decode mulx");
    check(dec({0xC4, 0xE2, 0x73, 0xF5, 0xC2}).op == be::Op::pdep, "decode pdep");
    check(dec({0xC4, 0xE2, 0xF2, 0xF5, 0xC2}).op == be::Op::pext, "decode pext");
    d = dec({0xC4, 0xE3, 0xFB, 0xF0, 0xC1, 0x23});
    check(d.op == be::Op::rorx && d.size == 8 && d.dst == 0 && d.rm == 1 && d.imm == 0x23 && d.length == 6,
          "decode rorx");
    check(dec({0xC4, 0xE2, 0x6A, 0xF7, 0xC1}).op == be::Op::sarx, "decode sarx");
    check(dec({0xC4, 0xE2, 0x69, 0xF7, 0xC1}).op == be::Op::shlx, "decode shlx");
    check(dec({0xC4, 0xE2, 0xEB, 0xF7, 0xC1}).op == be::Op::shrx, "decode shrx");

    // movbe 16/32/64 load & store
    d = dec({0x66, 0x0F, 0x38, 0xF0, 0x01});
    check(d.op == be::Op::movbe_load && d.size == 2 && d.dst == 0 && d.base == 1, "decode movbe load 16");
    d = dec({0x48, 0x0F, 0x38, 0xF1, 0x05, 0x10, 0x00, 0x00, 0x00});
    check(d.op == be::Op::movbe_store && d.size == 8 && d.base == -2 && d.disp == 16 && d.length == 9,
          "decode movbe store 64 rip-rel");

    // tzcnt / lzcnt
    check(dec({0xF3, 0x0F, 0xBC, 0xC1}).op == be::Op::tzcnt && dec({0xF3, 0x0F, 0xBC, 0xC1}).length == 4,
          "decode 4B tzcnt");
    check(dec({0xF3, 0x48, 0x0F, 0xBD, 0xC1}).op == be::Op::lzcnt && dec({0xF3, 0x48, 0x0F, 0xBD, 0xC1}).size == 8,
          "decode 5B lzcnt");

    // Rejections
    check(dec({0xF2, 0x0F, 0x38, 0xF0, 0x01}).op == be::Op::none, "F2 0F 38 F0 is CRC32, not MOVBE");
    check(dec({0x0F, 0x38, 0xF0, 0xC1}).op == be::Op::none, "MOVBE with register r/m stays #UD");
    check(dec({0xC4, 0xE2, 0x6C, 0xF7, 0xC1}).op == be::Op::none, "BMI with VEX.L=1 stays #UD");
    check(dec({0xC4, 0xE3, 0x73, 0xF0, 0xC1, 0x07}).op == be::Op::none, "RORX with non-zero vvvv stays #UD");
    check(dec({0xF0, 0x0F, 0x38, 0xF1, 0x01}).op == be::Op::none, "LOCK MOVBE stays #UD");
}

void test_apply() {
    CONTEXT c{};
    c.Rip = 0x2000;
    c.Rax = 0xDEAD000000000000ull;
    c.Rcx = 0xFF00FF00u;
    c.Rdx = 0x0F0F0F0Fu;
    // andn eax, ecx, edx -> ~0xFF00FF00 & 0x0F0F0F0F = 0x000F000F, zero-extended to RAX
    check(be::apply(dec({0xC4, 0xE2, 0x70, 0xF2, 0xC2}), &c), "apply andn");
    check(c.Rax == 0x000F000Full && c.Rip == 0x2005 && (c.EFlags & 0xC1u) == 0, "andn result, zero-ext and flags");

    // blsi / blsmsk / blsr on zero vs non-zero
    c.Rcx = 0;
    c.EFlags = 0;
    check(be::apply(dec({0xC4, 0xE2, 0x78, 0xF3, 0xD9}), &c), "apply blsi zero");
    check(c.Rax == 0 && (c.EFlags & 0x41u) == 0x40u, "blsi(0): CF=0, ZF=1");
    c.Rcx = 0;
    check(be::apply(dec({0xC4, 0xE2, 0x78, 0xF3, 0xC9}), &c), "apply blsr zero");
    check(c.Rax == 0 && (c.EFlags & 0x41u) == 0x41u, "blsr(0): CF=1, ZF=1");
    c.Rcx = 0x18;
    check(be::apply(dec({0xC4, 0xE2, 0x78, 0xF3, 0xD1}), &c) && c.Rax == 0x0F && (c.EFlags & 0xC1u) == 0,
          "blsmsk(0x18) == 0x0F, CF=0, ZF=0");

    // mulx with distinct and identical destinations (flags must stay untouched)
    c.Rdx = 0x8000000000000001ull;
    c.Rcx = 4;
    c.EFlags = 0x8D5u;
    // mulx rax(0), rbx(3), rcx(1): hi=2, lo=4
    check(be::apply(dec({0xC4, 0xE2, 0xE3, 0xF6, 0xC1}), &c), "apply mulx");
    check(c.Rax == 2 && c.Rbx == 4 && c.EFlags == 0x8D5u, "mulx 64x64->128 and untouched EFlags");
    // mulx rax(0), rax(0), rcx(1): identical destinations keep high half
    check(be::apply(dec({0xC4, 0xE2, 0xFB, 0xF6, 0xC1}), &c) && c.Rax == 2, "mulx same dst keeps high half");

    // pdep / pext round-trip (0x39 = 0b11_1001 -> low nibble 0x9, bits 13:12 = 0x3 -> 0x00003009)
    check(be::pdep_u64(0x39ull, 0x0F00F00Full) == 0x00003009ull, "pdep_u64");
    check(be::pext_u64(0x00003009ull, 0x0F00F00Full) == 0x39ull, "pext_u64");

    // SIB with R12 as index + MOVBE 16-bit preserving upper 48 bits
    uint16_t word_mem[4] = {0, 0, 0x3412, 0};
    c.R9 = reinterpret_cast<uint64_t>(word_mem);
    c.R12 = 2;
    c.Rax = 0xAABBCCDDEEFF0000ull;
    // movbe ax, [r9 + r12*2]: 66 43 0F 38 F0 04 61
    check(be::apply(dec({0x66, 0x43, 0x0F, 0x38, 0xF0, 0x04, 0x61}), &c), "apply movbe 16 with R12 index");
    check(c.Rax == 0xAABBCCDDEEFF1234ull, "movbe 16 swaps bytes and preserves bits [63:16]");

    // Fault propagation: MOVBE store to a read-only page leaves RIP and registers unchanged and reports write AV
    void* ro = VirtualAlloc(nullptr, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY);
    check(ro != nullptr, "alloc read-only page");
    if (ro) {
        c.Rip = 0x5000;
        c.Rcx = reinterpret_cast<uint64_t>(ro) + 8;
        c.Rax = 0x0102030405060708ull;
        EXCEPTION_RECORD fault{};
        const bool ok = be::apply(dec({0x48, 0x0F, 0x38, 0xF1, 0x01}), &c, &fault);
        check(!ok && c.Rip == 0x5000 && fault.ExceptionCode == DWORD(STATUS_ACCESS_VIOLATION) &&
                  fault.ExceptionInformation[0] == 1 && fault.ExceptionInformation[1] == c.Rcx,
              "read-only MOVBE store reports write access violation without advancing RIP");
        VirtualFree(ro, 0, MEM_RELEASE);
    }
}

void test_lzcnt_tzcnt_patcher() {
    check(be::install(), "install bmi_emulation VEH");

    auto* img = static_cast<uint8_t*>(VirtualAlloc(nullptr, 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    check(img != nullptr, "alloc synthetic image");
    if (!img) return;
    std::memset(img, 0xCC, 0x2000);

    // Function 0 at +0x100 (in .pdata):
    //   mov eax, 0xBD0F48F3   ; B8 F3 48 0F BD  <-- immediate contains `F3 48 0F BD`! Must NOT be patched!
    //   cmp eax, 0xBD0F48F3   ; 3D F3 48 0F BD
    //   jne fail              ; 75 0E
    //   lzcnt eax, ecx        ; F3 0F BD C1     <-- 4-byte LZCNT -> UD2 + side-table!
    //   shl rax, 32           ; 48 C1 E0 20
    //   lzcnt rdx, rdx        ; F3 48 0F BD D2  <-- 5-byte 64-bit LZCNT (same reg!) -> native stub!
    //   or rax, rdx           ; 48 09 D0
    //   ret                   ; C3
    //   fail: mov eax, -1; ret
    const uint8_t fn0[] = {
        0xB8, 0xF3, 0x48, 0x0F, 0xBD,
        0x3D, 0xF3, 0x48, 0x0F, 0xBD,
        0x75, 0x11,
        0xF3, 0x0F, 0xBD, 0xC1,
        0x48, 0xC1, 0xE0, 0x20,
        0xF3, 0x48, 0x0F, 0xBD, 0xD2,
        0x48, 0x09, 0xD0,
        0xC3,
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF,
        0xC3,
    };
    std::memcpy(img + 0x100, fn0, sizeof fn0);

    // Function 1 at +0x200 (in .pdata): tests TZCNT + incoming CF=1 + immediate `cmovb` flag consumer!
    //   stc                   ; F9              <-- poison incoming CF=1 (BSF on SNB leaves CF=1 on non-zero!)
    //   tzcnt rax, rcx        ; F3 48 0F BC C1  <-- 5-byte TZCNT -> native stub!
    //   mov r8d, 999          ; 41 B8 E7 03 00 00
    //   cmovb rax, r8         ; 49 0F 42 C0     <-- must pick 999 ONLY when rcx == 0!
    //   ret                   ; C3
    const uint8_t fn1[] = {
        0xF9,
        0xF3, 0x48, 0x0F, 0xBC, 0xC1,
        0x41, 0xB8, 0xE7, 0x03, 0x00, 0x00,
        0x49, 0x0F, 0x42, 0xC0,
        0xC3,
    };
    std::memcpy(img + 0x200, fn1, sizeof fn1);

    // Leaf function at +0x300 (OUTSIDE .pdata, 0xCC-delimited!):
    //   stc                   ; F9
    //   tzcnt eax, ecx        ; F3 0F BC C1     <-- 4-byte leaf TZCNT -> UD2 + side-table!
    //   mov edx, 777          ; BA 09 03 00 00
    //   cmovb eax, edx        ; 0F 42 C2
    //   ret                   ; C3
    const uint8_t leaf[] = {
        0xF9,
        0xF3, 0x0F, 0xBC, 0xC1,
        0xBA, 0x09, 0x03, 0x00, 0x00,
        0x0F, 0x42, 0xC2,
        0xC3,
    };
    std::memcpy(img + 0x300, leaf, sizeof leaf);

    RUNTIME_FUNCTION pdata[2]{};
    pdata[0] = {0x100, uint32_t(0x100 + sizeof fn0), 0};
    pdata[1] = {0x200, uint32_t(0x200 + sizeof fn1), 0};

    const auto pr = be::patch_image(img, 0x2000, {{0x100, 0x400}}, pdata, 2, true, true);
    check(pr.lzcnt_sites == 2 && pr.tzcnt_sites == 2 && pr.stub_sites == 2 && pr.ud2_sites == 2,
          "patcher finds exact 2 LZCNT + 2 TZCNT sites (2 stubs, 2 UD2) and skips immediate trap");
    check(img[0x100 + 1] == 0xF3 && img[0x100 + 6] == 0xF3, "immediate bytes containing F3 48 0F BD are untouched");
    check(img[0x100 + 12] == 0x0F && img[0x100 + 13] == 0x0B, "4-byte LZCNT becomes UD2");
    check(img[0x100 + 20] == 0xE9 && img[0x200 + 1] == 0xE9, "5-byte LZCNT and TZCNT become jmp rel32 stubs");
    FlushInstructionCache(GetCurrentProcess(), img, 0x2000);

    using Fn0 = uint64_t (*)(uint32_t, uint64_t);
    using Fn1 = uint64_t (*)(uint64_t);
    using Leaf = uint32_t (*)(uint32_t);
    const auto f0 = reinterpret_cast<Fn0>(img + 0x100);
    const auto f1 = reinterpret_cast<Fn1>(img + 0x200);
    const auto fl = reinterpret_cast<Leaf>(img + 0x300);

    check(f0(0u, 0ull) == ((uint64_t(32) << 32) | 64ull), "patched LZCNT(0, 0) -> (32, 64)");
    check(f0(0x80000000u, 1ull) == ((uint64_t(0) << 32) | 63ull), "patched LZCNT(MSB, 1) -> (0, 63)");
    check(f0(0x00100000u, 0x0001000000000000ull) == ((uint64_t(11) << 32) | 15ull), "patched LZCNT mid bits");

    check(f1(0ull) == 999ull, "stub TZCNT(0) sets CF=1 for cmovb");
    check(f1(8ull) == 3ull, "stub TZCNT(8) clears poisoned CF=0 and returns 3");
    check(f1(1ull) == 0ull, "stub TZCNT(1) clears CF=0 and returns 0");

    check(fl(0u) == 777u, "leaf UD2 TZCNT(0) sets CF=1 for cmovb");
    check(fl(0x40u) == 6u, "leaf UD2 TZCNT(0x40) clears poisoned CF=0 and returns 6");

    if (pr.stubs) VirtualFree(pr.stubs, 0, MEM_RELEASE);
    VirtualFree(img, 0, MEM_RELEASE);
}

void test_live_execution() {
    check(be::install(), "bmi_emulation handler installed");
    const auto host = be::detect_host();
    std::printf("host features: BMI1=%d BMI2=%d LZCNT=%d MOVBE=%d\n", host.bmi1, host.bmi2, host.lzcnt, host.movbe);

    auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    check(page != nullptr, "live exec page");
    if (!page) return;

    const auto run_u64 = [&](std::initializer_list<uint8_t> insns, uint64_t a, uint64_t b) -> uint64_t {
        std::memset(page, 0xCC, 128);
        std::memcpy(page, insns.begin(), insns.size());
        FlushInstructionCache(GetCurrentProcess(), page, 128);
        return reinterpret_cast<uint64_t (*)(uint64_t, uint64_t)>(page)(a, b);
    };

    // andn rax, rcx, rdx; ret
    const uint64_t before_andn = be::count(be::Op::andn).load();
    check(run_u64({0xC4, 0xE2, 0xF0, 0xF2, 0xC2, 0xC3}, 0xFF00FF00FF00FF00ull, 0x123456789ABCDEF0ull) ==
              0x0034007800BC00F0ull,
          "live ANDN 64-bit");
    if (!host.bmi1) check(be::count(be::Op::andn).load() > before_andn, "ANDN took emulated #UD path on pre-Haswell");

    // bextr rax, rcx, rdx; ret
    check(run_u64({0xC4, 0xE2, 0xE8, 0xF7, 0xC1, 0xC3}, 0xFEDCBA9876543210ull, (16u << 8) | 8u) == 0x5432ull,
          "live BEXTR 64-bit");

    // bzhi rax, rcx, rdx; ret
    check(run_u64({0xC4, 0xE2, 0xE8, 0xF5, 0xC1, 0xC3}, 0xFFFFFFFFFFFFFFFFull, 12) == 0xFFFull, "live BZHI 64-bit");

    // shlx / shrx / sarx / rorx / pdep / pext / mulx
    check(run_u64({0xC4, 0xE2, 0xE9, 0xF7, 0xC1, 0xC3}, 0x1234ull, 16) == 0x12340000ull, "live SHLX 64-bit");
    check(run_u64({0xC4, 0xE2, 0xEB, 0xF7, 0xC1, 0xC3}, 0x12340000ull, 16) == 0x1234ull, "live SHRX 64-bit");
    check(run_u64({0xC4, 0xE2, 0xEA, 0xF7, 0xC1, 0xC3}, 0x8000000000000000ull, 4) == 0xF800000000000000ull,
          "live SARX 64-bit");
    check(run_u64({0xC4, 0xE3, 0xFB, 0xF0, 0xC1, 0x08, 0xC3}, 0x0102030405060708ull, 0) == 0x0801020304050607ull,
          "live RORX 64-bit");
    check(run_u64({0xC4, 0xE2, 0xF3, 0xF5, 0xC2, 0xC3}, 0x39ull, 0x0F00F00Full) == 0x00003009ull,
          "live PDEP 64-bit");
    check(run_u64({0xC4, 0xE2, 0xF2, 0xF5, 0xC2, 0xC3}, 0x00003009ull, 0x0F00F00Full) == 0x39ull,
          "live PEXT 64-bit");
    // mulx rax(0), r8(8), rcx(1) with rdx = b: C4 E2 BB F6 C1; add rax, r8 (4C 01 C0); ret
    check(run_u64({0xC4, 0xE2, 0xBB, 0xF6, 0xC1, 0x4C, 0x01, 0xC0, 0xC3}, 4ull, 0x8000000000000003ull) == (2ull + 12ull),
          "live MULX 64-bit");

    // movbe rax, [rcx]; movbe [rdx], rax; ret
    uint64_t in_be = 0x0807060504030201ull, out_be = 0;
    check(run_u64({0x48, 0x0F, 0x38, 0xF0, 0x01, 0x48, 0x0F, 0x38, 0xF1, 0x02, 0xC3},
                  reinterpret_cast<uint64_t>(&in_be), reinterpret_cast<uint64_t>(&out_be)) == 0x0102030405060708ull &&
              out_be == 0x0807060504030201ull,
          "live MOVBE load and store 64-bit");

    VirtualFree(page, 0, MEM_RELEASE);
}

}  // namespace

int main() {
    test_decode();
    test_apply();
    test_lzcnt_tzcnt_patcher();
    test_live_execution();
    std::printf("bmi_emulation_test: %s (%d checks)\n", failures ? "FAILED" : "OK", checks);
    return failures ? 1 : 0;
}
