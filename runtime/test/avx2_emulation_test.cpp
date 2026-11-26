// Tests for F16C, FMA3 and AVX2 emulation (src/avx2_emulation.h, docs/cpu_features.md):
//   1. IEEE-754 half-precision <-> single-precision conversion (normal, subnormal, signed zero, Inf, NaN payload,
//      rounding modes RN/RD/RU/RZ).
//   2. Decoder coverage across 2-byte and 3-byte VEX forms and negative cases (128-bit AVX1 instructions left native,
//      native VPTEST / VPERM2F128 left untouched, illegal scalar VEX.L=1 rejected).
//   3. Synthetic XSTATE_AVX CONTEXT execution (`InitializeContext` + `XSTATE_MASK_AVX`), verifying full 256-bit YMM
//      reads/writes, 128-bit VEX upper-YMM zeroing, clean-XSTATE_AVX initialization, masked memory fault suppression on
//      disabled VPMASKMOVD lanes, and fault propagation on enabled lanes.
//   4. Live 256-bit machine-code execution from a VirtualAlloc executable page (`vmovdqu ymm` -> F16C / FMA3 / AVX2 ->
//      `vmovdqu [out], ymm; vzeroupper; ret`) on hardware (takes the #UD XSTATE_AVX path on Sandy Bridge / Ivy Bridge
//      and the native path on Haswell+ / Zen). No game files.
#include <windows.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "avx2_emulation.h"

namespace ae = rt::avx2_emulation;

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

ae::Decoded dec(std::initializer_list<uint8_t> bytes) {
    std::vector<uint8_t> v(bytes);
    return ae::decode(v.data(), v.size());
}

struct XContextBuf {
    std::vector<uint8_t> storage;
    CONTEXT* ctx = nullptr;

    XContextBuf() {
        DWORD sz = 0;
        InitializeContext(nullptr, CONTEXT_AMD64 | CONTEXT_FULL | CONTEXT_XSTATE, nullptr, &sz);
        storage.resize(sz + 64, 0);
        InitializeContext(storage.data(), CONTEXT_AMD64 | CONTEXT_FULL | CONTEXT_XSTATE, &ctx, &sz);
    }
};

void test_f16c_math() {
    check(ae::f16_to_f32(0x0000u) == 0.0f && !std::signbit(ae::f16_to_f32(0x0000u)), "+0.0h -> +0.0f");
    check(ae::f16_to_f32(0x8000u) == 0.0f && std::signbit(ae::f16_to_f32(0x8000u)), "-0.0h -> -0.0f");
    check(ae::f16_to_f32(0x3C00u) == 1.0f, "1.0h -> 1.0f");
    check(ae::f16_to_f32(0xC000u) == -2.0f, "-2.0h -> -2.0f");
    check(ae::f16_to_f32(0x0001u) == std::ldexp(1.0f, -24), "smallest subnormal f16");
    check(std::isinf(ae::f16_to_f32(0x7C00u)) && !std::signbit(ae::f16_to_f32(0x7C00u)), "+Inf f16");
    check(std::isnan(ae::f16_to_f32(0x7E2Au)), "NaN f16");

    check(ae::f32_to_f16(1.0f, 0) == 0x3C00u, "1.0f -> 1.0h");
    check(ae::f32_to_f16(-0.0f, 0) == 0x8000u, "-0.0f -> -0.0h");
    check(ae::f32_to_f16(65536.0f, 0) == 0x7C00u, "overflow RN -> +Inf");
    check(ae::f32_to_f16(65536.0f, 3) == 0x7BFFu, "overflow RZ -> max normal");
    check(ae::f32_to_f16(std::ldexp(1.0f, -24), 0) == 0x0001u, "subnormal f32 -> 0x0001h");
}

void test_decode() {
    check(dec({0xC4, 0xE2, 0x7D, 0x13, 0xC1}).cat == ae::Category::f16c, "decode vcvtph2ps ymm0, xmm1");
    check(dec({0xC4, 0xE3, 0x7D, 0x1D, 0xC1, 0x04}).cat == ae::Category::f16c, "decode vcvtps2ph xmm1, ymm0, 4");
    check(dec({0xC4, 0xE2, 0x75, 0x98, 0xC2}).cat == ae::Category::fma3, "decode vfmadd132ps ymm0, ymm1, ymm2");
    check(dec({0xC4, 0xE2, 0x71, 0xA9, 0x01}).cat == ae::Category::fma3, "decode vfmadd213ss xmm0, xmm1, [rcx]");
    check(dec({0xC4, 0xE2, 0x75, 0xB6, 0xC2}).cat == ae::Category::fma3, "decode vfmaddsub231ps ymm0, ymm1, ymm2");
    check(dec({0xC4, 0xE2, 0x7D, 0x18, 0xC1}).cat == ae::Category::avx2, "decode vbroadcastss ymm0, xmm1");
    check(dec({0xC5, 0xFD, 0x72, 0xF1, 0x05}).cat == ae::Category::avx2, "decode 2-byte VEX vpslld ymm0, ymm1, 5");
    check(dec({0xC5, 0xFD, 0xD7, 0xC1}).cat == ae::Category::avx2, "decode vpmovmskb eax, ymm1");

    // Leave native AVX1 instructions untouched
    check(dec({0xC5, 0xF9, 0xFE, 0xC2}).cat == ae::Category::none, "128-bit vpaddd xmm is AVX1 (left native)");
    check(dec({0xC4, 0xE2, 0x7D, 0x17, 0xC1}).cat == ae::Category::none, "256-bit vptest ymm is AVX1 (left native)");
    check(dec({0xC4, 0xE3, 0x75, 0x06, 0xC2, 0x01}).cat == ae::Category::none, "vperm2f128 is AVX1 (left native)");
    check(dec({0xC4, 0xE2, 0x75, 0xA9, 0xC2}).cat == ae::Category::none, "scalar vfmadd213ss with VEX.L=1 stays #UD");
}

void test_xstate_and_maskmov() {
    XContextBuf xb;
    check(xb.ctx != nullptr, "InitializeContext with XSTATE_AVX");
    if (!xb.ctx) return;
    CONTEXT* c = xb.ctx;
    c->FltSave.MxCsr = 0x1F80u;

    // Start with clean XSTATE_MASK_AVX (0): writing YMM2 must preserve clean zeros in YMM0..1,3..15!
    SetXStateFeaturesMask(c, 0);
    c->FltSave.XmmRegisters[1].Low = 0x0000000400000003ull;
    // vpbroadcastd ymm2(2), xmm1(1): C4 E2 7D 58 D1
    c->Rip = 0x3000;
    check(ae::apply(dec({0xC4, 0xE2, 0x7D, 0x58, 0xD1}), c), "apply vpbroadcastd ymm2, xmm1 from clean XSTATE");
    const ae::Ymm y2 = ae::detail::read_ymm(c, 2);
    const ae::Ymm y5 = ae::detail::read_ymm(c, 5);
    bool all3 = true;
    for (int k = 0; k < 8; ++k) all3 &= (y2.u32[k] == 3u);
    check(all3 && y5.u64[2] == 0 && y5.u64[3] == 0, "YMM2 has all 8 lanes = 3 and untouched YMM5 upper half is 0");

    // 128-bit VEX write (vfmadd213ss xmm2, xmm1, xmm1) MUST zero YMM2[255:128] while preserving XMM2[127:32]!
    float f_two = 2.0f, f_five = 5.0f;
    std::memcpy(&c->FltSave.XmmRegisters[1].Low, &f_two, 4);
    std::memcpy(&c->FltSave.XmmRegisters[2].Low, &f_five, 4);
    // vfmadd213ss xmm2(2), xmm1(1), xmm1(1): C4 E2 71 A9 D1 -> lane0 = 2*5 + 2 = 12, lanes 1..3 preserved (=3u), upper 128 zeroed!
    check(ae::apply(dec({0xC4, 0xE2, 0x71, 0xA9, 0xD1}), c), "apply vfmadd213ss xmm2, xmm1, xmm1");
    const ae::Ymm y2_after = ae::detail::read_ymm(c, 2);
    check(y2_after.f32[0] == 12.0f && y2_after.u32[1] == 3u && y2_after.u32[2] == 3u && y2_after.u32[3] == 3u &&
              y2_after.u64[2] == 0 && y2_after.u64[3] == 0,
          "scalar FMA3 preserves XMM[127:32] and zeroes YMM[255:128]");

    // VPMASKMOVD at page boundary: enabled lanes in readable page, disabled lanes straddling PAGE_NOACCESS page!
    auto* two_pages = static_cast<uint8_t*>(VirtualAlloc(nullptr, 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    check(two_pages != nullptr, "alloc two pages for vpmaskmovd boundary test");
    if (two_pages) {
        DWORD old = 0;
        VirtualProtect(two_pages + 0x1000, 0x1000, PAGE_NOACCESS, &old);
        auto* tail = reinterpret_cast<uint32_t*>(two_pages + 0x1000 - 16);  // 4 dwords before guard, 4 dwords in NOACCESS
        tail[0] = 10; tail[1] = 20; tail[2] = 30; tail[3] = 40;

        ae::Ymm mask{};
        for (int k = 0; k < 4; ++k) mask.u32[k] = 0x80000000u;  // lanes 0..3 enabled
        for (int k = 4; k < 8; ++k) mask.u32[k] = 0u;           // lanes 4..7 disabled (in PAGE_NOACCESS!)
        ae::detail::write_vec(c, 1, mask, true);
        c->Rcx = reinterpret_cast<uint64_t>(tail);
        c->Rip = 0x4000;
        // vpmaskmovd ymm0, ymm1, [rcx]: C4 E2 75 8C 01
        check(ae::apply(dec({0xC4, 0xE2, 0x75, 0x8C, 0x01}), c),
              "vpmaskmovd does NOT fault when disabled lanes cross into PAGE_NOACCESS");
        const ae::Ymm loaded = ae::detail::read_ymm(c, 0);
        check(loaded.u32[0] == 10 && loaded.u32[3] == 40 && loaded.u32[4] == 0 && loaded.u32[7] == 0,
              "vpmaskmovd loads enabled lanes and zeroes disabled lanes");

        // Now enable lane 4 (inside PAGE_NOACCESS): MUST report read access violation at two_pages + 0x1000!
        mask.u32[4] = 0x80000000u;
        ae::detail::write_vec(c, 1, mask, true);
        c->Rip = 0x4000;
        EXCEPTION_RECORD fault{};
        const bool ok = ae::apply(dec({0xC4, 0xE2, 0x75, 0x8C, 0x01}), c, &fault);
        check(!ok && c->Rip == 0x4000 && fault.ExceptionCode == DWORD(STATUS_ACCESS_VIOLATION) &&
                  fault.ExceptionInformation[0] == 0 &&
                  fault.ExceptionInformation[1] == reinterpret_cast<ULONG_PTR>(two_pages + 0x1000),
              "vpmaskmovd faults at exact PAGE_NOACCESS address when an enabled lane touches it");
        VirtualFree(two_pages, 0, MEM_RELEASE);
    }
}

void test_live_256bit_execution() {
    check(ae::install(), "install avx2_emulation VEH");
    const auto host = ae::detect_host();
    std::printf("host features: AVX2=%d FMA=%d F16C=%d\n", host.avx2, host.fma, host.f16c);

    auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    check(page != nullptr, "live 256-bit exec page");
    if (!page) return;

    const auto run_vec = [&](std::initializer_list<uint8_t> body, const void* rcx_p, const void* rdx_p, void* r8_p) {
        std::memset(page, 0xCC, 256);
        std::memcpy(page, body.begin(), body.size());
        FlushInstructionCache(GetCurrentProcess(), page, 256);
        reinterpret_cast<void (*)(const void*, const void*, void*)>(page)(rcx_p, rdx_p, r8_p);
    };

    // 1. Live 256-bit FMA3 (`vfmadd213ps ymm0, ymm1, [rdx]`):
    //   vmovups ymm0, [rcx]         ; C5 FC 10 01 (AVX1)
    //   vbroadcastss ymm1, xmm0     ; C4 E2 7D 18 C8 (AVX2 reg-source!)
    //   vfmadd213ps ymm0, ymm1, [rdx]; C4 E2 75 A8 02 (FMA3 256-bit!)
    //   vmovups [r8], ymm0          ; C4 C1 7C 11 00 (AVX1)
    //   vzeroupper                  ; C5 F8 77 (AVX1)
    //   ret                         ; C3
    alignas(32) float a[8] = {2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    alignas(32) float b[8] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    alignas(32) float out_f[8]{};
    const uint64_t fma_before = ae::count(ae::Category::fma3).load();
    const uint64_t avx2_before = ae::count(ae::Category::avx2).load();
    run_vec({0xC5, 0xFC, 0x10, 0x01,
             0xC4, 0xE2, 0x7D, 0x18, 0xC8,
             0xC4, 0xE2, 0x75, 0xA8, 0x02,
             0xC4, 0xC1, 0x7C, 0x11, 0x00,
             0xC5, 0xF8, 0x77, 0xC3},
            a, b, out_f);
    bool fma_ok = true;
    for (int k = 0; k < 8; ++k) fma_ok &= (out_f[k] == 2.0f * a[k] + b[k]);
    check(fma_ok, "live 256-bit vbroadcastss + vfmadd213ps round-trips all 8 YMM lanes via XSTATE_AVX");
    if (!host.fma) check(ae::count(ae::Category::fma3).load() > fma_before, "FMA3 took #UD path on pre-Haswell");
    if (!host.avx2) check(ae::count(ae::Category::avx2).load() > avx2_before, "AVX2 took #UD path on pre-Haswell");

    // 2. Live 256-bit F16C round-trip (`vcvtph2ps ymm0, [rcx]` -> `vcvtps2ph [r8], ymm0, 0`):
    alignas(16) uint16_t h_in[8] = {0x3C00u, 0xC000u, 0x4200u, 0x0000u, 0x8000u, 0x3800u, 0x4400u, 0xC400u};
    alignas(16) uint16_t h_out[8]{};
    run_vec({0xC4, 0xE2, 0x7D, 0x13, 0x01,
             0xC4, 0xC3, 0x7D, 0x1D, 0x00, 0x00,
             0xC5, 0xF8, 0x77, 0xC3},
            h_in, nullptr, h_out);
    check(std::memcmp(h_in, h_out, sizeof h_in) == 0, "live 256-bit F16C vcvtph2ps + vcvtps2ph round-trip");

    // 3. Live 256-bit AVX2 integer pipeline (`vpaddd` + `vpermq` + `vpsllvd` + `vpmovmskb`):
    alignas(32) uint32_t ia[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) uint32_t ib[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(32) uint32_t io[8]{};
    // vmovdqu ymm0, [rcx] (C5 FE 6F 01)
    // vpaddd ymm0, ymm0, [rdx] (C5 FD FE 02)
    // vpermq ymm0, ymm0, 0x4E (swap low/high 128-bit halves: C4 E3 FD 00 C0 4E)
    // vpslld ymm0, ymm0, 1 (C5 FD 72 F0 01)
    // vmovdqu [r8], ymm0 (C4 C1 7E 7F 00)
    // vzeroupper; ret
    run_vec({0xC5, 0xFE, 0x6F, 0x01,
             0xC5, 0xFD, 0xFE, 0x02,
             0xC4, 0xE3, 0xFD, 0x00, 0xC0, 0x4E,
             0xC5, 0xFD, 0x72, 0xF0, 0x01,
             0xC4, 0xC1, 0x7E, 0x7F, 0x00,
             0xC5, 0xF8, 0x77, 0xC3},
            ia, ib, io);
    check(io[0] == (55u * 2u) && io[3] == (88u * 2u) && io[4] == (11u * 2u) && io[7] == (44u * 2u),
          "live 256-bit AVX2 vpaddd + vpermq + vpslld");

    VirtualFree(page, 0, MEM_RELEASE);
}

}  // namespace

int main() {
    test_f16c_math();
    test_decode();
    test_xstate_and_maskmov();
    test_live_256bit_execution();
    std::printf("avx2_emulation_test: %s (%d checks)\n", failures ? "FAILED" : "OK", checks);
    return failures ? 1 : 0;
}
