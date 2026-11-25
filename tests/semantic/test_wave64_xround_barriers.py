"""Unit and 40/64/128-lane workgroup execution tests for coalesced and inline `Gen.x_round` exchanges
under `--wave64-emulate` (SHADER-W64-XROUND-BARRIERS).

Covers:
  1. Division-free inline `Gen._x_round_inline`: zero `%x_pre_*_fn` `OpFunctionCall`s, zero `OpUMod`s in
     scheduler-free modules (`%x_it` cycles `0..5`), `OpUMod` retained when `mini_blocks`/`tail_blocks` exist,
     `0x108` (`WorkgroupMemory`) barrier semantics when `"mem"` is clean vs `0x948` when `"mem"` is dirty, and
     omitted dead scheduler-word atomics in scheduler-free multi-wave workgroups.
  2. Block-local `Any`-vote and first-lane caching (`x_vcache`, `x_mcache`, `x_flcache`), preserved across
     mask-preserving `OpSelectionMerge` regions (`buffer_load`/`buffer_store`/`ds_*`):
     - live-SCC `s_and_saveexec_b64` / `s_and_b64` + `s_cbranch_execz`/`vccz` share one `Any` vote (`1` round
       barrier, `1` `OpAtomicOr`), emitting a bare `x_order_barrier` if memory became dirty in between;
     - `s_bcnt1_i32_b64` / `v_readfirstlane_b32` + `s_cbranch_execz` derive `Any` from `(lo | hi) != 0` with zero
       extra `OpAtomicOr`s;
     - cache invalidation on EXEC / SGPR-pair / half-SGPR rewrite and across basic-block labels;
     - 32-bit half-mask SGPR writes in wave64 never alias 64-bit full-wave `x_vcache` entries.
  3. Deferred SCC vote coalescing (`_pending_scc_vote` in `emulate_inline`):
     - live-SCC mask op + `s_cbranch_execz` on distinct masks coalesces into one 2-word `x_round` (`2 -> 1`
       round barrier) across all four `(SCC, EXEC != 0)` truth combinations and upper-half-only active lanes;
     - live-SCC mask op + cross-half `v_readlane_b32` coalesces `data` and `bits` into one `x_round` (`2 -> 1`
       round barrier);
     - live-SCC mask op + memory-hazard `x_order_barrier` coalesces into one `0x948` barrier.
  4. Contiguous `v_readfirstlane_b32` batching (`Gen._emit_rfl_run` in `emulate_inline`):
     - 8-register runs collapse `8 -> 2` (or `1` when EXEC's mask/first-lane is already cached) round barriers;
     - exact across upper-half-only active lanes (`32..63`), `EXEC == 0` lane-0 fallback, partial 40-thread waves,
       128-thread 2-wave workgroups with distinct first active lanes per wave, and piggybacked pending SCC votes.

Note: every `--wave64-emulate` module starts with 1 `x_prologue` `0x948` barrier after zeroing `%x_buf`, so total
`OpControlBarrier` count is `1 + <round_barriers>`.

Run from the repository root:  python -I tests/semantic/test_wave64_xround_barriers.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_asm  # noqa: E402
from test_wave64_emulate_opt import (  # noqa: E402
    VSHARP,
    _translate_encoded,
    run_emulated_workgroup,
)


def _with_labels(items):
    """Resolve `('label', name)` and `('branch', op, target)` into `(insn_text, nbytes)` pairs."""
    offsets = {}
    byte_pos = 0
    for it in items:
        if it[0] == "label":
            offsets[it[1]] = byte_pos
        elif it[0] == "branch":
            byte_pos += 4
        else:
            byte_pos += it[1]
    out = []
    byte_pos = 0
    for it in items:
        if it[0] == "label":
            continue
        if it[0] == "branch":
            _, op, target = it
            byte_pos += 4
            delta = offsets[target] - byte_pos
            assert delta % 4 == 0, f"unaligned branch target {target}: {delta}"
            out.append((f"{op} {delta // 4}", 4))
        else:
            byte_pos += it[1]
            out.append(it)
    return out


def _with_structured_tail(items):
    """Append a barrier-free 2-block scalar tail before `s_endpgm` so `len(eff_blocks) >= 2` and
    `plan_structured_emulation` engages `inline_all` / `structured_hybrid` with 0 extra barriers."""
    assert items[-1] == ("s_endpgm", 4)
    return _with_labels(list(items[:-1]) + [
        ("s_cmp_eq_u32 0, 1", 4),
        ("branch", "s_cbranch_scc1", "_tail_end"),
        ("s_nop 0", 4),
        ("label", "_tail_end"),
        ("s_endpgm", 4),
    ])


def _assemble_and_maybe_validate(spv):
    binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
    val = shutil.which("spirv-val")
    if val is not None:
        with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as tf:
            tf.write(binary)
            spv_path = tf.name
        try:
            res = subprocess.run(
                [val, "--target-env", "vulkan1.2", "--scalar-block-layout", spv_path],
                capture_output=True, text=True, timeout=30,
            )
            if res.returncode != 0:
                raise AssertionError(f"spirv-val failed:\n{res.stderr}\n{res.stdout}")
        finally:
            try:
                os.unlink(spv_path)
            except OSError:
                pass
    return binary


def _run_both_orders(tc, spv, local_size, memory_init=None):
    _assemble_and_maybe_validate(spv)
    mem_fwd, bars_fwd = run_emulated_workgroup(
        spv, local_size, push_constants=VSHARP + [0] * 28,
        memory=dict(memory_init or {}), reverse_order=False,
    )
    mem_rev, bars_rev = run_emulated_workgroup(
        spv, local_size, push_constants=VSHARP + [0] * 28,
        memory=dict(memory_init or {}), reverse_order=True,
    )
    tc.assertEqual(mem_fwd, mem_rev)
    tc.assertEqual(bars_fwd, bars_rev)
    return mem_fwd, bars_fwd


class InlineXRoundTests(unittest.TestCase):
    def test_inline_xround_eliminates_helper_calls_umod_and_dead_sched_atomics(self):
        # 128-thread (2-wave) shader with uniform spine blocks + sync-free tail (no mini_scheduler / tail_scheduler):
        # must emit 0 OpFunctionCall, 0 OpUMod, 0x108 barrier when mem is clean, 0x948 when mem is dirty,
        # and only 2 reset OpAtomicStore per x_round (mask words only, omitting the 2 dead sched words).
        insns = _with_structured_tail([
            ("s_mov_b64 exec, -1", 4),
            ("v_add_nc_u32_e32 v4, 100, v0", 4),
            ("v_readlane_b32 s10, v4, 45", 8),                      # clean mem -> 0x108 barrier
            ("v_lshlrev_b32_e32 v8, 2, v0", 4),
            ("buffer_load_dword v5, v8, s[0:3], 0 offen", 8),       # dirties ("mem", "r")
            ("v_add_nc_u32_e32 v5, s10, v5", 4),
            ("v_readlane_b32 s11, v5, 50", 8),                      # dirty mem -> 0x948 barrier
            ("v_add_nc_u32_e32 v6, s11, v4", 4),
            ("v_add_nc_u32_e32 v9, 1024, v8", 4),
            ("buffer_store_dword v6, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ])
        spv, gen = _translate_encoded("cs_inline_multiwave", insns, (128, 1, 1), emulate=True)
        self.assertEqual(gen.emulate_plan["mode"], "structured_hybrid")
        self.assertEqual(gen.emulate_plan["mini_blocks"], {})
        self.assertEqual(gen.emulate_plan["tail_blocks"], set())
        self.assertEqual(spv.count("OpFunctionCall"), 0)
        self.assertEqual(spv.count("OpUMod"), 0)
        self.assertEqual(spv.count("OpControlBarrier %c2 %c2 %c108"), 1)
        # 1 prologue 0x948 + 1 dirty-mem x_round 0x948 = 2
        self.assertEqual(spv.count("OpControlBarrier %c2 %c2 %c948"), 2)
        # 1 prologue zeroing store + 2 rounds * 2 mask-reset OpAtomicStores = 5 total (not 1 + 8 = 9 with dead sched resets).
        self.assertEqual(spv.count("OpAtomicStore "), 1 + 4)

        init = {i * 4: i * 10 for i in range(128)}
        mem, bars = _run_both_orders(self, spv, (128, 1, 1), init)
        self.assertEqual(bars, 1 + 2)
        for tid in range(128):
            w_base = tid & ~63
            s10 = 100 + (w_base + 45)
            s11 = s10 + init[(w_base + 50) * 4]
            self.assertEqual(mem[1024 + tid * 4], s11 + 100 + tid, f"tid {tid}")

    def test_six_plus_inline_rounds_wrap_mod6_ring_cleanly(self):
        # 8 sequential cross-half (s_bcnt1 + v_readlane_b32) pairs = 16 inline rounds cycling %x_it 0..5 -> 0..3.
        items = [("v_mul_lo_u32 v4, 3, v0", 8), ("s_mov_b32 s20, 0", 4)]
        for k in range(8):
            src_lane = 33 + k
            items.append((f"v_cmp_ge_u32_e64 s[10:11], v0, {src_lane}", 8))
            items.append(("s_bcnt1_i32_b64 s12, s[10:11]", 4))
            items.append((f"v_readlane_b32 s13, v4, {src_lane}", 8))
            items.append(("s_add_i32 s20, s20, s12", 4))
            items.append(("s_add_i32 s20, s20, s13", 4))
        items.extend([
            ("v_mov_b32_e32 v5, s20", 4),
            ("v_mov_b32_e32 v6, 0", 4),
            ("v_cmpx_eq_u32_e32 0, v0", 4),
            ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ])
        spv, gen = _translate_encoded("cs_mod6_wrap", _with_structured_tail(items), (64, 1, 1), emulate=True)
        self.assertEqual(gen.emulate_plan["mode"], "inline_all")
        self.assertEqual(spv.count("OpUMod"), 0)
        mem, bars = _run_both_orders(self, spv, (64, 1, 1))
        self.assertEqual(bars, 1 + 16)
        want = sum((64 - (33 + k)) + 3 * (33 + k) for k in range(8))
        self.assertEqual(mem[0], want)


class VoteAndMaskCacheTests(unittest.TestCase):
    def test_live_scc_saveexec_and_cbranch_execz_share_single_vote(self):
        # Live SCC from s_and_saveexec_b64 is read after s_cbranch_execz; only 1 round barrier + 1 OpAtomicOr!
        insns = _with_labels([
            ("v_cmp_ge_u32_e64 s[10:11], v0, 44", 8),        # upper-half lanes 44..63 only
            ("s_mov_b32 s15, 7", 4),
            ("s_and_saveexec_b64 s[12:13], s[10:11]", 4),
            ("branch", "s_cbranch_execz", "skip"),
            ("s_cselect_b32 s15, 91, 92", 4),
            ("label", "skip"),
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v4, s15", 4),
            ("v_mov_b32_e32 v5, 0", 4),
            ("v_cmpx_eq_u32_e32 0, v0", 4),
            ("buffer_store_dword v4, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ])
        spv, _ = _translate_encoded("cs_saveexec_reuse", insns, (64, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpAtomicOr"), 1)
        self.assertEqual(spv.count("OpControlBarrier"), 1 + 1)
        mem, bars = _run_both_orders(self, spv, (64, 1, 1))
        self.assertEqual(bars, 1 + 1)
        self.assertEqual(mem[0], 91)

    def test_cached_branch_condition_survives_buffer_store_and_orders_dirty_memory(self):
        # s_bcnt1_i32_b64 caches EXEC in x_mcache; buffer_store_dword emits an internal `if (%active)` selection
        # without touching mask registers (so x_gen is restored!) and dirties ("mem", "w"); s_cbranch_execz
        # derives Any from x_mcache with ZERO extra OpAtomicOrs and emits 1 bare 0x948 x_order_barrier!
        insns = _with_labels([
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),               # per-lane byte offset while all 64 lanes are active
            ("v_cmp_ge_u32_e64 s[10:11], v0, 32", 8),        # lanes 32..63 active
            ("s_mov_b64 exec, s[10:11]", 4),
            ("s_bcnt1_i32_b64 s14, exec", 4),                # caches EXEC in x_mcache (2 OpAtomicOrs, 0x108 barrier)
            ("v_add_nc_u32_e32 v4, 500, v0", 4),
            ("buffer_store_dword v4, v5, s[0:3], 0 offen", 8),  # lanes 32..63 store to 128..252
            ("branch", "s_cbranch_execz", "done"),           # cached Any -> 0 OpAtomicOrs, 1 bare 0x948 barrier!
            ("s_mov_b64 exec, -1", 4),
            ("v_sub_nc_u32_e32 v6, 63, v0", 4),
            ("v_lshlrev_b32_e32 v6, 2, v6", 4),
            ("buffer_load_dword v7, v6, s[0:3], 0 offen", 8),
            ("v_add_nc_u32_e32 v7, s14, v7", 4),
            ("v_add_nc_u32_e32 v8, 512, v5", 4),
            ("buffer_store_dword v7, v8, s[0:3], 0 offen", 8),
            ("label", "done"),
            ("s_endpgm", 4),
        ])
        spv, _ = _translate_encoded("cs_cached_branch_dirty_mem", insns, (64, 1, 1), emulate=True)
        # 1 prologue + 1 (s_bcnt1 0x108) + 1 (s_cbranch_execz dirty-mem 0x948) + 1 (blk1 load->store WAR 0x948) = 4.
        self.assertEqual(spv.count("OpAtomicOr"), 2)
        self.assertEqual(spv.count("OpControlBarrier"), 1 + 3)
        mem, bars = _run_both_orders(self, spv, (64, 1, 1))
        self.assertEqual(bars, 1 + 3)
        self.assertEqual(mem[512 + 0], 595)
        self.assertEqual(mem[512 + 15 * 4], 500 + 48 + 32)

    def test_half_mask_write_does_not_alias_full_wave_vote_cache(self):
        insns = _with_labels([
            ("v_cmp_ge_u32_e64 s[10:11], v0, 40", 8),
            ("s_and_b32 s10, s10, -1", 4),
            ("s_cselect_b32 s14, 10, 20", 4),
            ("s_mov_b64 vcc, s[10:11]", 4),
            ("s_mov_b32 s15, 30", 4),
            ("branch", "s_cbranch_vccz", "skip"),
            ("s_mov_b32 s15, 40", 4),
            ("label", "skip"),
            ("s_add_i32 s14, s14, s15", 4),
            ("v_mov_b32_e32 v4, s14", 4),
            ("v_mov_b32_e32 v5, 0", 4),
            ("v_cmpx_eq_u32_e32 0, v0", 4),
            ("buffer_store_dword v4, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ])
        spv, _ = _translate_encoded("cs_half_mask_no_alias", insns, (64, 1, 1), emulate=True)
        mem, _ = _run_both_orders(self, spv, (64, 1, 1))
        self.assertEqual(mem[0], 60)


class DeferredSccCoalesceTests(unittest.TestCase):
    def test_deferred_scc_coalesces_with_cbranch_execz_all_four_truth_cases(self):
        cases = [
            (70, 70, 0, 0),
            (70, 48, 0, 1),
            (48, 70, 1, 0),
            (48, 52, 1, 1),
        ]
        for idx, (scc_thr, exec_thr, want_scc, want_exec_nz) in enumerate(cases):
            with self.subTest(case=idx, want_scc=want_scc, want_exec_nz=want_exec_nz):
                insns = _with_labels([
                    (f"v_cmp_ge_u32_e64 s[10:11], v0, {scc_thr}", 8),
                    (f"v_cmp_ge_u32_e64 s[12:13], v0, {exec_thr}", 8),
                    ("s_mov_b64 exec, s[12:13]", 4),
                    ("s_mov_b32 s20, 100", 4),
                    ("s_and_b64 s[14:15], s[10:11], s[10:11]", 4),
                    ("branch", "s_cbranch_execz", "taken"),
                    ("s_mov_b32 s20, 200", 4),
                    ("label", "taken"),
                    ("s_cselect_b32 s21, 1, 0", 4),
                    ("s_mov_b64 exec, -1", 4),
                    ("s_add_i32 s20, s20, s21", 4),
                    ("v_mov_b32_e32 v4, s20", 4),
                    ("v_mov_b32_e32 v5, 0", 4),
                    ("v_cmpx_eq_u32_e32 0, v0", 4),
                    ("buffer_store_dword v4, v5, s[0:3], 0 offen", 8),
                    ("s_endpgm", 4),
                ])
                spv, _ = _translate_encoded(f"cs_coal_truth_{idx}", insns, (64, 1, 1), emulate=True)
                self.assertEqual(spv.count("OpControlBarrier"), 1 + 1)
                mem, bars = _run_both_orders(self, spv, (64, 1, 1))
                self.assertEqual(bars, 1 + 1)
                self.assertEqual(mem[0], (200 if want_exec_nz else 100) + want_scc)

    def test_deferred_scc_coalesces_with_readlane_and_memory_order_barrier(self):
        insns = _with_structured_tail([
            ("s_mov_b64 exec, -1", 4),
            ("v_add_nc_u32_e32 v4, 300, v0", 4),
            ("v_cmp_ge_u32_e64 s[10:11], v0, 50", 8),
            ("s_and_b64 s[12:13], s[10:11], s[10:11]", 4),           # deferred SCC = 1
            ("v_readlane_b32 s14, v4, 55", 8),                       # coalesces into 1st round barrier (0x108)
            ("s_cselect_b32 s15, 10, 20", 4),                        # s15 = 10
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v4, v5, s[0:3], 0 offen", 8),       # dirties ("mem", "w")
            ("v_cmp_ge_u32_e64 s[16:17], v0, 70", 8),
            ("s_and_b64 s[18:19], s[16:17], s[16:17]", 4),           # deferred SCC = 0 while mem is dirty
            ("v_sub_nc_u32_e32 v6, 63, v0", 4),
            ("v_lshlrev_b32_e32 v6, 2, v6", 4),
            ("buffer_load_dword v7, v6, s[0:3], 0 offen", 8),        # RAW hazard -> x_order_barrier coalesces SCC (0x948)!
            ("s_cselect_b32 s16, 3, 7", 4),                          # SCC = 0 -> s16 = 7
            ("s_add_i32 s14, s14, s15", 4),
            ("s_add_i32 s14, s14, s16", 4),
            ("v_add_nc_u32_e32 v7, s14, v7", 4),
            ("v_add_nc_u32_e32 v8, 512, v5", 4),
            ("buffer_store_dword v7, v8, s[0:3], 0 offen", 8),       # WAR hazard after load -> 1 more 0x948 barrier
            ("s_endpgm", 4),
        ])
        spv, _ = _translate_encoded("cs_coal_data_and_mem", insns, (64, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpControlBarrier %c2 %c2 %c108"), 1)
        # 1 prologue + 1 coalesced (RAW + SCC) + 1 (load->store WAR) = 3
        self.assertEqual(spv.count("OpControlBarrier %c2 %c2 %c948"), 1 + 2)
        mem, bars = _run_both_orders(self, spv, (64, 1, 1))
        self.assertEqual(bars, 1 + 3)
        self.assertEqual(mem[512], 735)


class BatchedReadfirstlaneTests(unittest.TestCase):
    def test_eight_readfirstlanes_batch_to_two_barriers_uncached_and_one_cached(self):
        setup = [("v_mbcnt_lo_u32_b32 v1, -1, 0", 8), ("v_mbcnt_hi_u32_b32 v1, -1, v1", 8)]
        for k in range(8):
            setup.append((f"v_mul_lo_u32 v{4 + k}, {k + 2}, v1", 8))
            setup.append((f"v_add_nc_u32_e32 v{4 + k}, {100 * (k + 1)}, v{4 + k}", 4))

        # 1) Uncached 8x RFL -> 2 round barriers total (down from 8)!
        insns_uncached = _with_structured_tail(setup + [
            ("v_cmp_ge_u32_e64 s[10:11], v1, 39", 8),
            ("s_mov_b64 exec, s[10:11]", 4),
        ] + [(f"v_readfirstlane_b32 s{20 + k}, v{4 + k}", 4) for k in range(8)] + [
            ("s_mov_b64 exec, -1", 4),
            ("s_mov_b32 s31, 0", 4),
        ] + [(f"s_add_i32 s31, s31, s{20 + k}", 4) for k in range(8)] + [
            ("v_mov_b32_e32 v15, s31", 4),
            ("v_mov_b32_e32 v16, 0", 4),
            ("v_cmpx_eq_u32_e32 0, v1", 4),
            ("buffer_store_dword v15, v16, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ])
        spv_u, _ = _translate_encoded("cs_rfl8_uncached", insns_uncached, (64, 1, 1), emulate=True)
        self.assertEqual(spv_u.count("OpControlBarrier"), 1 + 2)
        mem_u, bars_u = _run_both_orders(self, spv_u, (64, 1, 1))
        self.assertEqual(bars_u, 1 + 2)
        want_rfl = sum(100 * (k + 1) + (k + 2) * 39 for k in range(8))
        self.assertEqual(mem_u[0], want_rfl)

        # 2) Cached EXEC mask from s_bcnt1 + pending SCC vote piggybacking onto _emit_rfl_run's single barrier!
        insns_cached = _with_structured_tail(setup + [
            ("v_cmp_ge_u32_e64 s[10:11], v1, 39", 8),
            ("s_mov_b64 exec, s[10:11]", 4),
            ("s_bcnt1_i32_b64 s18, exec", 4),                # 1st round barrier (populates x_mcache[exec])
            ("v_cmp_ge_u32_e64 s[14:15], v1, 50", 8),
            ("s_and_b64 s[16:17], s[14:15], s[14:15]", 4),   # deferred SCC = 1 piggybacks onto _emit_rfl_run!
        ] + [(f"v_readfirstlane_b32 s{20 + k}, v{4 + k}", 4) for k in range(8)] + [
            ("s_cselect_b32 s30, 77, 88", 4),
            ("s_mov_b64 exec, -1", 4),
            ("s_add_i32 s31, s18, s30", 4),
        ] + [(f"s_add_i32 s31, s31, s{20 + k}", 4) for k in range(8)] + [
            ("v_mov_b32_e32 v15, s31", 4),
            ("v_mov_b32_e32 v16, 0", 4),
            ("v_cmpx_eq_u32_e32 0, v1", 4),
            ("buffer_store_dword v15, v16, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ])
        spv_c, _ = _translate_encoded("cs_rfl8_cached", insns_cached, (64, 1, 1), emulate=True)
        # 1 prologue + 1 (s_bcnt1) + 1 (8x RFL + deferred SCC vote) = 3 barriers total (down from 11)!
        self.assertEqual(spv_c.count("OpControlBarrier"), 1 + 2)
        mem_c, bars_c = _run_both_orders(self, spv_c, (64, 1, 1))
        self.assertEqual(bars_c, 1 + 2)
        self.assertEqual(mem_c[0], (64 - 39) + 77 + want_rfl)

    def test_batched_rfl_exec_zero_partial_wave_and_multiwave(self):
        # 1) Partial 40-thread wave with active lanes 35..39 (first = 35) and EXEC = 0 fallback (reads lane 0).
        insns_partial = _with_structured_tail([
            ("v_add_nc_u32_e32 v4, 400, v0", 4),
            ("v_add_nc_u32_e32 v5, 800, v0", 4),
            ("v_add_nc_u32_e32 v6, 900, v0", 4),
            ("v_cmp_ge_u32_e64 s[10:11], v0, 35", 8),
            ("s_mov_b64 exec, s[10:11]", 4),
            ("v_readfirstlane_b32 s20, v4", 4),
            ("v_readfirstlane_b32 s21, v5", 4),
            ("v_readfirstlane_b32 s24, v6", 4),
            ("v_cmp_ge_u32_e64 s[12:13], v0, 50", 8),
            ("s_mov_b64 exec, s[12:13]", 4),
            ("v_readfirstlane_b32 s22, v4", 4),
            ("v_readfirstlane_b32 s23, v5", 4),
            ("v_readfirstlane_b32 s25, v6", 4),
            ("s_mov_b64 exec, -1", 4),
            ("s_add_i32 s20, s20, s21", 4),
            ("s_add_i32 s20, s20, s24", 4),
            ("s_add_i32 s22, s22, s23", 4),
            ("s_add_i32 s22, s22, s25", 4),
            ("v_mov_b32_e32 v8, s20", 4),
            ("v_mov_b32_e32 v9, 0", 4),
            ("v_cmpx_eq_u32_e32 0, v0", 4),
            ("buffer_store_dword v8, v9, s[0:3], 0 offen", 8),
            ("v_mov_b32_e32 v8, s22", 4),
            ("v_mov_b32_e32 v9, 4", 4),
            ("buffer_store_dword v8, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ])
        spv_p, gen_p = _translate_encoded("cs_rfl_partial40", insns_partial, (40, 1, 1), emulate=True)
        self.assertEqual(gen_p.emulate_plan["mode"], "inline_all")
        # Each 3x RFL run takes 2 round barriers + 1 WAW barrier between the two stores -> 1 prologue + 5 = 6.
        self.assertEqual(spv_p.count("OpControlBarrier"), 1 + 5)
        mem_p, _ = _run_both_orders(self, spv_p, (40, 1, 1))
        self.assertEqual(mem_p[0], (400 + 35) + (800 + 35) + (900 + 35))
        self.assertEqual(mem_p[4], 400 + 800 + 900)

        # 2) 128-thread (2-wave) group where wave 0 has first active wave_lane = 42 (tid 42) and wave 1 has
        #    first active wave_lane = 7 (tid 71).
        insns_mw = _with_structured_tail([
            ("v_add_nc_u32_e32 v4, 1000, v0", 4),
            ("v_mul_lo_u32 v5, 5, v0", 8),
            ("v_cmp_ge_u32_e64 s[10:11], v0, 42", 8),
            ("v_cmp_ge_u32_e64 s[12:13], v0, 71", 8),
            ("v_cmp_lt_u32_e64 s[14:15], v0, 64", 8),
            ("s_and_b64 s[10:11], s[10:11], s[14:15]", 4),
            ("s_or_b64 exec, s[10:11], s[12:13]", 4),
            ("v_readfirstlane_b32 s20, v4", 4),
            ("v_readfirstlane_b32 s21, v5", 4),
            ("v_readfirstlane_b32 s22, v0", 4),
            ("s_mov_b64 exec, -1", 4),
            ("s_add_i32 s20, s20, s21", 4),
            ("s_add_i32 s20, s20, s22", 4),
            ("v_mov_b32_e32 v6, s20", 4),
            ("v_lshlrev_b32_e32 v7, 2, v0", 4),
            ("buffer_store_dword v6, v7, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ])
        spv_mw, gen_mw = _translate_encoded("cs_rfl_mw128", insns_mw, (128, 1, 1), emulate=True)
        self.assertEqual(gen_mw.emulate_plan["mode"], "structured_hybrid")
        self.assertEqual(spv_mw.count("OpControlBarrier"), 1 + 2)
        mem_mw, _ = _run_both_orders(self, spv_mw, (128, 1, 1))
        for tid in range(64):
            self.assertEqual(mem_mw[tid * 4], (1000 + 42) + 5 * 42 + 42)
        for tid in range(64, 128):
            self.assertEqual(mem_mw[tid * 4], (1000 + 71) + 5 * 71 + 71)


if __name__ == "__main__":
    unittest.main()
