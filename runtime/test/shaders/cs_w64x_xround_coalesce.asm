; Coalesced and cached wave-wide exchanges in --wave64-emulate (Gen.x_round):
;   v1 = lane 0..63; v4 = 100 + lane, v5 = 200 + 3*lane, v6 = 500 + 5*lane, v7 = 900 + 7*lane.
;   s[10:11] = lanes 36..51 (upper half only, lowest 36); s[14:15] = lanes 8..23 (low half only, lowest 8).
; 1) Live-SCC s_and_saveexec_b64 s[20:21], s[10:11] -> s_cbranch_execz (reuses cached Any vote) -> s_cselect_b32 s30,
;    11, 22 (SCC = 1 -> 11) + 4x contiguous v_readfirstlane_b32 from lane 36 (batched across one data round after
;    finding first = 36):
;      out[0] = s24 + s30 = 136 + 11 = 147, out[1] = s25 = 308, out[2] = s26 + s27 = 680 + 1152 = 1832.
; 2) Deferred SCC vote coalesced with s_cbranch_execz on distinct masks (opposite polarities):
;    2a) EXEC = s[10:11] (non-zero), s_and_b64 s[18:19], s[10:11], s[14:15] (0 -> SCC = 0), s_cbranch_execz not taken,
;        s_cselect_b32 s31, 333, 444 -> out[3] = 444.
;    2b) EXEC = s[18:19] (0), s_or_b64 s[22:23], s[10:11], s[14:15] (non-zero -> SCC = 1), s_cbranch_execz taken over
;        s32 = 666 (keeps 555), s_cselect_b32 s33, 70, 80 -> out[4] = 555 + 70 = 625.
; 3) Deferred SCC vote coalesced with cross-half v_readlane_b32 (lane 45 -> v5 = 335, SCC = 1 -> +1000):
;      out[5] = 1335.
; 4) Cached EXEC mask from s_bcnt1_i32_b64 (lanes 41..63 -> 23, first = 41) reused by a 1-round 4x v_readfirstlane_b32
;    batch (141 + 323 + 705 + 1187), then EXEC rewritten to s[14:15] (first = 8 -> 108 + 224) to check invalidation:
;      out[6] = 2379, out[7] = 332.
; Run: vk_compute_test cs_w64x_xround_coalesce.spv 1 dump --require-wave64
;   -> out[0..7] = 147 308 1832 444 625 1335 2379 332
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_add_nc_u32_e32 v4, 100, v1
v_mul_lo_u32 v5, 3, v1
v_add_nc_u32_e32 v5, 200, v5
v_mul_lo_u32 v6, 5, v1
v_add_nc_u32_e32 v6, 500, v6
v_mul_lo_u32 v7, 7, v1
v_add_nc_u32_e32 v7, 900, v7
v_cmp_ge_u32_e64 s[10:11], v1, 36
v_cmp_lt_u32_e64 s[12:13], v1, 52
s_and_b64 s[10:11], s[10:11], s[12:13]
v_cmp_ge_u32_e64 s[14:15], v1, 8
v_cmp_lt_u32_e64 s[16:17], v1, 24
s_and_b64 s[14:15], s[14:15], s[16:17]
; 1) Live-SCC saveexec + s_cbranch_execz + 4x batched v_readfirstlane_b32
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
s_mov_b32 s30, 0
s_and_saveexec_b64 s[20:21], s[10:11]
s_cbranch_execz 5
s_cselect_b32 s30, 11, 22
v_readfirstlane_b32 s24, v4
v_readfirstlane_b32 s25, v5
v_readfirstlane_b32 s26, v6
v_readfirstlane_b32 s27, v7
s_mov_b64 exec, -1
s_add_i32 s24, s24, s30
s_add_i32 s26, s26, s27
; 2a) Deferred SCC = 0 coalesced with s_cbranch_execz (EXEC != 0)
s_mov_b64 exec, s[10:11]
s_mov_b32 s31, 99
s_mov_b32 s43, 444
s_and_b64 s[18:19], s[10:11], s[14:15]
s_cbranch_execz 2
s_cselect_b32 s31, 333, s43
; 2b) Deferred SCC = 1 coalesced with s_cbranch_execz (EXEC == 0)
s_mov_b64 exec, s[18:19]
s_mov_b32 s32, 555
s_mov_b32 s43, 80
s_or_b64 s[22:23], s[10:11], s[14:15]
s_cbranch_execz 2
s_mov_b32 s32, 666
s_cselect_b32 s33, 70, s43
s_mov_b64 exec, -1
s_add_i32 s32, s32, s33
; 3) Deferred SCC = 1 coalesced with cross-half v_readlane_b32 (lane 45)
s_mov_b32 s43, 2000
s_and_b64 s[22:23], s[10:11], s[10:11]
v_readlane_b32 s34, v5, 45
s_cselect_b32 s35, 1000, s43
s_add_i32 s34, s34, s35
; 4) s_bcnt1 -> 1-round 4x v_readfirstlane_b32 batch, then EXEC rewrite invalidates first-lane cache
v_cmp_ge_u32_e64 s[28:29], v1, 41
s_mov_b64 exec, s[28:29]
s_bcnt1_i32_b64 s36, exec
v_readfirstlane_b32 s37, v4
v_readfirstlane_b32 s38, v5
v_readfirstlane_b32 s39, v6
v_readfirstlane_b32 s40, v7
s_mov_b64 exec, s[14:15]
v_readfirstlane_b32 s41, v4
v_readfirstlane_b32 s42, v5
s_mov_b64 exec, -1
s_add_i32 s36, s36, s37
s_add_i32 s36, s36, s38
s_add_i32 s36, s36, s39
s_add_i32 s36, s36, s40
s_add_i32 s41, s41, s42
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s24
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s25
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s26
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s31
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s32
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s34
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s36
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s41
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
