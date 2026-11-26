// Pure-CPU tests for runtime/src/spirv_bary_fallback.h (RT-BARY-FALLBACK-PSO):
//   1. Basic VS + PS pair: copies BuiltIn Position and pass-through vec4 Locations, emits one-hot vec3 barycentrics.
//   2. Extra unread VS outputs (including at bary_location itself): left unconsumed on VS->GS while GS->PS writes vec3.
//   3. Unexported PS Input Location: zero-filled via OpConstantNull without declaring a dangling GS Input.
//   4. VS ClipDistance[2] and CullDistance[3] forwarding (capabilities, BuiltIn arrays, no duplicate types).
//   5. Combined link_f16_halves + build_bary_fallback_gs: f16 relocation preserves bary_location and forwards relocated halves.
//   6. Clean refusals: mixed ClipDistance + CullDistance, ShaderLayer / ShaderViewportIndex, non-vec3 collision at bary_location.
//   7. Computed I/J pairs (RT-BARY-FALLBACK-PER-VERTEX): a PS `Flat vec4[3]` Input named by `per_vertex` gets the
//      triangle's three values of the VS output it maps, with shared GS Inputs, zero-fill and clean refusals.
//   8. Neither a relocated nor a static FP16 half can supply a missing raw computed-I/J attribute.
// Optional `--dump-dir <dir>` writes each synthesized GS binary for external `spirv-val` inspection.
// No GPU or game files required.
#include "../src/spirv_bary_fallback.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

namespace {

using namespace leonida::spirv_detail;

void append_inst(std::vector<std::uint32_t>& w, std::uint16_t op, std::initializer_list<std::uint32_t> args) {
    w.push_back((std::uint32_t(1 + args.size()) << 16) | op);
    for (std::uint32_t a : args) w.push_back(a);
}

void append_inst_vec(std::vector<std::uint32_t>& w, std::uint16_t op, const std::vector<std::uint32_t>& args) {
    w.push_back((std::uint32_t(1 + args.size()) << 16) | op);
    w.insert(w.end(), args.begin(), args.end());
}

struct VsSpec {
    std::vector<std::uint32_t> raw_locs;
    std::map<std::uint32_t, std::array<std::uint32_t, 2>> static_halves;
    std::uint32_t clip_count = 0;
    std::uint32_t cull_count = 0;
    std::vector<std::uint32_t> extra_caps;
};

std::vector<std::uint32_t> build_vs(const VsSpec& spec) {
    std::uint32_t next_id = 1;
    const auto alloc = [&]() { return next_id++; };

    const std::uint32_t id_void = alloc();
    const std::uint32_t id_fn = alloc();
    const std::uint32_t id_float = alloc();
    const std::uint32_t id_v4float = alloc();
    const std::uint32_t id_uint = alloc();
    const std::uint32_t id_ptr_out_v4 = alloc();
    const std::uint32_t id_null_v4 = alloc();
    const std::uint32_t id_main = alloc();
    const std::uint32_t id_entry = alloc();
    const std::uint32_t id_pos = alloc();

    std::uint32_t id_clip_len = 0, id_clip_arr = 0, id_ptr_clip = 0, id_clip_var = 0, id_null_clip = 0;
    if (spec.clip_count) {
        id_clip_len = alloc();
        id_clip_arr = alloc();
        id_ptr_clip = alloc();
        id_clip_var = alloc();
        id_null_clip = alloc();
    }
    std::uint32_t id_cull_len = 0, id_cull_arr = 0, id_ptr_cull = 0, id_cull_var = 0, id_null_cull = 0;
    if (spec.cull_count) {
        id_cull_len = alloc();
        id_cull_arr = alloc();
        id_ptr_cull = alloc();
        id_cull_var = alloc();
        id_null_cull = alloc();
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> out_vars;
    for (std::uint32_t loc : spec.raw_locs) out_vars.push_back({alloc(), loc});
    for (const auto& [param, pair] : spec.static_halves) {
        out_vars.push_back({alloc(), pair[0]});
        out_vars.push_back({alloc(), pair[1]});
    }

    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, next_id, 0};
    append_inst(w, kOpCapability, {1});  // Shader
    if (spec.clip_count) append_inst(w, kOpCapability, {32});
    if (spec.cull_count) append_inst(w, kOpCapability, {33});
    for (std::uint32_t cap : spec.extra_caps) append_inst(w, kOpCapability, {cap});
    append_inst(w, kOpMemoryModel, {0, 1});

    std::vector<std::uint32_t> ep = {kExecModelVertex, id_main, 0x6e69616du, 0x00000000u, id_pos};
    if (id_clip_var) ep.push_back(id_clip_var);
    if (id_cull_var) ep.push_back(id_cull_var);
    for (const auto& [v, loc] : out_vars) ep.push_back(v);
    append_inst_vec(w, kOpEntryPoint, ep);

    append_inst(w, kOpDecorate, {id_pos, 11, 0});  // BuiltIn Position
    if (id_clip_var) append_inst(w, kOpDecorate, {id_clip_var, 11, 3});  // BuiltIn ClipDistance
    if (id_cull_var) append_inst(w, kOpDecorate, {id_cull_var, 11, 4});  // BuiltIn CullDistance
    for (const auto& [v, loc] : out_vars) append_inst(w, kOpDecorate, {v, kDecorationLocation, loc});

    append_inst(w, 19, {id_void});
    append_inst(w, 33, {id_fn, id_void});
    append_inst(w, kOpTypeFloat, {id_float, 32});
    append_inst(w, kOpTypeVector, {id_v4float, id_float, 4});
    append_inst(w, kOpTypeInt, {id_uint, 32, 0});
    append_inst(w, kOpTypePointer, {id_ptr_out_v4, kStorageOutput, id_v4float});
    append_inst(w, kOpConstantNull, {id_v4float, id_null_v4});
    if (spec.clip_count) {
        append_inst(w, 43, {id_uint, id_clip_len, spec.clip_count});
        append_inst(w, 28, {id_clip_arr, id_float, id_clip_len});
        append_inst(w, kOpTypePointer, {id_ptr_clip, kStorageOutput, id_clip_arr});
        append_inst(w, kOpConstantNull, {id_clip_arr, id_null_clip});
    }
    if (spec.cull_count) {
        append_inst(w, 43, {id_uint, id_cull_len, spec.cull_count});
        append_inst(w, 28, {id_cull_arr, id_float, id_cull_len});
        append_inst(w, kOpTypePointer, {id_ptr_cull, kStorageOutput, id_cull_arr});
        append_inst(w, kOpConstantNull, {id_cull_arr, id_null_cull});
    }

    append_inst(w, kOpVariable, {id_ptr_out_v4, id_pos, kStorageOutput});
    if (id_clip_var) append_inst(w, kOpVariable, {id_ptr_clip, id_clip_var, kStorageOutput});
    if (id_cull_var) append_inst(w, kOpVariable, {id_ptr_cull, id_cull_var, kStorageOutput});
    for (const auto& [v, loc] : out_vars) append_inst(w, kOpVariable, {id_ptr_out_v4, v, kStorageOutput});

    append_inst(w, kOpFunction, {id_void, id_main, 0, id_fn});
    append_inst(w, 248, {id_entry});
    append_inst(w, kOpStore, {id_pos, id_null_v4});
    if (id_clip_var) append_inst(w, kOpStore, {id_clip_var, id_null_clip});
    if (id_cull_var) append_inst(w, kOpStore, {id_cull_var, id_null_cull});
    for (const auto& [v, loc] : out_vars) append_inst(w, kOpStore, {v, id_null_v4});
    append_inst(w, kOpReturn, {});
    append_inst(w, kOpFunctionEnd, {});
    return w;
}

std::vector<std::uint32_t> build_ps(const std::vector<std::uint32_t>& raw_locs,
                                    const std::set<std::uint32_t>& flat_locs,
                                    std::optional<std::uint32_t> bary_v3_loc,
                                    const std::map<std::uint32_t, std::array<std::uint32_t, 2>>& f16_halves = {},
                                    const std::vector<std::uint32_t>& pv_bases = {}) {
    std::uint32_t next_id = 1;
    const auto alloc = [&]() { return next_id++; };

    const std::uint32_t id_void = alloc();
    const std::uint32_t id_fn = alloc();
    const std::uint32_t id_float = alloc();
    const std::uint32_t id_v3float = alloc();
    const std::uint32_t id_v4float = alloc();
    const std::uint32_t id_ptr_in_v3 = alloc();
    const std::uint32_t id_ptr_in_v4 = alloc();
    const std::uint32_t id_ptr_out_v4 = alloc();
    const std::uint32_t id_null_v4 = alloc();
    const std::uint32_t id_out0 = alloc();
    const std::uint32_t id_main = alloc();
    const std::uint32_t id_entry = alloc();
    const std::uint32_t id_bary = bary_v3_loc ? alloc() : 0;

    std::vector<std::pair<std::uint32_t, std::uint32_t>> in_v4_vars;
    for (std::uint32_t loc : raw_locs) in_v4_vars.push_back({alloc(), loc});
    for (const auto& [param, pair] : f16_halves) {
        in_v4_vars.push_back({alloc(), pair[0]});
        in_v4_vars.push_back({alloc(), pair[1]});
    }
    // --bary-fallback computed I/J pairs: a Flat vec4[3] Input at each first location (rdna2spirv's pv_fallback).
    std::uint32_t id_uint = 0, id_uint_3 = 0, id_arr_v4_3 = 0, id_ptr_in_arr = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> in_pv_vars;
    if (!pv_bases.empty()) {
        id_uint = alloc();
        id_uint_3 = alloc();
        id_arr_v4_3 = alloc();
        id_ptr_in_arr = alloc();
        for (std::uint32_t base : pv_bases) in_pv_vars.push_back({alloc(), base});
    }

    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, next_id, 0};
    append_inst(w, kOpCapability, {1});
    append_inst(w, kOpMemoryModel, {0, 1});

    std::vector<std::uint32_t> ep = {kExecModelFragment, id_main, 0x6e69616du, 0x00000000u, id_out0};
    if (id_bary) ep.push_back(id_bary);
    for (const auto& [v, loc] : in_v4_vars) ep.push_back(v);
    for (const auto& [v, loc] : in_pv_vars) ep.push_back(v);
    append_inst_vec(w, kOpEntryPoint, ep);
    append_inst(w, kOpExecutionMode, {id_main, 7});  // OriginUpperLeft

    append_inst(w, kOpDecorate, {id_out0, kDecorationLocation, 0});
    if (id_bary) append_inst(w, kOpDecorate, {id_bary, kDecorationLocation, *bary_v3_loc});
    for (const auto& [v, loc] : in_v4_vars) {
        append_inst(w, kOpDecorate, {v, kDecorationLocation, loc});
        if (flat_locs.count(loc)) append_inst(w, kOpDecorate, {v, 14});  // Flat
    }
    for (const auto& [v, loc] : in_pv_vars) {
        append_inst(w, kOpDecorate, {v, kDecorationLocation, loc});
        append_inst(w, kOpDecorate, {v, 14});  // Flat
    }

    append_inst(w, 19, {id_void});
    append_inst(w, 33, {id_fn, id_void});
    append_inst(w, kOpTypeFloat, {id_float, 32});
    append_inst(w, kOpTypeVector, {id_v3float, id_float, 3});
    append_inst(w, kOpTypeVector, {id_v4float, id_float, 4});
    append_inst(w, kOpTypePointer, {id_ptr_in_v3, kStorageInput, id_v3float});
    append_inst(w, kOpTypePointer, {id_ptr_in_v4, kStorageInput, id_v4float});
    append_inst(w, kOpTypePointer, {id_ptr_out_v4, kStorageOutput, id_v4float});
    append_inst(w, kOpConstantNull, {id_v4float, id_null_v4});
    if (!pv_bases.empty()) {
        append_inst(w, kOpTypeInt, {id_uint, 32, 0});
        append_inst(w, 43, {id_uint, id_uint_3, 3});
        append_inst(w, 28, {id_arr_v4_3, id_v4float, id_uint_3});
        append_inst(w, kOpTypePointer, {id_ptr_in_arr, kStorageInput, id_arr_v4_3});
    }

    append_inst(w, kOpVariable, {id_ptr_out_v4, id_out0, kStorageOutput});
    if (id_bary) append_inst(w, kOpVariable, {id_ptr_in_v3, id_bary, kStorageInput});
    for (const auto& [v, loc] : in_v4_vars) append_inst(w, kOpVariable, {id_ptr_in_v4, v, kStorageInput});
    for (const auto& [v, loc] : in_pv_vars) append_inst(w, kOpVariable, {id_ptr_in_arr, v, kStorageInput});

    append_inst(w, kOpFunction, {id_void, id_main, 0, id_fn});
    append_inst(w, 248, {id_entry});
    append_inst(w, kOpStore, {id_out0, id_null_v4});
    append_inst(w, kOpReturn, {});
    append_inst(w, kOpFunctionEnd, {});
    return w;
}

struct GsInspection {
    std::set<std::uint32_t> caps;
    std::set<std::uint32_t> in_locs;
    std::map<std::uint32_t, std::uint32_t> out_loc_vec_len;  // loc -> 3 or 4
    std::set<std::uint32_t> out_pv_locs;                     // first location of each Flat Output vec4[3]
    int out_pv_stores = 0;                                   // stores through an element of such an Output
    int out_pv_null_stores = 0;                              // whole-array zero-fill stores
    int emit_vertices = 0;
    int end_primitives = 0;
    bool has_in_pos = false;
    bool has_out_pos = false;
    std::uint32_t builtin_dist_in = 0;
    std::uint32_t builtin_dist_out = 0;
};

bool verify_gs_module(const std::vector<std::uint32_t>& words, GsInspection& out) {
    Module mod;
    if (!mod.parse(words)) return false;
    if (mod.header[1] != 0x00010500u) return false;
    out = {};

    int section = 0;  // 0=cap, 1=mem, 2=ep, 3=execmode, 4=dec, 5=types/consts/vars, 6=fn
    std::set<std::uint32_t> defined_ids;
    std::set<std::uint32_t> ep_iface;
    std::set<std::uint32_t> global_vars;
    std::uint32_t float_ty = 0;
    std::set<std::uint32_t> exec_modes;
    std::map<std::uint32_t, std::uint32_t> vec_len;
    std::map<std::uint32_t, std::uint32_t> ptr_pointee;
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> var_info;  // var_id -> (storage, pointee)
    std::set<std::uint32_t> arr_v4_3_tys;       // vec4<f32>[3]
    std::map<std::uint32_t, std::uint32_t> uint_consts;
    std::set<std::uint32_t> flat_vars;
    std::set<std::uint32_t> null_arr_consts;    // OpConstantNull of vec4<f32>[3]
    std::set<std::uint32_t> out_arr_elem_ptrs;  // access chains into an Output vec4<f32>[3]
    bool saw_fn_end = false;

    const auto record_def = [&](std::uint32_t id) {
        return id > 0 && id < mod.header[3] && defined_ids.insert(id).second;
    };

    for (const Inst& inst : mod.insts) {
        if (saw_fn_end) return false;
        switch (inst.opcode) {
        case kOpCapability:
            if (section > 0 || inst.words.size() != 2) return false;
            out.caps.insert(inst.words[1]);
            break;
        case kOpMemoryModel:
            if (section > 1 || inst.words.size() != 3 || inst.words[1] != 0 || inst.words[2] != 1) return false;
            section = 1;
            break;
        case kOpEntryPoint: {
            if (section > 2 || inst.words.size() < 5 || inst.words[1] != 3) return false;
            section = 2;
            const std::size_t s = entry_point_interface_start(inst);
            for (std::size_t i = s; i < inst.words.size(); ++i) ep_iface.insert(inst.words[i]);
            break;
        }
        case kOpExecutionMode:
            if (section < 2 || section > 3 || inst.words.size() < 3) return false;
            if ((inst.words[2] == 0 || inst.words[2] == 26) ? (inst.words.size() != 4) : (inst.words.size() != 3))
                return false;
            exec_modes.insert(inst.words[2]);
            section = 3;
            break;
        case kOpDecorate:
            if (section < 3 || section > 4) return false;
            if (inst.words.size() == 3 && inst.words[2] == 14) flat_vars.insert(inst.words[1]);
            section = 4;
            break;
        case kOpFunction:
            if (section < 5 || inst.words.size() != 5) return false;
            if (!record_def(inst.words[2])) return false;
            section = 6;
            break;
        case kOpFunctionEnd:
            if (section != 6) return false;
            saw_fn_end = true;
            break;
        default:
            if (section < 6) {
                if (section < 4) return false;
                section = 5;
                if (inst.opcode == 19 || inst.opcode == 33 || inst.opcode == kOpTypeInt || inst.opcode == kOpTypeFloat ||
                    inst.opcode == kOpTypeVector || inst.opcode == 28 || inst.opcode == kOpTypePointer) {
                    if (!record_def(inst.words[1])) return false;
                    if (inst.opcode == kOpTypeFloat && inst.words[2] == 32) float_ty = inst.words[1];
                    if (inst.opcode == kOpTypeVector && inst.words[2] == float_ty) vec_len[inst.words[1]] = inst.words[3];
                    if (inst.opcode == kOpTypePointer) ptr_pointee[inst.words[1]] = inst.words[3];
                    if (inst.opcode == 28 && vec_len.count(inst.words[2]) && vec_len[inst.words[2]] == 4 &&
                        uint_consts.count(inst.words[3]) && uint_consts[inst.words[3]] == 3)
                        arr_v4_3_tys.insert(inst.words[1]);
                } else if (inst.opcode == 43 || inst.opcode == 44 || inst.opcode == kOpConstantNull) {
                    if (!record_def(inst.words[2])) return false;
                    if (inst.opcode == 43 && inst.words.size() == 4) uint_consts[inst.words[2]] = inst.words[3];
                    if (inst.opcode == kOpConstantNull && arr_v4_3_tys.count(inst.words[1]))
                        null_arr_consts.insert(inst.words[2]);
                } else if (inst.opcode == kOpVariable) {
                    if (!record_def(inst.words[2])) return false;
                    global_vars.insert(inst.words[2]);
                    var_info[inst.words[2]] = {inst.words[3], ptr_pointee[inst.words[1]]};
                } else {
                    return false;
                }
            } else {
                if (inst.opcode == 248) {
                    if (!record_def(inst.words[1])) return false;
                } else if (inst.opcode == kOpAccessChain || inst.opcode == kOpLoad) {
                    if (!record_def(inst.words[2])) return false;
                    if (inst.opcode == kOpAccessChain) {
                        const auto base = var_info.find(inst.words[3]);
                        if (base != var_info.end() && base->second.first == kStorageOutput &&
                            arr_v4_3_tys.count(base->second.second))
                            out_arr_elem_ptrs.insert(inst.words[2]);
                    }
                } else if (inst.opcode == kOpStore && out_arr_elem_ptrs.count(inst.words[1])) {
                    ++out.out_pv_stores;
                } else if (inst.opcode == kOpStore && null_arr_consts.count(inst.words[2])) {
                    ++out.out_pv_null_stores;
                } else if (inst.opcode == 218) {
                    ++out.emit_vertices;
                } else if (inst.opcode == 219) {
                    ++out.end_primitives;
                } else if (inst.opcode != kOpStore && inst.opcode != kOpReturn) {
                    return false;
                }
            }
            break;
        }
    }
    if (!saw_fn_end || !out.caps.count(2) || global_vars != ep_iface ||
        exec_modes != std::set<std::uint32_t>{0, 22, 26, 29})
        return false;

    for (const Inst& inst : mod.insts) {
        if (inst.opcode != kOpDecorate || inst.words.size() < 4) continue;
        const auto vit = var_info.find(inst.words[1]);
        if (vit == var_info.end()) return false;
        const std::uint32_t storage = vit->second.first;
        const std::uint32_t pt = vit->second.second;
        if (inst.words[2] == 11) {  // BuiltIn
            if (inst.words[3] == 0) {
                if (storage == kStorageInput) out.has_in_pos = true;
                if (storage == kStorageOutput) out.has_out_pos = true;
            } else if (inst.words[3] == 3 || inst.words[3] == 4) {
                if (storage == kStorageInput) out.builtin_dist_in = inst.words[3];
                if (storage == kStorageOutput) out.builtin_dist_out = inst.words[3];
            }
        } else if (inst.words[2] == kDecorationLocation) {
            const std::uint32_t loc = inst.words[3];
            if (storage == kStorageInput) {
                if (!out.in_locs.insert(loc).second) return false;
            } else if (storage == kStorageOutput && arr_v4_3_tys.count(pt)) {
                // A per-vertex block: Flat, and (checked below) three locations clear of every other output.
                if (!flat_vars.count(inst.words[1]) || !out.out_pv_locs.insert(loc).second) return false;
            } else if (storage == kStorageOutput) {
                const auto vlen = vec_len.find(pt);
                if (vlen == vec_len.end()) return false;
                if (!out.out_loc_vec_len.emplace(loc, vlen->second).second) return false;
            }
        }
    }
    for (std::uint32_t base : out.out_pv_locs) {
        for (std::uint32_t k = 0; k < 3; ++k) {
            if (out.out_loc_vec_len.count(base + k)) return false;
            if (k && out.out_pv_locs.count(base + k)) return false;
        }
    }
    return out.has_in_pos && out.has_out_pos && out.emit_vertices == 3 && out.end_primitives == 1;
}

void maybe_dump(const std::filesystem::path& dump_dir, const char* name, const std::vector<std::uint32_t>& words) {
    if (dump_dir.empty()) return;
    std::filesystem::create_directories(dump_dir);
    std::ofstream f(dump_dir / (std::string(name) + ".spv"), std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
}

}  // namespace

int main(int argc, char** argv) {
    using namespace leonida;
    std::filesystem::path dump_dir;
    if (argc == 3 && std::strcmp(argv[1], "--dump-dir") == 0) dump_dir = argv[2];

    // 1. Basic VS + PS pair: bary_location = 1, pass-through Location 0 and 2.
    {
        const auto vs = build_vs({.raw_locs = {0, 2}});
        const auto ps = build_ps({0, 2}, {}, 1);
        std::vector<std::uint32_t> gs;
        BaryFallbackGsInfo info{};
        std::string err;
        CHECK(build_bary_fallback_gs(vs, ps, 1, gs, &info, &err));
        CHECK(info.clip_count == 0 && info.cull_count == 0);
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        CHECK((ins.in_locs == std::set<std::uint32_t>{0, 2}));
        CHECK(ins.out_loc_vec_len.size() == 3);
        CHECK(ins.out_loc_vec_len[0] == 4 && ins.out_loc_vec_len[1] == 3 && ins.out_loc_vec_len[2] == 4);
        maybe_dump(dump_dir, "gs_basic", gs);
    }

    // 2. Extra unread VS outputs (including Location 1 == bary_location): left unconsumed on VS->GS.
    {
        const auto vs = build_vs({.raw_locs = {0, 1, 2, 3}});
        const auto ps = build_ps({0}, {}, 1);
        std::vector<std::uint32_t> gs;
        std::string err;
        CHECK(build_bary_fallback_gs(vs, ps, 1, gs, nullptr, &err));
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        CHECK((ins.in_locs == std::set<std::uint32_t>{0}));
        CHECK(ins.out_loc_vec_len.size() == 2);
        CHECK(ins.out_loc_vec_len[0] == 4 && ins.out_loc_vec_len[1] == 3);
        maybe_dump(dump_dir, "gs_unread_vs", gs);
    }

    // 3. Unexported PS Flat Input Location 2: zero-filled in GS without declaring a dangling GS Input.
    {
        const auto vs = build_vs({.raw_locs = {0}});
        const auto ps = build_ps({0, 2}, {2}, 1);
        std::vector<std::uint32_t> gs;
        std::string err;
        CHECK(build_bary_fallback_gs(vs, ps, 1, gs, nullptr, &err));
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        CHECK((ins.in_locs == std::set<std::uint32_t>{0}));
        CHECK(ins.out_loc_vec_len.size() == 3);
        CHECK(ins.out_loc_vec_len[0] == 4 && ins.out_loc_vec_len[1] == 3 && ins.out_loc_vec_len[2] == 4);
        maybe_dump(dump_dir, "gs_zero_fill", gs);
    }

    // 4a. ClipDistance[2] forwarding.
    {
        const auto vs = build_vs({.raw_locs = {0}, .clip_count = 2});
        const auto ps = build_ps({0}, {}, 1);
        std::vector<std::uint32_t> gs;
        BaryFallbackGsInfo info{};
        std::string err;
        CHECK(build_bary_fallback_gs(vs, ps, 1, gs, &info, &err));
        CHECK(info.clip_count == 2 && info.cull_count == 0);
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        CHECK(ins.caps.count(32) == 1 && !ins.caps.count(33));
        CHECK(ins.builtin_dist_in == 3 && ins.builtin_dist_out == 3);
        maybe_dump(dump_dir, "gs_clip2", gs);
    }

    // 4b. CullDistance[3] forwarding (dist_count == 3 shares %uint_3 cleanly).
    {
        const auto vs = build_vs({.raw_locs = {1}, .cull_count = 3});
        const auto ps = build_ps({1}, {}, 0);
        std::vector<std::uint32_t> gs;
        BaryFallbackGsInfo info{};
        std::string err;
        CHECK(build_bary_fallback_gs(vs, ps, 0, gs, &info, &err));
        CHECK(info.clip_count == 0 && info.cull_count == 3);
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        CHECK(ins.caps.count(33) == 1 && !ins.caps.count(32));
        CHECK(ins.builtin_dist_in == 4 && ins.builtin_dist_out == 4);
        CHECK(ins.out_loc_vec_len[0] == 3 && ins.out_loc_vec_len[1] == 4);
        maybe_dump(dump_dir, "gs_cull3", gs);
    }

    // 5. Combined link_f16_halves + build_bary_fallback_gs:
    //    16-param VS (raw 0..15) + PS reading raw 0..13, bary_location = 14, f16_halves[7] = [15, 16] (colliding with VS 15).
    {
        ShaderEntry vs_e, ps_e;
        ps_e.f16_halves = {{7, {15, 16}}};
        ps_e.bary_fallback_location = 14;
        std::vector<std::uint32_t> vs_raw, ps_raw;
        for (std::uint32_t i = 0; i < 16; ++i) vs_raw.push_back(i);
        for (std::uint32_t i = 0; i < 14; ++i) ps_raw.push_back(i);

        auto vs = build_vs({.raw_locs = vs_raw});
        auto ps = build_ps(ps_raw, {}, 14, ps_e.f16_halves);
        std::string err;
        CHECK(link_f16_halves(vs, vs_e, ps, ps_e, &err, 29));

        std::vector<std::uint32_t> gs;
        CHECK(build_bary_fallback_gs(vs, ps, *ps_e.bary_fallback_location, gs, nullptr, &err));
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        // Forwarded: raw 0..13 + relocated f16 halves 28, 29 (16 vec4 inputs); unconsumed VS 14, 15 omitted; bary 14 is vec3.
        CHECK(ins.in_locs.size() == 16);
        CHECK(!ins.in_locs.count(14) && !ins.in_locs.count(15));
        CHECK(ins.in_locs.count(28) == 1 && ins.in_locs.count(29) == 1);
        CHECK(ins.out_loc_vec_len[14] == 3);
        CHECK(ins.out_loc_vec_len[28] == 4 && ins.out_loc_vec_len[29] == 4);
        maybe_dump(dump_dir, "gs_f16_linked", gs);
    }

    // 6. Refusals leave caller's gs_words untouched.
    {
        const std::vector<std::uint32_t> sentinel = {0xdeadbeefu};
        std::vector<std::uint32_t> gs = sentinel;
        std::string err;

        // Mixed ClipDistance + CullDistance.
        const auto vs_mixed = build_vs({.raw_locs = {0}, .clip_count = 1, .cull_count = 1});
        const auto ps_ok = build_ps({0}, {}, 1);
        CHECK(!build_bary_fallback_gs(vs_mixed, ps_ok, 1, gs, nullptr, &err) && !err.empty() && gs == sentinel);

        // ShaderLayer (69) and ShaderViewportIndex (70).
        const auto vs_layer = build_vs({.raw_locs = {0}, .extra_caps = {69}});
        const auto vs_vp = build_vs({.raw_locs = {0}, .extra_caps = {70}});
        err.clear();
        CHECK(!build_bary_fallback_gs(vs_layer, ps_ok, 1, gs, nullptr, &err) && !err.empty() && gs == sentinel);
        err.clear();
        CHECK(!build_bary_fallback_gs(vs_vp, ps_ok, 1, gs, nullptr, &err) && !err.empty() && gs == sentinel);

        // bary_location colliding with a vec4 PS Input.
        const auto vs_ok = build_vs({.raw_locs = {0, 1}});
        const auto ps_collide = build_ps({0, 1}, {}, std::nullopt);
        err.clear();
        CHECK(!build_bary_fallback_gs(vs_ok, ps_collide, 1, gs, nullptr, &err) && !err.empty() && gs == sentinel);
    }

    // 7. Computed I/J pairs: bary_fallback_input.per_vertex (attribute -> first of three locations).
    using PerVertexMap = std::map<std::uint32_t, std::uint32_t>;
    // 7a. The shape of a translated computed-I/J PS: attribute 0 as a Flat vec4[3] at Locations 0..2, barycentrics
    //     at 3.
    {
        const auto vs = build_vs({.raw_locs = {0}});
        const auto ps = build_ps({}, {}, 3, {}, {0});
        std::vector<std::uint32_t> gs;
        std::string err;
        // Without the map the array is still refused, as before.
        CHECK(!build_bary_fallback_gs(vs, ps, 3, gs, nullptr, &err) && !err.empty() && gs.empty());
        err.clear();
        CHECK(build_bary_fallback_gs(vs, ps, 3, PerVertexMap{{0, 0}}, gs, nullptr, &err));
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        CHECK((ins.in_locs == std::set<std::uint32_t>{0}));
        CHECK((ins.out_pv_locs == std::set<std::uint32_t>{0}));
        CHECK(ins.out_loc_vec_len.size() == 1 && ins.out_loc_vec_len[3] == 3);
        CHECK(ins.out_pv_stores == 9 && ins.out_pv_null_stores == 0);  // three values at each of three vertices
        maybe_dump(dump_dir, "gs_per_vertex", gs);
    }

    // 7b. A relocated block next to ordinary inputs: attribute 0 -> Locations 3..5, pass-through 1, barycentrics 2;
    //     the PS also reads Location 0 as an ordinary input, so one GS Input at Location 0 feeds both.
    {
        const auto vs = build_vs({.raw_locs = {0, 1, 6}});
        const auto ps = build_ps({0, 1}, {}, 2, {}, {3});
        std::vector<std::uint32_t> gs;
        std::string err;
        CHECK(build_bary_fallback_gs(vs, ps, 2, PerVertexMap{{0, 3}}, gs, nullptr, &err));
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        CHECK((ins.in_locs == std::set<std::uint32_t>{0, 1}));  // 6 is unread and stays unconsumed
        CHECK((ins.out_pv_locs == std::set<std::uint32_t>{3}));
        CHECK(ins.out_loc_vec_len.size() == 3);
        CHECK(ins.out_loc_vec_len[0] == 4 && ins.out_loc_vec_len[1] == 4 && ins.out_loc_vec_len[2] == 3);
        CHECK(ins.out_pv_stores == 9);
        maybe_dump(dump_dir, "gs_per_vertex_relocated", gs);
    }

    // 7c. Two attributes, one of which the VS does not export: that array is zero-filled, with no dangling GS Input.
    {
        const auto vs = build_vs({.raw_locs = {2}});
        const auto ps = build_ps({}, {}, 0, {}, {1, 4});
        std::vector<std::uint32_t> gs;
        std::string err;
        CHECK(build_bary_fallback_gs(vs, ps, 0, PerVertexMap{{2, 1}, {5, 4}}, gs, nullptr, &err));
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        CHECK((ins.in_locs == std::set<std::uint32_t>{2}));
        CHECK((ins.out_pv_locs == std::set<std::uint32_t>{1, 4}));
        CHECK(ins.out_pv_stores == 9 && ins.out_pv_null_stores == 3);
        maybe_dump(dump_dir, "gs_per_vertex_zero_fill", gs);
    }

    // 7d. A mapping the PS does not declare is ignored: the module is the one built without it.
    {
        const auto vs = build_vs({.raw_locs = {0, 2}});
        const auto ps = build_ps({0, 2}, {}, 1);
        std::vector<std::uint32_t> plain, mapped;
        std::string err;
        CHECK(build_bary_fallback_gs(vs, ps, 1, plain, nullptr, &err));
        CHECK(build_bary_fallback_gs(vs, ps, 1, PerVertexMap{{0, 7}}, mapped, nullptr, &err));
        CHECK(plain == mapped);
    }

    // 7e. Refusals leave caller's gs_words untouched.
    {
        const std::vector<std::uint32_t> sentinel = {0xdeadbeefu};
        std::vector<std::uint32_t> gs = sentinel;
        std::string err;
        const auto vs = build_vs({.raw_locs = {0, 1}});
        const auto refused = [&](const std::vector<std::uint32_t>& ps, std::uint32_t bary, const PerVertexMap& pv) {
            err.clear();
            return !build_bary_fallback_gs(vs, ps, bary, pv, gs, nullptr, &err) && !err.empty() && gs == sentinel;
        };
        // The mapped first location holds a plain vec4, not vec4[3].
        CHECK(refused(build_ps({0}, {}, 3), 3, {{0, 0}}));
        // The block 0..2 runs into the barycentric location.
        CHECK(refused(build_ps({}, {}, 2, {}, {0}), 2, {{0, 0}}));
        // The block 0..2 runs into another PS Input (Location 1).
        CHECK(refused(build_ps({1}, {}, 3, {}, {0}), 3, {{0, 0}}));
        // Two attributes at one first location.
        CHECK(refused(build_ps({}, {}, 3, {}, {0}), 3, {{0, 0}, {1, 0}}));
        // A first location whose block would pass the last location.
        CHECK(refused(build_ps({}, {}, 3, {}, {0}), 3, {{0, 0}, {1, 0xfffffffeu}}));
        // An array the map names by its second location only.
        CHECK(refused(build_ps({}, {}, 3, {}, {0}), 3, {{0, 1}}));
    }

    // Relocating the two decoded halves to 13/14 must not make the VS appear to export raw attr14.
    {
        ShaderEntry vs_e, ps_e;
        ps_e.f16_halves = {{0, {28, 29}}};
        auto vs = build_vs({.raw_locs = {0}});
        auto ps = build_ps({}, {}, 3, ps_e.f16_halves, {0, 4});
        PerVertexMap raw_vs_outputs;
        std::vector<std::uint32_t> gs;
        std::string err;
        CHECK(link_f16_halves(vs, vs_e, ps, ps_e, &err, 14, &raw_vs_outputs));
        CHECK(raw_vs_outputs.size() == 1 && raw_vs_outputs.count(0));
        CHECK(build_bary_fallback_gs(vs, ps, 3, PerVertexMap{{14, 0}, {0, 4}}, gs,
                                     nullptr, &err, &raw_vs_outputs));
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        // Keep both decoded halves and the genuine per-vertex attr0, but zero-fill the array for attr14.
        CHECK((ins.in_locs == std::set<std::uint32_t>{0, 13, 14}));
        CHECK((ins.out_pv_locs == std::set<std::uint32_t>{0, 4}));
        CHECK(ins.out_pv_stores == 9 && ins.out_pv_null_stores == 3);
        maybe_dump(dump_dir, "gs_per_vertex_relocated_half_alias", gs);
        maybe_dump(dump_dir, "ps_per_vertex_relocated_half_alias", ps);
    }

    // The same rule applies when the FP16 linker leaves unused static VS halves in place on its no-op path.
    {
        ShaderEntry vs_e;
        vs_e.f16_halves = {{8, {12, 13}}};
        auto vs = build_vs({.raw_locs = {8}, .static_halves = vs_e.f16_halves});
        const auto original_vs = vs;
        auto ps = build_ps({}, {}, 3, {}, {0});
        PerVertexMap raw_vs_outputs;
        std::vector<std::uint32_t> gs;
        std::string err;
        CHECK(link_f16_halves(vs, vs_e, ps, {}, &err, 14, &raw_vs_outputs));
        CHECK(vs == original_vs && raw_vs_outputs.size() == 1 && raw_vs_outputs.count(8));
        CHECK(build_bary_fallback_gs(vs, ps, 3, PerVertexMap{{13, 0}}, gs, nullptr, &err, &raw_vs_outputs));
        GsInspection ins;
        CHECK(verify_gs_module(gs, ins));
        CHECK(ins.in_locs.empty() && ins.out_pv_stores == 0 && ins.out_pv_null_stores == 3);
        maybe_dump(dump_dir, "gs_per_vertex_static_half_alias", gs);
    }

    return 0;
}
