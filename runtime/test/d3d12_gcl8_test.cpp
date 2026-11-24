// ID3D12GraphicsCommandList8 and 9 on the Vulkan-backed COM surface: QueryInterface for the public IIDs, their
// methods with the values the layer records, and the reference count. Creating an allocator and a command list needs
// neither a GPU nor the Vulkan loader, so this runs as a plain CPU test.
//
// The stencil references are recorded but not applied: the backend has no pipeline or draw path that could consume
// per-face stencil references yet (docs/d3d12_vulkan.md section 3.2). The GCL9 depth bias and strip-cut value are
// list state, applied only by a pipeline carrying the matching DYNAMIC_* flag (d3d12_depth24_draw_test and
// d3d12_draw_test draw with them).
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_trace.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--trace-d3d12") == 0) rt::d3d12_trace::set_enabled(true);
    else if (argc != 1) return 2;
    using namespace rt::d3d12;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList7> list7;
    EXPECT(SUCCEEDED(create_command_allocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(create_command_list(D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, IID_PPV_ARGS(&list7.p))));

    // Null bindings exercise the public entry points without creating a pipeline or loading Vulkan.
    list7.p->SetPipelineState(nullptr);
    list7.p->SetGraphicsRootSignature(nullptr);

    // AddRef through the GCL7 pointer gives the count this test expects while the QueryInterface reference is held.
    const ULONG held = list7.p->AddRef();
    list7.p->Release();
    {
        Ref<ID3D12GraphicsCommandList8> list8;
        EXPECT(SUCCEEDED(list7.p->QueryInterface(IID_PPV_ARGS(&list8.p))));
        EXPECT(list8.p != nullptr);
        EXPECT(static_cast<IUnknown*>(list8.p) == static_cast<IUnknown*>(list7.p));
        EXPECT(list7.p->AddRef() == held + 1);  // QueryInterface holds one reference
        list7.p->Release();

        list8.p->OMSetFrontAndBackStencilRef(1, 2);
        UINT front = 0, back = 0;
        EXPECT(front_back_stencil_ref(list8.p, front, back));
        EXPECT(front == 1 && back == 2);

        // Both parameters are UINTs, not bytes.
        list8.p->OMSetFrontAndBackStencilRef(0xffffffffu, 0x80000000u);
        EXPECT(front_back_stencil_ref(list8.p, front, back));
        EXPECT(front == 0xffffffffu && back == 0x80000000u);

        // The method really is at the SDK's GCL8 slot: IUnknown 0-2, ID3D12Object 3-6, ID3D12DeviceChild 7,
        // ID3D12CommandList 8, GCL 9-59, GCL1 60-65, GCL2 66, GCL3 67, GCL4 68-76, GCL5 77-78, GCL6 79,
        // GCL7 80 (Barrier), GCL8 81 (byte offset 0x288) in d3d12.h 10.0.26100.0.
        constexpr std::size_t kStencilRefSlot = 81;
        using StencilRefFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList8*, UINT, UINT);
        void** vtable = *reinterpret_cast<void***>(list8.p);
        reinterpret_cast<StencilRefFn>(vtable[kStencilRefSlot])(list8.p, 3, 4);
        EXPECT(front_back_stencil_ref(list8.p, front, back));
        EXPECT(front == 3 && back == 4);

        // A marker is traced by name, never by its label. A method still unsupported (protected sessions) goes
        // through its generated stub on each call, even when the legacy diagnostic is deduplicated.
        constexpr char marker[] = "synthetic-marker-argument";
        list8.p->SetMarker(1, marker, static_cast<UINT>(sizeof(marker)));
        list8.p->SetProtectedResourceSession(nullptr);
        list8.p->SetProtectedResourceSession(nullptr);
        EXPECT(list8.p->Close() == S_OK);
        EXPECT(list8.p->Close() == E_FAIL);
        EXPECT(SUCCEEDED(list7.p->Reset(allocator.p, nullptr)));
        EXPECT(front_back_stencil_ref(list8.p, front, back));
        EXPECT(front == 0 && back == 0);  // Reset returns list state to its default

        // GCL9: the depth bias and the strip-cut value are list state, at slots 82 and 83 (after GCL8's 81).
        Ref<ID3D12GraphicsCommandList9> list9;
        EXPECT(SUCCEEDED(list7.p->QueryInterface(IID_PPV_ARGS(&list9.p))));
        EXPECT(static_cast<IUnknown*>(list9.p) == static_cast<IUnknown*>(list7.p));
        using BiasFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList9*, FLOAT, FLOAT, FLOAT);
        using CutFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList9*, D3D12_INDEX_BUFFER_STRIP_CUT_VALUE);
        void** vtable9 = *reinterpret_cast<void***>(list9.p);
        reinterpret_cast<BiasFn>(vtable9[82])(list9.p, 2.5f, 0.25f, -1.5f);
        reinterpret_cast<CutFn>(vtable9[83])(list9.p, D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF);
        float constant = 0.0f, clamp = 0.0f, slope = 0.0f;
        std::uint32_t cut = 0;
        EXPECT(depth_bias_state(list9.p, constant, clamp, slope));
        EXPECT(constant == 2.5f && clamp == 0.25f && slope == -1.5f);
        EXPECT(strip_cut_value(list9.p, cut) && cut == D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF);
        EXPECT(list9.p->Close() == S_OK);  // Reset takes only a closed list
        EXPECT(SUCCEEDED(list7.p->Reset(allocator.p, nullptr)));
        EXPECT(depth_bias_state(list9.p, constant, clamp, slope));
        EXPECT(constant == 0.0f && clamp == 0.0f && slope == 0.0f);
        EXPECT(strip_cut_value(list9.p, cut) && cut == D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED);

        // A value outside the enum makes Close report the invalid call, as D3D12's own validation does.
        reinterpret_cast<CutFn>(vtable9[83])(list9.p, static_cast<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>(7));
        EXPECT(list9.p->Close() == E_INVALIDARG);
        EXPECT(SUCCEEDED(list7.p->Reset(allocator.p, nullptr)));  // leave the list open for the calls below
    }
    EXPECT(list7.p->AddRef() == held);  // QueryInterface released its reference
    list7.p->Release();

    // Calls made before pipeline setup still belong in the trace. They invalidate this list without needing a GPU.
    list7.p->SetDescriptorHeaps(0, nullptr);
    list7.p->SetGraphicsRootDescriptorTable(0, {});
    list7.p->SetComputeRootSignature(nullptr);
    list7.p->Dispatch(1, 1, 1);
    list7.p->DrawInstanced(3, 1, 0, 0);
    list7.p->DrawIndexedInstanced(3, 1, 0, 0, 0);
    EXPECT(list7.p->Close() == E_INVALIDARG);

    UINT ignored_front = 0, ignored_back = 0;
    EXPECT(!front_back_stencil_ref(nullptr, ignored_front, ignored_back));

    std::printf("GCL8/9: QueryInterface, OMSetFrontAndBackStencilRef, RSSetDepthBias, IASetIndexBufferStripCutValue, "
                "refcount balanced\n");
    return 0;
}
