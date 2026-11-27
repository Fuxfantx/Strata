// vk-rdc.cpp - one decode step through RenderDoc's in-application capture API.
//
// The x64 build runs under Prism with the Adreno still native, so RenderDoc's x64 layer can capture it; there is no
// ARM64 RenderDoc.  Captures are requested with STRATA_VK_RDC=<decode step> (and optional STRATA_VK_RDC_PATH), which
// wraps exactly that step so the capture holds one graph: the pipelines and descriptors of a single token.
#include "vk-common.h"

#ifdef _WIN32
#include <windows.h>
#endif
#ifdef STRATA_HAVE_RDC
#include "renderdoc_app.h"
#endif

#include <cstdlib>
#include <cstdio>

namespace strata {

namespace {

#ifdef STRATA_HAVE_RDC
RENDERDOC_API_1_6_0 * g_api = nullptr;
bool g_probed = false;

RENDERDOC_API_1_6_0 * rdc_api() {
    if (g_probed) {
        return g_api;
    }
    g_probed = true;
    HMODULE mod = LoadLibraryA("renderdoc.dll");
    if (mod == nullptr) {
        logf("[strata-vk] RenderDoc: renderdoc.dll not found next to the executable; no capture\n");
        return nullptr;
    }
    pRENDERDOC_GetAPI get_api = (pRENDERDOC_GetAPI) GetProcAddress(mod, "RENDERDOC_GetAPI");
    if (get_api == nullptr) {
        logf("[strata-vk] RenderDoc: renderdoc.dll has no RENDERDOC_GetAPI\n");
        return nullptr;
    }
    void * api = nullptr;
    if (!get_api(eRENDERDOC_API_Version_1_6_0, &api) || api == nullptr) {
        logf("[strata-vk] RenderDoc: this build does not expose the 1.6.0 capture API\n");
        return nullptr;
    }
    g_api = (RENDERDOC_API_1_6_0 *) api;
    return g_api;
}
#endif

// which decode step to capture, and where to put it; 0 (the default) disables capture entirely
int capture_step() {
    static const int step = [] {
        const char * e = std::getenv("STRATA_VK_RDC");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    return step;
}

} // namespace

bool rdc_capture_wanted(int step) {
#ifdef STRATA_HAVE_RDC
    return capture_step() > 0 && step == capture_step() && rdc_api() != nullptr;
#else
    (void) step;
    return false;
#endif
}

void rdc_capture_begin() {
#ifdef STRATA_HAVE_RDC
    RENDERDOC_API_1_6_0 * api = rdc_api();
    if (api == nullptr) {
        return;
    }
    if (const char * path = std::getenv("STRATA_VK_RDC_PATH")) {
        api->SetCaptureFilePathTemplate(path);
    } else {
        api->SetCaptureFilePathTemplate("strata_capture");
    }
    // A decode step references the 542 MiB compute buffer, the 32k-token KV cache and every weight, and an unbounded
    // capture of that never finished (17 GB resident, ten minutes).  We are after event timings and pipeline state, not
    // resource contents, so cap the serialisation: RenderDoc then truncates buffers instead of copying them.
    uint32_t limit_mb = 2048;
    if (const char * mb = std::getenv("STRATA_VK_RDC_MB")) {
        limit_mb = (uint32_t) std::atoi(mb);
    }
    api->SetCaptureOptionU32(eRENDERDOC_Option_SoftMemoryLimit, limit_mb);
    logf("[strata-vk] RenderDoc: capturing decode step %d (soft memory limit %u MB)\n", capture_step(), limit_mb);
    api->StartFrameCapture(nullptr, nullptr);
#endif
}

void rdc_capture_end() {
#ifdef STRATA_HAVE_RDC
    RENDERDOC_API_1_6_0 * api = rdc_api();
    if (api == nullptr) {
        return;
    }
    if (api->EndFrameCapture(nullptr, nullptr)) {
        logf("[strata-vk] RenderDoc: capture written (%u so far)\n", api->GetNumCaptures());
    } else {
        logf("[strata-vk] RenderDoc: EndFrameCapture failed\n");
    }
#endif
}

} // namespace strata
