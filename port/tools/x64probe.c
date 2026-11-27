// x64probe.c - asks one question: can an x64 (Prism-emulated) process on this machine see the Adreno GPU through
// Vulkan?  It loads the system loader at run time, so it needs no import library and no SDK: if the loader and the
// driver's ICD both carry an x64 slice (ARM64X), the device count is non-zero.
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

typedef struct VkInstance_T * VkInstance;

typedef struct {
    uint32_t sType, pNext, flags, apiVersion, applicationVersion, engineVersion;
} VkApplicationInfo_lite;

typedef struct {
    uint32_t sType, pNext, flags;
    const void * pApplicationInfo;
    uint32_t enabledLayerCount;
    const char * const * ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char * const * ppEnabledExtensionNames;
} VkInstanceCreateInfo_lite;

typedef struct {
    uint32_t apiVersion, driverVersion, vendorID, deviceID, deviceType;
    char     name[256];
    uint8_t  uuid[16];
} VkPhysicalDeviceProperties_lite;

typedef int32_t (*PFN_create)(const void *, const void *, VkInstance *);
typedef int32_t (*PFN_enum)(VkInstance, uint32_t *, void **);
typedef void    (*PFN_getprops)(void *, void *);

int main(void) {
    HMODULE vk = LoadLibraryA("vulkan-1.dll");
    if (!vk) {
        printf("x64: cannot load vulkan-1.dll (error %lu)\n", GetLastError());
        return 2;
    }
    printf("x64: vulkan-1.dll loaded in this process\n");
    fflush(stdout);

    void * gipa = GetProcAddress(vk, "vkGetInstanceProcAddr");
    if (!gipa) {
        printf("x64: no vkGetInstanceProcAddr\n");
        return 3;
    }
    PFN_create  create  = (PFN_create)  ((void * (*)(void *, const char *)) gipa)(NULL, "vkCreateInstance");
    if (!create) {
        printf("x64: no vkCreateInstance\n");
        return 4;
    }

    VkApplicationInfo_lite app;
    memset(&app, 0, sizeof(app));
    app.sType       = 1;
    app.apiVersion  = (1u << 22);   // 1.0
    app.applicationVersion = 1;
    app.engineVersion      = 1;

    VkInstanceCreateInfo_lite ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = 1;
    ci.pApplicationInfo = &app;

    VkInstance inst = NULL;
    int32_t r = create(&ci, NULL, &inst);
    printf("x64: vkCreateInstance -> %d%s\n", (int) r, r == 0 ? " (VK_SUCCESS)" : "");
    fflush(stdout);
    if (r != 0 || !inst) {
        return 5;
    }

    PFN_enum enumerate = (PFN_enum) ((void * (*)(void *, const char *)) gipa)(inst, "vkEnumeratePhysicalDevices");
    if (!enumerate) {
        printf("x64: no vkEnumeratePhysicalDevices\n");
        return 6;
    }

    uint32_t n = 0;
    int32_t e1 = enumerate(inst, &n, NULL);
    printf("x64: vkEnumeratePhysicalDevices -> %d, %u device(s)\n", (int) e1, n);
    fflush(stdout);
    if (n == 0) {
        return 7;
    }

    void * devs[8];
    memset(devs, 0, sizeof(devs));
    uint32_t m = n > 8 ? 8 : n;
    int32_t e2 = enumerate(inst, &m, devs);
    printf("x64: second call -> %d, %u handle(s):", (int) e2, m);
    for (uint32_t i = 0; i < m; ++i) {
        printf(" %p", devs[i]);
    }
    printf("\n");
    fflush(stdout);

    PFN_getprops getprops = (PFN_getprops) ((void * (*)(void *, const char *)) gipa)(inst, "vkGetPhysicalDeviceProperties");
    for (uint32_t i = 0; getprops && i < m; ++i) {
        VkPhysicalDeviceProperties_lite p;
        memset(&p, 0, sizeof(p));
        getprops(devs[i], &p);
        printf("x64:   [%u] %s (vendor 0x%04x, api %u.%u.%u)\n", i, p.name, p.vendorID,
               (p.apiVersion >> 22) & 0x7f, (p.apiVersion >> 12) & 0x3ff, p.apiVersion & 0xfff);
        fflush(stdout);
    }
    return 0;
}
