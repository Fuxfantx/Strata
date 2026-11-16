// x64debug.c - a minimal debugger: launches a command line with DEBUG_ONLY_THIS_PROCESS and reports the first
// exceptions with module, offset and a stack walk.  Built as x64 so it can debug the x64 engine under Prism; nothing
// has to be installed.  Modules are resolved through psapi (dbghelp's own module table is unreliable for a debuggee
// here), and each frame is printed as module+offset so the offsets can be symbolized offline, e.g. with
// llvm-symbolizer against the .pdb next to the module.
#include <windows.h>
#include <dbghelp.h>
#include <psapi.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static const char * exception_name(DWORD code) {
    switch (code) {
        case 0xC0000409: return "STATUS_STACK_BUFFER_OVERRUN (fail-fast)";
        case 0xC0000005: return "STATUS_ACCESS_VIOLATION";
        case 0xC000001D: return "STATUS_ILLEGAL_INSTRUCTION";
        case 0xC0000094: return "STATUS_INTEGER_DIVIDE_BY_ZERO";
        case 0xC00000FD: return "STATUS_STACK_OVERFLOW";
        case 0xC0000135: return "STATUS_DLL_NOT_FOUND";
        case 0x80000003: return "breakpoint";
        case 0x40080201: return "SEH app-raise (WIL/HRESULT carry)";
        case 0xE06D7363: return "C++ exception";
        default:         return "(other)";
    }
}

// Resolve the module that contains `a` in the debuggee, through psapi.
static DWORD64 base_of(HANDLE proc, DWORD64 a, char * name, size_t name_len) {
    if (name != NULL && name_len > 0) {
        name[0] = 0;
    }
    HMODULE mods[512];
    DWORD needed = 0;
    DWORD64 base = 0;
    if (!EnumProcessModules(proc, mods, sizeof(mods), &needed)) {
        return 0;
    }
    for (DWORD i = 0; i < needed / sizeof(HMODULE); ++i) {
        MODULEINFO mi;
        if (!GetModuleInformation(proc, mods[i], &mi, sizeof(mi))) {
            continue;
        }
        const DWORD64 lo = (DWORD64) (uintptr_t) mi.lpBaseOfDll;
        if (a >= lo && a < lo + mi.SizeOfImage) {
            base = lo;
            if (name != NULL && name_len > 0) {
                GetModuleBaseNameA(proc, mods[i], name, (DWORD) name_len);
            }
            break;
        }
    }
    return base;
}

static void describe(HANDLE proc, DWORD64 a) {
    char name[MAX_PATH] = { 0 };
    const DWORD64 base = base_of(proc, a, name, sizeof(name));
    if (base != 0) {
        printf("    module %s + 0x%llx\n", name, (unsigned long long) (a - base));
    } else {
        printf("    module unresolved (address 0x%llx)\n", (unsigned long long) a);
    }
    char buf[sizeof(SYMBOL_INFO) + 512] = { 0 };
    SYMBOL_INFO * si = (SYMBOL_INFO *) buf;
    si->SizeOfStruct = sizeof(SYMBOL_INFO);
    si->MaxNameLen   = 512;
    DWORD64 disp = 0;
    if (SymFromAddr(proc, a, &disp, si)) {
        printf("    symbol %s + 0x%llx\n", si->Name, (unsigned long long) disp);
    }
}

static void walk_stack(HANDLE proc, DWORD thread_id) {
    HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, thread_id);
    if (thread == NULL) {
        printf("    (no thread handle: %lu)\n", GetLastError());
        return;
    }
    CONTEXT ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.ContextFlags = CONTEXT_ALL;
    if (!GetThreadContext(thread, &ctx)) {
        printf("    (GetThreadContext failed: %lu)\n", GetLastError());
        CloseHandle(thread);
        return;
    }
    STACKFRAME64 frame;
    memset(&frame, 0, sizeof(frame));
    frame.AddrPC.Offset    = ctx.Rip;
    frame.AddrPC.Mode      = AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp;
    frame.AddrStack.Mode   = AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp;
    frame.AddrFrame.Mode   = AddrModeFlat;

    for (int i = 0; i < 16; ++i) {
        char name[MAX_PATH] = { 0 };
        const DWORD64 base = base_of(proc, frame.AddrPC.Offset, name, sizeof(name));
        printf("    #%02d %s + 0x%llx\n", i, base != 0 ? name : "(unknown)",
               (unsigned long long) (base != 0 ? frame.AddrPC.Offset - base : frame.AddrPC.Offset));
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, thread, &frame, &ctx, NULL,
                         SymFunctionTableAccess64, SymGetModuleBase64, NULL) ||
            frame.AddrPC.Offset == 0) {
            break;
        }
    }
    fflush(stdout);
    CloseHandle(thread);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: x64debug.exe <command line>\n");
        return 2;
    }
    char cmd[4096] = { 0 };
    for (int i = 1; i < argc; ++i) {
        if (i > 1) strcat_s(cmd, sizeof(cmd), " ");
        strcat_s(cmd, sizeof(cmd), argv[i]);
    }

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    memset(&pi, 0, sizeof(pi));
    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE,
                        DEBUG_ONLY_THIS_PROCESS | CREATE_NEW_PROCESS_GROUP, NULL, NULL, &si, &pi)) {
        printf("CreateProcess failed: %lu\n", GetLastError());
        return 3;
    }
    printf("launched pid %lu\n", (unsigned long) pi.dwProcessId);
    fflush(stdout);

    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(pi.hProcess, NULL, FALSE);

    DEBUG_EVENT de;
    int reports = 0;
    for (;;) {
        if (!WaitForDebugEvent(&de, INFINITE)) break;
        DWORD cont = DBG_CONTINUE;
        if (de.dwDebugEventCode == LOAD_DLL_DEBUG_EVENT) {
            // keep the handle: dbghelp maps the image through it
            SymLoadModuleEx(pi.hProcess, de.u.LoadDll.hFile, NULL, NULL,
                            (DWORD64) (uintptr_t) de.u.LoadDll.lpBaseOfDll, 0, NULL, 0);
        } else if (de.dwDebugEventCode == EXCEPTION_DEBUG_EVENT) {
            const EXCEPTION_RECORD * er = &de.u.Exception.ExceptionRecord;
            const int first_chance = de.u.Exception.dwFirstChance != 0;
            if (reports < 8) {
                printf("[exception] code 0x%08lx %s%s\n", er->ExceptionCode, exception_name(er->ExceptionCode),
                       first_chance ? "" : " (second chance)");
                if (er->ExceptionCode == 0xC0000409 && er->NumberParameters > 0) {
                    printf("    fast-fail subcode: %llu\n", (unsigned long long) er->ExceptionInformation[0]);
                }
                describe(pi.hProcess, (DWORD64) (uintptr_t) er->ExceptionAddress);
                if (!first_chance) {
                    walk_stack(pi.hProcess, de.dwThreadId);
                }
                fflush(stdout);
                ++reports;
            }
            cont = DBG_EXCEPTION_NOT_HANDLED;
        } else if (de.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT) {
            printf("[exit] code 0x%08lx\n", de.u.ExitProcess.dwExitCode);
            fflush(stdout);
            ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
            break;
        }
        ContinueDebugEvent(de.dwProcessId, de.dwThreadId, cont);
    }
    SymCleanup(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return 0;
}
