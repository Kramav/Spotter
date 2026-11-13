// The console guard (D49, the user's rule): a program's console window never closes by itself on an
// error, so the cause can be read. Linked into every program in examples/ by examples/CMakeLists.txt
// (an OBJECT library, so this file always links and its set-up below runs before main). Nothing to
// call: a program gets it by being built there. The check_* programs don't link it (ctest never waits).
//
//   An uncaught exception (any thread)   prints its type and message, then waits for Enter.
//   A crash (Windows: access violation,  prints the exception code, its name, the address and where it
//   stack overflow, ...)                 happened (module + offset, to look up in the .pdb), then waits.
//   abort() / an assert; on Linux also   prints the signal, then waits.
//   a segfault or a divide by zero
//   Any other end (return 1, Q, ...)     prints "--- <program> has ended ---" and waits for Enter, but only
//                                        when the program owns its console (double-clicked or started from
//                                        Explorer). Run from a terminal it never waits: the terminal stays.
//
// The exit code of a plain `return 1` can't be seen from here (atexit isn't told it), so every error exit
// prints its own reason first; this only keeps the window open under it. All messages go to stderr,
// unbuffered. A Kinect that hangs while stopping in a destructor still hangs before this runs.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <typeinfo>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace
{

std::atomic<bool> crashed{ false };   // the first crash handler to run prints; the rest stay quiet

//std::string ProgramName(){
//
//@brief - this program's file name (13e_Room_Scan.exe), or "the program". O(1).
std::string ProgramName()
{
#ifdef _WIN32
    char path[MAX_PATH] = {};
    if (!GetModuleFileNameA(nullptr, path, MAX_PATH)) return "the program";
#else
    char path[4096] = {};
    if (readlink("/proc/self/exe", path, sizeof path - 1) <= 0) return "the program";
#endif
    std::string name = path;
    size_t slash = name.find_last_of("\\/");
    return slash == std::string::npos ? name : name.substr(slash + 1);
}

//bool OwnsConsole(){
//
//@brief - true when no other program shares this console (Windows: double-clicked or started from Explorer, so
//         the window closes when this program ends). False in a terminal, and always on Linux. O(1).
bool OwnsConsole()
{
#ifdef _WIN32
    DWORD ids[2];
    return GetConsoleProcessList(ids, 2) == 1;
#else
    return false;
#endif
}

//void WaitForEnter(const std::string& line){
//
//@param1 - line: what to print first, framed
//
//@brief - prints the line, then, if this program owns its console, closes the OpenCV windows (frozen otherwise),
//         drops any keys typed earlier and waits for Enter. O(1) plus the wait.
void WaitForEnter(const std::string& line)
{
    std::fflush(stdout);
    std::fprintf(stderr, "\n%s\n", line.c_str());
    if (!OwnsConsole()) return;
    try { cv::destroyAllWindows(); } catch (...) {}
    std::fprintf(stderr, "Press Enter to close this window.\n");
#ifdef _WIN32
    FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
#endif
    for (int c = std::getchar(); c != '\n' && c != EOF; c = std::getchar()) {}
}

//void AtExit(){
//
//@brief - any normal end (main returned, exit()): the closing line, and the wait when the console is ours. O(1).
void AtExit()
{
    if (!crashed) WaitForEnter("--- " + ProgramName() + " has ended ---");
}

//std::string CurrentException(){
//
//@brief - the exception being handled right now, as "uncaught exception <type>: <message>", or "" if there's none.
//         O(1).
std::string CurrentException()
{
    std::exception_ptr error = std::current_exception();
    if (!error) return "";
    try { std::rethrow_exception(error); }
    catch (const cv::Exception& e) { return std::string("uncaught exception cv::Exception: ") + e.what(); }
    catch (const std::exception& e) { return std::string("uncaught exception ") + typeid(e).name() + ": " + e.what(); }
    catch (...) { return "uncaught exception of an unknown type (not a std::exception)"; }
}

//void OnTerminate(){
//
//@brief - std::terminate, i.e. an exception nobody caught (or a joinable thread destroyed): its type and message,
//         the wait, then the end (no atexit, so no second wait). Never returns. (MSVC keeps this per thread, so
//         other threads reach OnSignal's SIGABRT instead, and an exception leaving main reaches OnCrash.)
void OnTerminate()
{
    if (!crashed.exchange(true))
    {
        std::string what = CurrentException();
        if (what.empty()) what = "std::terminate was called with no exception (e.g. a std::thread destroyed while still running)";
        WaitForEnter("=== " + ProgramName() + " CRASHED: " + what + " ===");
    }
    std::_Exit(3);
}

//void OnSignal(int signal){
//
//@param1 - signal: the signal's number
//
//@brief - abort() or an assert (and on Linux a segfault or a divide by zero): the signal's name, the wait, then
//         the end. Never returns.
void OnSignal(int signal)
{
    if (!crashed.exchange(true))
    {
        const char* name = signal == SIGABRT ? "SIGABRT (abort() or a failed assert)" : signal == SIGSEGV ? "SIGSEGV (a bad memory access)"
                         : signal == SIGFPE ? "SIGFPE (e.g. an integer divide by zero)" : "an unexpected signal";
        const std::string what = CurrentException();   // an exception that left another thread ends here on MSVC
        WaitForEnter("=== " + ProgramName() + " CRASHED: " + (what.empty() ? std::string("signal ") + name : what + " (on another thread)") + " ===");
    }
    std::_Exit(3);
}

#ifdef _WIN32
//std::string CodeName(DWORD code){
//
//@param1 - code: a Windows exception code
//
//@brief - a readable name for the common ones, else "". O(1).
std::string CodeName(DWORD code)
{
    switch (code)
    {
    case EXCEPTION_ACCESS_VIOLATION: return "access violation";
    case EXCEPTION_STACK_OVERFLOW: return "stack overflow (endless recursion?)";
    case EXCEPTION_INT_DIVIDE_BY_ZERO: return "integer divide by zero";
    case EXCEPTION_ILLEGAL_INSTRUCTION: return "illegal instruction";
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "array bounds exceeded";
    case EXCEPTION_IN_PAGE_ERROR: return "in-page error (a file or disk read failed)";
    case 0xE06D7363: return "C++ exception that reached the system";
    case 0xC0000409: return "stack buffer overrun / fail-fast";
    default: return "";
    }
}

//std::string CppException(const EXCEPTION_RECORD& record){
//
//@param1 - record: a crash with code 0xE06D7363 (a C++ throw nobody caught)
//
//@brief - the thrown object's type and, for a std::exception (cv::Exception too), its message, read from the
//         record the way MSVC lays a throw out (x64: ThrowInfo -> catchable types, as offsets from the image
//         base). "" if the record isn't one MSVC made. An exception leaving main ends here, not in std::terminate.
//         O(the type's bases).
std::string CppException(const EXCEPTION_RECORD& record)
{
#if defined(_WIN64)
    if (record.NumberParameters < 4 || record.ExceptionInformation[0] != 0x19930520) return "";
    struct ThrowInfo { unsigned attributes; int unwind, forwardCompat, catchableTypes; };
    struct CatchableType { unsigned properties; int type; int mdisp, pdisp, vdisp; int size, copy; };
    struct TypeDescriptor { const void* vtable; void* spare; char name[1]; };
    const char* object = (const char*)record.ExceptionInformation[1];
    const ThrowInfo* info = (const ThrowInfo*)record.ExceptionInformation[2];
    const char* base = (const char*)record.ExceptionInformation[3];
    if (!object || !info || !base || !info->catchableTypes) return "";
    const int* types = (const int*)(base + info->catchableTypes);   // [count, offset, offset, ...]
    auto name = [&](int i) { return std::string(((const TypeDescriptor*)(base + ((const CatchableType*)(base + types[1 + i]))->type))->name); };
    auto readable = [](const std::string& mangled) {   // ".?AVruntime_error@std@@" -> "std::runtime_error"
        std::vector<std::string> parts(1);
        for (char c : mangled.substr(mangled.size() > 4 ? 4 : 0))
        {
            if (c == '@') parts.emplace_back();
            else parts.back().push_back(c);
        }
        std::string out;
        for (auto part = parts.rbegin(); part != parts.rend(); ++part)
            if (!part->empty()) out += (out.empty() ? "" : "::") + *part;
        return out;
    };
    std::string text = "uncaught exception " + readable(name(0));
    for (int i = 0; i < types[0]; i++)
        if (name(i) == ".?AVexception@std@@")   // it is a std::exception: ask it what happened
        {
            const CatchableType* c = (const CatchableType*)(base + types[1 + i]);
            if (c->pdisp == -1) text += std::string(": ") + ((const std::exception*)(object + c->mdisp))->what();
            break;
        }
    return text;
#else
    (void)record;
    return "";
#endif
}

//LONG WINAPI OnCrash(EXCEPTION_POINTERS* info){
//
//@param1 - info: the crash, from Windows
//
//@brief - a crash no one handled: its code and name, the bad address for an access violation, and the module +
//         offset where it happened; the wait; then the end (EXCEPTION_EXECUTE_HANDLER: no atexit). O(1).
LONG WINAPI OnCrash(EXCEPTION_POINTERS* info)
{
    if (!crashed.exchange(true))
    {
        const EXCEPTION_RECORD& record = *info->ExceptionRecord;
        const std::string thrown = record.ExceptionCode == 0xE06D7363 ? CppException(record) : "";
        if (!thrown.empty())   // a C++ exception: its type and message say more than where Windows raised it
        {
            WaitForEnter("=== " + ProgramName() + " CRASHED: " + thrown + " ===");
            return EXCEPTION_EXECUTE_HANDLER;
        }
        std::string text = cv::format("exception code 0x%08lX", (unsigned long)record.ExceptionCode);
        const std::string name = CodeName(record.ExceptionCode);
        if (!name.empty()) text += " (" + name;
        if (record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record.NumberParameters >= 2)
        {
            const ULONG_PTR kind = record.ExceptionInformation[0];
            text += cv::format(", %s address 0x%llX", kind == 0 ? "reading" : kind == 1 ? "writing" : "executing",
                               (unsigned long long)record.ExceptionInformation[1]);
        }
        if (!name.empty()) text += ")";
        HMODULE module = nullptr;
        char path[MAX_PATH] = "an unknown module";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)record.ExceptionAddress, &module) && module)
            GetModuleFileNameA(module, path, MAX_PATH);
        std::string where = path;
        size_t slash = where.find_last_of("\\/");
        if (slash != std::string::npos) where = where.substr(slash + 1);
        text += cv::format(" at %s+0x%llX", where.c_str(), (unsigned long long)((const char*)record.ExceptionAddress - (const char*)module));
        WaitForEnter("=== " + ProgramName() + " CRASHED: " + text + " ===");
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

// The set-up: runs before main, because this file's static object is always linked (an OBJECT library).
struct Install
{
    Install()
    {
        std::setvbuf(stderr, nullptr, _IONBF, 0);
        std::set_terminate(OnTerminate);
        std::signal(SIGABRT, OnSignal);
#ifdef _WIN32
        // Not SIGSEGV / SIGFPE here: the CRT would hand crashes to them without the code and address.
        _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);   // no extra pop-up: the console says it
        SetUnhandledExceptionFilter(OnCrash);
        ULONG reserve = 64 * 1024;   // room on the stack for OnCrash after a stack overflow
        SetThreadStackGuarantee(&reserve);
#else
        std::signal(SIGSEGV, OnSignal);
        std::signal(SIGFPE, OnSignal);
#endif
        std::atexit(AtExit);
    }
} install;

}
