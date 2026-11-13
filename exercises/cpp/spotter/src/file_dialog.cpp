#include "spotter/file_dialog.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shobjidl.h>
#else
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;

namespace spotter
{

std::string Ask(const std::string& question)
{
    std::printf("%s", question.c_str());
    std::fflush(stdout);
    std::string answer;
    if (!std::getline(std::cin, answer)) return "cancel";
    while (!answer.empty() && (answer.back() == ' ' || answer.back() == '"' || answer.back() == '\r')) answer.pop_back();
    size_t start = answer.find_first_not_of(" \"");
    return start == std::string::npos ? "" : answer.substr(start);
}

//bool ConsoleOnly(){
//
//@brief - true when SPOTTER_CONSOLE_DIALOGS=1 is set: every question goes to the console (scripts, a headless
//         Pi, the self-checks). O(1).
static bool ConsoleOnly()
{
    const char* value = std::getenv("SPOTTER_CONSOLE_DIALOGS");
    return value && std::string(value) == "1";
}

//fs::path ExistingFolder(fs::path folder){
//
//@param1 - folder: a folder that may not exist yet
//
//@brief - the folder itself if it exists, else the nearest one above it that does (empty if none). O(depth).
static fs::path ExistingFolder(fs::path folder)
{
    std::error_code error;
    while (!folder.empty() && !fs::is_directory(folder, error))
    {
        if (folder == folder.parent_path()) return {};
        folder = folder.parent_path();
    }
    return folder;
}

//fs::path FromConsole(const std::string& typed, const fs::path& base){
//
//@param1 - typed: a path typed in the console
//@param2 - base: the folder a relative path is taken from
//
//@brief - the typed path made absolute from base (never from the folder the program runs in). O(1).
static fs::path FromConsole(const std::string& typed, const fs::path& base)
{
    fs::path p = fs::u8path(typed);
    return (p.is_absolute() ? p : base / p).lexically_normal();
}

#ifdef _WIN32

//std::wstring Wide(const std::string& text){
//
//@param1 - text: UTF-8 text
//
//@brief - the same text as UTF-16, as Windows wants it. O(length).
static std::wstring Wide(const std::string& text)
{
    if (text.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), (int)text.size(), nullptr, 0);
    std::wstring wide(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), (int)text.size(), wide.data(), n);
    return wide;
}

// COM on this thread for as long as a dialog is open (the shell's dialogs need a single-threaded apartment).
struct Com
{
    HRESULT result;
    Com() : result(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)) {}
    ~Com() { if (SUCCEEDED(result)) CoUninitialize(); }
    bool Ok() const { return SUCCEEDED(result); }   // RPC_E_CHANGED_MODE (already multi-threaded) is not: the console asks
};

//bool ShowDialog(IFileDialog* dialog, const std::string& title, const fs::path& folder, const std::string& name, fs::path& chosen){
//
//@param1 - dialog: a Save or Open dialog, its options set
//@param2 - title: the window's title
//@param3 - folder: where it opens (the nearest existing folder)
//@param4 - name: the suggested name (or empty)
//@param5 - chosen: set to the chosen path, or empty if cancelled
//
//@brief - shows the dialog in front of the program's windows and reads the answer. False if it couldn't be shown
//         (the console asks instead). Releases the dialog. O(1) plus the wait.
static bool ShowDialog(IFileDialog* dialog, const std::string& title, const fs::path& folder, const std::string& name, fs::path& chosen)
{
    chosen.clear();
    dialog->SetTitle(Wide(title).c_str());
    const fs::path start = ExistingFolder(folder);
    IShellItem* item = nullptr;
    if (!start.empty() && SUCCEEDED(SHCreateItemFromParsingName(start.wstring().c_str(), nullptr, IID_PPV_ARGS(&item))))
    {
        dialog->SetFolder(item);   // always here, not wherever the dialog was last used
        item->Release();
    }
    if (!name.empty()) dialog->SetFileName(Wide(name).c_str());
    HRESULT shown = dialog->Show(GetForegroundWindow());
    bool ok = SUCCEEDED(shown) || shown == HRESULT_FROM_WIN32(ERROR_CANCELLED);
    IShellItem* result = nullptr;
    if (SUCCEEDED(shown) && SUCCEEDED(dialog->GetResult(&result)))
    {
        PWSTR path = nullptr;
        if (SUCCEEDED(result->GetDisplayName(SIGDN_FILESYSPATH, &path))) { chosen = fs::path(path); CoTaskMemFree(path); }
        else ok = false;
        result->Release();
    }
    dialog->Release();
    return ok;
}

//bool WindowsSaveAs(const std::string& title, const fs::path& folder, const std::string& name, const std::string& filterName, const std::string& extension, fs::path& chosen){
//
//@brief - AskSaveAs with the shell's Save As window. False if it couldn't be shown. O(1) plus the wait.
static bool WindowsSaveAs(const std::string& title, const fs::path& folder, const std::string& name, const std::string& filterName,
                          const std::string& extension, fs::path& chosen)
{
    Com com;
    IFileSaveDialog* dialog = nullptr;
    if (!com.Ok() || FAILED(CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return false;
    DWORD options = 0;
    dialog->GetOptions(&options);
    options |= FOS_FORCEFILESYSTEM | FOS_NOREADONLYRETURN | FOS_PATHMUSTEXIST;
    options = extension.empty() ? options & ~FOS_OVERWRITEPROMPT : options | FOS_OVERWRITEPROMPT;
    dialog->SetOptions(options);
    // A folder name: a pattern no file has, so only folders are listed. A file: *.<extension>, added when missing.
    const std::wstring label = Wide(extension.empty() ? filterName + " (type the new folder's name)" : filterName + " (*." + extension + ")");
    const std::wstring pattern = Wide(extension.empty() ? std::string("*.spotter-folder-name") : "*." + extension);
    COMDLG_FILTERSPEC filter{ label.c_str(), pattern.c_str() };
    dialog->SetFileTypes(1, &filter);
    if (!extension.empty()) dialog->SetDefaultExtension(Wide(extension).c_str());
    return ShowDialog(dialog, title, folder, name, chosen);
}

//bool WindowsOpenFolder(const std::string& title, const fs::path& startIn, const std::string& preselect, fs::path& chosen){
//
//@brief - AskOpenFolder with the shell's Select Folder window. False if it couldn't be shown. O(1) plus the wait.
static bool WindowsOpenFolder(const std::string& title, const fs::path& startIn, const std::string& preselect, fs::path& chosen)
{
    Com com;
    IFileOpenDialog* dialog = nullptr;
    if (!com.Ok() || FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return false;
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    return ShowDialog(dialog, title, startIn, preselect, chosen);
}

#else

//std::string Quoted(const std::string& text){
//
//@param1 - text: any text
//
//@brief - the text in single quotes for the shell, its own single quotes escaped. O(length).
static std::string Quoted(const std::string& text)
{
    std::string quoted = "'";
    for (char c : text) quoted += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return quoted + "'";
}

//bool Zenity(const std::string& arguments, std::string& answer, bool& yes){
//
//@param1 - arguments: zenity's arguments
//@param2 - answer: set to what it printed (its last newline removed)
//@param3 - yes: set to true if it ended with OK / Yes, false for Cancel / No / closed
//
//@brief - runs zenity when there's a desktop. False if it couldn't (no desktop, zenity not installed): the console
//         asks instead. O(1) plus the wait.
static bool Zenity(const std::string& arguments, std::string& answer, bool& yes)
{
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY")) return false;
    FILE* pipe = popen(("zenity " + arguments + " 2>/dev/null").c_str(), "r");
    if (!pipe) return false;
    answer.clear();
    char buffer[512];
    while (std::fgets(buffer, sizeof buffer, pipe)) answer += buffer;
    const int status = pclose(pipe);
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (code != 0 && code != 1) return false;   // 127 = not installed, -1 = killed
    while (!answer.empty() && answer.back() == '\n') answer.pop_back();
    yes = code == 0;
    return true;
}

#endif

fs::path AskSaveAs(const std::string& title, const fs::path& folder, const std::string& name, const std::string& filterName, const std::string& extension)
{
    fs::path chosen;
    bool shown = false;
    if (!ConsoleOnly())
    {
#ifdef _WIN32
        shown = WindowsSaveAs(title, folder, name, filterName, extension, chosen);
#else
        std::string answer;
        bool yes = false;
        shown = Zenity("--file-selection --save" + std::string(extension.empty() ? "" : " --confirm-overwrite") + " --title=" + Quoted(title) +
                       " --filename=" + Quoted((folder / name).string()), answer, yes);
        if (shown && yes && !answer.empty()) chosen = answer;
#endif
    }
    if (!shown)   // the console instead
    {
        std::printf("\n=== %s (cancel = don't save) ===\n", title.c_str());
        std::string where = Ask("Folder [" + folder.string() + "]: ");
        if (where == "cancel") return {};
        std::string file = Ask("Name [" + name + "]: ");
        if (file == "cancel") return {};
        chosen = FromConsole(where.empty() ? folder.string() : where, folder) / fs::u8path(file.empty() ? name : file);
    }
    if (!chosen.empty() && !extension.empty() && chosen.extension() != "." + extension) chosen += "." + extension;
    std::printf("%s: %s\n", title.c_str(), chosen.empty() ? "cancelled" : chosen.string().c_str());
    return chosen;
}

fs::path AskOpenFolder(const std::string& title, const fs::path& startIn, const std::string& preselect)
{
    fs::path chosen;
    bool shown = false;
    if (!ConsoleOnly())
    {
#ifdef _WIN32
        shown = WindowsOpenFolder(title, startIn, preselect, chosen);
#else
        std::string answer;
        bool yes = false;
        shown = Zenity("--file-selection --directory --title=" + Quoted(title) + " --filename=" + Quoted((startIn / preselect).string() + "/"), answer, yes);
        if (shown && yes && !answer.empty()) chosen = answer;
#endif
    }
    if (!shown)
    {
        std::printf("\n=== %s (cancel = none) ===\n", title.c_str());
        const fs::path suggested = preselect.empty() ? startIn : startIn / preselect;
        std::string where = Ask("Folder [" + suggested.string() + "]: ");
        if (where == "cancel") return {};
        chosen = where.empty() ? suggested : FromConsole(where, startIn);
    }
    std::printf("%s: %s\n", title.c_str(), chosen.empty() ? "cancelled" : chosen.string().c_str());
    return chosen;
}

bool Confirm(const std::string& title, const std::string& text)
{
    std::printf("\n%s\n%s\n", title.c_str(), text.c_str());
    bool yes = false, shown = false;
    if (!ConsoleOnly())
    {
#ifdef _WIN32
        shown = true;
        yes = MessageBoxW(GetForegroundWindow(), Wide(text).c_str(), Wide(title).c_str(),
                          MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2 | MB_SETFOREGROUND | MB_TOPMOST) == IDYES;
#else
        std::string answer;
        shown = Zenity("--question --default-cancel --title=" + Quoted(title) + " --text=" + Quoted(text), answer, yes);
#endif
    }
    if (!shown) yes = Ask("[y/N]: ") == "y";
    std::printf("-> %s\n", yes ? "yes" : "no");
    return yes;
}

void Tell(const std::string& title, const std::string& text)
{
    std::printf("\n=== %s ===\n%s\n", title.c_str(), text.c_str());
    if (ConsoleOnly()) return;
#ifdef _WIN32
    MessageBoxW(GetForegroundWindow(), Wide(text).c_str(), Wide(title).c_str(), MB_OK | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST);
#else
    std::string answer;
    bool yes = false;
    Zenity("--error --title=" + Quoted(title) + " --text=" + Quoted(text), answer, yes);
#endif
}

}
