/**
 * tools/hwinfo_npu_bridge.cpp -- a live monitor for the AMD XDNA1 NPU (Phoenix / Hawk Point),
 * and a bridge that publishes the same numbers to HWiNFO64's Custom Sensors interface.
 *
 * WHAT IT SHOWS, AND WHERE EACH NUMBER COMES FROM
 * ------------------------------------------------
 * Every figure on screen is read from one of three sources, and the screen says which:
 *
 *   1. Windows' own GPU-engine statistics (the D3DKMT counters Task Manager reads), through
 *      PDH:  \GPU Engine(*)\Utilization Percentage, \GPU Adapter Memory(*)\{Shared,Dedicated}
 *      Usage, \GPU Process Memory(*)\Shared Usage. The NPU is an MCDM compute-only adapter,
 *      so it has a LUID and engine counters like any GPU. This is the ONLY live utilization
 *      number available for this NPU: measured on the Phoenix desktop 2026-09-07, an IRON
 *      (XRT hardware-context) GEMM run showed 88.5 % on
 *      pid_<pid>_luid_0x00000000_0x0000d6bf_phys_0_eng_0_engtype_compute while xrt-smi's own
 *      GOPS/FPS/latency columns for the same context read N/A. The NPU adapter is identified
 *      through DXCore (a D3D12_CORE_COMPUTE adapter whose driver description names an
 *      NPU/IPU/XDNA device); --adapter / --luid override that if the match fails.
 *
 *   2. xrt-smi examine -r aie-partitions (AMD's signed CLI over the XRT driver query
 *      interface), per read: memory in use, which columns each partition holds, and the
 *      per-hardware-context table -- pid, process, status, submissions, completions,
 *      migrations, suspensions, errors, priority, and whatever GOPS/EGOPS/FPS/latency the
 *      context reports (VitisAI EP contexts report GOPS; IRON/XRT ones report N/A). The
 *      completions/s and submissions/s columns are deltas of xrt-smi's counters between
 *      reads, computed here.
 *
 *   3. XRT's own C++ query API (xrt::device::get_info, the interface xrt-smi is a CLI over),
 *      in-process, when built with HAVE_XRT: device name and BDF once; and per poll, at
 *      ~0.05 ms each, the power mode and **the NPU clock** -- max_clock_frequency_mhz is a
 *      live readback on this driver, not a nameplate: measured 2026-09-07 on the Phoenix
 *      desktop it reads 800 MHz idle, 1800 MHz for as long as a hardware context is active,
 *      and 800 again afterwards (the other session's in-kernel clock probe puts the busy
 *      core clock at 1.80 GHz in the default power mode, results/aie/clock_probe_npu.log).
 *      Without HAVE_XRT the same static fields come from xrt-smi examine -r platform / -r host
 *      once at start-up, and no clock is shown. Firmware version always comes from the host
 *      report. The XRT queries for AIE core/shim/mem status, electrical, thermal and memory
 *      return "No such query request" or "No sensors present" on this driver and are not used.
 *
 * WHAT IT DOES NOT SHOW, AND WHY
 * ------------------------------
 * Voltage and power are not exposed for this NPU by any documented interface: xrt-smi prints
 * "Estimated Power: N/A", its electrical query fails at the driver escape ("0xc0000023: The
 * data area passed to a system call is too small"), and thermal/mechanical report "No sensors
 * present". HWiNFO's native 0.001 V for the NPU is a placeholder, not a reading, and its flat
 * 800 MHz is the idle clock. Nothing here invents a value for the missing ones.
 *
 * HWiNFO64 side: writes HKCU\Software\HWiNFO64\Sensors\Custom\<group>\{Usage0,Clock0,Other0..N}
 * with Name/Value/Unit, the interface HWiNFO documents for custom sensors (enabled by default
 * since v6.10; HWiNFO must be running with its Sensors window open to pick them up).
 * Two rules about what gets written, both because HWiNFO's Min/Max/Average columns average
 * every sample they are given and a custom sensor has no way to say "no reading":
 *   - a value xrt-smi reports as N/A (GOPS/EGOPS for IRON/XRT contexts) is never published
 *     as 0 -- the key is removed until some context reports a number;
 *   - --idle hide removes the activity sensors (utilization, clock, completions/s,
 *     submissions/s, GOPS) while no hardware context is active, so HWiNFO's Average covers
 *     the time the NPU was doing something. The default, --idle zero, keeps publishing the
 *     true idle readings (0 %, 800 MHz, 0/s), which is what pulls a whole-session Average
 *     down. The memory, context-count and column-count sensors are always published.
 * The previous build of this bridge named its group "XDNA NPU" when the platform name did
 * not parse and used a different sensor schema (Other0 = "NPU GOPS"); that group is
 * removed at start-up when found, so HWiNFO does not show two NPUs.
 *
 * POLLING
 * -------
 * Two cadences. The fast sources -- the Windows engine counters and XRT's clock readback --
 * cost a few ms and are sampled every --interval seconds (default 0.5, minimum 0.1: up to ten
 * samples a second). xrt-smi is a child process that takes a few hundred ms per report, so it
 * runs on its own thread every --smi-interval seconds (default 2, 0 = never) and a frame uses
 * the newest xrt-smi sample, showing its age; completions/s and submissions/s are deltas
 * between consecutive xrt-smi reads, not per frame. The footer prints the requested and the
 * measured poll period. Dashboard keys: + and - halve and double the poll interval, [ and ]
 * the xrt-smi interval, s turns xrt-smi off and on, p pauses (the activity sensors are
 * removed from HWiNFO while paused, for the Average rule above), q quits.
 * Measured 2026-09-07 on the Phoenix desktop, three monitors side by side across one 2048^3
 * bf16 GEMM hold (results/aie/npu_monitor_poll_rate_npu.log): the engine counter read a mean
 * 88.2 % over 312 polls at 0.1 s (single polls 82-94 %), 87.9 % at 0.25 s and 87.8 % at 0.5 s
 * (86-90 %) -- the same figure at every rate, a little more scatter at the fastest, no
 * dropouts; the first poll of a load is a partial or a 100 % window. The measured period is
 * exact at all three rates (0.100 / 0.250 / 0.500 s) with timeBeginPeriod(1); on the default
 * 15.6 ms tick the 20 ms naps ran ~31 ms and every period carried ~22 ms over the request.
 * When a kernel hung (ERT_CMD_STATE_TIMEOUT in the hold) the counter fell to 0 with sporadic
 * 100 % samples while the process waited out the timeout.
 *
 * USAGE
 *   hwinfo_npu_bridge.exe                     live dashboard: 0.5 s polls, xrt-smi every 2 s, publishes to HWiNFO
 *   hwinfo_npu_bridge.exe --interval 0.1      ten utilization/clock samples a second
 *   hwinfo_npu_bridge.exe --smi-interval 5    xrt-smi every 5 s (0 = never run it)
 *   hwinfo_npu_bridge.exe --once              one sample, printed plainly, then exit
 *   hwinfo_npu_bridge.exe --json              one JSON object per sample on stdout (for scripts)
 *   hwinfo_npu_bridge.exe --plain             one text line per sample, no screen redraw
 *   hwinfo_npu_bridge.exe --no-hwinfo         monitor only, touch no registry key
 *   hwinfo_npu_bridge.exe --idle hide         remove activity sensors from HWiNFO while the NPU is idle
 *   hwinfo_npu_bridge.exe --background        hide the console window (bridge only)
 *   hwinfo_npu_bridge.exe --clean             remove this group's registry keys on exit
 *
 * Build: scripts/build_hwinfo_bridge.bat (MSVC; nlohmann/json + boost headers from the
 * npu_monitor_build conda env; XRT SDK from C:\Xilinx\XRT\xrt_sdk when present -> HAVE_XRT).
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <unknwn.h>
#include <initguid.h>
#include <dxcore.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <timeapi.h>
#include <conio.h>
#include <io.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#ifdef HAVE_XRT
#include <xrt/xrt_device.h>
#endif

#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "dxcore.lib")
#pragma comment(lib, "winmm.lib")

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

static const wchar_t* XRT_SMI_PATH = L"C:\\Windows\\System32\\AMD\\xrt-smi.exe";
static const wchar_t* REG_ROOT = L"Software\\HWiNFO64\\Sensors\\Custom";

static std::atomic<bool> g_running{true};

BOOL WINAPI ConsoleCtrlHandler(DWORD dwCtrlType) {
    switch (dwCtrlType) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        g_running = false;
        return TRUE;
    default:
        return FALSE;
    }
}

// ---------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------

static std::wstring Utf8ToWide(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}

static std::string WideToUtf8(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}

static std::string Trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

static std::string Lower(std::string s) {
    for (auto& c : s) c = (char)tolower((unsigned char)c);
    return s;
}

// json value -> string, whatever its type ("N/A" stays "N/A")
static std::string JStr(const json& j, const char* key, const std::string& def = "") {
    if (!j.is_object() || !j.contains(key)) return def;
    const auto& v = j[key];
    if (v.is_string()) return v.get<std::string>();
    if (v.is_number_integer()) return std::to_string(v.get<long long>());
    if (v.is_number_float()) { char b[64]; snprintf(b, sizeof b, "%g", v.get<double>()); return b; }
    return def;
}

// json value -> number; -1 when absent or "N/A"
static double JNum(const json& j, const char* key) {
    std::string s = Trim(JStr(j, key, ""));
    if (s.empty() || Lower(s) == "n/a") return -1.0;
    try { return std::stod(s); } catch (...) { return -1.0; }
}

// "96 MB" -> 96, "96 KB" -> 0.09375 (MB), "N/A" -> -1
static double ParseMb(const std::string& s) {
    std::string t = Lower(Trim(s));
    if (t.empty() || t == "n/a") return -1.0;
    double v = 0; char unit[8] = {0};
    if (sscanf(t.c_str(), "%lf %7s", &v, unit) < 1) return -1.0;
    std::string u = unit;
    if (u.rfind("kb", 0) == 0) return v / 1024.0;
    if (u.rfind("gb", 0) == 0) return v * 1024.0;
    if (u.rfind("b", 0) == 0) return v / (1024.0 * 1024.0);
    return v;  // MB or no unit
}

static std::string FmtNum(double v, int prec, const char* na = "n/a") {
    if (v < 0) return na;
    char b[64]; snprintf(b, sizeof b, "%.*f", prec, v); return b;
}

static std::string ProcessNameOf(int pid) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (!h) return "";
    wchar_t buf[MAX_PATH]; DWORD n = MAX_PATH;
    std::string name;
    if (QueryFullProcessImageNameW(h, 0, buf, &n)) {
        std::wstring w(buf, n);
        size_t p = w.find_last_of(L"\\/");
        name = WideToUtf8(p == std::wstring::npos ? w : w.substr(p + 1));
    }
    CloseHandle(h);
    return name;
}

// ---------------------------------------------------------------------------------------
// xrt-smi (source 2 and 3)
// ---------------------------------------------------------------------------------------

static std::wstring TempJsonPath(const wchar_t* prefix) {
    wchar_t dir[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, dir)) wcscpy_s(dir, L".");
    wchar_t file[MAX_PATH];
    if (GetTempFileNameW(dir, prefix, 0, file) != 0) return file;
    return std::wstring(dir) + L"\\" + prefix + L".json";
}

// Run `xrt-smi examine -r <report> -f JSON -o <file> --force` with no console flash.
// Returns false (with err text) on launch failure, non-zero exit, or timeout.
static bool RunReport(const std::wstring& report, const std::wstring& outPath, std::string& err, double& ms) {
    std::wstring cmd = L"\"" + std::wstring(XRT_SMI_PATH) + L"\" examine -r " + report +
                       L" -f JSON -o \"" + outPath + L"\" --force";
    STARTUPINFOW si = { sizeof(si) };
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi = { 0 };
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(L'\0');
    auto t0 = Clock::now();
    if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        err = "CreateProcess failed (" + std::to_string(GetLastError()) + ")";
        ms = 0;
        return false;
    }
    DWORD w = WaitForSingleObject(pi.hProcess, 15000);
    DWORD code = 0;
    if (w == WAIT_TIMEOUT) { TerminateProcess(pi.hProcess, 1); err = "xrt-smi timed out (15 s)"; }
    else { GetExitCodeProcess(pi.hProcess, &code); if (code != 0) err = "xrt-smi exit code " + std::to_string(code); }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return w != WAIT_TIMEOUT && code == 0;
}

static bool LoadJsonFile(const std::wstring& path, json& out, std::string& err) {
    std::ifstream f(path);
    if (!f.is_open()) { err = "report file not written"; return false; }
    try { f >> out; return true; }
    catch (const std::exception& e) { err = std::string("bad JSON: ") + e.what(); return false; }
}

static bool Report(const wchar_t* report, json& out, std::string& err, double& ms) {
    std::wstring tmp = TempJsonPath(L"npu");
    bool ok = RunReport(report, tmp, err, ms);
    if (ok) ok = LoadJsonFile(tmp, out, err);
    DeleteFileW(tmp.c_str());
    return ok;
}

struct StaticInfo {
    std::string name = "NPU (name unknown)";
    std::string bdf, powerMode, xrtVersion, driverVersion, firmware;
    int totalCols = 0;
    std::string staticHow = "xrt-smi";   // "XRT API" when the in-process queries answered
    bool xrtApi = false;                 // in-process XRT device open succeeded
    // NPU adapter as Windows sees it (source 1)
    bool haveLuid = false;
    LUID luid{};
    std::string luidTag;       // "luid_0x00000000_0x0000d6bf", as PDH spells it
    std::string adapterDesc;   // DXCore driver description, e.g. "AMD IPU Device"
    std::string adapterHow;    // how the adapter was picked
};

#ifdef HAVE_XRT
static xrt::device* g_xrtDev = nullptr;

// Open device 0 through XRT and take what it answers. Any failure leaves the fields for
// the xrt-smi path to fill.
static bool XrtStatic(StaticInfo& si, std::vector<std::string>& notes) {
    try {
        g_xrtDev = new xrt::device(0);
        std::string n = Trim(g_xrtDev->get_info<xrt::info::device::name>());
        if (!n.empty()) si.name = n;
        si.bdf = g_xrtDev->get_info<xrt::info::device::bdf>();
        json plat = json::parse(g_xrtDev->get_info<xrt::info::device::platform>());
        const auto& p0 = plat.at("platforms").at(0);
        si.totalCols = (int)JNum(p0.at("static_region"), "total_columns");
        if (p0.contains("status")) si.powerMode = JStr(p0["status"], "power_mode");
        json host = json::parse(g_xrtDev->get_info<xrt::info::device::host>());
        si.xrtVersion = JStr(host, "version");
        if (host.contains("drivers"))
            for (const auto& drv : host["drivers"])
                if (JStr(drv, "name").find("NPU") != std::string::npos) si.driverVersion = JStr(drv, "version");
        si.xrtApi = true;
        si.staticHow = "XRT API";
        return true;
    } catch (const std::exception& e) {
        notes.push_back(std::string("XRT API unavailable, using xrt-smi: ") + e.what());
        delete g_xrtDev; g_xrtDev = nullptr;
        return false;
    }
}

// Per poll: clock (live) and power mode, ~0.05 ms and ~0.3 ms.
static void XrtLive(double& clockMhz, std::string& powerMode, double& ms) {
    clockMhz = -1;
    if (!g_xrtDev) return;
    auto t0 = Clock::now();
    try {
        clockMhz = (double)g_xrtDev->get_info<xrt::info::device::max_clock_frequency_mhz>();
        json plat = json::parse(g_xrtDev->get_info<xrt::info::device::platform>());
        const auto& p0 = plat.at("platforms").at(0);
        if (p0.contains("status")) powerMode = JStr(p0["status"], "power_mode");
    } catch (...) {}
    ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
#else
static bool XrtStatic(StaticInfo&, std::vector<std::string>&) { return false; }
static void XrtLive(double& clockMhz, std::string&, double& ms) { clockMhz = -1; ms = 0; }
#endif

static void QueryStatic(StaticInfo& si, std::vector<std::string>& notes) {
    json d; std::string err; double ms;
    bool viaApi = XrtStatic(si, notes);
    if (!viaApi && Report(L"platform", d, err, ms)) {
        try {
            const auto& dev = d.at("devices").at(0);
            si.bdf = JStr(dev, "device_id");
            const auto& plat = dev.at("platforms").at(0);
            const auto& sr = plat.at("static_region");
            std::string n = Trim(JStr(sr, "name"));
            if (!n.empty()) si.name = n;
            si.totalCols = (int)JNum(sr, "total_columns");
            if (plat.contains("status")) si.powerMode = JStr(plat["status"], "power_mode");
        } catch (...) { notes.push_back("platform report: unexpected shape"); }
    } else if (!viaApi) {
        notes.push_back("platform report: " + err);
    }
    if (Report(L"host", d, err, ms)) {   // firmware version lives only here
        try {
            const auto& host = d.at("system").at("host");
            if (host.contains("xrt") && !viaApi) {
                si.xrtVersion = JStr(host["xrt"], "version");
                if (host["xrt"].contains("drivers"))
                    for (const auto& drv : host["xrt"]["drivers"])
                        if (JStr(drv, "name").find("NPU") != std::string::npos) si.driverVersion = JStr(drv, "version");
            }
            if (host.contains("devices"))
                for (const auto& dev : host["devices"])
                    if (JStr(dev, "bdf") == si.bdf || si.bdf.empty()) si.firmware = JStr(dev, "firmware_version");
        } catch (...) { notes.push_back("host report: unexpected shape"); }
    } else {
        notes.push_back("host report: " + err);
    }
}

struct Ctx {
    int pid = 0;
    std::string process, status, priority;
    int ctxId = -1, partIdx = 0, startCol = 0, numCols = 0;
    double subs = -1, compl = -1, migr = -1, susp = -1, err = -1;
    double gops = -1, egops = -1, fps = -1, latency = -1;
    double memMb = -1, instrMb = -1;
    double subsPerS = -1, complPerS = -1;   // computed here from deltas
    double enginePct = -1;                  // from source 1, joined by pid
    double procSharedMb = -1;               // from source 1, joined by pid
    bool fromXrt = true;                    // false = seen only in the engine counters
};

struct XrtSample {
    bool ok = false;
    std::string err;
    double ms = 0;
    double totalMemMb = -1;
    int partitions = 0;
    std::vector<int> cols;      // columns held by any partition
    std::vector<Ctx> ctx;
    // set by the sampler (TakeXrt), not by the parser
    int seq = 0;                                     // 0 = no xrt-smi read yet
    Clock::time_point takenAt{};
    double totalComplPerS = -1, totalSubsPerS = -1;  // deltas against the previous read
};

static XrtSample SampleXrt() {
    XrtSample s;
    json d;
    if (!Report(L"aie-partitions", d, s.err, s.ms)) return s;
    s.ok = true;
    try {
        const auto& dev = d.at("devices").at(0);
        const auto& ap = dev.at("aie_partitions");
        s.totalMemMb = ParseMb(JStr(ap, "total_memory_usage", "N/A"));
        if (ap.contains("partitions") && ap["partitions"].is_array()) {
            for (const auto& part : ap["partitions"]) {
                s.partitions++;
                int start = (int)JNum(part, "start_col"), n = (int)JNum(part, "num_cols");
                int pidx = (int)JNum(part, "partition_index");
                for (int c = 0; c < n; ++c) s.cols.push_back(start + c);
                if (!part.contains("hw_contexts") || !part["hw_contexts"].is_array()) continue;
                for (const auto& h : part["hw_contexts"]) {
                    Ctx c;
                    c.pid = (int)JNum(h, "pid");
                    c.process = JStr(h, "process_name");
                    c.ctxId = (int)JNum(h, "context_id");
                    c.status = JStr(h, "status");
                    c.priority = JStr(h, "priority");
                    c.partIdx = pidx; c.startCol = start; c.numCols = n;
                    c.subs = JNum(h, "command_submissions");
                    c.compl = JNum(h, "command_completions");
                    c.migr = JNum(h, "migrations");
                    c.susp = JNum(h, "suspensions");
                    c.err = JNum(h, "errors");
                    c.gops = JNum(h, "gops");
                    c.egops = JNum(h, "egops");
                    c.fps = JNum(h, "fps");
                    c.latency = JNum(h, "latency");
                    c.memMb = ParseMb(JStr(h, "memory_usage", "N/A"));
                    c.instrMb = ParseMb(JStr(h, "instr_bo_mem", "N/A"));
                    s.ctx.push_back(c);
                }
            }
        }
        std::sort(s.cols.begin(), s.cols.end());
        s.cols.erase(std::unique(s.cols.begin(), s.cols.end()), s.cols.end());
    } catch (const std::exception& e) {
        s.ok = false; s.err = std::string("aie-partitions: unexpected shape: ") + e.what();
    }
    return s;
}

// ---------------------------------------------------------------------------------------
// The NPU adapter's LUID (DXCore), and its engine/memory counters (PDH)  -- source 1
// ---------------------------------------------------------------------------------------

static std::string LuidTag(const LUID& l) {
    char b[64]; snprintf(b, sizeof b, "luid_0x%08lx_0x%08lx", (unsigned long)l.HighPart, (unsigned long)l.LowPart);
    return b;
}

struct AdapterInfo { std::string desc; LUID luid{}; bool hardware = false; bool graphics = false; bool npuType = false; bool genericMl = false; };

// Every adapter DXCore will list under any of: NPU hardware type (Windows 11 24H2+ NPUs),
// D3D12 generic-ML, or D3D12 core-compute. An MCDM NPU is not a graphics adapter, so the
// compute-only list alone can miss it (it did on the Phoenix desktop, where core-compute
// returned only the Radeon 780M and the Basic Render Driver).
static std::vector<AdapterInfo> EnumAdapters(std::string& err) {
    std::vector<AdapterInfo> out;
    IDXCoreAdapterFactory* factory = nullptr;
    HRESULT hr = DXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr) || !factory) { err = "DXCoreCreateAdapterFactory failed"; return out; }
    const GUID* filters[] = { &DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
    for (const GUID* attr : filters) {
        IDXCoreAdapterList* list = nullptr;
        if (FAILED(factory->CreateAdapterList(1, attr, IID_PPV_ARGS(&list))) || !list) continue;   // older DXCore: unknown attribute
        uint32_t n = list->GetAdapterCount();
        for (uint32_t i = 0; i < n; ++i) {
            IDXCoreAdapter* a = nullptr;
            if (FAILED(list->GetAdapter(i, IID_PPV_ARGS(&a))) || !a) continue;
            AdapterInfo ai;
            size_t sz = 0;
            if (SUCCEEDED(a->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &sz)) && sz > 0) {
                std::string d(sz, '\0');
                if (SUCCEEDED(a->GetProperty(DXCoreAdapterProperty::DriverDescription, sz, &d[0]))) ai.desc = Trim(d.c_str());
            }
            a->GetProperty(DXCoreAdapterProperty::InstanceLuid, &ai.luid);
            bool hw = false; a->GetProperty(DXCoreAdapterProperty::IsHardware, &hw); ai.hardware = hw;
            ai.graphics = a->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS);
            ai.npuType = a->IsAttributeSupported(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU);
            ai.genericMl = a->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML);
            a->Release();
            bool dup = false;
            for (const auto& o : out) if (o.luid.LowPart == ai.luid.LowPart && o.luid.HighPart == ai.luid.HighPart) { dup = true; break; }
            if (!dup) out.push_back(ai);
        }
        list->Release();
    }
    factory->Release();
    return out;
}

// LUIDs the GPU-engine statistics know about (every adapter with a memory segment, NPU included).
static std::vector<LUID> PdhAdapterLuids() {
    std::vector<LUID> out;
    PDH_HQUERY q = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &q) != ERROR_SUCCESS) return out;
    PDH_HCOUNTER c = nullptr;
    if (PdhAddEnglishCounterW(q, L"\\GPU Adapter Memory(*)\\Shared Usage", 0, &c) == ERROR_SUCCESS && PdhCollectQueryData(q) == ERROR_SUCCESS) {
        DWORD size = 0, count = 0;
        if (PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, &size, &count, nullptr) == PDH_MORE_DATA && size > 0) {
            std::vector<BYTE> buf(size);
            auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
            if (PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, &size, &count, items) == ERROR_SUCCESS) {
                for (DWORD i = 0; i < count; ++i) {
                    std::string inst = Lower(WideToUtf8(items[i].szName));
                    unsigned long hi = 0, lo = 0;
                    size_t p = inst.find("luid_");
                    if (p != std::string::npos && sscanf(inst.c_str() + p, "luid_0x%lx_0x%lx", &hi, &lo) == 2) {
                        LUID l; l.HighPart = (LONG)hi; l.LowPart = lo; out.push_back(l);
                    }
                }
            }
        }
    }
    PdhCloseQuery(q);
    return out;
}

static void PickNpuAdapter(StaticInfo& si, const std::string& match, const std::string& luidOverride, std::vector<std::string>& notes) {
    if (!luidOverride.empty()) {
        unsigned long lo = 0, hi = 0;
        std::string s = Lower(luidOverride);
        if (sscanf(s.c_str(), "0x%lx_0x%lx", &hi, &lo) == 2 || sscanf(s.c_str(), "0x%lx", &lo) == 1) {
            si.luid.HighPart = (LONG)hi; si.luid.LowPart = lo; si.haveLuid = true;
            si.luidTag = LuidTag(si.luid); si.adapterDesc = "(from --luid)"; si.adapterHow = "--luid";
            return;
        }
        notes.push_back("--luid not understood, expected 0x<low> or 0x<high>_0x<low>");
    }
    std::string err;
    auto adapters = EnumAdapters(err);
    const AdapterInfo* pick = nullptr;
    std::string m = Lower(match);
    if (!m.empty()) {
        for (const auto& a : adapters) if (Lower(a.desc).find(m) != std::string::npos) { pick = &a; si.adapterHow = "DXCore: description matches --adapter"; break; }
    } else {
        for (const auto& a : adapters) if (a.npuType) { pick = &a; si.adapterHow = "DXCore: NPU hardware type"; break; }
        if (!pick) for (const auto& a : adapters) {
            std::string d = Lower(a.desc);
            if (d.find("npu") != std::string::npos || d.find("ipu") != std::string::npos || d.find("xdna") != std::string::npos) { pick = &a; si.adapterHow = "DXCore: description names an NPU/IPU/XDNA"; break; }
        }
        if (!pick) for (const auto& a : adapters) if (a.hardware && !a.graphics) { pick = &a; si.adapterHow = "DXCore: the compute-only hardware adapter"; break; }
    }
    if (pick) { si.luid = pick->luid; si.haveLuid = true; si.luidTag = LuidTag(si.luid); si.adapterDesc = pick->desc; return; }

    // DXCore did not list the NPU. Windows' GPU-engine statistics still track it as an
    // adapter LUID, so take the LUID that has a memory segment but no DXCore adapter.
    if (m.empty()) {
        std::vector<LUID> left;
        for (const LUID& l : PdhAdapterLuids()) {
            bool known = false;
            for (const auto& a : adapters) if (a.luid.LowPart == l.LowPart && a.luid.HighPart == l.HighPart) { known = true; break; }
            if (!known) left.push_back(l);
        }
        if (left.size() == 1) {
            std::string listed;
            for (const auto& a : adapters) listed += " [" + a.desc + "]";
            si.luid = left[0]; si.haveLuid = true; si.luidTag = LuidTag(si.luid);
            si.adapterDesc = "adapter DXCore does not list"; si.adapterHow = "PDH: the only adapter LUID DXCore does not enumerate";
            notes.push_back("NPU adapter taken as " + si.luidTag + ": the only GPU-statistics adapter DXCore does not list (DXCore listed" +
                            (listed.empty() ? std::string(" nothing") : listed) + ")");
            return;
        }
    }
    std::string all;
    for (const auto& a : adapters) all += " [" + a.desc + " " + LuidTag(a.luid) + (a.npuType ? " NPU" : "") + "]";
    notes.push_back("NPU adapter not identified. DXCore adapters:" + (all.empty() ? std::string(" none") : all) + (err.empty() ? "" : " (" + err + ")") + " -- use --adapter <substring> or --luid 0x<low>");
}

struct PdhSample {
    bool ok = false;
    std::string err;
    double ms = 0;
    double utilPct = -1;                     // sum over the NPU adapter's engines and pids
    std::map<int, double> pidPct;            // per pid, NPU adapter only
    std::map<std::string, double> engPct;    // per engine instance (eng_N_engtype_X), NPU only
    double adapterSharedMb = -1, adapterDedicatedMb = -1;
    std::map<int, double> procSharedMb;      // per pid on the NPU adapter
};

class PdhReader {
public:
    bool Init(std::string& err) {
        if (PdhOpenQueryW(nullptr, 0, &q_) != ERROR_SUCCESS) { err = "PdhOpenQuery failed"; return false; }
        struct { const wchar_t* path; PDH_HCOUNTER* h; } want[] = {
            { L"\\GPU Engine(*)\\Utilization Percentage", &cEng_ },
            { L"\\GPU Adapter Memory(*)\\Shared Usage", &cAdShared_ },
            { L"\\GPU Adapter Memory(*)\\Dedicated Usage", &cAdDed_ },
            { L"\\GPU Process Memory(*)\\Shared Usage", &cProcShared_ },
        };
        for (auto& w : want) {
            PDH_STATUS st = PdhAddEnglishCounterW(q_, w.path, 0, w.h);
            if (st != ERROR_SUCCESS) { char b[128]; snprintf(b, sizeof b, "PdhAddEnglishCounter failed 0x%08lx for %ls", (unsigned long)st, w.path); err = b; return false; }
        }
        PdhCollectQueryData(q_);   // baseline; rates need two collections
        return true;
    }
    ~PdhReader() { if (q_) PdhCloseQuery(q_); }

    PdhSample Sample(const StaticInfo& si) {
        PdhSample s;
        auto t0 = Clock::now();
        PDH_STATUS st = PdhCollectQueryData(q_);
        if (st != ERROR_SUCCESS) { char b[64]; snprintf(b, sizeof b, "PdhCollectQueryData 0x%08lx", (unsigned long)st); s.err = b; return s; }
        if (!si.haveLuid) { s.err = "NPU adapter LUID unknown"; return s; }
        s.ok = true;
        std::string tag = Lower(si.luidTag);
        // engines
        double util = 0;
        ForEach(cEng_, [&](const std::string& inst, double v) {
            if (Lower(inst).find(tag) == std::string::npos) return;
            // A per-engine value far above 100 % is a counter reset (a process leaving), not a
            // reading: seen as 7e14 % once in a 0.25 s run. Drop it; cap the rest at 100.
            if (v < 0 || v > 1000.0) return;
            v = std::min(100.0, v);
            int pid = PidOf(inst);
            util += v;
            s.pidPct[pid] += v;
            size_t e = inst.find("_eng_");
            if (e != std::string::npos) s.engPct[inst.substr(e + 1)] += v;
        });
        s.utilPct = std::min(100.0, util);
        ForEach(cAdShared_, [&](const std::string& inst, double v) { if (Lower(inst).find(tag) != std::string::npos) s.adapterSharedMb = v / (1024.0 * 1024.0); });
        ForEach(cAdDed_, [&](const std::string& inst, double v) { if (Lower(inst).find(tag) != std::string::npos) s.adapterDedicatedMb = v / (1024.0 * 1024.0); });
        ForEach(cProcShared_, [&](const std::string& inst, double v) { if (Lower(inst).find(tag) != std::string::npos) s.procSharedMb[PidOf(inst)] = v / (1024.0 * 1024.0); });
        s.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return s;
    }

private:
    static int PidOf(const std::string& inst) {
        if (inst.rfind("pid_", 0) != 0) return -1;
        try { return std::stoi(inst.substr(4)); } catch (...) { return -1; }
    }
    template <typename F> void ForEach(PDH_HCOUNTER c, F fn) {
        DWORD size = 0, count = 0;
        PDH_STATUS st = PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &size, &count, nullptr);
        if (st != PDH_MORE_DATA || size == 0) return;
        std::vector<BYTE> buf(size);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
        st = PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &size, &count, items);
        if (st != ERROR_SUCCESS) return;
        for (DWORD i = 0; i < count; ++i) {
            if (items[i].FmtValue.CStatus != PDH_CSTATUS_VALID_DATA && items[i].FmtValue.CStatus != PDH_CSTATUS_NEW_DATA) continue;
            fn(WideToUtf8(items[i].szName), items[i].FmtValue.doubleValue);
        }
    }
    PDH_HQUERY q_ = nullptr;
    PDH_HCOUNTER cEng_ = nullptr, cAdShared_ = nullptr, cAdDed_ = nullptr, cProcShared_ = nullptr;
};

// ---------------------------------------------------------------------------------------
// HWiNFO64 custom sensors (registry)
// ---------------------------------------------------------------------------------------

struct Sensor { std::wstring key; std::wstring name; std::wstring unit; double value; };

static bool WriteSensor(const std::wstring& group, const Sensor& s) {
    std::wstring sub = std::wstring(REG_ROOT) + L"\\" + group + L"\\" + s.key;
    HKEY h = nullptr;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, sub.c_str(), 0, nullptr, REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, nullptr, &h, nullptr) != ERROR_SUCCESS)
        return false;
    wchar_t val[64]; swprintf_s(val, L"%.2f", s.value);
    bool ok = true;
    ok &= RegSetValueExW(h, L"Name", 0, REG_SZ, (const BYTE*)s.name.c_str(), (DWORD)((s.name.size() + 1) * sizeof(wchar_t))) == ERROR_SUCCESS;
    ok &= RegSetValueExW(h, L"Unit", 0, REG_SZ, (const BYTE*)s.unit.c_str(), (DWORD)((s.unit.size() + 1) * sizeof(wchar_t))) == ERROR_SUCCESS;
    ok &= RegSetValueExW(h, L"Value", 0, REG_SZ, (const BYTE*)val, (DWORD)((wcslen(val) + 1) * sizeof(wchar_t))) == ERROR_SUCCESS;
    RegCloseKey(h);
    return ok;
}

static void DeleteSensor(const std::wstring& group, const std::wstring& key) {
    RegDeleteKeyW(HKEY_CURRENT_USER, (std::wstring(REG_ROOT) + L"\\" + group + L"\\" + key).c_str());
}

static std::wstring ReadSensorName(const std::wstring& group, const std::wstring& key) {
    std::wstring sub = std::wstring(REG_ROOT) + L"\\" + group + L"\\" + key;
    HKEY h = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, sub.c_str(), 0, KEY_QUERY_VALUE, &h) != ERROR_SUCCESS) return L"";
    wchar_t buf[256]; DWORD size = sizeof(buf), type = 0;
    std::wstring out;
    if (RegQueryValueExW(h, L"Name", nullptr, &type, (BYTE*)buf, &size) == ERROR_SUCCESS && type == REG_SZ) out.assign(buf, size / sizeof(wchar_t));
    RegCloseKey(h);
    while (!out.empty() && out.back() == L'\0') out.pop_back();
    return out;
}

static void CleanRegistry(const std::wstring& group) {
    const wchar_t* types[] = { L"Usage", L"Other", L"Temp", L"Volt", L"Fan", L"Current", L"Power", L"Clock" };
    for (auto t : types)
        for (int i = 0; i < 16; ++i)
            RegDeleteKeyW(HKEY_CURRENT_USER, (std::wstring(REG_ROOT) + L"\\" + group + L"\\" + t + std::to_wstring(i)).c_str());
    RegDeleteKeyW(HKEY_CURRENT_USER, (std::wstring(REG_ROOT) + L"\\" + group).c_str());
}

// The previous build of this bridge wrote a different schema (Other0 = "NPU GOPS", Other1 =
// "NPU EGOPS", ...) under "XDNA NPU" (its fallback name when the platform name did not
// parse) or under the device name. Those keys are never updated again, so HWiNFO would keep
// showing a second, frozen NPU. Remove the legacy group, and any legacy keys in our own group
// that this schema does not overwrite.
static void RemoveLegacySensors(const std::wstring& ourGroup, std::vector<std::string>& notes) {
    if (ReadSensorName(L"XDNA NPU", L"Other0") == L"NPU GOPS") {
        CleanRegistry(L"XDNA NPU");
        notes.push_back("removed the previous build's frozen HWiNFO group \"XDNA NPU\"");
    }
    if (ourGroup != L"XDNA NPU" && ReadSensorName(ourGroup, L"Other0") == L"NPU GOPS") {
        for (int i = 0; i < 16; ++i) DeleteSensor(ourGroup, L"Other" + std::to_wstring(i));
        notes.push_back("replaced the previous build's sensors in HWiNFO group \"" + WideToUtf8(ourGroup) + "\"");
    }
}

// ---------------------------------------------------------------------------------------
// xrt-smi on its own cadence, and one poll = the fast sources joined with its newest sample
// ---------------------------------------------------------------------------------------

// xrt-smi is a child process that takes a few hundred ms per report, so it never runs inside
// the poll loop: XrtWorker reads it every intervalMs on its own thread and a frame takes the
// newest sample from here, with its age. Completions/s and submissions/s are deltas between
// consecutive xrt-smi reads, computed where the reads happen rather than per frame.
struct RateKey { int pid; int ctx; bool operator<(const RateKey& o) const { return pid != o.pid ? pid < o.pid : ctx < o.ctx; } };
struct RatePrev { double subs, compl; Clock::time_point t; };

struct XrtShared {
    std::mutex m;
    XrtSample latest;                    // seq 0 until the first read
    std::atomic<int> intervalMs{2000};   // 0 = xrt-smi off
    std::atomic<bool> paused{false};
    std::map<RateKey, RatePrev> prev;    // sampler side only
};

// One xrt-smi read, with rates against the previous one. Called from the worker thread, or
// from the main thread in --once mode -- never from both.
static void TakeXrt(XrtShared& sh, bool resetRates) {
    if (resetRates) sh.prev.clear();
    auto now = Clock::now();
    XrtSample s = SampleXrt();
    s.takenAt = now;
    std::map<RateKey, RatePrev> next;
    double totC = 0, totS = 0; bool any = false;
    for (auto& c : s.ctx) {
        RateKey k{ c.pid, c.ctxId };
        auto it = sh.prev.find(k);
        if (it != sh.prev.end()) {
            double dt = std::chrono::duration<double>(now - it->second.t).count();
            if (dt > 0 && c.compl >= 0 && it->second.compl >= 0 && c.compl >= it->second.compl) { c.complPerS = (c.compl - it->second.compl) / dt; totC += c.complPerS; any = true; }
            if (dt > 0 && c.subs >= 0 && it->second.subs >= 0 && c.subs >= it->second.subs) { c.subsPerS = (c.subs - it->second.subs) / dt; totS += c.subsPerS; }
        }
        next[k] = RatePrev{ c.subs, c.compl, now };
    }
    if (s.ok) sh.prev.swap(next);   // a failed read keeps the last good counters
    if (any) { s.totalComplPerS = totC; s.totalSubsPerS = totS; }
    std::lock_guard<std::mutex> lk(sh.m);
    s.seq = sh.latest.seq + 1;
    sh.latest = std::move(s);
}

static void XrtWorker(XrtShared* sh) {
    Clock::time_point last{};
    int reads = 0;
    bool gap = false;   // off or paused since the last read: the next read must not rate against stale counters
    while (g_running) {
        int iv = sh->intervalMs;
        if (iv <= 0 || sh->paused) { gap = true; std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }
        // the second read follows the first within 500 ms so the rates show without a full interval's wait
        int wait = reads == 0 ? 0 : (reads == 1 ? std::min(iv, 500) : iv);
        if (std::chrono::duration<double, std::milli>(Clock::now() - last).count() >= wait) {
            TakeXrt(*sh, gap);
            gap = false;
            last = Clock::now();
            ++reads;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

struct Frame {
    XrtSample xrt;              // newest xrt-smi sample (seq 0: none yet, or xrt-smi off)
    double xrtAgeS = -1;        // age of that sample at this frame
    PdhSample pdh;
    double clockMhz = -1;       // XRT API readback; -1 without HAVE_XRT
    std::string powerMode;      // XRT API, per poll
    double xrtApiMs = 0;
    std::vector<Ctx> rows;      // xrt contexts + engine-only pids, sorted by engine %
    int activeCtx = 0;
    double totalComplPerS = -1, totalSubsPerS = -1;
    std::string when;
    int seq = 0;
    Clock::time_point at{};
};

static Frame Poll(const StaticInfo& si, PdhReader* pdh, XrtShared& sh, int seq) {
    Frame f;
    f.seq = seq;
    f.at = Clock::now();
    if (sh.intervalMs > 0) { std::lock_guard<std::mutex> lk(sh.m); f.xrt = sh.latest; }
    if (f.xrt.seq > 0) f.xrtAgeS = std::chrono::duration<double>(f.at - f.xrt.takenAt).count();
    if (pdh) f.pdh = pdh->Sample(si);
    f.powerMode = si.powerMode;
    XrtLive(f.clockMhz, f.powerMode, f.xrtApiMs);
    {
        SYSTEMTIME st; GetLocalTime(&st);
        char b[32]; snprintf(b, sizeof b, "%02d:%02d:%02d.%03d", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds); f.when = b;
    }
    f.totalComplPerS = f.xrt.totalComplPerS;
    f.totalSubsPerS = f.xrt.totalSubsPerS;
    for (auto c : f.xrt.ctx) {
        if (Lower(c.status) == "active") f.activeCtx++;
        if (f.pdh.ok) {
            auto p = f.pdh.pidPct.find(c.pid);
            if (p != f.pdh.pidPct.end()) c.enginePct = p->second;
            auto m = f.pdh.procSharedMb.find(c.pid);
            if (m != f.pdh.procSharedMb.end()) c.procSharedMb = m->second;
        }
        f.rows.push_back(c);
    }
    // pids the engine counters see that xrt-smi does not list (a context torn down between the two reads, or xrt-smi off)
    if (f.pdh.ok) {
        for (const auto& [pid, pct] : f.pdh.pidPct) {
            if (pid < 0 || pct < 0.05) continue;
            bool listed = false;
            for (const auto& r : f.rows) if (r.pid == pid) { listed = true; break; }
            if (listed) continue;
            Ctx c; c.pid = pid; c.process = ProcessNameOf(pid); c.status = "(engine only)"; c.enginePct = pct; c.fromXrt = false;
            auto m = f.pdh.procSharedMb.find(pid);
            if (m != f.pdh.procSharedMb.end()) c.procSharedMb = m->second;
            f.rows.push_back(c);
        }
    }
    std::stable_sort(f.rows.begin(), f.rows.end(), [](const Ctx& a, const Ctx& b) { return a.enginePct > b.enginePct; });
    return f;
}

// Returns the sensors written this poll. Keys not in the returned list were deleted, so a
// sensor HWiNFO cannot be given a real value for is absent rather than 0.
static std::vector<Sensor> Publish(const std::wstring& group, const Frame& f, bool hideIdle, int& failed) {
    bool idle = f.activeCtx == 0 && (!f.pdh.ok || f.pdh.utilPct < 0.5);
    bool hide = hideIdle && idle;
    std::vector<Sensor> v;
    std::vector<std::wstring> drop;
    double util = f.pdh.ok ? f.pdh.utilPct : -1;
    if (!hide && util >= 0) v.push_back({ L"Usage0", L"NPU Utilization", L"%", util }); else drop.push_back(L"Usage0");
    if (!hide && f.clockMhz >= 0) v.push_back({ L"Clock0", L"NPU Clock", L"MHz", f.clockMhz }); else drop.push_back(L"Clock0");
    // always while xrt-smi answers: memory and what is allocated -- 0 is a true reading here.
    // With no xrt-smi sample (--smi-interval 0, or a failed read) these are absent rather than 0.
    v.push_back({ L"Other0", L"NPU Memory (adapter, shared)", L"MB", f.pdh.adapterSharedMb < 0 ? 0.0 : f.pdh.adapterSharedMb });
    if (f.xrt.ok) {
        v.push_back({ L"Other1", L"NPU Memory (xrt-smi)", L"MB", f.xrt.totalMemMb < 0 ? 0.0 : f.xrt.totalMemMb });
        v.push_back({ L"Other2", L"NPU Active Contexts", L"", (double)f.activeCtx });
        v.push_back({ L"Other3", L"NPU Columns In Use", L"cols", (double)f.xrt.cols.size() });
    } else { drop.push_back(L"Other1"); drop.push_back(L"Other2"); drop.push_back(L"Other3"); }
    // rates: a real 0 while idle in --idle zero, absent in --idle hide
    if (!hide && f.xrt.ok) {
        v.push_back({ L"Other4", L"NPU Completions", L"/s", f.totalComplPerS < 0 ? 0.0 : f.totalComplPerS });
        v.push_back({ L"Other5", L"NPU Submissions", L"/s", f.totalSubsPerS < 0 ? 0.0 : f.totalSubsPerS });
    } else { drop.push_back(L"Other4"); drop.push_back(L"Other5"); }
    // GOPS/EGOPS only when some context actually reports them (xrt-smi says N/A for IRON/XRT contexts)
    double gops = 0, egops = 0; bool anyG = false, anyE = false;
    for (const auto& r : f.rows) { if (r.gops >= 0) { gops += r.gops; anyG = true; } if (r.egops >= 0) { egops += r.egops; anyE = true; } }
    if (!hide && anyG) v.push_back({ L"Other6", L"NPU GOPS (xrt-smi)", L"GOPS", gops }); else drop.push_back(L"Other6");
    if (!hide && anyE) v.push_back({ L"Other7", L"NPU EGOPS (xrt-smi)", L"GOPS", egops }); else drop.push_back(L"Other7");
    failed = 0;
    for (const auto& x : v) if (!WriteSensor(group, x)) failed++;
    for (const auto& k : drop) DeleteSensor(group, k);
    return v;
}

// While the dashboard is paused nothing is sampled, so the activity sensors are removed rather
// than left frozen at their last value, which HWiNFO would keep averaging.
static void Unpublish(const std::wstring& group) {
    for (const wchar_t* k : { L"Usage0", L"Clock0", L"Other4", L"Other5", L"Other6", L"Other7" }) DeleteSensor(group, k);
}

// ---------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------

struct Term {
    bool vt = false;      // ANSI escapes usable
    bool unicode = true;  // block/box glyphs
    int width = 100;
};

static std::string Colr(const Term& t, const char* code) { return t.vt ? std::string("\x1b[") + code + "m" : ""; }
static std::string Reset(const Term& t) { return t.vt ? "\x1b[0m" : ""; }

static std::string Bar(const Term& t, double pct, int width) {
    if (pct < 0) return std::string(width, t.unicode ? ' ' : ' ');
    int filled = (int)std::lround(std::max(0.0, std::min(100.0, pct)) / 100.0 * width);
    std::string s;
    const char* on = t.unicode ? "\xE2\x96\x88" : "#";   // █
    const char* off = t.unicode ? "\xE2\x96\x91" : ".";  // ░
    for (int i = 0; i < width; ++i) s += (i < filled) ? on : off;
    return s;
}

static std::string Rule(const Term& t, int width) {
    std::string s;
    const char* g = t.unicode ? "\xE2\x94\x80" : "-";    // ─
    for (int i = 0; i < width; ++i) s += g;
    return s;
}

static std::string Spark(const Term& t, const std::vector<double>& vals) {
    static const char* lv[] = { "\xE2\x96\x81", "\xE2\x96\x82", "\xE2\x96\x83", "\xE2\x96\x84", "\xE2\x96\x85", "\xE2\x96\x86", "\xE2\x96\x87", "\xE2\x96\x88" };
    static const char* la[] = { "_", ".", "-", "=", "+", "*", "%", "#" };
    std::string s;
    for (double v : vals) {
        if (v < 0) { s += " "; continue; }
        int i = (int)std::min(7.0, std::floor(std::max(0.0, std::min(100.0, v)) / 100.0 * 7.999));
        s += t.unicode ? lv[i] : la[i];
    }
    return s;
}

static std::string ColsText(const std::vector<int>& cols) {
    if (cols.empty()) return "none";
    std::string s; int start = cols[0], last = cols[0];
    auto flush = [&]() { if (!s.empty()) s += ","; s += (start == last) ? std::to_string(start) : std::to_string(start) + "-" + std::to_string(last); };
    for (size_t i = 1; i < cols.size(); ++i) { if (cols[i] == last + 1) last = cols[i]; else { flush(); start = last = cols[i]; } }
    flush();
    return s;
}

static std::string Pad(const std::string& s, int w, bool right = false) {
    if ((int)s.size() >= w) return s.substr(0, w);
    return right ? std::string(w - s.size(), ' ') + s : s + std::string(w - s.size(), ' ');
}

struct View { double interval = 0.5; double smiInterval = 2.0; double actualPeriod = -1; bool paused = false; };
struct HistPt { double t; double util; };   // seconds since start; utilization %, -1 = no reading

static std::vector<std::string> RenderFrame(const Term& t, const StaticInfo& si, const Frame& f, const std::deque<HistPt>& hist,
                                            const std::wstring& group, bool hwinfo, int hwCount, int hwFailed, bool hideIdle,
                                            const std::vector<std::string>& notes, const View& v) {
    std::vector<std::string> L;
    std::string B = Colr(t, "1"), C = Colr(t, "36"), D = Colr(t, "2"), G = Colr(t, "32"), Y = Colr(t, "33"), R = Colr(t, "31"), Z = Reset(t);
    const std::string dot = t.unicode ? " \xC2\xB7 " : " | ";
    int W = std::max(60, std::min(t.width - 1, 160));
    bool smiOn = v.smiInterval > 0;

    // header
    {
        std::ostringstream o;
        o << B << C << "AMD NPU monitor" << Z << dot << B << si.name << Z;
        if (!si.bdf.empty()) o << dot << si.bdf;
        if (si.totalCols > 0) o << dot << si.totalCols << " columns";
        if (!f.powerMode.empty()) o << dot << "power mode " << f.powerMode;
        o << dot << D << f.when << "  poll #" << f.seq << Z;
        if (v.paused) o << dot << Y << B << "PAUSED" << Z << Y << " (p resumes)" << Z;
        L.push_back(o.str());
    }
    {
        std::ostringstream o;
        o << D;
        if (!si.driverVersion.empty()) o << "NPU driver " << si.driverVersion << dot;
        if (!si.firmware.empty()) o << "firmware " << si.firmware << dot;
        if (!si.xrtVersion.empty()) o << "XRT " << si.xrtVersion << " (" << si.staticHow << ")" << dot;
        if (si.haveLuid) o << "adapter \"" << si.adapterDesc << "\" " << si.luidTag;
        else o << "adapter: not identified";
        o << Z;
        L.push_back(o.str());
    }
    L.push_back(D + Rule(t, W) + Z);

    // utilization, and its history over as many polls as fit the width
    {
        double u = f.pdh.ok ? f.pdh.utilPct : -1;
        std::string col = (u < 0) ? D : (u < 50 ? G : (u < 85 ? Y : R));
        std::ostringstream o;
        o << B << "utilization  " << Z << col << Bar(t, u, 30) << Z << " " << B << Pad(u < 0 ? "  n/a" : FmtNum(u, 1) + " %", 8, true) << Z
          << "  " << D << (f.pdh.ok ? "Windows GPU-engine statistics, NPU adapter" : "unavailable: " + f.pdh.err) << Z;
        L.push_back(o.str());
        int sparkW = std::max(20, std::min(120, W - 13 - 48));
        size_t n = std::min(hist.size(), (size_t)sparkW);
        std::vector<double> vals; double sum = 0, mx = -1; int cnt = 0;
        for (size_t i = hist.size() - n; i < hist.size(); ++i) {
            double x = hist[i].util; vals.push_back(x);
            if (x >= 0) { sum += x; ++cnt; mx = std::max(mx, x); }
        }
        double span = n >= 2 ? hist.back().t - hist[hist.size() - n].t : 0.0;
        std::ostringstream h;
        h << D << "history      " << Z << col << Spark(t, vals) << Z << D << "  last " << FmtNum(span, 1) << " s";
        if (cnt > 0) h << dot << "avg " << FmtNum(sum / cnt, 1) << " %" << dot << "max " << FmtNum(mx, 1) << " %";
        h << Z;
        L.push_back(h.str());
    }
    // clock
    {
        std::ostringstream o;
        o << B << "clock        " << Z;
        if (f.clockMhz >= 0) {
            std::string col = f.clockMhz >= 1500 ? G : (f.clockMhz > 850 ? Y : D);
            o << col << B << FmtNum(f.clockMhz, 0) << " MHz" << Z << "  " << D << "XRT max_clock_frequency_mhz, a live readback here: 800 idle, 1800 with an active context" << Z;
        } else {
            o << D << "n/a  (built without the XRT SDK; see HAVE_XRT in scripts/build_hwinfo_bridge.bat)" << Z;
        }
        L.push_back(o.str());
    }
    // memory
    {
        std::ostringstream o;
        o << B << "memory       " << Z;
        if (f.pdh.ok) o << FmtNum(f.pdh.adapterSharedMb, 1) << " MB shared" << dot << FmtNum(f.pdh.adapterDedicatedMb, 1) << " MB dedicated  " << D << "(adapter, Windows)" << Z;
        else o << D << "adapter: n/a" << Z;
        o << "     " << (f.xrt.ok ? (f.xrt.totalMemMb < 0 ? std::string("0 MB") : FmtNum(f.xrt.totalMemMb, 0) + " MB") : std::string("n/a")) << " in use  " << D << "(xrt-smi)" << Z;
        L.push_back(o.str());
    }
    // contexts summary
    {
        std::ostringstream o;
        o << B << "contexts     " << Z;
        if (!smiOn) {
            o << D << "xrt-smi off (--smi-interval 0, or the s key): contexts, columns and counters are not sampled" << Z;
        } else if (f.xrt.seq == 0) {
            o << D << "waiting for the first xrt-smi read" << Z;
        } else if (f.xrt.ok) {
            o << f.activeCtx << " active / " << f.xrt.ctx.size() << " listed" << dot
              << "columns in use " << f.xrt.cols.size() << (si.totalCols > 0 ? "/" + std::to_string(si.totalCols) : "") << " [" << ColsText(f.xrt.cols) << "]" << dot
              << "completions " << FmtNum(f.totalComplPerS, 1) << "/s" << dot << "submissions " << FmtNum(f.totalSubsPerS, 1) << "/s"
              << dot << D << "xrt-smi read " << FmtNum(f.xrtAgeS, 1) << " s ago" << Z;
        } else {
            o << R << "xrt-smi failed: " << f.xrt.err << Z;
        }
        L.push_back(o.str());
    }
    L.push_back(D + Rule(t, W) + Z);

    // table
    {
        std::ostringstream h;
        h << B << Pad("PID", 7, true) << " " << Pad("process", 18) << " " << Pad("status", 13) << " " << Pad("cols", 6) << " "
          << Pad("compl/s", 9, true) << " " << Pad("subm/s", 9, true) << " " << Pad("migr", 5, true) << " " << Pad("susp", 5, true) << " "
          << Pad("err", 4, true) << " " << Pad("prio", 7) << " " << Pad("GOPS", 7, true) << " " << Pad("EGOPS", 7, true) << " "
          << Pad("mem MB", 7, true) << " " << Pad("engine%", 8, true) << Z;
        L.push_back(h.str());
        if (f.rows.empty()) {
            L.push_back(D + (smiOn ? "  no hardware contexts running on the device (xrt-smi), and no process shows NPU engine time (Windows)"
                                   : "  no process shows NPU engine time (Windows); xrt-smi is off") + Z);
        }
        for (const auto& r : f.rows) {
            std::ostringstream o;
            std::string cols = r.fromXrt ? (r.numCols > 0 ? std::to_string(r.startCol) + "-" + std::to_string(r.startCol + r.numCols - 1) : "-") : "";
            std::string statusCol = Lower(r.status) == "active" ? G : (r.fromXrt ? Y : D);
            o << Pad(std::to_string(r.pid), 7, true) << " " << Pad(r.process.empty() ? "?" : r.process, 18) << " " << statusCol << Pad(r.status, 13) << Z << " " << Pad(cols, 6) << " "
              << Pad(FmtNum(r.complPerS, 1, r.fromXrt ? "-" : ""), 9, true) << " " << Pad(FmtNum(r.subsPerS, 1, r.fromXrt ? "-" : ""), 9, true) << " "
              << Pad(FmtNum(r.migr, 0, ""), 5, true) << " " << Pad(FmtNum(r.susp, 0, ""), 5, true) << " " << Pad(FmtNum(r.err, 0, ""), 4, true) << " "
              << Pad(r.priority, 7) << " " << Pad(FmtNum(r.gops, 0, r.fromXrt ? "n/a" : ""), 7, true) << " " << Pad(FmtNum(r.egops, 0, r.fromXrt ? "n/a" : ""), 7, true) << " "
              << Pad(FmtNum(r.memMb >= 0 ? r.memMb : r.procSharedMb, 0, ""), 7, true) << " " << Pad(FmtNum(r.enginePct, 1), 8, true);
            L.push_back(o.str());
        }
    }
    L.push_back(D + Rule(t, W) + Z);

    // footer
    {
        L.push_back(D + "not exposed on this NPU: voltage, power (xrt-smi: Estimated Power N/A; the electrical query fails at the driver escape)" + Z);
        std::ostringstream p;
        p << D;
        if (hwinfo) {
            p << "HWiNFO: " << (v.paused ? std::string("paused, activity sensors removed") : (hwFailed == 0 ? std::to_string(hwCount) + " sensors" : (std::to_string(hwFailed) + " sensor writes FAILED")))
              << " -> HKCU\\" << WideToUtf8(REG_ROOT) << "\\" << WideToUtf8(group) << (hideIdle ? " (idle: hidden)" : " (idle: zeros)");
        } else {
            p << "HWiNFO publish off (--no-hwinfo)";
        }
        p << Z;
        L.push_back(p.str());
        std::ostringstream q;
        q << D << "poll every " << FmtNum(v.interval, 2) << " s";
        if (v.actualPeriod >= 0) q << " (last " << FmtNum(v.actualPeriod, 2) << " s)";
        q << dot << "pdh " << FmtNum(f.pdh.ms, 1) << " ms";
        if (si.xrtApi) q << dot << "xrt api " << FmtNum(f.xrtApiMs, 2) << " ms";
        if (smiOn) { q << dot << "xrt-smi every " << FmtNum(v.smiInterval, 1) << " s"; if (f.xrt.seq > 0) q << ", " << FmtNum(f.xrt.ms, 0) << " ms a read"; }
        else q << dot << "xrt-smi off";
        q << Z;
        L.push_back(q.str());
        if (t.vt) L.push_back(D + "keys  + / - poll faster / slower" + dot + "[ / ] xrt-smi faster / slower" + dot + "s xrt-smi on / off" + dot + "p pause" + dot + "q quit" + Z);
        for (const auto& n : notes) L.push_back(Y + "note: " + n + Z);
    }
    return L;
}

static json FrameJson(const StaticInfo& si, const Frame& f, const View& v) {
    json j;
    j["time"] = f.when; j["poll"] = f.seq; j["poll_interval_s"] = v.interval; j["period_s"] = v.actualPeriod;
    j["device"] = { {"name", si.name}, {"bdf", si.bdf}, {"total_columns", si.totalCols}, {"power_mode", f.powerMode},
                    {"npu_driver", si.driverVersion}, {"firmware", si.firmware}, {"xrt", si.xrtVersion},
                    {"adapter", si.adapterDesc}, {"luid", si.luidTag} };
    j["xrt_api"] = { {"available", si.xrtApi}, {"clock_mhz", f.clockMhz}, {"power_mode", f.powerMode}, {"ms", f.xrtApiMs} };
    j["windows"] = { {"ok", f.pdh.ok}, {"error", f.pdh.err}, {"utilization_pct", f.pdh.utilPct},
                     {"adapter_shared_mb", f.pdh.adapterSharedMb}, {"adapter_dedicated_mb", f.pdh.adapterDedicatedMb}, {"ms", f.pdh.ms} };
    json eng = json::object();
    for (const auto& [k, v2] : f.pdh.engPct) eng[k] = v2;
    j["windows"]["engines"] = eng;
    // sample/age say how fresh the xrt-smi section is: it is read every interval_s, not per poll
    j["xrt_smi"] = { {"ok", f.xrt.ok}, {"error", f.xrt.err}, {"enabled", v.smiInterval > 0}, {"interval_s", v.smiInterval},
                     {"sample", f.xrt.seq}, {"age_s", f.xrtAgeS},
                     {"memory_mb", f.xrt.totalMemMb}, {"partitions", f.xrt.partitions},
                     {"columns_in_use", f.xrt.cols}, {"active_contexts", f.activeCtx},
                     {"completions_per_s", f.totalComplPerS}, {"submissions_per_s", f.totalSubsPerS}, {"ms", f.xrt.ms} };
    json rows = json::array();
    for (const auto& r : f.rows) {
        rows.push_back({ {"pid", r.pid}, {"process", r.process}, {"status", r.status}, {"listed_by_xrt_smi", r.fromXrt},
                         {"context_id", r.ctxId}, {"partition", r.partIdx}, {"start_col", r.startCol}, {"num_cols", r.numCols},
                         {"submissions", r.subs}, {"completions", r.compl}, {"submissions_per_s", r.subsPerS}, {"completions_per_s", r.complPerS},
                         {"migrations", r.migr}, {"suspensions", r.susp}, {"errors", r.err}, {"priority", r.priority},
                         {"gops", r.gops}, {"egops", r.egops}, {"fps", r.fps}, {"latency", r.latency},
                         {"memory_mb", r.memMb}, {"instr_bo_mb", r.instrMb}, {"engine_pct", r.enginePct}, {"process_shared_mb", r.procSharedMb} });
    }
    j["contexts"] = rows;
    return j;
}

// ---------------------------------------------------------------------------------------

static void PrintUsage() {
    printf(
        "AMD XDNA1 NPU monitor + HWiNFO64 custom-sensor bridge\n\n"
        "Usage: hwinfo_npu_bridge.exe [options]\n\n"
        "  --interval <sec>      seconds between polls of utilization, memory and clock\n"
        "                        (default 0.5, min 0.1, max 60)\n"
        "  --smi-interval <sec>  seconds between xrt-smi reads for contexts, columns and counters\n"
        "                        (default 2, min 0.5, max 60; 0 = never run xrt-smi)\n"
        "  --once                one sample, printed plainly, then exit\n"
        "  --json                one JSON object per sample on stdout (implies no redraw)\n"
        "  --plain               one text line per sample, no ANSI, no redraw\n"
        "  --ascii               dashboard without Unicode block/box glyphs\n"
        "  --no-hwinfo           do not write HWiNFO custom-sensor registry keys\n"
        "  --idle zero|hide      while no hardware context is active: publish the true idle readings\n"
        "                        (0 %%, 800 MHz, 0/s; default) or remove those sensors so HWiNFO's\n"
        "                        Average covers active time only; memory/contexts/columns stay either way\n"
        "  --group <name>        HWiNFO sensor group name (default: device name, e.g. \"NPU Phoenix\")\n"
        "  --background          hide the console window (bridge only, no display)\n"
        "  --clean               remove this group's registry keys on exit\n"
        "  --adapter <substr>    pick the NPU adapter by DXCore driver description (default: NPU/IPU/XDNA)\n"
        "  --luid <0x..>         pick the NPU adapter by LUID low part, as in PDH's luid_0x00000000_0x0000d6bf\n"
        "  -h, --help            this text\n\n"
        "Dashboard keys: + / - halve / double the poll interval, [ / ] the xrt-smi interval, s xrt-smi\n"
        "on/off, p pause (activity sensors are removed from HWiNFO while paused), q quit.\n\n"
        "Sources: utilization and adapter memory come from Windows' GPU-engine statistics for the NPU\n"
        "adapter (what Task Manager reads); contexts, columns, counters and GOPS come from xrt-smi\n"
        "examine -r aie-partitions, read on its own thread every --smi-interval; clock and power mode\n"
        "from XRT's in-process query API (the clock is a live readback: 800 MHz idle, 1800 MHz with an\n"
        "active context); firmware from xrt-smi's host report.\n"
        "Not available on this NPU from any documented interface, so never shown: voltage, power.\n"
        "Requires the AMD NPU driver (%s) and, for the bridge, HWiNFO64 with its Sensors window open.\n",
        WideToUtf8(XRT_SMI_PATH).c_str());
}

int main(int argc, char* argv[]) {
    double interval = 0.5, smiInterval = 2.0;
    std::string group, adapterMatch, luidOverride, idleMode = "zero";
    bool once = false, jsonMode = false, plain = false, ascii = false, background = false, noHwinfo = false, cleanOnExit = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](std::string& dst) { if (i + 1 < argc) dst = argv[++i]; else { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } };
        auto seconds = [&](double& dst) { std::string v; next(v); try { dst = std::stod(v); } catch (...) { fprintf(stderr, "bad %s value \"%s\"\n", a.c_str(), v.c_str()); exit(2); } };
        if (a == "--interval") seconds(interval);
        else if (a == "--smi-interval") seconds(smiInterval);
        else if (a == "--group") next(group);
        else if (a == "--adapter") next(adapterMatch);
        else if (a == "--luid") next(luidOverride);
        else if (a == "--idle") { next(idleMode); if (idleMode != "zero" && idleMode != "hide") { fprintf(stderr, "--idle takes zero or hide\n"); return 2; } }
        else if (a == "--once") once = true;
        else if (a == "--json") jsonMode = true;
        else if (a == "--plain") plain = true;
        else if (a == "--ascii") ascii = true;
        else if (a == "--background" || a == "--silent") background = true;
        else if (a == "--no-hwinfo") noHwinfo = true;
        else if (a == "--clean") cleanOnExit = true;
        else if (a == "--help" || a == "-h" || a == "/?") { PrintUsage(); return 0; }
        else { fprintf(stderr, "unknown option %s (try --help)\n", a.c_str()); return 2; }
    }
    interval = std::max(0.1, std::min(60.0, interval));
    smiInterval = smiInterval > 0 ? std::max(0.5, std::min(60.0, smiInterval)) : 0.0;

    if (GetFileAttributesW(XRT_SMI_PATH) == INVALID_FILE_ATTRIBUTES) {
        fprintf(stderr, "error: %s not found -- no AMD XDNA NPU driver on this machine\n", WideToUtf8(XRT_SMI_PATH).c_str());
        return 1;
    }
    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);
    SetConsoleOutputCP(CP_UTF8);
    // Windows' default timer tick is 15.6 ms, which turns the loop's 20 ms naps into ~31 ms and
    // put ~22 ms on every measured poll period (0.122 s for --interval 0.1); ask for 1 ms.
    timeBeginPeriod(1);

    // terminal capabilities
    Term term;
    HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
    bool isConsole = _isatty(_fileno(stdout)) != 0;
    bool dashboard = isConsole && !once && !jsonMode && !plain && !background;
    if (dashboard) {
        DWORD mode = 0;
        if (GetConsoleMode(hout, &mode) && SetConsoleMode(hout, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) term.vt = true;
    }
    term.unicode = !ascii && (isConsole || jsonMode);
    if (plain || once) term.unicode = !ascii;

    std::vector<std::string> notes;
    StaticInfo si;
    QueryStatic(si, notes);
    PickNpuAdapter(si, adapterMatch, luidOverride, notes);

    PdhReader pdh;
    std::string pdhErr;
    bool havePdh = pdh.Init(pdhErr);
    if (!havePdh) notes.push_back("PDH: " + pdhErr);

    std::wstring wgroup = Utf8ToWide(group.empty() ? si.name : group);
    bool hideIdle = idleMode == "hide";
    if (!noHwinfo) RemoveLegacySensors(wgroup, notes);
    if (background) { HWND w = GetConsoleWindow(); if (w) ShowWindow(w, SW_HIDE); }

    if (dashboard) { fputs("\x1b[?25l\x1b[2J\x1b[H", stdout); fflush(stdout); }

    XrtShared sh;
    sh.intervalMs = (int)std::lround(smiInterval * 1000);
    std::thread worker;
    View view; view.interval = interval; view.smiInterval = smiInterval;
    std::deque<HistPt> hist;
    Clock::time_point t0 = Clock::now();
    int seq = 0;
    // Prime: the engine-utilization rate needs a second PDH collection after the baseline, and
    // completions/s needs two xrt-smi reads. --once takes both reads here; otherwise the worker
    // takes them and the first frame waits (up to 3 s) for the first one.
    if (once) {
        if (smiInterval > 0) TakeXrt(sh, false);
        Frame prime = Poll(si, havePdh ? &pdh : nullptr, sh, 0); (void)prime;
        std::this_thread::sleep_for(std::chrono::milliseconds(std::max(500, (int)std::lround(interval * 1000))));
        if (smiInterval > 0) TakeXrt(sh, false);
    } else {
        worker = std::thread(XrtWorker, &sh);
        Frame prime = Poll(si, havePdh ? &pdh : nullptr, sh, 0); (void)prime;
        auto until = Clock::now() + std::chrono::milliseconds(std::max(200, (int)std::lround(interval * 1000)));
        while (g_running && Clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        auto giveUp = Clock::now() + std::chrono::seconds(3);
        while (g_running && smiInterval > 0 && Clock::now() < giveUp) {
            { std::lock_guard<std::mutex> lk(sh.m); if (sh.latest.seq > 0) break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    Frame f;
    Clock::time_point lastPoll{}, nextPoll = Clock::now();
    bool paused = false, wasPaused = false, needRender = false;
    int hwFailed = 0, hwCount = 0;
    double smiSaved = smiInterval > 0 ? smiInterval : 2.0;   // what the s key restores
    while (g_running) {
        bool polled = false;
        if (!paused && Clock::now() >= nextPoll) {
            ++seq;
            auto now = Clock::now();
            view.actualPeriod = seq > 1 ? std::chrono::duration<double>(now - lastPoll).count() : -1;
            lastPoll = now;
            // fixed-rate schedule: the next deadline steps from the previous one, so the mean period
            // is the interval rather than the interval plus the poll's own cost; only a poll more
            // than one interval late resets it
            auto step = std::chrono::milliseconds((int)std::lround(interval * 1000));
            nextPoll = (nextPoll + step > now) ? nextPoll + step : now + step;
            f = Poll(si, havePdh ? &pdh : nullptr, sh, seq);
            hist.push_back({ std::chrono::duration<double>(f.at - t0).count(), f.pdh.ok ? f.pdh.utilPct : -1 });
            while (hist.size() > 600) hist.pop_front();
            if (!noHwinfo) hwCount = (int)Publish(wgroup, f, hideIdle, hwFailed).size();
            polled = true;
        }
        if (paused && !wasPaused && !noHwinfo) Unpublish(wgroup);
        wasPaused = paused;
        view.interval = interval; view.smiInterval = smiInterval; view.paused = paused;

        if (polled || needRender) {
            needRender = false;
            if (jsonMode) {
                json j = FrameJson(si, f, view);
                j["hwinfo"] = { {"published", !noHwinfo}, {"sensors", hwCount}, {"failed_writes", hwFailed}, {"group", WideToUtf8(wgroup)}, {"idle", idleMode} };
                printf("%s\n", j.dump().c_str());
                fflush(stdout);
            } else if (!background) {
                if (dashboard) {
                    CONSOLE_SCREEN_BUFFER_INFO csbi;
                    if (GetConsoleScreenBufferInfo(hout, &csbi)) term.width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
                    auto lines = RenderFrame(term, si, f, hist, wgroup, !noHwinfo, hwCount, hwFailed, hideIdle, notes, view);
                    std::string out = "\x1b[H";
                    for (const auto& l : lines) out += l + "\x1b[K\n";
                    out += "\x1b[J";
                    fputs(out.c_str(), stdout);
                    fflush(stdout);
                } else if (plain) {
                    std::string smiAge = smiInterval <= 0 ? "off" : (f.xrt.seq == 0 ? "n/a" : FmtNum(f.xrtAgeS, 1) + "s");
                    printf("%s util=%s%% clk=%sMHz mem=%sMB(win) %sMB(xrt) ctx=%d/%zu cols=%s compl/s=%s subm/s=%s period=%ss smi-age=%s%s\n",
                           f.when.c_str(), FmtNum(f.pdh.ok ? f.pdh.utilPct : -1, 1).c_str(), FmtNum(f.clockMhz, 0).c_str(), FmtNum(f.pdh.adapterSharedMb, 1).c_str(),
                           f.xrt.ok ? FmtNum(std::max(0.0, f.xrt.totalMemMb), 0).c_str() : "n/a", f.activeCtx, f.xrt.ctx.size(), ColsText(f.xrt.cols).c_str(),
                           FmtNum(f.totalComplPerS, 1).c_str(), FmtNum(f.totalSubsPerS, 1).c_str(), FmtNum(view.actualPeriod, 2).c_str(), smiAge.c_str(),
                           (f.xrt.ok || f.xrt.seq == 0) ? "" : (" xrt-smi:" + f.xrt.err).c_str());
                    fflush(stdout);
                } else {  // --once
                    Term t2 = term; t2.vt = false;
                    for (const auto& l : RenderFrame(t2, si, f, hist, wgroup, !noHwinfo, hwCount, hwFailed, hideIdle, notes, view)) printf("%s\n", l.c_str());
                    fflush(stdout);
                }
            }
        }
        if (once) break;

        // Wait for the next poll in 20 ms steps, watching Ctrl+C and, on a live console, the keys.
        // A key that changes a rate takes effect now, not after the old interval has elapsed.
        while (g_running && !needRender) {
            if (!paused && Clock::now() >= nextPoll) break;
            if (dashboard && _kbhit()) {
                int c = _getch();
                if (c == 'q' || c == 'Q' || c == 27) g_running = false;
                else if (c == '+' || c == '=') { interval = std::max(0.1, interval / 2); needRender = true; }
                else if (c == '-' || c == '_') { interval = std::min(60.0, interval * 2); needRender = true; }
                else if (c == '[' && smiInterval > 0) { smiInterval = std::max(0.5, smiInterval / 2); needRender = true; }
                else if (c == ']' && smiInterval > 0) { smiInterval = std::min(60.0, smiInterval * 2); needRender = true; }
                else if (c == 's' || c == 'S') { if (smiInterval > 0) { smiSaved = smiInterval; smiInterval = 0; } else smiInterval = smiSaved; needRender = true; }
                else if (c == 'p' || c == 'P') { paused = !paused; sh.paused = paused; needRender = true; }
                if (needRender) {
                    sh.intervalMs = (int)std::lround(smiInterval * 1000);
                    nextPoll = std::min(nextPoll, lastPoll + std::chrono::milliseconds((int)std::lround(interval * 1000)));
                }
            }
            // nap up to 20 ms so a key is seen promptly, but never past the next deadline
            auto nap = std::chrono::milliseconds(20);
            if (!paused) {
                auto left = std::chrono::duration_cast<std::chrono::milliseconds>(nextPoll - Clock::now());
                if (left < nap) nap = std::max(std::chrono::milliseconds(1), left);
            }
            std::this_thread::sleep_for(nap);
        }
    }
    g_running = false;
    if (worker.joinable()) worker.join();
    timeEndPeriod(1);

    if (dashboard) { fputs("\x1b[?25h", stdout); fflush(stdout); }
#ifdef HAVE_XRT
    delete g_xrtDev; g_xrtDev = nullptr;
#endif
    if (cleanOnExit) {
        CleanRegistry(wgroup);
        if (!background && !jsonMode) printf("removed HKCU\\%s\\%s\n", WideToUtf8(REG_ROOT).c_str(), WideToUtf8(wgroup).c_str());
    }
    return 0;
}
