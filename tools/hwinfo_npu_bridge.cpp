/**
 * HWiNFO64 AMD XDNA1 NPU Custom Sensor Bridge (Native C++ Executable)
 *
 * Reads real XDNA1 NPU telemetry via AMD's signed xrt-smi examine tool
 * and publishes it to HWiNFO64's Custom Sensors registry interface:
 * HKCU\Software\HWiNFO64\Sensors\Custom\<group>\Other0..5
 *
 * Sourced metrics:
 *   - NPU GOPS             (active context compute rate)
 *   - NPU EGOPS            (effective compute rate)
 *   - NPU Completions/s    (command completion rate)
 *   - NPU Columns Active   (active hardware columns)
 *   - NPU Array Utilization (active columns / total columns %)
 *   - NPU Memory           (NPU allocated memory in MB)
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winreg.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <regex>
#include <atomic>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

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

// Convert string to wstring
std::wstring Utf8ToWide(const std::string& str) {
    if (str.empty()) return std::wstring();
    int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    std::wstring wstr(sizeNeeded, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], sizeNeeded);
    return wstr;
}

// Convert wstring to string
std::string WideToUtf8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string str(sizeNeeded, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &str[0], sizeNeeded, NULL, NULL);
    return str;
}

// Get temporary file path
std::wstring GetTempJsonPath(const wchar_t* prefix) {
    wchar_t tempDir[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, tempDir)) {
        wcscpy_s(tempDir, L".");
    }
    wchar_t tempFile[MAX_PATH];
    if (GetTempFileNameW(tempDir, prefix, 0, tempFile) != 0) {
        return std::wstring(tempFile);
    }
    return std::wstring(tempDir) + L"\\" + prefix + L".json";
}

// Execute xrt-smi without flashing console window
bool RunReport(const std::wstring& reportType, const std::wstring& outPath) {
    std::wstring cmd = L"\"" + std::wstring(XRT_SMI_PATH) + L"\" examine -r " +
                       reportType + L" -f JSON -o \"" + outPath + L"\" --force";

    STARTUPINFOW si = { sizeof(si) };
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    PROCESS_INFORMATION pi = { 0 };

    std::vector<wchar_t> cmdBuf(cmd.begin(), cmd.end());
    cmdBuf.push_back(L'\0');

    BOOL success = CreateProcessW(
        NULL, cmdBuf.data(), NULL, NULL, FALSE,
        CREATE_NO_WINDOW, NULL, NULL, &si, &pi
    );

    if (!success) {
        return false;
    }

    WaitForSingleObject(pi.hProcess, 15000);
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return (exitCode == 0);
}

// Safely load JSON from file
bool LoadJson(const std::wstring& filePath, json& outJson) {
    std::ifstream f(filePath);
    if (!f.is_open()) return false;
    try {
        f >> outJson;
        f.close();
        return true;
    } catch (...) {
        f.close();
        return false;
    }
}

// Helper to safely extract integer from json value that might be string or number
int64_t GetInt(const json& j, const std::string& key, int64_t defaultVal = 0) {
    if (!j.contains(key)) return defaultVal;
    const auto& v = j[key];
    if (v.is_number_integer()) return v.get<int64_t>();
    if (v.is_number_float()) return (int64_t)v.get<double>();
    if (v.is_string()) {
        try {
            return std::stoll(v.get<std::string>());
        } catch (...) {
            return defaultVal;
        }
    }
    return defaultVal;
}

// Parse "XX MB" string
int ParseMb(const std::string& s) {
    std::regex re("(\\d+)\\s*MB", std::regex_constants::icase);
    std::smatch match;
    if (std::regex_search(s, match, re) && match.size() > 1) {
        try {
            return std::stoi(match.str(1));
        } catch (...) {
            return 0;
        }
    }
    return 0;
}

// Query platform info: device name and total columns
bool QueryDeviceInfo(std::string& outName, int& outTotalCols) {
    std::wstring tmpPath = GetTempJsonPath(L"plat");
    bool ok = RunReport(L"platform", tmpPath);
    if (!ok) {
        DeleteFileW(tmpPath.c_str());
        return false;
    }

    json data;
    if (!LoadJson(tmpPath, data)) {
        DeleteFileW(tmpPath.c_str());
        return false;
    }
    DeleteFileW(tmpPath.c_str());

    outName = "XDNA NPU";
    outTotalCols = 5;

    try {
        if (data.contains("devices") && data["devices"].is_array() && !data["devices"].empty()) {
            const auto& dev = data["devices"][0];
            if (dev.contains("platforms") && dev["platforms"].is_array() && !dev["platforms"].empty()) {
                const auto& plat = dev["platforms"][0];
                if (plat.contains("static_region") && plat["static_region"].is_object()) {
                    const auto& sr = plat["static_region"];
                    if (sr.contains("name") && sr["name"].is_string()) {
                        std::string n = sr["name"].get<std::string>();
                        // Trim whitespace
                        size_t first = n.find_first_not_of(" \t\r\n");
                        size_t last = n.find_last_not_of(" \t\r\n");
                        if (first != std::string::npos && last != std::string::npos) {
                            outName = n.substr(first, (last - first + 1));
                        }
                    }
                    outTotalCols = (int)GetInt(sr, "total_columns", 5);
                }
            }
        }
    } catch (...) {
    }

    return true;
}

struct AieSample {
    int64_t gops = 0;
    int64_t egops = 0;
    int64_t completions = 0;
    int cols_active = 0;
    int mem_mb = 0;
};

// Sample AIE partitions
AieSample SampleAie() {
    AieSample sample;
    std::wstring tmpPath = GetTempJsonPath(L"aie");
    if (!RunReport(L"aie-partitions", tmpPath)) {
        DeleteFileW(tmpPath.c_str());
        return sample;
    }

    json data;
    if (!LoadJson(tmpPath, data)) {
        DeleteFileW(tmpPath.c_str());
        return sample;
    }
    DeleteFileW(tmpPath.c_str());

    try {
        if (data.contains("devices") && data["devices"].is_array() && !data["devices"].empty()) {
            const auto& devObj = data["devices"][0];
            if (devObj.contains("aie_partitions") && devObj["aie_partitions"].is_object()) {
                const auto& aie = devObj["aie_partitions"];
                if (aie.contains("total_memory_usage") && aie["total_memory_usage"].is_string()) {
                    sample.mem_mb = ParseMb(aie["total_memory_usage"].get<std::string>());
                }

                if (aie.contains("partitions") && aie["partitions"].is_array()) {
                    for (const auto& part : aie["partitions"]) {
                        sample.cols_active += (int)GetInt(part, "num_cols", 0);
                        if (part.contains("hw_contexts") && part["hw_contexts"].is_array()) {
                            for (const auto& ctx : part["hw_contexts"]) {
                                std::string status = ctx.value("status", "");
                                if (status == "Active") {
                                    sample.gops += GetInt(ctx, "gops", 0);
                                    sample.egops += GetInt(ctx, "egops", 0);
                                    sample.completions += GetInt(ctx, "command_completions", 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (...) {
    }

    return sample;
}

// Write a single sensor entry to HWiNFO Custom Sensors registry
void WriteSensor(const std::wstring& group, int index, const std::wstring& name,
                 const std::wstring& unit, double value) {
    std::wstring subKey = std::wstring(REG_ROOT) + L"\\" + group + L"\\Other" + std::to_wstring(index);
    HKEY hKey = NULL;
    LSTATUS status = RegCreateKeyExW(
        HKEY_CURRENT_USER, subKey.c_str(), 0, NULL,
        REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, NULL, &hKey, NULL
    );

    if (status == ERROR_SUCCESS) {
        wchar_t valBuf[64];
        swprintf_s(valBuf, L"%.2f", value);

        RegSetValueExW(hKey, L"Name", 0, REG_SZ, (const BYTE*)name.c_str(), (DWORD)((name.length() + 1) * sizeof(wchar_t)));
        RegSetValueExW(hKey, L"Unit", 0, REG_SZ, (const BYTE*)unit.c_str(), (DWORD)((unit.length() + 1) * sizeof(wchar_t)));
        RegSetValueExW(hKey, L"Value", 0, REG_SZ, (const BYTE*)valBuf, (DWORD)((wcslen(valBuf) + 1) * sizeof(wchar_t)));

        RegCloseKey(hKey);
    }
}

// Clean up registry keys on exit
void CleanRegistry(const std::wstring& group) {
    for (int i = 0; i < 6; ++i) {
        std::wstring subKey = std::wstring(REG_ROOT) + L"\\" + group + L"\\Other" + std::to_wstring(i);
        RegDeleteKeyW(HKEY_CURRENT_USER, subKey.c_str());
    }
    std::wstring groupKey = std::wstring(REG_ROOT) + L"\\" + group;
    RegDeleteKeyW(HKEY_CURRENT_USER, groupKey.c_str());
}

struct SensorRow {
    std::wstring name;
    std::wstring unit;
    double value;
};

std::vector<SensorRow> PublishMetrics(const std::wstring& group, int totalCols,
                                      const AieSample& sample, double completionsPerSec) {
    double utilPct = (totalCols > 0) ? (100.0 * sample.cols_active / totalCols) : 0.0;

    std::vector<SensorRow> rows = {
        { L"NPU GOPS", L"GOPS", (double)sample.gops },
        { L"NPU EGOPS", L"GOPS", (double)sample.egops },
        { L"NPU Completions", L"/s", completionsPerSec },
        { L"NPU Columns Active", L"cols", (double)sample.cols_active },
        { L"NPU Array Utilization", L"%", utilPct },
        { L"NPU Memory", L"MB", (double)sample.mem_mb }
    };

    for (int i = 0; i < (int)rows.size(); ++i) {
        WriteSensor(group, i, rows[i].name, rows[i].unit, rows[i].value);
    }

    return rows;
}

void PrintUsage(const char* progName) {
    std::cout << "HWiNFO64 AMD XDNA1 NPU Custom Sensor Bridge (Native Windows EXE)\n\n"
              << "Usage: " << progName << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --interval <sec>   Seconds between polls (default: 2.0)\n"
              << "  --group <name>     Override the HWiNFO sensor group name (default: device name)\n"
              << "  --once             Sample once, print to stdout, and exit\n"
              << "  --background       Run silently in background (hides console window)\n"
              << "  --clean            Remove custom sensor registry keys on exit\n"
              << "  --help, -h         Show this help message\n\n"
              << "Requirements:\n"
              << "  HWiNFO64 running with Sensors window open (Custom Sensors enabled by default).\n"
              << "  AMD NPU driver installed with " << WideToUtf8(XRT_SMI_PATH) << ".\n";
}

int main(int argc, char* argv[]) {
    double interval = 2.0;
    std::string groupOverride = "";
    bool once = false;
    bool background = false;
    bool cleanOnExit = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--interval" && i + 1 < argc) {
            interval = std::stod(argv[++i]);
        } else if (arg == "--group" && i + 1 < argc) {
            groupOverride = argv[++i];
        } else if (arg == "--once") {
            once = true;
        } else if (arg == "--background" || arg == "--silent") {
            background = true;
        } else if (arg == "--clean") {
            cleanOnExit = true;
        } else if (arg == "--help" || arg == "-h" || arg == "/?") {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    if (GetFileAttributesW(XRT_SMI_PATH) == INVALID_FILE_ATTRIBUTES) {
        std::cerr << "Error: xrt-smi not found at " << WideToUtf8(XRT_SMI_PATH)
                  << " -- this machine has no AMD XDNA1 driver installed.\n";
        return 1;
    }

    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);

    std::string devName;
    int totalCols = 5;
    if (!QueryDeviceInfo(devName, totalCols)) {
        devName = "NPU Phoenix";
    }

    std::string groupName = groupOverride.empty() ? devName : groupOverride;
    std::wstring wGroup = Utf8ToWide(groupName);

    if (background) {
        HWND hWnd = GetConsoleWindow();
        if (hWnd) {
            ShowWindow(hWnd, SW_HIDE);
        }
    } else {
        std::cout << "device: " << devName << "  total_columns: " << totalCols
                  << "  registry group: " << groupName << "\n";
    }

    int64_t lastCompletions = -1;
    auto lastTime = std::chrono::steady_clock::now();

    while (g_running) {
        AieSample sample = SampleAie();
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - lastTime).count();

        double completionsPerSec = 0.0;
        if (lastCompletions >= 0 && sample.completions >= lastCompletions && dt > 0.0) {
            completionsPerSec = (double)(sample.completions - lastCompletions) / dt;
        }
        lastCompletions = sample.completions;
        lastTime = now;

        auto rows = PublishMetrics(wGroup, totalCols, sample, completionsPerSec);

        if (!background) {
            std::cout << " ";
            for (const auto& r : rows) {
                char buf[64];
                snprintf(buf, sizeof(buf), "  %s=%.1f%s",
                         WideToUtf8(r.name).c_str(), r.value, WideToUtf8(r.unit).c_str());
                std::cout << buf;
            }
            std::cout << "\n" << std::flush;
        }

        if (once) break;

        // Sleep with interrupt check
        int sleepMs = (int)(interval * 1000);
        int elapsed = 0;
        while (g_running && elapsed < sleepMs) {
            int step = (std::min)(100, sleepMs - elapsed);
            std::this_thread::sleep_for(std::chrono::milliseconds(step));
            elapsed += step;
        }
    }

    if (cleanOnExit) {
        CleanRegistry(wGroup);
        if (!background) {
            std::cout << "Cleaned HWiNFO custom sensor registry entries.\n";
        }
    }

    return 0;
}
