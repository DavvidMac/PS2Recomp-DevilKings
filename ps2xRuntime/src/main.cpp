#include "ps2_runtime.h"
#include "games_database.h"
#include "runtime/ps2_settings.h"
#if defined(PS2X_ENABLE_DEBUG_UI) && !defined(PLATFORM_VITA)
#include "ps2_debug_panel.h"
#include "ps2_settings_menu.h"
#endif

#ifdef _DEBUG
#include "ps2_log.h"
#endif

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <exception>
#include <algorithm>
#include <cstdlib>

#if defined(__ANDROID__)
#include <android/log.h>
#include <android/native_activity.h>
#include <android/window.h>
#include <android_native_app_glue.h>
#include <unistd.h>
extern "C" struct android_app *GetAndroidApp(void); // raylib (rcore_android.c)
#include <thread>
#include <cstdio>
#include <cstring>
#endif

namespace
{
#if defined(__ANDROID__)
    int g_logcatPipeFds[2]{-1, -1};
    std::thread g_logcatThread;

    void stopLogcatRedirect()
    {
        std::fflush(stdout);
        std::fflush(stderr);
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
        if (g_logcatPipeFds[1] >= 0)
        {
            close(g_logcatPipeFds[1]);
            g_logcatPipeFds[1] = -1;
        }
        if (g_logcatThread.joinable())
        {
            g_logcatThread.join();
        }
    }

    // stderr goes to logcat (tag ps2x) and to logFile; stdout, where RUNTIME_LOG writes a line
    // for nearly every DMA and disc request, is dropped unless PS2X_STDOUT=1 sends it along.
    void redirectStdioToLogcat(const std::filesystem::path &logFile)
    {
        if (pipe(g_logcatPipeFds) != 0)
        {
            return;
        }

        setvbuf(stdout, nullptr, _IOLBF, 0);
        setvbuf(stderr, nullptr, _IONBF, 0);
        const char *keepStdout = std::getenv("PS2X_STDOUT");
        if (keepStdout && keepStdout[0] == '1')
            dup2(g_logcatPipeFds[1], STDOUT_FILENO);
        else if (FILE *null = std::fopen("/dev/null", "w"))
            dup2(fileno(null), STDOUT_FILENO);
        dup2(g_logcatPipeFds[1], STDERR_FILENO);

        FILE *log = logFile.empty() ? nullptr : std::fopen(logFile.string().c_str(), "w");
        g_logcatThread = std::thread([log]()
                                     {
                                         FILE *reader = fdopen(g_logcatPipeFds[0], "r");
                                         if (!reader)
                                         {
                                             return;
                                         }
                                         char line[1024];
                                         while (fgets(line, sizeof(line), reader))
                                         {
                                             if (log)
                                             {
                                                 std::fputs(line, log);
                                                 std::fflush(log);
                                             }
                                             size_t len = std::strlen(line);
                                             if (len > 0 && line[len - 1] == '\n')
                                             {
                                                 line[len - 1] = '\0';
                                             }
                                             __android_log_write(ANDROID_LOG_INFO, "ps2x", line);
                                         }
                                         fclose(reader);
                                         if (log)
                                             std::fclose(log);
                                         g_logcatPipeFds[0] = -1;
                                     });
        if (std::atexit(stopLogcatRedirect) != 0)
        {
            close(STDOUT_FILENO);
            close(STDERR_FILENO);
            close(g_logcatPipeFds[1]);
            g_logcatPipeFds[1] = -1;
            if (g_logcatThread.joinable())
            {
                g_logcatThread.join();
            }
        }
    }

    // An app has no command line to set PS2X_* variables on: ps2x.env next to the game's ELF
    // holds NAME=value lines (# starts a comment); a variable already set is left alone.
    void loadEnvironmentFile(const std::filesystem::path &file)
    {
        std::ifstream in(file);
        std::string text;
        while (std::getline(in, text))
        {
            while (!text.empty() && (text.back() == '\r' || text.back() == ' '))
                text.pop_back();
            const size_t equals = text.find('=');
            if (text.empty() || text[0] == '#' || equals == std::string::npos || equals == 0)
                continue;
            setenv(text.substr(0, equals).c_str(), text.substr(equals + 1).c_str(), 0);
        }
    }
#endif

    void setupTerminateLogger() // to help on release build crashs
    {
        std::set_terminate([]()
                           {
                               std::cerr << "[terminate] unhandled exception" << std::endl;
                               const std::exception_ptr ep = std::current_exception();
                               if (ep)
                               {
                                   try
                                   {
                                       std::rethrow_exception(ep);
                                   }
                                   catch (const std::system_error &e)
                                   {
                                       std::cerr << "[terminate] std::system_error code=" << e.code().value()
                                                 << " category=" << e.code().category().name()
                                                 << " message=" << e.what() << std::endl;
                                   }
                                   catch (const std::exception &e)
                                   {
                                       std::cerr << "[terminate] std::exception: " << e.what() << std::endl;
                                   }
                                   catch (...)
                                   {
                                       std::cerr << "[terminate] non-std exception" << std::endl;
                                   }
                               }
                               std::abort(); });
    }

    std::string normalizeGameId(const std::string &folderName)
    {
        std::string result = folderName;

        size_t underscore = result.find('_');
        if (underscore != std::string::npos)
            result[underscore] = '-';

        size_t dot = result.find('.');
        if (dot != std::string::npos)
            result.erase(dot, 1);

        std::ranges::transform(result, result.begin(), [](unsigned char character)
                               { return static_cast<char>(std::toupper(character)); });

        return result;
    }

    std::filesystem::path getExecutablePath(int argc, char *argv[])
    {
        if (argc >= 2 && argv[1] && argv[1][0] != '\0')
        {
            std::cout << "Using argv boot path" << std::endl;
            return std::filesystem::path(argv[1]);
        }
#if defined(PS2X_DEFAULT_BOOT_ELF)
        std::cout << "Using default boot file" << std::endl;
        const std::filesystem::path configuredPath = std::filesystem::path(PS2X_DEFAULT_BOOT_ELF);
#if defined(PLATFORM_VITA)
        return configuredPath;
#endif
        if (configuredPath.is_absolute())
        {
            return configuredPath;
        }
        return (std::filesystem::current_path() / configuredPath).lexically_normal();
#else
        throw std::runtime_error("Unable to determine executable path. Pass the guest ELF as argv[1] or define PS2X_DEFAULT_BOOT_ELF.");
#endif
    }
}

int main(int argc, char *argv[])
{
#if defined(__ANDROID__)
    {
        std::filesystem::path bootElf;
        try
        {
            bootElf = getExecutablePath(argc, argv);
        }
        catch (const std::exception &)
        {
        }
        if (!bootElf.empty())
        {
            // An app starts in "/"; the runtime's default paths (the IOP's cdrom0: among them)
            // come from the current folder, which the Windows launcher sets to the game's.
            std::error_code error;
            std::filesystem::current_path(bootElf.parent_path(), error);
            loadEnvironmentFile(bootElf.parent_path() / "ps2x.env");
        }
        redirectStdioToLogcat(bootElf.empty() ? std::filesystem::path() : bootElf.parent_path() / "ultima-sessao.log");
        // Keep the screen on while the game runs: once it sleeps, Android moves the app to
        // the background, where it only gets the slow cores.
        if (android_app *app = GetAndroidApp())
            ANativeActivity_setWindowFlags(app->activity, AWINDOW_FLAG_KEEP_SCREEN_ON, 0);
    }
#endif
    setupTerminateLogger();

    try
    {
        std::filesystem::path pathObj = getExecutablePath(argc, argv);

        std::string filePathStr = pathObj.string();
        std::string elfName = pathObj.filename().string();
        std::string normalizedId = normalizeGameId(elfName);

        std::string windowTitle = "PS2-Recomp | ";
        const char *gameName = getGameName(normalizedId);

#if !defined(PLATFORM_VITA)
        if (gameName)
        {
            windowTitle += std::string(gameName) + " | " + elfName;
        }
        else
#endif
        {
            windowTitle += elfName;
        }

        // settings.ini next to the ELF, before anything reads the PS2X_* variables it sets.
        ps2x::settings::load(pathObj);

        PS2Runtime runtime;
#if defined(PS2X_ENABLE_DEBUG_UI) && !defined(PLATFORM_VITA)
        // This hook is to prevent leak rlimgui deps to recompiler etc
        struct UiHost
        {
            PS2DebugPanel debugPanel;   // F2
            PS2SettingsMenu settingsMenu; // F1
        } ui;
        runtime.setDebugUiCallbacks(
            [](PS2Runtime &rt, void *userData)
            {
                (void)rt;
                auto *host = static_cast<UiHost *>(userData);
                host->debugPanel.initialize();
                host->settingsMenu.initialize();
            },
            [](PS2Runtime &rt, void *userData)
            {
                auto *host = static_cast<UiHost *>(userData);
                host->debugPanel.draw(rt);
                host->settingsMenu.draw(rt);
            },
            [](PS2Runtime &rt, void *userData)
            {
                (void)rt;
                static_cast<UiHost *>(userData)->debugPanel.shutdown();
            },
            &ui);
#endif
        if (!runtime.initialize(windowTitle.c_str()))
        {
            std::cerr << "Failed to initialize PS2 runtime" << std::endl;
            return 1;
        }

        if (!runtime.loadELF(filePathStr))
        {
            std::cerr << "Failed to load ELF file: " << filePathStr << std::endl;
            return 1;
        }

        // Memory card folder from the settings (relative to the ELF's folder).
        const std::string memcardFolder = ps2x::settings::get("saves", "memcard_folder");
        if (!memcardFolder.empty())
        {
            PS2Runtime::IoPaths paths = PS2Runtime::getIoPaths();
            std::filesystem::path folder(memcardFolder);
            if (folder.is_relative())
                folder = paths.elfDirectory / folder;
            paths.mcRoot = folder;
            PS2Runtime::setIoPaths(paths);
        }
        ps2x::settings::backupMemoryCards(PS2Runtime::getIoPaths().mcRoot,
                                          ps2x::settings::getNumber("saves", "backups"));

        runtime.run();

#ifdef _DEBUG
        ps2_log::print_saved_location();
#endif
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[main] fatal exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[main] fatal exception: unknown" << std::endl;
    }

    std::cout.flush();
    std::cerr.flush();
    std::_Exit(1);
}
