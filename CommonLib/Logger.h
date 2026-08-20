#pragma once

#include <iostream>
#include <string>
#include <array>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

class Logger
{
private:
    static constexpr size_t MaxLines = 1024;

    struct Line
    {
        std::string label;
        std::string text;
        bool dirty = false;
    };

    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::array<Line, MaxLines> lines{};
    std::unordered_map<std::string, size_t> labelToLine;

    size_t nextLine = 0;

    bool anchored = false;
    int anchorRow = 0;
    int anchorColumn = 0;

    bool enabled = true;

    float interval = 1.0f;

    std::chrono::steady_clock::time_point lastFlush =
        std::chrono::steady_clock::now();
	std::mutex write_lock;
public:

    static Logger& getInstance()
    {
        static Logger instance;
        return instance;
    }

    void SetInterval(float seconds)
    {
        interval = std::max(0.0f, seconds);
    }

    void setEnabled(bool value)
    {
        enabled = value;
    }

    bool isEnabled() const
    {
        return enabled;
    }

    /*
        Write a labeled line.

        The first argument is always the label.
        Everything after it is concatenated into the line.

        Examples:

            write("FPS", fps);

            write("Position", x, ", ", y, ", ", z);

            write("Memory", used, " / ", total, " MB");
    */
    template <typename... Args>
    void write(const std::string& label, Args&&... args)
    {
		std::lock_guard<std::mutex> lock(write_lock);
        if (!enabled)
            return;

        size_t line = getLine(label);

        lines[line].text = format(std::forward<Args>(args)...);
        lines[line].dirty = true;
    }

    /*
        Clear a labeled line.
    */
    void clear(const std::string& label)
    {
        if (!enabled)
            return;

        auto it = labelToLine.find(label);

        if (it == labelToLine.end())
            return;

        size_t line = it->second;

        lines[line].text.clear();
        lines[line].dirty = true;
    }

    /*
        Call once per frame/update.

        No console output happens until the interval expires.
    */
    void update()
    {
        if (!enabled)
            return;

        auto now = std::chrono::steady_clock::now();

        float elapsed =
            std::chrono::duration<float>(now - lastFlush).count();

        if (interval <= 0.0f || elapsed >= interval)
            flush();
    }

    /*
        Immediately write all pending changes.
    */
    void flush()
    {
        if (!enabled)
            return;

        if (!anchored)
            anchor();

        for (size_t i = 0; i < nextLine; ++i)
        {
            if (!lines[i].dirty)
                continue;

            moveTo(anchorRow + static_cast<int>(i), anchorColumn);

            // Clear entire terminal line.
            std::cout << "\033[2K";

            std::cout
                << lines[i].label
                << ": "
                << lines[i].text;

            lines[i].dirty = false;
        }

        std::cout.flush();

        lastFlush = std::chrono::steady_clock::now();
    }

    /*
        Completely reset the logger.

        All labels are forgotten and the next label
        starts again at line 0.
    */
    void reset()
    {
        for (auto& line : lines)
        {
            line.label.clear();
            line.text.clear();
            line.dirty = false;
        }

        labelToLine.clear();
        nextLine = 0;

        anchored = false;
    }

private:

    /*
        Find an existing label or allocate a new line.
    */
    size_t getLine(const std::string& label)
    {
        auto it = labelToLine.find(label);

        if (it != labelToLine.end())
            return it->second;

        if (nextLine >= MaxLines)
            throw std::runtime_error(
                "Logger exceeded maximum number of lines."
            );

        size_t line = nextLine++;

        labelToLine.emplace(label, line);

        lines[line].label = label;

        return line;
    }

    /*
        Convert one value to a string.
    */
    template <typename T>
    static std::string toString(T&& value)
    {
        std::ostringstream stream;
        stream << std::forward<T>(value);
        return stream.str();
    }

    /*
        Base case.
    */
    static std::string format()
    {
        return {};
    }

    /*
        Convert an arbitrary number of arguments
        into one string.
    */
    template <typename First, typename... Rest>
    static std::string format(First&& first, Rest&&... rest)
    {
        return toString(std::forward<First>(first)) +
            format(std::forward<Rest>(rest)...);
    }

    void anchor()
    {
#ifdef _WIN32
        HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);

        CONSOLE_SCREEN_BUFFER_INFO info{};

        if (GetConsoleScreenBufferInfo(handle, &info))
        {
            anchorRow = info.dwCursorPosition.Y;
            anchorColumn = info.dwCursorPosition.X;
        }
#else
        /*
            ANSI cursor-position querying could be implemented here.
        */
        anchorRow = 0;
        anchorColumn = 0;
#endif

        anchored = true;
    }

    void moveTo(int row, int column)
    {
        std::cout
            << "\033["
            << (row + 1)
            << ";"
            << (column + 1)
            << "H";
    }
};