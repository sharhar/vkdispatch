#ifndef LOG_HH_SRC_LOG_HH
#define LOG_HH_SRC_LOG_HH

#include "base.hh"

#include <mutex>

#include <stdarg.h>

extern std::mutex __log_mutex;
extern LogLevel __log_level_limit;

extern const char* prefixes[];

inline void log_message(LogLevel log_level, const char* postfix, const char* file_str, int line_str, const char* format, ...) {
    if(log_level < __log_level_limit) {
        return;
    }

    __log_mutex.lock();

    va_list args;
    va_start(args, format);

    const char* level_str = NULL;

    if(log_level < LOG_LEVEL_VERBOSE || log_level > LOG_LEVEL_ERROR) {
        level_str = "INVALID";
    } else {
        level_str = prefixes[log_level];
    }

    if(file_str != NULL) {
        printf("[%s %s:%d] ", level_str, file_str, line_str);
    } else {
        printf("[%s] ", level_str);
    }

    vprintf(format, args);
    printf("%s", postfix);

    va_end(args);

    __log_mutex.unlock();
}

//#define LOG_VERBOSE_ENABLED

#ifdef LOG_VERBOSE_ENABLED
#define LOG_VERBOSE(format, ...) log_message(LOG_LEVEL_VERBOSE, "\n", __FILE__, __LINE__, format, ##__VA_ARGS__)
#else
#define LOG_VERBOSE(format, ...)
#endif

#define LOG_INFO(format, ...) log_message(LOG_LEVEL_INFO, "\n", __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARNING(format, ...) log_message(LOG_LEVEL_WARNING, "\n", __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_message(LOG_LEVEL_ERROR, "\n", __FILE__, __LINE__, format, ##__VA_ARGS__)



#endif