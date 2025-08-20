#ifndef LOG_HH_SRC_LOG_HH
#define LOG_HH_SRC_LOG_HH

#include <mutex>
#include <stdarg.h>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <cstdint>
#include <cstdio>

enum LogLevel {
    LOG_LEVEL_VERBOSE = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3
};

extern std::mutex __log_mutex;
extern std::mutex __log_level_mutex;
extern LogLevel __log_level_limit;

extern const char* prefixes[];

// -------- NEW: thread-id registry (decls only) --------
extern std::mutex __tid_mutex;
extern std::atomic<uint32_t> __next_tid;
//extern thread_local uint32_t __thread_tid; // UINT32_MAX means "unassigned"
extern std::unordered_map<std::thread::id, uint32_t> __tid_map;

// Assign an ID to the calling thread on first use; then return it fast.
inline uint32_t get_thread_tid() {
    return 0;

    // if(__thread_tid == 0) 
    //     return 0; // Main thread always gets ID 0.

    // return 12;
    
    // constexpr uint32_t kUnassigned = UINT32_MAX;
    // if (__thread_tid != kUnassigned) return __thread_tid;

    // std::lock_guard<std::mutex> lock(__tid_mutex);
    // auto tid = std::this_thread::get_id();
    // auto it = __tid_map.find(tid);
    // if (it != __tid_map.end()) {
    //     __thread_tid = it->second;
    //     return __thread_tid;
    // }
    // uint32_t id = __next_tid.fetch_add(1, std::memory_order_relaxed);
    // __tid_map.emplace(tid, id);
    // __thread_tid = id;
    // return id;
}
    
// Call this once early in main() *before* starting any threads to guarantee
// main thread gets ID 0.
inline void init_logging_main_thread() {
    // std::lock_guard<std::mutex> lock(__tid_mutex);
    // if (__tid_map.empty()) {
    //     __tid_map.emplace(std::this_thread::get_id(), 0);
    //     __thread_tid = 0;
    //     __next_tid.store(1, std::memory_order_relaxed);
    // }
}
#define LOG_INIT_MAIN_THREAD() init_logging_main_thread()

inline void log_message(LogLevel log_level, const char* postfix, const char* file_str, int line_str, const char* format, ...) {
    //__log_level_mutex.lock();

    if(log_level < __log_level_limit) {
        return;
    }

    //__log_level_mutex.unlock();

    //std::lock_guard<std::mutex> lock(__log_mutex);

    __log_mutex.lock();

    va_list args;
    va_start(args, format);

    const char* level_str = NULL;

    if(log_level < LOG_LEVEL_VERBOSE || log_level > LOG_LEVEL_ERROR) {
        level_str = "INVALID";
    } else {
        level_str = prefixes[log_level];
    }

    const uint32_t tid = get_thread_tid();

    if(file_str != NULL) {
        printf("[%s T%u %s:%d] ", level_str, tid, file_str, line_str);
    } else {
        printf("[%s] ", level_str);
    }

    vprintf(format, args);
    printf("%s", postfix);

    va_end(args);

    __log_mutex.unlock();
}

#define LOG_VERBOSE_ENABLED

#ifdef LOG_VERBOSE_ENABLED
#define LOG_VERBOSE(format, ...) log_message(LOG_LEVEL_VERBOSE, "\n", __FILE__, __LINE__, format, ##__VA_ARGS__)
#else
#define LOG_VERBOSE(format, ...)
#endif

#define LOG_INFO(format, ...) log_message(LOG_LEVEL_INFO, "\n", __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARNING(format, ...) log_message(LOG_LEVEL_WARNING, "\n", __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_message(LOG_LEVEL_ERROR, "\n", __FILE__, __LINE__, format, ##__VA_ARGS__)



#endif