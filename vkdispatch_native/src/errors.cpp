#include "../include/internal.hh"

std::mutex __error_mutex = {};
const char* __error_string = NULL;

const char* get_error_string_extern() {
    const char* result = NULL;
    __error_mutex.lock();
    result = __error_string;
    __error_mutex.unlock();
    return result;
}

void set_error(const char* format, ...) {
    __error_mutex.lock();
    
    va_list args;
    va_start(args, format);

    if (__error_string != NULL) {
        free((void*)__error_string);
    }

    #ifdef _WIN32
    int length = _vscprintf(format, args) + 1;
    __error_string = (const char*)malloc(length * sizeof(char));
    vsprintf_s((char*)__error_string, length, format, args);
    #else
    vasprintf((char**)&__error_string, format, args);
    #endif
    va_end(args);

    __error_mutex.unlock();
}