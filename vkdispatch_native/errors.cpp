#include "internal.h"

std::mutex __error_mutex = {};
const char* __error_string = NULL;

const char* get_error_string_extern() {
    return __error_string;
}