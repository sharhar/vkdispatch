import vkdispatch as vd
import vkdispatch_native

def check_for_errors():
    error = vkdispatch_native.get_error_string()

    if error == 0:
        return
    
    if isinstance(error, str):
        raise RuntimeError(error)
    else:
        raise RuntimeError("Unknown error occurred")