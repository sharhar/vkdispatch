import vkdispatch_native

def check_for_errors():
    """
    Check for errors in the vkdispatch_native library and raise a RuntimeError if found.
    """

    error = vkdispatch_native.get_error_string()

    if error == 0:
        return
    
    if isinstance(error, str):
        raise RuntimeError(error)
    else:
        raise RuntimeError("Unknown error occurred")
    
def check_for_compute_stage_errors():
    """
    Check for errors in the shader compilation stage of the vkdispatch_native library and raise a RuntimeError if found.
    """

    error = vkdispatch_native.get_error_string()

    if error == 0:
        return

    if not isinstance(error, str):
        raise RuntimeError("Unknown error occurred")

    result = ""

    for ii, line in enumerate(error.split("\n")):
        result += f"{ii + 1:4d}: {line}\n"

    print(result)

    raise RuntimeError("Error occurred in compute stage")