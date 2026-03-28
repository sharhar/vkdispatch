import vkdispatch as vd

try:
    import click
except ImportError:
    raise ImportError("vkdispatch.cli requires the 'click' package to be installed. Please install it using 'pip install click' or 'pip install vkdispatch[cli]'.")

@click.command()
@click.option('--verbose', is_flag=True, help="Will print verbose messages.")
@click.option('--log_info', is_flag=True, help="Will print verbose messages.")
@click.option('--vulkan_loader_debug_logs', '--loader_debug', is_flag=True, help="Enable debug logs for the vulkan loader.")
@click.option('--debug', is_flag=True, help="Enable debug logs for the vulkan loader.")
@click.option('--vulkan', is_flag=True, help="Use the Vulkan backend (this is the default backend).")
@click.option('--cuda', is_flag=True, help="Use the CUDA backend.")
@click.option('--opencl', is_flag=True, help="Use the OpenCL backend.")
@click.version_option(version=vd.__version__)
def cli_entrypoint(verbose, log_info, vulkan_loader_debug_logs, debug, vulkan, cuda, opencl):
    selected_backend = None

    if vulkan:
        assert not cuda and not opencl, "Multiple backends selected. Please select only one backend."
        selected_backend = "vulkan"
    
    if cuda:
        assert not vulkan and not opencl, "Multiple backends selected. Please select only one backend."
        selected_backend = "cuda"
    
    if opencl:
        assert not vulkan and not cuda, "Multiple backends selected. Please select only one backend."
        selected_backend = "opencl"

    log_level = vd.LogLevel.INFO if log_info or debug else vd.LogLevel.WARNING
    loader_debug_logs = vulkan_loader_debug_logs or debug

    vd.initialize(log_level=log_level, loader_debug_logs=loader_debug_logs, backend=selected_backend)

    for dev in vd.get_devices():
        print(dev.get_info_string(verbose))


    

