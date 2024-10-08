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
@click.option('--context_info', is_flag=True, help="Print info about the context")
@click.option('--max_streams', is_flag=True, help="Make a context with the maximum number of streams.")
@click.option('--use_cpu', is_flag=True, help="Make a context out of a CPU device")
@click.version_option(version=vd.__version__)
def cli_entrypoint(verbose, log_info, vulkan_loader_debug_logs, debug, context_info, max_streams, use_cpu):
    if log_info or debug:
        vd.initialize(log_level=vd.LogLevel.INFO, loader_debug_logs=vulkan_loader_debug_logs or debug)
    else:
        vd.initialize(log_level=vd.LogLevel.WARNING, loader_debug_logs=vulkan_loader_debug_logs or debug)

    for dev in vd.get_devices():
        print(dev.get_info_string(verbose))
    
    if context_info:
        vd.make_context(max_streams=max_streams, use_cpu=use_cpu)

        context = vd.get_context()

        print("Context Info:")

        print(f"\tMax Workgroup Sizes: {context.max_workgroup_size}")
        print(f"\tSubgroup Size: {context.subgroup_size}")
        print(f"\tUniform Buffer Alignment: {context.uniform_buffer_alignment} bytes")

    

