import vkdispatch as vd
import vkdispatch.codegen as vc

def shader(*args, local_size=None, workgroups=None, exec_size=None, annotations: tuple = None):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def process_function(func):
        shader_name = f"{func.__module__}.{func.__name__}"

        builder = vc.ShaderBuilder()
        signature = vd.ShaderSignature()

        old_builder = vc.set_global_builder(builder)
        func(*signature.make_for_decorator(builder, func, annotations))
        vc.set_global_builder(old_builder)

        shader_object = vd.ShaderObject(shader_name, builder.build(shader_name), signature)

        build_func = lambda: shader_object.build(local_size=local_size, workgroups=workgroups, exec_size=exec_size)

        if not vd.is_context_initialized():
            vd.stage_for_postinit(build_func)
        else:
            build_func()

        #shader_object.build(local_size=local_size, workgroups=workgroups, exec_size=exec_size)

        wrapper: str = shader_object # type: ignore

        return wrapper
    
    if len(args) == 1 and callable(args[0]):
        return process_function(args[0])
    return process_function