import vkdispatch as vd
import vkdispatch.codegen as vc

def shader(*args, local_size=None, workgroups=None, exec_size=None, signature: tuple = None):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def process_function(func):
        shader_object = vd.ShaderObject(f"{func.__module__}.{func.__name__}", local_size=local_size, workgroups=workgroups, exec_size=exec_size)
        
        func_args = (
            shader_object.args_from_inspectable_function(func)
            if signature is None
            else shader_object.args_from_type_annotations(signature)
        )

        old_builder = vc.set_global_builder(shader_object.builder)
        func(*func_args)
        vc.set_global_builder(old_builder)

        wrapper: str = shader_object # type: ignore

        return wrapper
    
    if len(args) == 1 and callable(args[0]):
        return process_function(args[0])
    return process_function