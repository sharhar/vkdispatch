import vkdispatch as vd
import vkdispatch.codegen as vc

import uuid

def shader(*args, local_size=None, workgroups=None, exec_size=None, signature: tuple = None):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def process_function(func):
        my_local_size = (
            local_size
            if local_size is not None
            else [vd.get_context().max_workgroup_size[0], 1, 1]
        )

        vc.builder_obj.reset()

        shader_signature = vd.ShaderSignature()

        if signature is None:
            func(*shader_signature.make_from_inspectable_function(func, vc.builder_obj))
        else:
            func(*shader_signature.make_from_type_annotations(signature, builder=vc.builder_obj))

        shader_description = vc.builder_obj.build(
            my_local_size[0], my_local_size[1], my_local_size[2], f"{func.__module__}.{func.__name__}.{uuid.uuid4()}"
        )

        wrapper: str = vd.ShaderLauncher(
            shader_description,
            shader_signature,
            my_local_size,
            workgroups,
            exec_size
        )  # type: ignore

        return wrapper
    
    if len(args) == 1 and callable(args[0]):
        return process_function(args[0])
    return process_function