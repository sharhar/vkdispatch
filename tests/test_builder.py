import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

vd.initialize(log_level=vd.LogLevel.WARNING)

def test_builder_basic():
    buff = vd.asbuffer(np.array([1, 2, 3, 4], dtype=np.float32))
    buff2 = vd.asbuffer(np.array([10, 20, 30, 40], dtype=np.float32))

    uniform_buffer = vd.Buffer((vd.get_context().uniform_buffer_alignment, ), vd.float32)

    my_builder = vc.ShaderBuilder()

    var_buff = my_builder.declare_buffer(vc.f32)
    var_buff2 = my_builder.declare_buffer(vc.f32)

    uniform_var = my_builder.declare_constant(vc.f32)

    var_buff[my_builder.global_invocation.x] += var_buff2[my_builder.global_invocation.x] - uniform_var

    shader_description = my_builder.build("my_shader")

    source = shader_description.make_source(4, 1, 1)

    compute_plan = vd.ComputePlan(source, shader_description.binding_type_list, shader_description.pc_size, shader_description.name)

    descriptor_set = vd.DescriptorSet(compute_plan)

    descriptor_set.bind_buffer(uniform_buffer, 0, uniform=True)
    descriptor_set.bind_buffer(buff, var_buff.binding)
    descriptor_set.bind_buffer(buff2, var_buff2.binding)

    uniform_buffer_builder = vd.BufferBuilder(usage=vd.BufferUsage.UNIFORM_BUFFER)
    uniform_buffer_builder.register_struct("my_shader", shader_description.uniform_structure)
    uniform_buffer_builder.prepare(1)
    uniform_buffer_builder[("my_shader", shader_description.exec_count_name)] = [2, 1, 1, 0]
    uniform_buffer_builder[("my_shader", uniform_var.raw_name)] = 5

    uniform_buffer.write(uniform_buffer_builder.tobytes())

    cmd_list = vd.CommandList()

    cmd_list.record_compute_plan(compute_plan, descriptor_set, [1, 1, 1])

    cmd_list.submit(instance_count=1)
    cmd_list.submit(instance_count=1)

    assert np.allclose(buff.read(0), np.array([11, 32, 3, 4], dtype=np.float32))


def test_custom_GLSL_shader():
    buff = vd.asbuffer(np.array([1, 2, 3, 4], dtype=np.float32))
    buff2 = vd.asbuffer(np.array([10, 20, 30, 40], dtype=np.float32))

    uniform_buffer = vd.Buffer((vd.get_context().uniform_buffer_alignment, ), vd.float32)

    source = """
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_debug_printf : enable

layout(set = 0, binding = 0) uniform UniformObjectBuffer {
        uvec4 exec_count;
        float var0; 
} UBO;
layout(set = 0, binding = 1) buffer Buffer1 { float data[]; } buf1;
layout(set = 0, binding = 2) buffer Buffer2 { float data[]; } buf2;

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
void main() {
        if((UBO.exec_count.x <= gl_GlobalInvocationID.x)) {
                return ;
        }
        buf1.data[gl_GlobalInvocationID.x] += (buf2.data[gl_GlobalInvocationID.x] - UBO.var0);

}
"""

    shader_uniform_structure = [
        vc.StructElement("exec_count", vc.uv4, 1),
        vc.StructElement("var0", vc.f32, 1)
    ]

    compute_plan = vd.ComputePlan(source, [3, 1, 1], 0, "my_shader")

    descriptor_set = vd.DescriptorSet(compute_plan)

    descriptor_set.bind_buffer(uniform_buffer, 0, uniform=True)
    descriptor_set.bind_buffer(buff, 1)
    descriptor_set.bind_buffer(buff2, 2)

    uniform_buffer_builder = vd.BufferBuilder(usage=vd.BufferUsage.UNIFORM_BUFFER)
    uniform_buffer_builder.register_struct("my_shader", shader_uniform_structure)
    uniform_buffer_builder.prepare(1)
    uniform_buffer_builder[("my_shader", "exec_count")] = [2, 1, 1, 0]
    uniform_buffer_builder[("my_shader", "var0")] = 5

    uniform_buffer.write(uniform_buffer_builder.tobytes())

    cmd_list = vd.CommandList()

    cmd_list.record_compute_plan(compute_plan, descriptor_set, [1, 1, 1])

    cmd_list.submit(instance_count=1)
    cmd_list.submit(instance_count=1)

    assert np.allclose(buff.read(0), np.array([11, 32, 3, 4], dtype=np.float32))