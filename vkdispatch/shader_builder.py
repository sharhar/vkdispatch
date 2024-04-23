import vkdispatch

import numpy as np

class shader_variable:
    def __init__(self, name: str, builder: 'shader_builder') -> None:
        print("Shader variable created: ", name)
        self.name = name
        self.builder = builder
    
    def __repr__(self) -> str:
        return self.name
    
    def __iadd__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} += {other};\n")
        return self
    
    def __getitem__(self, index: 'shader_variable') -> 'shader_variable':
        return self.builder.make_var(f"{self}.data[{index}]")
    
    def __setitem__(self, index: 'shader_variable', value: 'shader_variable') -> None:
        if f"{self}.data[{index}]" == str(value):
            return
        self.builder.append_contents(f"{self}.data[{index}] = {value};\n")

class shader_builder:
    def __init__(self) -> None:
        self.var_count = 0
        self.binding_count = 0
        self.pc_size = 0
        self.bindings: list[tuple[vkdispatch.buffer, int]] = []
        self.global_x = self.make_var("gl_GlobalInvocationID.x")
        self.contents = ""
        self.header = r"""
#version 450
#extension GL_ARB_separate_shader_objects : enable
"""

    def make_var(self, var_name: str = None) -> shader_variable:
        new_var = f"var{self.var_count}" if var_name is None else var_name
        if var_name is None:
            self.var_count += 1
        return shader_variable(new_var, self)

    def static_buffer(self, buff: vkdispatch.buffer, var_name: str = None) -> shader_variable:
        new_var = self.buffer(var_name, buff.dtype)
        self.bindings.append((buff, self.binding_count - 1))
        return new_var

    def buffer(self, var_name: str = None, data_type: type = np.float32) -> shader_variable:
        if not data_type == np.float32 and not data_type == np.complex64:
            raise ValueError("Data type must be float32 or complex64")

        data_type = "float" if data_type == np.float32 else "vec2"
        new_var = self.make_var(var_name)

        self.header += f"""
layout(set = 0, binding = {self.binding_count}) buffer Buffer{self.binding_count} {{
    {data_type} data[];
}} {new_var};"""
        
        self.binding_count += 1

        return new_var

    def append_contents(self, contents: str) -> None:
        self.contents += contents

    def build(self, x: int, y: int, z: int) -> str:
        return self.header + f"\nlayout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;\nvoid main() {'{'}\n" + self.contents + "\n}"