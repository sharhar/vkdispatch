from vkdispatch.types.dimentions import _NDims, _0D
from vkdispatch.types.numbers import _Number, _NumberSignedInt, _NumberUnsignedInt, _NumberFloat
from vkdispatch.types.bits import _NBits, _32Bit, _64Bit
from vkdispatch.types.base import BaseType

class Scalar(BaseType[ _NBits, _Number, _0D]):
    name: str
    glsl_type: str
    item_size: int
    format_str: str

    def __init__(self) -> None:
        super().__init__()

        if self.nbits.count != 32:
            raise ValueError("Scalar must be 32 bits")

        self.name = f"{self.number.glsl_name}{self.nbits.count}"
        self.glsl_type = self.number.glsl_name
        self.item_size = self.nbits.count // 8
        
        if self.glsl_type == "float":
            self.format_str = "%f"
        elif self.glsl_type == "int":
            self.format_str = "%d"
        elif self.glsl_type == "uint":
            self.format_str = "%u"

int32 = Scalar[_32Bit, _NumberSignedInt]
uint32 = Scalar[_32Bit, _NumberUnsignedInt]
float32 = Scalar[_32Bit, _NumberFloat]

class Complex(BaseType[_NBits, _Number, _0D]):
    name: str
    glsl_type: str
    item_size: int
    format_str: str

    def __init__(self) -> None:
        super().__init__()

        if self.bits.count != 64:
            raise ValueError("Complex must be 64 bits")

        self.item_size = self.bits.count // 8
        
        if self.number.glsl_name == "float":
            self.format_str = "(%f, %f)"
            self.name = f"complex{self.bits.count}"
            self.glsl_type = "vec2"
        elif self.number.glsl_name == "int":
            self.format_str = "(%d, %d)"
            self.name = f"icomplex{self.bits.count}"
            self.glsl_type = "ivec2"
        elif self.number.glsl_name == "uint":
            self.format_str = "(%u, %u)"
            self.name = f"ucomplex{self.bits.count}"
            self.glsl_type = "uvec2"

complex64 = Complex[_64Bit, _NumberFloat]
icomplex64 = Complex[_64Bit, _NumberSignedInt]
ucomplex64 = Complex[_64Bit, _NumberUnsignedInt]

vv = complex64()

print(vv.dims)

        