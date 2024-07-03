import typing

@typing.final
class NumberBase:
    glsl_name: str = None

    def __init_subclass__(cls) -> None:
        allowed_names = {
            "NumberBase", "_NumberUnsignedInt",
            "_NumberSignedInt", "_NumberFloat"
        }
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()

class _NumberUnsignedInt(NumberBase):  # type: ignore[misc]
    glsl_name = "uint"

class _NumberSignedInt(NumberBase):  # type: ignore[misc]
    glsl_name = "int"

class _NumberFloat(NumberBase):  # type: ignore[misc]
    glsl_name = "float"

#class _FormatDouble(FormatBase):  # type: ignore[misc]
#    glsl_name = "double"

_Number = typing.TypeVar('_Number', bound=NumberBase)
