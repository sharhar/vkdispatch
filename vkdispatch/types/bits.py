import typing

@typing.final
class NBitBase:
    count: int = 0

    def __init_subclass__(cls) -> None:
        #allowed_names = {
        #    "NBitBase", "_256Bit", "_128Bit", "_96Bit", "_80Bit",
        #    "_64Bit", "_32Bit", "_16Bit", "_8Bit",
        #}
        allowed_names = {
            "NBitBase", "_512Bit", "_256Bit", 
            "_128Bit", "_64Bit", "_32Bit",
        }
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()

class _512Bit(NBitBase):  # type: ignore[misc]
    count = 512

class _256Bit(NBitBase):  # type: ignore[misc]
    count = 256

class _128Bit(NBitBase):  # type: ignore[misc]
    count = 128

class _64Bit(NBitBase):  # type: ignore[misc]
    count = 64

class _32Bit(NBitBase):  # type: ignore[misc]
    count = 32

_NBits = typing.TypeVar('_NBits', bound=NBitBase)
