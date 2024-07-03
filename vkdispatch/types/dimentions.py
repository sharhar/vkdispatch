import typing

@typing.final
class NDimBase:
    count: int = -1

    def __init_subclass__(cls) -> None:
        allowed_names = {
            "NDimBase", "_3D", "_2D", "_1D", "_0D",
        }
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()

class _3D(NDimBase):  # type: ignore[misc]
    count = 3

class _2D(NDimBase):  # type: ignore[misc]
    count = 2

class _1D(NDimBase):  # type: ignore[misc]
    count = 1

class _0D(NDimBase):  # type: ignore[misc]
    count = 0

_NDims = typing.TypeVar('_NDims', bound=NDimBase)
