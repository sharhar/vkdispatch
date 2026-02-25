import vkdispatch as vd
import vkdispatch.codegen as vc

import dataclasses

from typing import Callable
from typing import Union
from typing import Optional

@dataclasses.dataclass
class ReduceOp:
    name: str
    reduction: Callable[[vc.ShaderVariable, vc.ShaderVariable], vc.ShaderVariable]
    identity: Union[int, float, str]
    subgroup_reduction: Optional[Callable[[vc.ShaderVariable], vc.ShaderVariable]] = None

SubgroupAdd = ReduceOp(
    name="add",
    reduction=lambda x, y: x + y,
    identity=0,
    subgroup_reduction=vc.subgroup_add
)

SubgroupMul = ReduceOp(
    name="mul",
    reduction=lambda x, y: x * y,
    identity=1,
    subgroup_reduction=vc.subgroup_mul
)

SubgroupMin = ReduceOp(
    name="min",
    reduction=lambda x, y: vc.min(x, y),
    identity=float("inf"),
    subgroup_reduction=vc.subgroup_min
)

SubgroupMax = ReduceOp(
    name="max",
    reduction=lambda x, y: vc.max(x, y),
    identity=float("-inf"),
    subgroup_reduction=vc.subgroup_max
)

SubgroupAnd = ReduceOp(
    name="and",
    reduction=lambda x, y: x & y,
    identity=-1,
    subgroup_reduction=vc.subgroup_and
)

SubgroupOr = ReduceOp(
    name="or",
    reduction=lambda x, y: x | y,
    identity=0,
    subgroup_reduction=vc.subgroup_or
)

SubgroupXor = ReduceOp(
    name="xor",
    reduction=lambda x, y: x ^ y,
    identity=0,
    subgroup_reduction=vc.subgroup_xor
)
