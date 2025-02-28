import vkdispatch as vd
import vkdispatch.codegen as vc

import dataclasses

from typing import Callable
from typing import Union
from typing import Optional

@dataclasses.dataclass
class ReductionOperation:
    name: str
    reduction: Callable[[vc.ShaderVariable, vc.ShaderVariable], vc.ShaderVariable]
    identity: Union[int, float, str]
    subgroup_reduction: Optional[Callable[[vc.ShaderVariable], vc.ShaderVariable]] = None

SubgroupAdd = ReductionOperation(
    name="add",
    reduction=lambda x, y: x + y,
    identity=0,
    subgroup_reduction=vc.subgroup_add
)

SubgroupMul = ReductionOperation(
    name="mul",
    reduction=lambda x, y: x * y,
    identity=1,
    subgroup_reduction=vc.subgroup_mul
)

SubgroupMin = ReductionOperation(
    name="min",
    reduction=lambda x, y: vc.min(x, y),
    identity=vc.inf_f32,
    subgroup_reduction=vc.subgroup_min
)

SubgroupMax = ReductionOperation(
    name="max",
    reduction=lambda x, y: vc.max(x, y),
    identity=vc.ninf_f32,
    subgroup_reduction=vc.subgroup_max
)

SubgroupAnd = ReductionOperation(
    name="and",
    reduction=lambda x, y: x & y,
    identity=-1,
    subgroup_reduction=vc.subgroup_and
)

SubgroupOr = ReductionOperation(
    name="or",
    reduction=lambda x, y: x | y,
    identity=0,
    subgroup_reduction=vc.subgroup_or
)

SubgroupXor = ReductionOperation(
    name="xor",
    reduction=lambda x, y: x ^ y,
    identity=0,
    subgroup_reduction=vc.subgroup_xor
)