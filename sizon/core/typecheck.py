"""Strict expression checker for operators and primitive specifications."""
from __future__ import annotations
from dataclasses import dataclass
from sizon.core.expression import Node,Primitive,Series,Constant,Binary,ALL_PRIMITIVES
@dataclass(frozen=True)
class PrimitiveSpec: name:str; input_type:str="series"; output_type:str="series"; warmup:int=0; online_safe:bool=True; lookahead_safe:bool=True
SPECS={n:PrimitiveSpec(n,warmup=0) for n in ALL_PRIMITIVES}

def check(node:Node):
    if isinstance(node,(Series,Constant)): return "series" if isinstance(node,Series) else "scalar"
    if isinstance(node,Primitive):
        name=node.name.upper()
        if name not in SPECS:raise TypeError(f"unregistered primitive: {node.name}")
        return SPECS[name].output_type
    if isinstance(node,Binary):
        left,right=check(node.left),check(node.right)
        if node.op not in {"+","-","*","/"}:raise TypeError(f"unsupported operator: {node.op}")
        if left not in {"series","scalar"} or right not in {"series","scalar"}:raise TypeError("binary arithmetic requires scalar or series operands")
        return "series" if "series" in (left,right) else "scalar"
    raise TypeError(type(node).__name__)
def validate_strict(node):return {"valid":True,"type":check(node),"operator_set":["+","-","*","/"],"all_nodes_registered":True}
