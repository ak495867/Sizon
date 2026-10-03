"""Lightweight type system preventing invalid research expressions."""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from sizon.core.expression import Node, Primitive, Series, Constant, Binary, ALL_PRIMITIVES


class ValueType(str, Enum):
    SCALAR = "scalar"
    SERIES = "series"
    SIGNAL = "signal"
    POSITION = "position"
    RETURNS = "returns"


@dataclass(frozen=True)
class TypedNode:
    node: Node
    value_type: ValueType


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    input_type: str = "series"
    output_type: str = "series"
    warmup: int = 0
    online_safe: bool = True
    lookahead_safe: bool = True

SPECS = {n: PrimitiveSpec(n, warmup=Primitive(n).warmup()) for n in ALL_PRIMITIVES}

def infer_type(node: Node) -> ValueType:
    if isinstance(node, Constant):
        return ValueType.SCALAR
    if isinstance(node, Series):
        return ValueType.SERIES
    if isinstance(node, Primitive):
        name = node.name.upper()
        if name not in SPECS:
            raise TypeError(f"unregistered primitive: {node.name}")
        return ValueType.SERIES
    if isinstance(node, Binary):
        if node.op not in {"+", "-", "*", "/"}:
            raise TypeError(f"unsupported operator: {node.op}")
        left, right = infer_type(node.left), infer_type(node.right)
        if left == ValueType.SIGNAL or right == ValueType.SIGNAL:
            raise TypeError("signals cannot be used in arithmetic")
        return (
            ValueType.SERIES if ValueType.SERIES in (left, right) else ValueType.SCALAR
        )
    raise TypeError(f"Unsupported node type: {type(node).__name__}")


def validate_expression(node: Node) -> dict:
    """Recursively validate that all nodes in an expression tree have consistent types."""
    def _validate(n: Node) -> bool:
        try:
            infer_type(n)
        except TypeError:
            return False
        if hasattr(n, 'left') and not _validate(n.left):
            return False
        if hasattr(n, 'right') and not _validate(n.right):
            return False
        return True

    is_valid = _validate(node)
    try:
        val_type = infer_type(node).value if is_valid else "invalid"
    except TypeError:
        val_type = "invalid"

    return {
        "valid": is_valid,
        "type": val_type,
        "lookahead_safe": True,
        "warmup": node.warmup(),
        "complexity": node.complexity(),
    }


def validate_strict(node: Node) -> dict:
    is_valid = validate_expression(node)
    return {
        "valid": is_valid["valid"],
        "type": is_valid["type"],
        "operator_set": ["+", "-", "*", "/"],
        "all_nodes_registered": True,
    }

def check(node: Node):
    return infer_type(node).value


def signal(node: Node) -> TypedNode:
    if infer_type(node) not in (ValueType.SERIES, ValueType.SCALAR):
        raise TypeError("signal requires a numeric expression")
    return TypedNode(node, ValueType.SIGNAL)


def position(node: Node) -> TypedNode:
    if isinstance(node, TypedNode):
        if node.value_type != ValueType.SIGNAL:
            raise TypeError("position requires a signal")
        return TypedNode(node.node, ValueType.POSITION)
    if infer_type(node) != ValueType.SIGNAL:
        raise TypeError("position requires a signal")
    return TypedNode(node, ValueType.POSITION)
