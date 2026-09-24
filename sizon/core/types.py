"""Lightweight type system preventing invalid research expressions."""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from sizon.core.expression import Node, Primitive, Series, Constant, Binary


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


def infer_type(node: Node) -> ValueType:
    if isinstance(node, Constant):
        return ValueType.SCALAR
    if isinstance(node, Series):
        return ValueType.SERIES
    if isinstance(node, Primitive):
        return ValueType.SERIES
    if isinstance(node, Binary):
        left, right = infer_type(node.left), infer_type(node.right)
        if left == ValueType.SIGNAL or right == ValueType.SIGNAL:
            raise TypeError("signals cannot be used in arithmetic")
        return (
            ValueType.SERIES if ValueType.SERIES in (left, right) else ValueType.SCALAR
        )
    raise TypeError(f"Unsupported node type: {type(node).__name__}")


def validate_expression(node: Node) -> dict:
    value_type = infer_type(node)
    return {
        "valid": True,
        "type": value_type.value,
        "lookahead_safe": True,
        "warmup": node.warmup(),
        "complexity": node.complexity(),
    }


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
