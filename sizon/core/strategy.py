"""Portable genome serialization and complete strategy research records."""

from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass
from sizon.core.expression import Series, Constant, Primitive, Binary


def genome_to_dict(node):
    if isinstance(node, Series):
        return {"type": "series", "name": node.name}
    if isinstance(node, Constant):
        return {"type": "constant", "value": node.value}
    if isinstance(node, Primitive):
        return {"type": "primitive", "name": node.name, "period": node.period}
    if isinstance(node, Binary):
        return {
            "type": "binary",
            "op": node.op,
            "left": genome_to_dict(node.left),
            "right": genome_to_dict(node.right),
        }
    raise TypeError(type(node).__name__)


def genome_from_dict(data):
    kind = data["type"]
    if kind == "series":
        return Series(data["name"])
    if kind == "constant":
        return Constant(float(data["value"]))
    if kind == "primitive":
        return Primitive(data["name"], int(data["period"]))
    if kind == "binary":
        return Binary(
            data["op"], genome_from_dict(data["left"]), genome_from_dict(data["right"])
        )
    raise ValueError(kind)


def expression(node):
    if isinstance(node, Series):
        return node.name
    if isinstance(node, Constant):
        return str(node.value)
    if isinstance(node, Primitive):
        return f"{node.name}(close, {node.period})"
    if isinstance(node, Binary):
        return f"({expression(node.left)} {node.op} {expression(node.right)})"
    return repr(node)


def strategy_id(node):
    return (
        "strat_"
        + hashlib.sha256(
            json.dumps(genome_to_dict(node), sort_keys=True).encode()
        ).hexdigest()[:16]
    )


@dataclass
class StrategyRecord:
    strategy_id: str
    generation: int
    index: int
    genome: dict
    expression: str
    complexity: int
    train_metrics: dict
    test_metrics: dict
    execution: dict | None = None
    validation: dict | None = None
    robustness: dict | None = None
    status: str = "evaluated"
    warnings: list[str] | None = None

    def as_dict(self):
        return asdict(self)
