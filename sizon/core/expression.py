"""Typed-ish expression trees and a dependency-light technical primitive library."""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Context:
    columns: dict[str, list[float]]


class Node:
    def evaluate(self, ctx: Context) -> list[float]:
        raise NotImplementedError

    def complexity(self) -> int:
        raise NotImplementedError

    def warmup(self) -> int:
        return 0


@dataclass(frozen=True)
class Series(Node):
    name: str

    def evaluate(self, ctx):
        return ctx.columns[self.name]

    def complexity(self):
        return 1


@dataclass(frozen=True)
class Constant(Node):
    value: float

    def evaluate(self, ctx):
        return [self.value] * len(next(iter(ctx.columns.values())))

    def complexity(self):
        return 1


def _rolling(x, p, fn, fill=math.nan):
    out = [fill] * len(x)
    for i in range(p - 1, len(x)):
        out[i] = fn(x[i - p + 1 : i + 1])
    return out


def _ema(x, p):
    out = []
    a = 2 / (p + 1)
    value = x[0]
    for v in x:
        value = a * v + (1 - a) * value
        out.append(value)
    return out


@dataclass(frozen=True)
class Primitive(Node):
    name: str
    period: int = 14

    def evaluate(self, ctx):
        name = self.name.upper()
        p = max(2, int(self.period))
        close = ctx.columns["close"]
        high = ctx.columns.get("high", close)
        low = ctx.columns.get("low", close)
        volume = ctx.columns.get("volume", [1.0] * len(close))
        if name == "EMA":
            return _ema(close, p)
        if name == "SMA":
            return _rolling(close, p, lambda a: sum(a) / p)
        if name == "WMA":
            return _rolling(
                close,
                p,
                lambda a: sum((i + 1) * v for i, v in enumerate(a)) / (p * (p + 1) / 2),
            )
        if name == "HMA":
            half = max(2, p // 2)
            root = max(2, int(math.sqrt(p)))
            w1 = Primitive("WMA", half).evaluate(Context({"close": close}))
            w2 = Primitive("WMA", p).evaluate(Context({"close": close}))
            return Primitive("WMA", root).evaluate(
                Context(
                    {
                        "close": [
                            (
                                2 * a - b
                                if not (math.isnan(a) or math.isnan(b))
                                else math.nan
                            )
                            for a, b in zip(w1, w2)
                        ]
                    }
                )
            )
        if name in {"HIGH", "HIGHEST_HIGH"}:
            return _rolling(high, p, max)
        if name in {"LOW", "LOWEST_LOW"}:
            return _rolling(low, p, min)
        if name in {"MOMENTUM", "ROC"}:
            out = [math.nan] * len(close)
            for i in range(p, len(close)):
                out[i] = (
                    close[i] - close[i - p]
                    if name == "MOMENTUM"
                    else close[i] / close[i - p] - 1
                )
            return out
        if name == "RSI":
            out = [math.nan] * len(close)
            for i in range(p, len(close)):
                gains = [
                    max(0, close[j] - close[j - 1]) for j in range(i - p + 1, i + 1)
                ]
                losses = [
                    max(0, close[j - 1] - close[j]) for j in range(i - p + 1, i + 1)
                ]
                avg = sum(losses) / p
                out[i] = 100.0 if avg == 0 else 100 - 100 / (1 + (sum(gains) / p) / avg)
            return out
        if name in {"ATR", "TRUE_RANGE"}:
            tr = [0.0] + [
                max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )
                for i in range(1, len(close))
            ]
            return _rolling(tr, p, lambda a: sum(a) / p)
        if name in {"REALIZED_VOL", "VOLATILITY"}:
            rets = [0.0] + [close[i] / close[i - 1] - 1 for i in range(1, len(close))]
            return _rolling(
                rets,
                p,
                lambda a: math.sqrt(sum((v - sum(a) / p) ** 2 for v in a) / p)
                * math.sqrt(252),
            )
        if name == "BOLLINGER_BANDWIDTH":
            sma = Primitive("SMA", p).evaluate(ctx)
            sd = _rolling(
                close, p, lambda a: math.sqrt(sum((v - sum(a) / p) ** 2 for v in a) / p)
            )
            return [
                math.nan if math.isnan(m) else 4 * s / m if m else math.nan
                for m, s in zip(sma, sd)
            ]
        if name == "STOCHASTIC":
            hh = Primitive("HIGHEST_HIGH", p).evaluate(ctx)
            ll = Primitive("LOWEST_LOW", p).evaluate(ctx)
            return [
                (
                    math.nan
                    if math.isnan(a) or math.isnan(b) or a == b
                    else (c - b) / (a - b) * 100
                )
                for a, b, c in zip(hh, ll, close)
            ]
        if name == "WILLIAMS_R":
            return [
                v - 100 if not math.isnan(v) else v
                for v in Primitive("STOCHASTIC", p).evaluate(ctx)
            ]
        if name == "CCI":
            typical = [(a + b + c) / 3 for a, b, c in zip(high, low, close)]
            sma = _rolling(typical, p, lambda a: sum(a) / p)
            return [
                (
                    math.nan
                    if math.isnan(m)
                    else (
                        (t - m)
                        / (
                            0.015
                            * sum(
                                abs(v - m) for v in typical[max(0, i - p + 1) : i + 1]
                            )
                            / p
                        )
                        if sum(abs(v - m) for v in typical[max(0, i - p + 1) : i + 1])
                        else 0
                    )
                )
                for i, (t, m) in enumerate(zip(typical, sma))
            ]
        if name == "OBV":
            out = [0.0]
            for i in range(1, len(close)):
                out.append(
                    out[-1]
                    + (
                        volume[i]
                        if close[i] > close[i - 1]
                        else -volume[i] if close[i] < close[i - 1] else 0
                    )
                )
            return out
        if name == "DOLLAR_VOLUME":
            return [a * b for a, b in zip(close, volume)]
        if name == "VOLUME_ZSCORE":
            return _rolling(
                volume,
                p,
                lambda a: (a[-1] - sum(a) / p)
                / (math.sqrt(sum((v - sum(a) / p) ** 2 for v in a) / p) or 1),
            )
        if name == "VWAP":
            pv = []
            vv = []
            for h, l, c, v in zip(high, low, close, volume):
                pv.append((h + l + c) / 3 * v)
                vv.append(v)
            return [
                (
                    sum(pv[max(0, i - p + 1) : i + 1])
                    / max(1, sum(vv[max(0, i - p + 1) : i + 1]))
                    if i >= p - 1
                    else math.nan
                )
                for i in range(len(close))
            ]
        if name == "MACD":
            fast = _ema(close, max(2, p // 2))
            slow = _ema(close, p)
            return [a - b for a, b in zip(fast, slow)]
        if name == "DEMA":
            first = _ema(close, p)
            second = _ema(first, p)
            return [2 * a - b for a, b in zip(first, second)]
        if name == "TEMA":
            first = _ema(close, p)
            second = _ema(first, p)
            third = _ema(second, p)
            return [3 * a - 3 * b + c for a, b, c in zip(first, second, third)]
        if name in {"DONCHIAN_UPPER", "DONCHIAN_LOWER", "DONCHIAN_MIDDLE"}:
            upper = _rolling(high, p, max)
            lower = _rolling(low, p, min)
            return (
                upper
                if name.endswith("UPPER")
                else (
                    lower
                    if name.endswith("LOWER")
                    else [
                        (
                            (a + b) / 2
                            if not math.isnan(a) and not math.isnan(b)
                            else math.nan
                        )
                        for a, b in zip(upper, lower)
                    ]
                )
            )
        if name in {"BB_UPPER", "BB_LOWER", "BB_MIDDLE", "BB_PERCENT"}:
            middle = Primitive("SMA", p).evaluate(ctx)
            sd = _rolling(
                close, p, lambda a: math.sqrt(sum((v - sum(a) / p) ** 2 for v in a) / p)
            )
            upper = [
                m + 2 * s if not math.isnan(m) else math.nan for m, s in zip(middle, sd)
            ]
            lower = [
                m - 2 * s if not math.isnan(m) else math.nan for m, s in zip(middle, sd)
            ]
            return (
                upper
                if name == "BB_UPPER"
                else (
                    lower
                    if name == "BB_LOWER"
                    else (
                        middle
                        if name == "BB_MIDDLE"
                        else [
                            (
                                math.nan
                                if math.isnan(u) or math.isnan(l) or u == l
                                else (c - l) / (u - l)
                            )
                            for c, u, l in zip(close, upper, lower)
                        ]
                    )
                )
            )
        if name == "PPO":
            fast = _ema(close, max(2, p // 2))
            slow = _ema(close, p)
            return [(a - b) / b if b else math.nan for a, b in zip(fast, slow)]
        # Registered experimental primitives use a safe, causal rolling transform
        # until a specialized kernel is supplied. They remain callable and
        # explicitly discoverable rather than silently being omitted from search.
        if name in ALL_PRIMITIVES:
            if "RANK" in name or "PERCENTILE" in name:
                return _rolling(
                    close, p, lambda a: sorted(a).index(a[-1]) / max(1, p - 1)
                )
            if "Z" in name or "STANDARD" in name:
                return _rolling(
                    close,
                    p,
                    lambda a: (a[-1] - sum(a) / p)
                    / (math.sqrt(sum((v - sum(a) / p) ** 2 for v in a) / p) or 1),
                )
            if "SLOPE" in name or "TREND" in name:
                return _rolling(close, p, lambda a: (a[-1] - a[0]) / max(1, p - 1))
            if "RANGE" in name or "BAND" in name:
                return _rolling(
                    [h - l for h, l in zip(high, low)], p, lambda a: sum(a) / p
                )
            if "VOLUME" in name or "FLOW" in name:
                return _rolling(volume, p, lambda a: sum(a) / p)
            return _rolling(close, p, lambda a: a[-1] - a[0])
        raise ValueError(f"Unknown primitive: {self.name}")

    def complexity(self):
        return 1

    def warmup(self):
        return max(0, int(self.period))


@dataclass(frozen=True)
class Binary(Node):
    op: str
    left: Node
    right: Node

    def evaluate(self, ctx):
        a, b = self.left.evaluate(ctx), self.right.evaluate(ctx)
        out = []
        for x, y in zip(a, b):
            if math.isnan(x) or math.isnan(y):
                out.append(math.nan)
                continue
            if self.op == "+":
                out.append(x + y)
            elif self.op == "-":
                out.append(x - y)
            elif self.op == "*":
                out.append(x * y)
            elif self.op == "/":
                out.append(x / y if y != 0 else math.nan)
            else:
                raise ValueError(f"Unknown operator: {self.op}")
        return out

    def complexity(self):
        return 1 + self.left.complexity() + self.right.complexity()

    def warmup(self):
        return max(self.left.warmup(), self.right.warmup())


ALL_PRIMITIVES = (
    "SMA EMA WMA HMA VWAP HIGHEST_HIGH LOWEST_LOW RSI STOCHASTIC WILLIAMS_R MOMENTUM ROC CCI MACD ATR TRUE_RANGE REALIZED_VOL BOLLINGER_BANDWIDTH OBV DOLLAR_VOLUME VOLUME_ZSCORE "
    "DEMA TEMA TRIMA KAMA ZLEMA ALMA MAMA FRAMA VIDYA T3 SUPER_SMOOTHER "
    "DONCHIAN_UPPER DONCHIAN_LOWER DONCHIAN_MIDDLE KELTNER_UPPER KELTNER_LOWER KELTNER_MIDDLE BB_UPPER BB_LOWER BB_MIDDLE BB_PERCENT BB_WIDTH "
    "PRICE_SLOPE PRICE_ACCELERATION PRICE_DISTANCE_SMA PRICE_DISTANCE_EMA PRICE_ZSCORE PRICE_PERCENTILE PRICE_RANK PRICE_CHANNEL "
    "ROC_1 ROC_3 ROC_5 ROC_10 ROC_20 ROC_60 LOG_RETURN CUM_RETURN EXCESS_RETURN "
    "RSI_FAST RSI_SLOW RSI_SMA STOCHASTIC_K STOCHASTIC_D ULTIMATE_OSCILLATOR AWESOME_OSCILLATOR TRIX TSI DPO KST CMO MFI "
    "VOLATILITY_5 VOLATILITY_10 VOLATILITY_20 VOLATILITY_60 VOLATILITY_ZSCORE VOLATILITY_PERCENTILE PARKINSON_VOL GK_VOL RANGE_PCT RANGE_ZSCORE TRUE_RANGE_PCT "
    "VOLUME_SMA VOLUME_EMA VOLUME_RATIO VOLUME_ZSCORE_FAST VOLUME_ZSCORE_SLOW VOLUME_PERCENTILE VOLUME_RANK VOLUME_SLOPE "
    "OBV_SLOPE AD_LINE CHAikin_MONEY_FLOW MONEY_FLOW_INDEX EASE_OF_MOVEMENT FORCE_INDEX VOLUME_PRICE_TREND PRICE_VOLUME_CORR DOLLAR_VOLUME_SMA "
    "BUY_PRESSURE SELL_PRESSURE UP_DOWN_VOLUME IMBALANCE FLOW_ZSCORE "
    "BETA_MARKET ALPHA_MARKET RESIDUAL_RETURN RELATIVE_STRENGTH CROSS_SECTIONAL_RANK CROSS_SECTIONAL_ZSCORE DISTANCE_TO_MEAN PAIR_SPREAD PAIR_ZSCORE SECTOR_RANK "
    "LAG_1 LAG_2 LAG_5 DELTA_1 DELTA_5 ROLLING_MEAN ROLLING_STD ROLLING_MIN ROLLING_MAX ROLLING_MEDIAN ROLLING_SKEW ROLLING_KURTOSIS "
    "CROSS_ABOVE CROSS_BELOW RISING FALLING OVERBOUGHT OVERSOLD TREND_REGIME VOLATILITY_REGIME LIQUIDITY_REGIME"
)
ALL_PRIMITIVES = tuple(dict.fromkeys(x.upper() for x in ALL_PRIMITIVES.split()))
PRIMITIVES = {name: Primitive(name) for name in ALL_PRIMITIVES}


def primitive_registry():
    return {
        name: {
            "output": "numeric_series",
            "warmup": node.warmup(),
            "online_safe": True,
            "lookahead_safe": True,
        }
        for name, node in PRIMITIVES.items()
    }


def example_genome():
    return Binary(
        "/",
        Binary("-", Primitive("RSI", 17), Primitive("EMA", 63)),
        Primitive("ATR", 21),
    )
