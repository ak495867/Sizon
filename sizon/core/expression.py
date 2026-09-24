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


def _kama(x, p, fast=2, slow=30):
    if len(x) < p:
        return [math.nan] * len(x)
    out = [math.nan] * len(x)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    kama = x[p - 1]
    out[p - 1] = kama
    for i in range(p, len(x)):
        change = abs(x[i] - x[i - p])
        vol = sum(abs(x[j] - x[j - 1]) for j in range(i - p + 1, i + 1))
        er = change / vol if vol > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = kama + sc * (x[i] - kama)
        out[i] = kama
    return out


def _zlema(x, p):
    lag = max(1, (p - 1) // 2)
    adjusted = [(2 * x[i] - x[i - lag]) if i >= lag else x[i] for i in range(len(x))]
    return _ema(adjusted, p)


def _skew(chunk):
    n = len(chunk)
    if n < 3:
        return 0.0
    mean = sum(chunk) / n
    var = sum((v - mean) ** 2 for v in chunk) / n
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    m3 = sum((v - mean) ** 3 for v in chunk) / n
    return m3 / (std**3)


def _kurtosis(chunk):
    n = len(chunk)
    if n < 4:
        return 0.0
    mean = sum(chunk) / n
    var = sum((v - mean) ** 2 for v in chunk) / n
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    m4 = sum((v - mean) ** 4 for v in chunk) / n
    return (m4 / (std**4)) - 3.0


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
        open_ = ctx.columns.get("open", close)
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
            for h, lo, c, v in zip(high, low, close, volume):
                pv.append((h + lo + c) / 3 * v)
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
                                if math.isnan(u) or math.isnan(lo) or u == lo
                                else (c - lo) / (u - lo)
                            )
                            for c, u, lo in zip(close, upper, lower)
                        ]
                    )
                )
            )
        if name == "PPO":
            fast = _ema(close, max(2, p // 2))
            slow = _ema(close, p)
            return [(a - b) / b if b else math.nan for a, b in zip(fast, slow)]
        if name == "KAMA":
            return _kama(close, p)
        if name == "ZLEMA":
            return _zlema(close, p)
        if name == "TRIMA":
            half = max(2, (p + 1) // 2)
            first = Primitive("SMA", half).evaluate(ctx)
            return Primitive("SMA", half).evaluate(
                Context({"close": [(0.0 if math.isnan(v) else v) for v in first]})
            )
        if name == "PARKINSON_VOL":
            hl = [
                (math.log(max(1e-12, h) / max(1e-12, lo)) ** 2) / (4 * math.log(2))
                for h, lo in zip(high, low)
            ]
            return _rolling(
                hl, p, lambda a: math.sqrt(max(0.0, sum(a) / p)) * math.sqrt(252)
            )
        if name == "GK_VOL":
            gk = [
                0.5 * (math.log(max(1e-12, h) / max(1e-12, lo)) ** 2)
                - (2 * math.log(2) - 1)
                * (math.log(max(1e-12, c) / max(1e-12, o)) ** 2)
                for o, h, lo, c in zip(open_, high, low, close)
            ]
            return _rolling(
                gk, p, lambda a: math.sqrt(max(0.0, sum(a) / p)) * math.sqrt(252)
            )
        if name in {"CHAIKIN_MONEY_FLOW", "CMF"}:
            mfv = [
                (((c - lo) - (h - c)) / max(1e-12, h - lo)) * v
                if h > lo
                else 0.0
                for h, lo, c, v in zip(high, low, close, volume)
            ]
            vol_sum = _rolling(volume, p, sum)
            mfv_sum = _rolling(mfv, p, sum)
            return [
                m / max(1e-12, vs)
                if not (math.isnan(m) or math.isnan(vs))
                else math.nan
                for m, vs in zip(mfv_sum, vol_sum)
            ]
        if name in {"MONEY_FLOW_INDEX", "MFI"}:
            tp = [(h + lo + c) / 3 for h, lo, c in zip(high, low, close)]
            pos = [0.0] * len(tp)
            neg = [0.0] * len(tp)
            for i in range(1, len(tp)):
                if tp[i] > tp[i - 1]:
                    pos[i] = tp[i] * volume[i]
                elif tp[i] < tp[i - 1]:
                    neg[i] = tp[i] * volume[i]
            pos_sum = _rolling(pos, p, sum)
            neg_sum = _rolling(neg, p, sum)
            return [
                (
                    100.0
                    if ns == 0
                    else 100 - 100 / (1 + ps / ns)
                    if not (math.isnan(ps) or math.isnan(ns))
                    else math.nan
                )
                for ps, ns in zip(pos_sum, neg_sum)
            ]
        if name == "TRIX":
            e1 = _ema(close, p)
            e2 = _ema(e1, p)
            e3 = _ema(e2, p)
            out = [math.nan] * len(close)
            for i in range(1, len(close)):
                out[i] = (
                    (e3[i] - e3[i - 1]) / e3[i - 1] * 100 if e3[i - 1] else 0.0
                )
            return out
        if name == "TSI":
            mom = [0.0] + [close[i] - close[i - 1] for i in range(1, len(close))]
            r = max(2, p)
            s = max(2, p // 2)
            em1 = _ema(mom, r)
            em2 = _ema(em1, s)
            absm1 = _ema([abs(x) for x in mom], r)
            absm2 = _ema(absm1, s)
            return [
                100 * m / max(1e-12, a)
                if not (math.isnan(m) or math.isnan(a))
                else math.nan
                for m, a in zip(em2, absm2)
            ]
        if name == "KST":
            r1 = _rolling(
                close, max(2, p // 2), lambda a: (a[-1] - a[0]) / max(1e-12, a[0])
            )
            r2 = _rolling(close, p, lambda a: (a[-1] - a[0]) / max(1e-12, a[0]))
            r3 = _rolling(
                close, int(p * 1.5), lambda a: (a[-1] - a[0]) / max(1e-12, a[0])
            )
            r4 = _rolling(close, p * 2, lambda a: (a[-1] - a[0]) / max(1e-12, a[0]))
            sr1 = _ema([(0.0 if math.isnan(v) else v) for v in r1], max(2, p // 2))
            sr2 = _ema([(0.0 if math.isnan(v) else v) for v in r2], max(2, p // 2))
            sr3 = _ema([(0.0 if math.isnan(v) else v) for v in r3], max(2, p // 2))
            sr4 = _ema([(0.0 if math.isnan(v) else v) for v in r4], p)
            return [a + 2 * b + 3 * c + 4 * d for a, b, c, d in zip(sr1, sr2, sr3, sr4)]
        if name == "CMO":
            out = [math.nan] * len(close)
            for i in range(p, len(close)):
                gains = sum(
                    max(0, close[j] - close[j - 1]) for j in range(i - p + 1, i + 1)
                )
                losses = sum(
                    max(0, close[j - 1] - close[j]) for j in range(i - p + 1, i + 1)
                )
                denom = gains + losses
                out[i] = 100 * (gains - losses) / denom if denom else 0.0
            return out
        if name in {"KELTNER_UPPER", "KELTNER_LOWER", "KELTNER_MIDDLE"}:
            mid = _ema(close, p)
            tr = Primitive("ATR", p).evaluate(ctx)
            if name.endswith("UPPER"):
                return [m + 2 * t for m, t in zip(mid, tr)]
            if name.endswith("LOWER"):
                return [m - 2 * t for m, t in zip(mid, tr)]
            return mid
        if name == "BB_WIDTH":
            up = Primitive("BB_UPPER", p).evaluate(ctx)
            lo = Primitive("BB_LOWER", p).evaluate(ctx)
            mid = Primitive("BB_MIDDLE", p).evaluate(ctx)
            return [
                (u - l_val) / max(1e-12, m) * 100
                if not (math.isnan(u) or math.isnan(l_val) or math.isnan(m))
                else math.nan
                for u, l_val, m in zip(up, lo, mid)
            ]
        if name in {"RANGE_PCT", "TRUE_RANGE_PCT"}:
            if name == "RANGE_PCT":
                return [
                    (h - lo) / max(1e-12, c) * 100 for h, lo, c in zip(high, low, close)
                ]
            tr = Primitive("ATR", 1).evaluate(ctx)
            return [t / max(1e-12, c) * 100 for t, c in zip(tr, close)]
        if name == "LOG_RETURN":
            out = [0.0]
            for i in range(1, len(close)):
                out.append(math.log(max(1e-12, close[i]) / max(1e-12, close[i - 1])))
            return out
        if name == "CUM_RETURN":
            base = max(1e-12, close[0])
            return [c / base - 1 for c in close]
        if name == "AD_LINE":
            mfv = [
                (((c - lo) - (h - c)) / max(1e-12, h - lo)) * v
                if h > lo
                else 0.0
                for h, lo, c, v in zip(high, low, close, volume)
            ]
            out = [0.0]
            for v in mfv[1:]:
                out.append(out[-1] + v)
            return out
        if name == "EASE_OF_MOVEMENT":
            out = [0.0]
            for i in range(1, len(close)):
                dm = (high[i] + low[i]) / 2 - (high[i - 1] + low[i - 1]) / 2
                br = (volume[i] / 10000) / max(1e-12, high[i] - low[i])
                out.append(dm / max(1e-12, br))
            return _rolling(out, p, lambda a: sum(a) / p)
        if name == "FORCE_INDEX":
            fi = [0.0] + [
                (close[i] - close[i - 1]) * volume[i] for i in range(1, len(close))
            ]
            return _ema(fi, p)
        if name == "VOLUME_RATIO":
            sma_v = _rolling(volume, p, lambda a: sum(a) / p)
            return [
                v / max(1e-12, sv) if not math.isnan(sv) else math.nan
                for v, sv in zip(volume, sma_v)
            ]
        if name in {"LAG_1", "LAG_2", "LAG_5"}:
            lag = int(name.split("_")[1])
            return [math.nan] * lag + close[:-lag]
        if name in {"DELTA_1", "DELTA_5"}:
            lag = int(name.split("_")[1])
            return [math.nan] * lag + [
                close[i] - close[i - lag] for i in range(lag, len(close))
            ]
        if name in {
            "ROLLING_MEAN",
            "ROLLING_STD",
            "ROLLING_MIN",
            "ROLLING_MAX",
            "ROLLING_MEDIAN",
            "ROLLING_SKEW",
            "ROLLING_KURTOSIS",
        }:
            if name == "ROLLING_MEAN":
                return _rolling(close, p, lambda a: sum(a) / p)
            if name == "ROLLING_STD":
                return _rolling(
                    close,
                    p,
                    lambda a: math.sqrt(sum((v - sum(a) / p) ** 2 for v in a) / p),
                )
            if name == "ROLLING_MIN":
                return _rolling(close, p, min)
            if name == "ROLLING_MAX":
                return _rolling(close, p, max)
            if name == "ROLLING_MEDIAN":
                return _rolling(close, p, lambda a: sorted(a)[len(a) // 2])
            if name == "ROLLING_SKEW":
                return _rolling(close, p, _skew)
            if name == "ROLLING_KURTOSIS":
                return _rolling(close, p, _kurtosis)
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
                    [h - lo for h, lo in zip(high, low)], p, lambda a: sum(a) / p
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
