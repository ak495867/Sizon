"""Live paper-trading execution runner with fail-closed safety and audit logging."""

from __future__ import annotations
from dataclasses import dataclass
import math
from sizon.core.expression import Context, Node
from sizon.platform.deployment import KillSwitch
from sizon.platform.production import DeploymentGate, SafetyConfig
from sizon.platform.security import AuditLog
from sizon.simulation.accounting import PortfolioLedger
from sizon.simulation.execution import ExecutionModel, DEFAULT_EXECUTION
from sizon.simulation.portfolio import (
    Order,
    PortfolioConfig,
    execute_order,
    apply_risk_limits,
    liquidation_check,
    PortfolioState,
)


@dataclass(frozen=True)
class LiveBar:
    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PaperTradingRunner:
    """Stateful, event-driven paper execution engine for streaming bar feeds."""

    def __init__(
        self,
        genome: Node,
        portfolio_config: PortfolioConfig | None = None,
        execution: ExecutionModel | None = None,
        safety_config: SafetyConfig | None = None,
        audit_log: AuditLog | None = None,
        kill_switch: KillSwitch | None = None,
        buffer_size: int = 300,
    ):
        self.genome = genome
        self.config = portfolio_config or PortfolioConfig()
        self.execution = execution or DEFAULT_EXECUTION
        self.safety = safety_config or SafetyConfig(
            allow_live_orders=True,
            max_order_notional=max(1_000_000.0, self.config.initial_cash * 2.0),
        )
        self.gate = DeploymentGate(self.safety)
        self.audit = audit_log or AuditLog()
        self.kill_switch = kill_switch or KillSwitch()
        self.buffer_size = buffer_size

        self.ledger = PortfolioLedger(cash=self.config.initial_cash)
        self.bars: list[LiveBar] = []
        self.current_position: float = 0.0
        self.trades_count: int = 0

    def step(self, bar: LiveBar, manual_approval: bool = True) -> dict:
        """Process one incoming live bar, evaluate AST signal, and execute state transitions."""
        # 1. Kill-switch check
        if not self.kill_switch.allow():
            self.audit.append(
                "paper_runner",
                "blocked_by_kill_switch",
                {"reason": self.kill_switch.reason, "bar": bar.timestamp},
            )
            return {
                "status": "rejected",
                "reason": f"kill switch active: {self.kill_switch.reason}",
            }

        # 2. Update buffer
        self.bars.append(bar)
        if len(self.bars) > self.buffer_size:
            self.bars = self.bars[-self.buffer_size :]

        # Warmup check
        warmup_required = max(2, self.genome.warmup())
        if len(self.bars) < warmup_required:
            return {
                "status": "warming_up",
                "bars_collected": len(self.bars),
                "warmup_required": warmup_required,
            }

        # 3. Evaluate signal
        columns = {
            "close": [b.close for b in self.bars],
            "high": [b.high for b in self.bars],
            "low": [b.low for b in self.bars],
            "open": [b.open for b in self.bars],
            "volume": [b.volume for b in self.bars],
        }
        ctx = Context(columns)
        signals = self.genome.evaluate(ctx)
        latest_signal = signals[-1] if signals else 0.0
        if math.isnan(latest_signal):
            latest_signal = 0.0

        # 4. Target positioning and sizing
        direction = 1.0 if latest_signal > 0.0 else (-1.0 if latest_signal < 0.0 else 0.0)
        current_prices = {bar.symbol: bar.close}
        current_equity = self.ledger.mark_to_market(current_prices)

        # Check liquidation
        state = PortfolioState(
            cash=self.ledger.cash,
            equity=current_equity,
            gross_exposure=abs(self.current_position * bar.close),
        )
        if liquidation_check(state, self.config):
            self.kill_switch.trip("portfolio_liquidated")
            self.audit.append(
                "paper_runner", "liquidated", {"equity": current_equity}
            )
            return {"status": "liquidated", "equity": current_equity}

        target_units = apply_risk_limits(
            direction * (current_equity / max(1e-12, bar.close)),
            current_equity,
            bar.close,
            self.config,
        )

        delta_units = target_units - self.current_position
        if abs(delta_units) < (self.config.min_order_size or 1e-6):
            return {
                "status": "no_change",
                "equity": current_equity,
                "position": self.current_position,
                "signal": latest_signal,
            }

        side = "buy" if delta_units > 0 else "sell"
        notional = abs(delta_units * bar.close)

        # 5. Deployment gate authorization
        try:
            self.gate.authorize(notional, manual_approval=manual_approval)
        except PermissionError as exc:
            self.audit.append(
                "paper_runner",
                "gate_rejected",
                {"error": str(exc), "notional": notional},
            )
            return {"status": "gate_rejected", "reason": str(exc)}

        # 6. Execute order & update double-entry ledger
        order = Order(
            symbol=bar.symbol,
            side=side,
            quantity=delta_units,
            order_type="market",
        )
        filled_order = execute_order(
            order, bar.close, liquidity=bar.volume * 0.5, config=self.config
        )

        if filled_order.status in {"filled", "partial"}:
            fill_price = filled_order.fill_price or bar.close
            fill_qty = filled_order.filled_quantity * (1 if side == "buy" else -1)
            fee = abs(fill_qty * fill_price) * (self.execution.commission_bps / 10000.0)
            self.ledger.trade(
                bar.timestamp, bar.symbol, fill_qty, fill_price, fees=fee
            )
            self.current_position += fill_qty
            self.trades_count += 1
            self.audit.append(
                "paper_runner",
                "order_filled",
                {
                    "symbol": bar.symbol,
                    "side": side,
                    "quantity": fill_qty,
                    "price": fill_price,
                    "fees": fee,
                    "new_position": self.current_position,
                },
            )
            new_equity = self.ledger.mark_to_market(current_prices)
            return {
                "status": filled_order.status,
                "order": filled_order.__dict__,
                "position": self.current_position,
                "equity": new_equity,
                "cash": self.ledger.cash,
            }

        return {"status": filled_order.status, "reason": "unfilled"}

    def snapshot(self) -> dict:
        """Return ledger balance, positions, audit log integrity, and trade count."""
        last_price = (
            {self.bars[-1].symbol: self.bars[-1].close} if self.bars else {}
        )
        return {
            "cash": self.ledger.cash,
            "equity": self.ledger.mark_to_market(last_price),
            "positions": dict(self.ledger.positions),
            "trades": self.trades_count,
            "audit_chain_verified": self.audit.verify(),
            "kill_switch_active": self.kill_switch.tripped,
            "bars_buffered": len(self.bars),
        }
