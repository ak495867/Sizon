# Sizon

Sizon is a reproducible quantitative research platform organized by domain rather than a flat module directory. It evolves alpha expressions, simulates execution, validates research chronologically, preserves every strategy, and provides controlled deployment interfaces.

## Clean architecture

```text
sizon/
├── core/          # ASTs, strategy records, primitive specs, strict typing
├── data/          # local feeds, point-in-time data, actions, rolls, connectors
├── simulation/    # execution, order books, backtests, portfolio, accounting
├── research/      # validation, robustness, statistics, NSGA-II, genealogy
├── platform/      # storage, DuckDB/SQLite, reports, exports, Studio, safety
└── cli/           # command-line entry point
```

## Implemented phases

### Phase 1 — Correctness

- 137 registry primitives
- Specialized kernels for EMA, SMA, WMA, HMA, RSI, ATR, MACD, DEMA, TEMA, Donchian, Bollinger, PPO, VWAP, OBV, CCI, volatility, and volume features
- Strict primitive and binary-operator validation
- Scalar/series/signal/position type foundations
- Warmup and lookahead-safety metadata
- Point-in-time visibility, corporate-action, futures-roll, and connector contracts

### Phase 2 — Realistic simulation

- Event-oriented order-book primitives
- Market and limit fills
- Queue-ahead simulation
- Participation-limited partial fills
- Impact curves
- Commission, spread, slippage, delay, funding, and borrow costs
- Portfolio ledger and mark-to-market accounting
- Cross-sectional constrained allocation
- Exposure, leverage, margin, liquidation, and tick-size controls

### Phase 3 — Better discovery

- NSGA-II-style non-dominated sorting and crowding selection
- Pareto tracking
- Structural crossover, diversity, and bloat controls
- Serialized checkpoints and resume-aware Engine
- Strategy genealogy
- Population analytics
- Multi-objective robustness score
- SQLite/DuckDB experiment database facade

### Phase 4 — Usability

- HTML and Markdown reports
- Strategy comparison
- JSON, YAML-like, and Python exports
- Notebook template generation
- DuckDB when installed, SQLite fallback otherwise
- Local Studio HTTP service with health endpoint

### Phase 5 — Controlled deployment

- Paper broker
- Shadow mode
- Simulated broker adapter
- Kill switch
- Health monitor and alert records
- Deployment gates and notional limits
- Secret-loading boundary
- Live orders disabled by default

## Every strategy is preserved

Each candidate record stores its genome, expression, execution assumptions, train/test metrics, walk-forward and purged-CV results, cost and parameter sensitivity, Monte Carlo diagnostics, generation, lineage fields, and status. Low-performing candidates are not silently discarded.

## Usage

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
python -m sizon.cli.main validate examples/sample.csv
python -m sizon.cli.main run examples/sample.csv --output-dir runs --run-id research-001
```

Or use the Python API:

```python
import sizon as sz
feed = sz.DataFeed.from_csv("examples/sample.csv")
engine = sz.Engine(feed, population=20, generations=5, output_dir="runs")
best = engine.run()
```

## Verification

The current repository has **18 passing tests**, including clean domain imports, 137 primitive evaluations, strict type validation, order-book partial fills, point-in-time visibility, portfolio accounting, NSGA-II/checkpoint behavior, genealogy, comparison, exports, notebook generation, experiment querying, shadow mode, kill-switch behavior, and complete per-strategy persistence.

## Safety boundary

This is a complete research and controlled-deployment baseline, not exchange-certified software. Live integrations still require venue-specific calibration, credentials, security review, operational approvals, incident response, and staged testing. Sizon fails closed for live orders by default.

## License

Apache-2.0.

## Production-readiness layer

The latest baseline adds explicit contracts for the remaining operational areas:

- Vendor connector configuration, retry handling, dataset lineage, and source hashes
- Survivorship-safe point-in-time universe membership
- Venue-specific tick sizes, lot sizes, fees, margin, and order limits
- Event-driven order-book replay with calibration records
- Broker adapter boundaries, account snapshots, and reconciliation
- Environment-backed secrets with namespace restrictions
- Hash-chained audit events
- Alert sinks and incident management
- Paper/shadow soak-test harnesses
- Reference fixture validation for statistical implementations
- Performance benchmark helpers and budgets
- Deployment readiness checklist and approval records
- Studio dashboard scaffold at `studio-ui/index.html`

These modules are intentionally **fail-closed**. They provide the production integration boundaries and verification harnesses, but they do not claim that a generic connector is calibrated for a particular venue or that a live broker is safe without credentials, operational approval, and venue-specific testing.

## Staged deployment checklist

```text
Research validation
  -> Paper soak test
  -> Shadow soak test
  -> Reconciliation verification
  -> Risk-gate and kill-switch verification
  -> Incident and alert-route verification
  -> Manual approval record
  -> Limited live test
```

The `ReadinessChecklist` refuses to report readiness until all required items are explicitly marked complete.
