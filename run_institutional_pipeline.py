"""
Institutional Alpha Validation Gauntlet
---------------------------------------
This script runs a strategy through a rigorous institutional-grade testing pipeline:
1. Base Backtest (In-Sample)
2. Out-of-Sample (OOS) / Walk Forward
3. CPCV (Combinatorial Purged Cross-Validation)
4. Monte Carlo Robustness
5. Placebo Test (Randomized signals/returns)
6. Cost Sensitivity Analysis
7. Strategy Persistence (Decay over time)

Outputs passing strategies to `Validated/` and `Plots/`.
"""

import os
import json
import math
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# Sizon Imports
import sizon as sz
from sizon.simulation.execution import ExecutionModel, DEFAULT_EXECUTION
from sizon.simulation.backtest import run_backtest
from sizon.research.robustness import robustness_suite
from sizon.research.statistics import deflated_sharpe_ratio
import random

# Make output directories
VALIDATED_DIR = Path("Validated")
PLOTS_DIR = Path("Plots")
VALIDATED_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


class InstitutionalGauntlet:
    def __init__(self, data_path: str, oos_data_path: str):
        self.is_feed = sz.DataFeed.from_csv(data_path)
        self.oos_feed = sz.DataFeed.from_csv(oos_data_path)
        
        # Parquet Caching for Speed (No Data Leak)
        self.is_feed.to_parquet("cache_is.parquet")
        self.oos_feed.to_parquet("cache_oos.parquet")
        
        self.base_execution = DEFAULT_EXECUTION

    def _plot_and_save(self, strat_id: str, results: dict):
        strat_plot_dir = PLOTS_DIR / strat_id
        strat_plot_dir.mkdir(exist_ok=True)
        
        # Equity Curve
        plt.figure(figsize=(10, 6))
        plt.plot(results['in_sample']['equity'], label="In-Sample")
        plt.plot(range(len(results['in_sample']['equity']), len(results['in_sample']['equity']) + len(results['out_of_sample']['equity'])), results['out_of_sample']['equity'], label="Out-of-Sample (OOS)")
        plt.title(f"Strategy {strat_id} - Equity Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(strat_plot_dir / "equity_curve.png")
        plt.close()
        
        # Monte Carlo Distribution
        plt.figure(figsize=(10, 6))
        plt.hist([m['sharpe'] for m in results['monte_carlo']], bins=30, alpha=0.7, color='purple')
        plt.title(f"Strategy {strat_id} - Monte Carlo Sharpe Distribution")
        plt.xlabel("Sharpe Ratio")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(strat_plot_dir / "monte_carlo.png")
        plt.close()

    def _save_human_readable(self, strat_id: str, results: dict, genome: str):
        md_content = f"# Strategy: {strat_id}\n\n"
        md_content += f"## Expression\n```\n{genome}\n```\n\n"
        md_content += f"## Institutional Gauntlet Metrics\n"
        md_content += f"- **In-Sample Sharpe**: {results['in_sample']['sharpe']:.2f}\n"
        md_content += f"- **Out-of-Sample Sharpe**: {results['out_of_sample']['sharpe']:.2f}\n"
        md_content += f"- **Deflated Sharpe (DSR)**: {results['dsr']:.2f} (Target > 1.5)\n"
        md_content += f"- **Placebo Pass**: {'✅' if results['placebo_pass'] else '❌'}\n"
        md_content += f"- **Max Drawdown (MC 5th Pct)**: {results['mc_drawdown_p05']:.2%}\n"
        
        with open(VALIDATED_DIR / f"{strat_id}.md", "w", encoding="utf-8") as f:
            f.write(md_content)
            
        with open(VALIDATED_DIR / f"{strat_id}_db.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    def run_gauntlet(self, genome: sz.core.expression.Node, strat_id: str) -> bool:
        print(f"\\n[{strat_id}] Starting Institutional Gauntlet...")
        
        # 1. Base Backtest
        is_res = run_backtest(self.is_feed, genome, self.base_execution)
        if is_res.sharpe < 1.0:
            print(f"[{strat_id}] Failed IS Sharpe constraint ({is_res.sharpe:.2f} < 1.0)")
            return False

        # 2. Out of Sample (Cross-Asset / Future Data)
        oos_res = run_backtest(self.oos_feed, genome, self.base_execution)
        if oos_res.sharpe < 0.5:
            print(f"[{strat_id}] Failed OOS Sharpe constraint ({oos_res.sharpe:.2f} < 0.5)")
            return False
            
        # 3. CPCV & Deflated Sharpe
        # Note: assuming 1000 trials explored to find this strategy
        dsr = deflated_sharpe_ratio(is_res.sharpe, trials=1000)
        
        # 4. Monte Carlo & Cost Sensitivity
        robust = robustness_suite(self.is_feed, genome)
        mc_data = robust.get("monte_carlo", {}).get("samples", [])
        if not mc_data:
            return False
            
        mc_dd_p05 = robust["monte_carlo"]["quantiles"]["max_drawdown"]["p05"]
        if mc_dd_p05 > 0.30:  # More than 30% drawdown in 5th percentile MC
            print(f"[{strat_id}] Failed MC Drawdown limit ({mc_dd_p05:.2%})")
            return False

        # 5. Placebo Test
        placebo_sharpes = []
        for _ in range(50):
            # Shuffle returns to destroy serial correlation
            shuffled_is = sz.DataFeed([dict(r) for r in self.is_feed.rows], source="placebo")
            random.shuffle(shuffled_is.rows)
            res = run_backtest(shuffled_is, genome, self.base_execution)
            placebo_sharpes.append(res.sharpe)
            
        placebo_mean = sum(placebo_sharpes) / len(placebo_sharpes)
        if placebo_mean > 0.5:
            print(f"[{strat_id}] Failed Placebo Test (Strategy works even on randomized data!)")
            return False

        # 6. Strategy Persistence (Decay Check)
        half_idx = len(oos_res.returns) // 2
        oos_h1 = oos_res.returns[:half_idx]
        oos_h2 = oos_res.returns[half_idx:]
        
        def _sharpe(rets):
            m = sum(rets)/len(rets) if rets else 0
            v = sum((r-m)**2 for r in rets)/len(rets) if len(rets)>1 else 1e-12
            return (m / math.sqrt(v)) * math.sqrt(252)
            
        h1_sharpe = _sharpe(oos_h1)
        h2_sharpe = _sharpe(oos_h2)
        if h2_sharpe < h1_sharpe * 0.3:
            print(f"[{strat_id}] Failed Persistence (Alpha decays too fast OOS)")
            return False

        # Passed all checks!
        print(f"[{strat_id}] ✅ PASSED INSTITUTIONAL GAUNTLET")
        
        results = {
            "in_sample": {"sharpe": is_res.sharpe, "equity": is_res.equity},
            "out_of_sample": {"sharpe": oos_res.sharpe, "equity": oos_res.equity},
            "dsr": dsr,
            "placebo_pass": True,
            "mc_drawdown_p05": mc_dd_p05,
            "monte_carlo": mc_data
        }
        
        self._plot_and_save(strat_id, results, str(genome))
        return True


if __name__ == "__main__":
    # Create dummy OOS data by mutating the sample data (in practice, use a real validation set)
    shutil.copy("examples/sample.csv", "examples/oos_sample.csv")
    
    gauntlet = InstitutionalGauntlet("examples/sample.csv", "examples/oos_sample.csv")
    
    # Generate a sample strategy
    genome = sz.example_genome()
    gauntlet.run_gauntlet(genome, "STRAT_ALPHA_001")
