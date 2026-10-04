"""
End-to-End Alpha Generation and Validation Script
-------------------------------------------------
1. Runs NSGA-II to evolve a population of candidate strategies.
2. Extracts the best genomes.
3. Passes them through the Institutional Gauntlet.
"""

import sizon as sz
import json
from pathlib import Path
from run_institutional_pipeline import InstitutionalGauntlet

def main():
    print("[START] Starting Alpha Hunt...")
    
    # 1. Load Data
    feed = sz.DataFeed.from_csv("Data/btcusd_is.csv")
    gauntlet = InstitutionalGauntlet("Data/btcusd_is.csv", "Data/btcusd_oos.csv")
    
    # 2. Configure Evolutionary Engine
    engine = sz.Engine(
        feed,
        population=30,      # Small enough for rapid generation
        generations=3,      # Small enough for rapid generation
        seed=1337,
        output_dir="runs",
        run_id="alpha_hunt_v1",
        robustness_simulations=30
    )
    
    # 3. Search for Alpha (NSGA-II)
    best_candidate = engine.run()
    
    # Sort the final population by In-Sample Sharpe
    candidates = sorted(engine.population, key=lambda c: c.scores.get("sharpe", 0.0), reverse=True)
    print(f"\n[DONE] Evolutionary Search Complete. Found {len(candidates)} candidates.")
    
    # 4. Pass top candidates through the Institutional Gauntlet
    print("\n[GAUNTLET] Passing Top Candidates to the Institutional Gauntlet...\n")
    
    # We will test the top 5 candidates
    for rank_idx, candidate in enumerate(candidates[:5]):
        genome = candidate.genome
        strat_id = sz.strategy_id(genome)
        
        print(f"Testing Candidate #{rank_idx+1}: {strat_id}")
        passed = gauntlet.run_gauntlet(genome, strat_id)
        if passed:
            print(f"[PASS] {strat_id} SURVIVED THE GAUNTLET and is saved to Validated/\n")
        else:
            print(f"[FAIL] {strat_id} FAILED THE GAUNTLET.\n")

if __name__ == "__main__":
    main()
