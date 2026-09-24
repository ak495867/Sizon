"""Reproducible evolutionary search with complete candidate persistence and robustness tests."""

from __future__ import annotations
from dataclasses import dataclass
import random
from sizon.simulation.backtest import fitness, run_backtest
from sizon.data.data import DataFeed
from sizon.simulation.execution import ExecutionModel, DEFAULT_EXECUTION
from sizon.core.expression import Node, Primitive, Binary, example_genome
from sizon.platform.experiments import ExperimentStore
from sizon.research.robustness import robustness_suite
from sizon.core.strategy import StrategyRecord, genome_to_dict, expression, strategy_id
from sizon.research.validation import walk_forward, purged_cross_validation
from sizon.research.nsga2 import NSGA2, save_checkpoint, load_checkpoint


@dataclass
class Candidate:
    genome: Node
    scores: dict[str, float]


class Engine:
    def __init__(
        self,
        feed: DataFeed,
        population=20,
        generations=5,
        seed=7,
        output_dir="runs",
        run_id=None,
        execution=None,
        robustness_simulations=30,
        resume_from=None,
    ):
        self.feed, self.population, self.generations = feed, population, generations
        self.seed = seed
        self.rng = random.Random(seed)
        self.execution = execution or DEFAULT_EXECUTION
        self.robustness_simulations = robustness_simulations
        self.store = ExperimentStore(output_dir, run_id)
        self.train, self.test = feed.split()
        self.nsga = NSGA2()
        self.resume_from = resume_from
        self.start_generation = 0
        if resume_from:
            checkpoint = load_checkpoint(resume_from)
            self.start_generation = int(checkpoint["generation"]) + 1
            self.seed = int(checkpoint["seed"])
            self.rng = random.Random(self.seed)
            self._resume_population = checkpoint["population"]
        else:
            self._resume_population = None

    def _mutate(self, node):
        if isinstance(node, Primitive):
            return Primitive(node.name, max(2, node.period + self.rng.randint(-5, 5)))
        if isinstance(node, Binary):
            return (
                Binary(node.op, self._mutate(node.left), node.right)
                if self.rng.random() < 0.5
                else Binary(node.op, node.left, self._mutate(node.right))
            )
        return node

    def _evaluate_and_save(self, generation, index, genome):
        train_result = run_backtest(self.train, genome, self.execution)
        test_result = run_backtest(self.test, genome, self.execution)
        complexity = genome.complexity()
        validation = {
            "walk_forward": walk_forward(
                self.feed, genome, n_splits=3, test_size=0.2, execution=self.execution
            ),
            "purged_cross_validation": purged_cross_validation(
                self.feed,
                genome,
                n_splits=3,
                purge_bars=genome.warmup(),
                embargo_bars=1,
                execution=self.execution,
            ),
        }
        robustness = robustness_suite(
            self.test,
            genome,
            self.execution,
            self.robustness_simulations,
            self.seed + generation * 1000 + index,
        )
        record = StrategyRecord(
            strategy_id(genome),
            generation,
            index,
            genome_to_dict(genome),
            expression(genome),
            complexity,
            fitness(train_result, complexity),
            fitness(test_result, complexity),
            self.execution.as_dict(),
            validation,
            robustness,
            "evaluated",
            [],
        )
        self.store.save(record)
        return Candidate(genome, fitness(train_result, complexity))

    def run(self):
        self.store.write_manifest(
            {
                "seed": self.seed,
                "population": self.population,
                "generations": self.generations,
                "execution": self.execution.as_dict(),
                "dataset": self.feed.manifest(),
                "quality": self.feed.quality_report().as_dict(),
            }
        )
        population = self._resume_population or [
            self._mutate(example_genome()) for _ in range(self.population)
        ]
        best = None
        saved = 0
        pareto_sizes = []
        for generation in range(self.start_generation, self.generations):
            candidates = [
                self._evaluate_and_save(generation, i, g)
                for i, g in enumerate(population)
            ]
            saved += len(candidates)
            candidates.sort(
                key=lambda c: (
                    c.scores["sharpe"],
                    -c.scores["max_drawdown"],
                    -c.scores["complexity"],
                ),
                reverse=True,
            )
            pareto_sizes.append(len(self.nsga.pareto_front(candidates)))
            save_checkpoint(
                self.store.path / f"checkpoint_g{generation}.json",
                population,
                generation,
                self.seed,
            )
            if best is None or candidates[0].scores["sharpe"] > best.scores["sharpe"]:
                best = candidates[0]
            elites = [
                c.genome
                for c in self.nsga.select(candidates, max(1, self.population // 4))
            ]
            population = elites + [
                self._mutate(self.rng.choice(elites))
                for _ in range(self.population - len(elites))
            ]
        self.store.finalize(
            {
                "strategies_saved": saved,
                "best_train_metrics": best.scores if best else {},
                "pareto_sizes": pareto_sizes,
                "run_id": self.store.run_id,
                "execution": self.execution.as_dict(),
            }
        )
        return best
