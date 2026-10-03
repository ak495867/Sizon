from __future__ import annotations
import math
from typing import List

import lightgbm as lgb

from sizon.core.expression import Context, Node
from sizon.data.data import DataFeed

class MetaLearner:
    def __init__(self, **lgbm_kwargs):
        """Initialize MetaLearner with LGBMRegressor arguments."""
        self.model = lgb.LGBMRegressor(**lgbm_kwargs)
        self.genomes: List[Node] = []

    def _get_context(self, feed: DataFeed) -> Context:
        """Create a Context object for expression evaluation from a DataFeed."""
        return Context({k: feed.column(k) for k in ("open", "high", "low", "close", "volume")})

    def fit(self, feed: DataFeed, genomes: List[Node]) -> "MetaLearner":
        """
        Fit the MetaLearner using the genomes as features.
        
        Args:
            feed: The historical DataFeed.
            genomes: A list of GP Nodes to be used as feature generators.
        """
        self.genomes = genomes
        ctx = self._get_context(feed)
        
        # Evaluate genomes -> features
        features_by_genome = [g.evaluate(ctx) for g in genomes]
        
        n_bars = len(feed.rows)
        n_genomes = len(genomes)
        
        # Assemble feature matrix X
        X = []
        for t in range(n_bars):
            row_features = [features_by_genome[g_idx][t] for g_idx in range(n_genomes)]
            X.append(row_features)
            
        # Compute target Y: forward log returns
        close = feed.column("close")
        Y = []
        for t in range(n_bars - 1):
            if close[t] > 0 and close[t+1] > 0:
                Y.append(math.log(close[t+1] / close[t]))
            else:
                Y.append(math.nan)
        Y.append(math.nan) # Last bar has no forward return
        
        # Filter valid rows (where Y is not NaN)
        X_train = []
        Y_train = []
        for t in range(n_bars):
            if not math.isnan(Y[t]):
                X_train.append(X[t])
                Y_train.append(Y[t])
                
        # Fit model
        self.model.fit(X_train, Y_train)
        return self

    def predict(self, feed: DataFeed) -> List[float]:
        """
        Predict the target for the given feed.
        
        Args:
            feed: The DataFeed to predict on.
            
        Returns:
            List of float predictions for each bar.
        """
        if not self.genomes:
            raise ValueError("MetaLearner must be fitted before predicting.")
            
        ctx = self._get_context(feed)
        features_by_genome = [g.evaluate(ctx) for g in self.genomes]
        
        n_bars = len(feed.rows)
        n_genomes = len(self.genomes)
        
        X = []
        for t in range(n_bars):
            row_features = [features_by_genome[g_idx][t] for g_idx in range(n_genomes)]
            X.append(row_features)
            
        preds = self.model.predict(X)
        return [float(p) for p in preds]
