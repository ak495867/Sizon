import numpy as np
from sklearn.mixture import GaussianMixture
from sizon.data.data import DataFeed

class RegimeClassifier:
    def __init__(self, n_components: int = 2, window: int = 20, random_state: int = 42):
        self.n_components = n_components
        self.window = window
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def _extract_features(self, feed: DataFeed) -> np.ndarray:
        closes = np.array(feed.column("close"), dtype=float)
        
        # Calculate returns
        returns = np.zeros_like(closes)
        returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        # Replace nan or inf
        returns = np.nan_to_num(returns)
        
        # Calculate rolling volatility
        volatility = np.zeros_like(closes)
        for i in range(len(closes)):
            start_idx = max(0, i - self.window + 1)
            if i > 0:
                volatility[i] = np.std(returns[start_idx:i+1])
            else:
                volatility[i] = 0.0
                
        # Stack features: Returns and Volatility
        features = np.column_stack((returns, volatility))
        return features

    def fit(self, feed: DataFeed):
        features = self._extract_features(feed)
        self.gmm.fit(features)
        self.is_fitted = True

    def predict(self, feed: DataFeed) -> list[int]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        features = self._extract_features(feed)
        labels = self.gmm.predict(features)
        return labels.tolist()

    def describe_regimes(self) -> dict[int, str]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        
        descriptions = {}
        means = self.gmm.means_  # Shape: (n_components, 2) [return, vol]
        
        # Analyze relative return and vol
        vol_median = np.median(means[:, 1])
        
        for i in range(self.n_components):
            ret, vol = means[i]
            
            ret_desc = "Positive Return" if ret >= 0 else "Negative Return"
            vol_desc = "High Vol" if vol >= vol_median else "Low Vol"
            
            descriptions[i] = f"Regime {i}: {vol_desc}, {ret_desc}"
            
        return descriptions
