from sizon.simulation.orderbook import OrderBookSnapshot, BookLevel
import math

class SyntheticBookGenerator:
    def __init__(self, levels: int = 10, decay_rate: float = 0.5):
        self.levels = levels
        self.decay_rate = decay_rate

    def generate(self, high: float, low: float, close: float, volume: float) -> OrderBookSnapshot:
        if self.levels <= 0:
            return OrderBookSnapshot(bids=(), asks=())

        # Distribute half volume to bids, half to asks
        side_volume = volume / 2.0

        # Calculate decay weights
        weights = [math.exp(-self.decay_rate * i) for i in range(self.levels)]
        total_weight = sum(weights)
        
        bids = []
        asks = []
        
        # Calculate price steps
        bid_step = (close - low) / self.levels if self.levels > 0 and close > low else 0.01
        ask_step = (high - close) / self.levels if self.levels > 0 and high > close else 0.01

        for i in range(self.levels):
            # Volume for this level based on exponential decay
            level_vol = side_volume * (weights[i] / total_weight) if total_weight > 0 else 0
            
            # Prices moving away from close
            # We ensure we don't place bids/asks exactly at close by offsetting by one step or a fraction
            bid_price = close - bid_step * (i + 1)
            ask_price = close + ask_step * (i + 1)
            
            bids.append(BookLevel(price=bid_price, quantity=level_vol))
            asks.append(BookLevel(price=ask_price, quantity=level_vol))
        
        return OrderBookSnapshot(bids=tuple(bids), asks=tuple(asks))
