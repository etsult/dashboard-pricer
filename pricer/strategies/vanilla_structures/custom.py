# pricer/strategies/vanilla_structures/custom.py

from __future__ import annotations
from pricer.strategies.strategy import Strategy


class CustomStrategy(Strategy):
    def __init__(self, name: str = "Custom Strategy"):
        super().__init__(name)
