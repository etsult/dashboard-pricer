# pricer/models/factory.py

from typing import Type, Any

from pricer.models.black76 import Black76Option
# from pricer.models.blackscholes import BlackScholesOption
# from pricer.models.bachelier import BachelierOption


class OptionFactory:
    """
    Generic option creation factory.
    Dynamically instantiates the right pricing model.
    """

    MODEL_MAP = {
        "black76": Black76Option,
        # "bs": BlackScholesOption,
        # "bachelier": BachelierOption,
    }

    @classmethod
    def create(cls, model: str, **kwargs: Any):
        """
        Create a new option using the selected pricing model.
        kwargs must include: F or S, K, r, sigma, expiry, option_type, valuation_date.
        """
        if model not in cls.MODEL_MAP:
            raise ValueError(f"Unknown model: {model}")

        option_cls: Type = cls.MODEL_MAP[model]
        return option_cls(**kwargs)