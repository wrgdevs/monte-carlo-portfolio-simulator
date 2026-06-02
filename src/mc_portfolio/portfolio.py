import numpy as np
from typing import Tuple


def portfolio_values_from_price_paths(paths: np.ndarray, holdings: float = 1.0, cash: float = 0.0) -> np.ndarray:
    """Convert price paths to portfolio values given holdings (units of the asset) and cash.

    Returns same shape as paths with values = holdings * price + cash (cash is static here).
    """
    return holdings * paths + cash