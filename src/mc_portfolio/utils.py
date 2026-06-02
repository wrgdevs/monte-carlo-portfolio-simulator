import re
from typing import Optional

ticker_re = re.compile(r"^[A-Za-z0-9\.\-]{1,10}$")


def validate_ticker(t: str) -> bool:
    if t is None:
        return False
    t = t.strip()
    return bool(ticker_re.match(t))


def ensure_dir(path: str):
    import os

    os.makedirs(path, exist_ok=True)