# Re-export AutoDDG utils to allow users to import from ExtendDDG only
from autoddg.utils import (
    get_log_time,
    get_various_descriptions,
    log_print,
)

from .prompts import load_prompts
from .sampling import get_sample

__all__ = [
    "get_log_time",
    "get_various_descriptions",
    "log_print",
    "load_prompts",
    "get_sample",
]
