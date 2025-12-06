# Re-export AutoDDG utils to allow users to import from ExtendDDG only
from autoddg.utils import (
    get_log_time,
    get_various_descriptions,
    load_prompts,
    log_print,
)

from .sampling import get_sample

__all__ = [
    "get_log_time",
    "get_various_descriptions",
    "log_print",
    "load_prompts",
    "get_sample",
]
