# Re-export AutoDDG modules to allow users to import from ExtendDDG only
from autoddg import GPTEvaluator

from .extendddg import ExtendDDG
from .profiling import DocumentationProfiler

__version__ = "0.1.0"

__all__ = ["ExtendDDG", "GPTEvaluator", "DocumentationProfiler"]
