from dataclasses import dataclass
import pandas as pd
from pipeline.vector_store import VectorStoreManager


@dataclass
class ToolContext:
    """Injected into every tool factory. Replaces module-level globals."""
    df: pd.DataFrame
    vsm: VectorStoreManager
