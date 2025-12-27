import pandas as pd
import hashlib

class DataVersioner:
    @staticmethod
    def compute_hash(df: pd.DataFrame) -> str:
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()
