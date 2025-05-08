# -*- coding: utf-8 -*-

import pandas as pd

class WaveformLoader:

    @staticmethod
    def load_single(path: str):
        return pd.read_parquet(path)
