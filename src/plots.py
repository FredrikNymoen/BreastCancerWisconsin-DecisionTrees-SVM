from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paths import FIGURES

def boxplot_features(X: pd.DataFrame, title: str, fname: Optional[str] = None, showfliers: bool = False):
    plt.figure(figsize=(14, 6))
    plt.boxplot(X.values, showfliers=showfliers)
    plt.xticks(ticks=np.arange(1, X.shape[1]+1), labels=X.columns, rotation=90)
    plt.title(title)
    plt.tight_layout()
    if fname:
        out = FIGURES / fname
        plt.savefig(out)
    plt.show()