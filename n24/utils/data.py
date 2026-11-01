import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray


def load_datajson(
    data_dir: str | Path,
) -> tuple[np.ndarray | ExtensionArray, list[int], int]:
    """
    Load and preprocess the N24News dataset from JSON for the PyTorch dataset
    """
    ## Reading JSON
    with open(data_dir) as fp:
        data = json.load(fp)

    data_df = pd.DataFrame(data)
    label_dict = {cat: idx for idx, cat in enumerate(data_df["section"].unique())}

    ## Convert the labels into numeric:
    labels = [label_dict[section] for section in data_df["section"]]

    ## Get the article texts:
    texts = data_df["article"].values

    del data_df
    return texts, labels, len(label_dict)
