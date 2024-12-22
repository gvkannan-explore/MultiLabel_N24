import pandas as pd
import json
from pathlib import Path
from typing import Union, Tuple, List


def load_datajson(data_dir: Union[str, Path]) -> Tuple[List, List, int]:
    """
    Load and preprocess the N24News dataset from JSON for the PyTorch dataset
    """
    ## Reading JSON
    with open(data_dir, "r") as fp:
        data = json.load(fp)

    data_df = pd.DataFrame(data)
    label_dict = {cat: idx for idx, cat in enumerate(data_df["section"].unique())}

    ## Convert the labels into numeric:
    labels = [label_dict[section] for section in data_df["section"]]

    ## Get the article texts:
    texts = data_df["article"].values

    del data_df
    return texts, labels, len(label_dict)