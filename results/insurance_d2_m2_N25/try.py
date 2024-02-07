from functools import cached_property

import pandas as pd


class DataLoader:
    def __init__(self, path):
        self.path = path

    @cached_property
    def dataset(self):
        # load the dataset here
        # this will only be executed once when the dataset property is first accessed
        print("loading the dataset")
        return pd.read_csv(self.path)
