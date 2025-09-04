
## generation model based only on the stats of the dataset

import pandas as pd
import numpy as np

from synthloop.competitors.competitor import Competitor
from synthloop.dataloader import Dataloader

class StatGeneration(Competitor):

    def __init__(self,
                 dataloader: Dataloader,
                 ):
        self.dataloader = dataloader
        self.data : pd.DataFrame = self.dataloader.data
        self.cat_features = self.dataloader.cat_features.copy()
        self.num_features = self.dataloader.num_features.copy()

        if self.dataloader.task == "classification":
            self.cat_features.append(self.dataloader.target)
        else:
            self.num_features.append(self.dataloader.target)

        self.cat_stats = {}
        self.num_stats = {}

        # get the values and stats for each categorical feature
        for feat in self.cat_features:
            stats = self.data[feat].value_counts(normalize=True).to_dict()
            self.cat_stats[feat] = {"values": [k for k, _ in stats.items()],
                                    "counts": [v for _, v in stats.items()]}
        # get the stats for each numerical feature
        for feat in self.num_features:
            stats = self.data[feat].describe().to_dict()
            self.num_stats[feat] = {"mean": stats["mean"],
                                    "std": stats["std"],
                                    "min": stats["min"],
                                    "max": stats["max"]}
        

    def generate(self, n_examples: int):
        fake_examples = pd.DataFrame(columns=self.dataloader.data.columns)

        # get cat features
        for feat in self.cat_features:
            fake_examples[feat] = np.random.choice(self.cat_stats[feat]["values"],
                                                  size=n_examples,
                                                  p=self.cat_stats[feat]["counts"])
        # get num features
        for feat in self.num_features:
            fake_examples[feat] = np.random.normal(loc=self.num_stats[feat]["mean"],
                                                  scale=self.num_stats[feat]["std"],
                                                  size=n_examples)
            fake_examples[feat] = np.clip(fake_examples[feat], self.num_stats[feat]["min"], self.num_stats[feat]["max"])
            fake_examples[feat] = fake_examples[feat].astype(self.data[feat].dtype)

        return fake_examples

                                                      


