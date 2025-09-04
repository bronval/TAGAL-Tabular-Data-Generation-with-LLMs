#################################################################################################
#
# Load the data given by name and create a dataloader object with the data and their information
#
#################################################################################################

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data.datasets_infos import datasets_infos


class Dataloader:

    def __init__(self, dname : str, parent_directory : str = "data", test_size: float = 0.2):
        """
        Creates a dataloader object with the data given by name

        Input:
        - dname: string, name of the data to load
        - parent_directory: string, path to the directory where the data is stored, default="DATASETS"
        """
        self.dname = dname
        
        try:
            data_info = datasets_infos[dname]
        except KeyError:
            raise ValueError(f"Dataset {dname} not found in datasets_infos. Possible datasets are: {datasets_infos.keys()}")
        
        try:
            self.data_full = pd.read_csv(f"{parent_directory}/{data_info['filename']}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {parent_directory}/{data_info['filename']} not found.\nMake sure the dataset has the correct filename and is in the correct directory.")
        
        # put train split in self.data
        self.data = self.data_full.sample(frac=1-test_size, random_state=42).copy(deep=True)
        self.data_test = self.data_full.drop(self.data.index).copy(deep=True)
        
        self.data.reset_index(drop=True, inplace=True)
        self.data_test.reset_index(drop=True, inplace=True)

        self.target = data_info["target"]
        self.task = data_info["task"]
        if self.task not in ["classification", "regression"]:
            raise ValueError(f"Task {self.task} not recognized. Possible tasks are: ['classification', 'regression']")

        if self.task == "classification":
            self.pos_label = data_info["pos_label"]
        else:
            self.pos_label = None

        self.dropped_features = data_info["drop_features"]
        if len(self.dropped_features) > 0:
            self.data = self.data.drop(columns=self.dropped_features)
        
        self.renamed_features = data_info["rename_features"]
        if len(self.renamed_features) > 0:
            self.data = self.data.rename(columns=self.renamed_features)
        
        self.cat_features = []
        for feat in data_info["categorical_features"]:
            if feat in self.data.columns and feat != self.target:
                self.cat_features.append(feat)
        
        self.num_features = [feat for feat in self.data.columns if feat not in self.cat_features + [self.target]]


    def get_features_information(self) -> str:
        """
        Returns a string with the full description of each feature.
        The description includes the feature name, type, categorical or numerical, unique values (with % of apparition) or range (with mean and std)
        Formatted as one line per feature.
        """
        desc = ""
        for col in self.data.columns:
            cat = "categorical" if col in self.cat_features else "numerical"
            if (cat == "categorical") or (col == self.target and self.task == "classification"):
                unique_values = self.data[col].value_counts(normalize=True)
                feat_info = ", ".join(f"{name} ({count*100:.2f}%)" for name, count in zip(unique_values.index, unique_values.values))
            elif (cat == "numerical") or (col == self.target and self.task == "regression"):
                values = self.data[col].dropna()
                feat_info = f"range from {values.min()} to {values.max()}, mean: {values.mean():.2f}, std: {values.std():.2f}, median: {values.median()}"
            else:
                raise ValueError(f"Feature {col} is not recognized as categorical or numerical. Check the information provided for this dataset {self.dname} in file datasets_infos.py")

            desc += f"- feature name: {col}, type: {self.data[col].dtype}, {cat}, {feat_info}\n"
        desc = desc.strip("\n")
        return desc


    def __str__(self):
        s = f"Dataset: {self.dname}\n"
        s += f"Number of examples: {self.data.shape[0]}\n"
        s += f"Number of features: {self.data.shape[1]}\n"
        s += f"Target: {self.target}\n"
        s += f"{len(self.cat_features)} categorical features: {self.cat_features}\n"
        s += f"{len(self.num_features)} numerical features: {self.num_features}\n"
        return s



