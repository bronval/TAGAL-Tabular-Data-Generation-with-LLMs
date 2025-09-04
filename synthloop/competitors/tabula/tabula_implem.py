## implem for TABULA competitor
## from https://github.com/zhao-zilong/Tabula/tree/main

from competitors.competitor import Competitor
from dataloader import Dataloader
from competitors.tabula.tabula import Tabula

import torch
import numpy as np


class Tabula_implem(Competitor):

    def __init__(self, dataloader: Dataloader,
                 llm: str ="distilgpt2",
                 batch_size: int = 32,
                 epochs: int = 400,
                 load_model_path: str = None,
                 save_model_path: str = None
                 ):
        self.dataloader = dataloader
        cat_feats = self.dataloader.cat_features
        # if self.dataloader.task == "classification":
        #     cat_feats.append(self.dataloader.target)
        self.cat_features = cat_feats
        self.data = self.dataloader.data
        self.llm = llm
        self.batch_size = batch_size
        self.epochs = epochs

        self.tabula = Tabula(llm=self.llm, batch_size=self.batch_size, epochs=self.epochs, categorical_columns=self.cat_features,
                            experiment_dir=f"tabula_{self.dataloader.dname}")

        if load_model_path is None:
            self.tabula.fit(self.data)
            if save_model_path is not None:
                torch.save(self.tabula.model.state_dict(), save_model_path)
        else:
            self.tabula.model.load_state_dict(torch.load(load_model_path), strict=False)
            self.tabula.columns = self.data.columns.to_list()
            self.tabula.num_cols = self.data.select_dtypes(include=np.number).columns.to_list()
            self.tabula.encode_categorical_column(self.data)


    def generate(self, n_examples):
        fake_data = self.tabula.sample(n_samples=n_examples, max_length=4096)
        return fake_data
