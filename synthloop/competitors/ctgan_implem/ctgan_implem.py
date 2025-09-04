
## implem CTGAN as a method to compare to

from competitors.competitor import Competitor
from ctgan import CTGAN

# https://github.com/sdv-dev/CTGAN


class CTGAN_implem(Competitor):

    def __init__(self, dataloader, epochs=10):
        self.dataloader = dataloader
        self.epochs = epochs
        self.ctgan = CTGAN(epochs=self.epochs)

        cat_feats = self.dataloader.cat_features
        if self.dataloader.task == "classification":
            cat_feats = cat_feats + [self.dataloader.target]
        self.data = self.dataloader.data

        self.ctgan.fit(self.data, cat_feats)

    def generate(self, n_examples):
        fake_data = self.ctgan.sample(n_examples)
        return fake_data
