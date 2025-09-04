
## Use GReaT to generate tabular data
## from https://github.com/kathrinse/be_great


from be_great import GReaT

from dataloader import Dataloader
from competitors.competitor import Competitor


class Great(Competitor):

    def __init__(self, dataloader: Dataloader,
                 llm: str = "distilgpt2",
                 batch_size: int = 32,
                 epochs: int = 10,
                 fp16: bool = True,
                 load_model_path: str = None,
                 save_model_path: str = None):
        self.dataloader = dataloader
        self.llm = llm
        self.batch_size = batch_size
        self.epochs = epochs
        self.fp16 = fp16

        if load_model_path is not None:
            self.great = GReaT.load_from_dir(load_model_path)
        else:
            self.great = GReaT(llm=llm, batch_size=batch_size, epochs=epochs, fp16=fp16, use_cpu=False, save_steps=20000)
            self.great.fit(self.dataloader.data)
            if save_model_path is not None:
                self.great.save(save_model_path)


    def generate(self, n_examples: int):
        fake_examples = self.great.sample(n_samples=n_examples, max_length=1000)
        return fake_examples



