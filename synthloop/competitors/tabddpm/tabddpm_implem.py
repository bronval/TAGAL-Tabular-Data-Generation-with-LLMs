
## implem for tabddpm competitor
## original code from https://github.com/yandex-research/tab-ddpm/tree/main
## use implementation from synthcity https://github.com/vanderschaarlab/synthcity

from competitors.competitor import Competitor
from dataloader import Dataloader

from synthcity.plugins import Plugins
from synthcity.utils.serialization import save_to_file, load_from_file


class TabDDPM_implem(Competitor):

    def __init__(self, dataloader: Dataloader, load_model_path: str = None, save_model_path: str = None):
        self.dataloader = dataloader
        self.data = self.dataloader.data
        self.model = Plugins().get("ddpm")

        if load_model_path is not None:
            self.model = load_from_file(load_model_path)
        else:
            self.model.fit(self.data)
            if save_model_path is not None:
                save_to_file(save_model_path, self.model)

    
    def generate(self, n_examples: int):
        fake_data = self.model.generate(count=n_examples)
        fake_data = fake_data.data
        return fake_data

