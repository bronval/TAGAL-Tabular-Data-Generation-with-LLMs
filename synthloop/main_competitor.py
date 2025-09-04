
from competitors.competitor import Competitor
from competitors.great.great import Great
from competitors.epic.epic import EPIC
from competitors.ctgan_implem.ctgan_implem import CTGAN_implem
from competitors.tabula.tabula_implem import Tabula_implem
from competitors.tabddpm.tabddpm_implem import TabDDPM_implem

from dataloader import Dataloader

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Competitors")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()


    dname = args.dataset
    n_gen = 2500

    MAX_DATA_LIMIT = 40

    print(f"dname: {dname}, n_gen: {n_gen}", flush=True)


    ## run great exp
    print("Running great...", flush=True)
    dataloader = Dataloader(dname, parent_directory="data")
    dataloader.data = dataloader.data.sample(n=MAX_DATA_LIMIT)

    great = Great(dataloader, epochs=50) # save_model_path=f"great_{dname}"
    # great = Great(dataloader, epochs=25, load_model_path=f"great_{dname}")

    fake_data = great.generate(n_gen)
    fake_data.to_csv(f"generated_examples/great_gen_data_{dname}_limit{MAX_DATA_LIMIT}.csv", index=False)
    print("saved great generated data!", flush=True)


    ################################

    ## run epic
    print("Running epic...", flush=True)
    dataloader = Dataloader(dname, parent_directory="data")
    dataloader.data = dataloader.data.sample(n=MAX_DATA_LIMIT)

    epic = EPIC("llama3.1", dataloader)
    fake_data = epic.generate(n_gen)
    fake_data.to_csv(f"generated_examples/epic_gen_data_{dname}_limit{MAX_DATA_LIMIT}.csv", index=False)
    print("saved epic generated data!", flush=True)


    ################################

    ## run ctgan
    print("Running ctgan...", flush=True)
    dataloader = Dataloader(dname, parent_directory="data")
    dataloader.data = dataloader.data.sample(n=MAX_DATA_LIMIT)

    model = CTGAN_implem(dataloader, epochs=50)
    fake_data = model.generate(n_gen)
    fake_data.to_csv(f"generated_examples/ctgan_gen_data_{dname}_limit{MAX_DATA_LIMIT}.csv", index=False)
    print("saved ctgan generated data!", flush=True)


    ################################

    ## run tabddpm
    print("Running tabddpm...", flush=True)
    dataloader = Dataloader(dname, parent_directory="data")
    dataloader.data = dataloader.data.sample(n=MAX_DATA_LIMIT)

    model = TabDDPM_implem(dataloader)  # save_model_path="tabddpm_adult.pkl"
    fake_data = model.generate(n_gen)
    fake_data.to_csv(f"generated_examples/tabddpm_gen_data_{dname}_limit{MAX_DATA_LIMIT}.csv", index=False)
    print("saved tabddpm generated data!", flush=True)


    ################################

    ## run tabula
    print("Running tabula...", flush=True)
    dataloader = Dataloader(dname, parent_directory="data")
    dataloader.data = dataloader.data.sample(n=MAX_DATA_LIMIT)

    model = Tabula_implem(dataloader, llm="distilgpt2", epochs=50, batch_size=8)  # save_model_path=f"tabula_{dname}.pt"
    # model = Tabula_implem(dataloader, llm="distilgpt2", epochs=50, batch_size=8, load_model_path=f"tabula_{dname}.pt")
    fake_data = model.generate(n_gen)
    fake_data.to_csv(f"generated_examples/tabula_gen_data_{dname}_limit{MAX_DATA_LIMIT}.csv", index=False)
    print("saved tabula generated data!", flush=True)



