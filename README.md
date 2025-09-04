# TAGAL: Tabular Data Generation using Agentic LLM Methods


This repository contains the code and resources used for TAGAL.
All of TAGAL source code, datasets, log files, and other resources will be made public on GitHub after the double-blind reviewing phase.


## Folders and files

- data: all the original datasets.
    - datasets_infos.py: contains detailed information about datasets to use them in TAGAL.
- evaluation_gen_data: results of the evaluation pipeline for each dataset and each group of test (refer to filename for the group)
- generated_examples: all the generated examples by TAGAL and competitors.
- outputs: *some* output logs from the different TAGAL methods and runs. Logs include prompts, few-shot examples, generated examples, and progress in the generation of synthetic examples.
- synthloop: **main folder**, contains the implementation of TAGAL
    - competitors: contains the source codes for all competitors.
    - dataloader.py: implements the dataloader class to represent the data in TAGAL.
    - evaluation.py: implements the evaluation pipeline used to realize the tables from the paper.
    - feedback_maker.py: implements the feedback part of TAGAL.
    - llm_handler.py: util file to handle all LLM calls, open or closed-source.
    - llms_infos.py: contains the information required to load or call the API of LLMs. Used snapshots for LLMs called through API are indicated there.
    - main_competitor.py: util file, runs the generation of data with all competitors.
    - main.py: launches TAGAL method with given parameters.
    - models.py: impletements the SynthLoop, ReducedLoop, and Prompt-Refine models of TAGAL.
    - prompt_templates.py: contains all the prompt templates used in TAGAL. Can be modified to include additional information if needed.
    - serializer.py: implements the serializer class used to transform the tabular examples into text.

- tex_tables: all the source files for the tables in the paper.
- README.md: this file.
- requirements.txt: list of Python packages needed to run TAGAL.
- utils_tex.py: code used to create the tex tables.
- deepseek_key.txt: contains the API key for DeepSeek API in the first line.
- openai_key.txt: contains the API key for OpenAI API in the first line.

Note: for the logs, only some of them are included due to the huge number of text line and the limited size for the supplementary material.


## Instructions to run the code

1) Install Python packages using `pip install -r requirements.txt`

2) Generate synthetic data with a TAGAL method by using

```python3 synthloop/main.py --dataset [dataset_name] --gen_method [TAGAL_model] --gen_model [LLM] --n_examples [value]```

where `dataset_name` is in [adult, bank, thyroid, german], `TAGAL_model` is in [synthloop, reduced, promptrefine], `LLM` is in [llama3.1, gpt4o, deepseek-v3], and `value` is the number of synthetic examples to generate.

A csv file will be named based on the given parameters and saved in the generated_examples folder.

Other parameters can be seen with `python3 synthloop/main.py --help`

Other LLMs and datasets can be added by respectively editing the *synthloop/llms_infos.py* and *data/datasets_infos.py* files by respecting the format.

3) [OPTIONAL] Run the evaluation of the generated data in folder *generated_examples* by using `python3 synthloop/evaluation.py`



