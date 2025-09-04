#################################################################################################
#
# Implements the classes used to obtain the feedback on the data during the generation loop
#
#################################################################################################

import pandas as pd

from llm_handler import LLMHandler
from serializer import Serializer
from prompt_templates import *
from dataloader import Dataloader



class Feedback:
    
    def get_feedback(self, gen_examples: pd.DataFrame, add_to_history: bool = False) -> str:
        raise NotImplementedError


class UserFeedback(Feedback):
    
    def __init__(self):
        ...

    def get_feedback(self, gen_examples: pd.DataFrame) -> str:
        ...


class LLMFeedback(Feedback):
    
    def __init__(self,
                 llm_name: str,
                 dataloader: Dataloader,
                 serializer: Serializer,
                 give_dataset_info: bool = True,
                 only_weakness: bool = False,
                 temperature: float = 0.7,
                 max_new_tokens: int = 2048,
                 do_sample: bool = True,
                 seed: int = 42
                 ):
        """
        Creates an LLMFeedback object to get feedback on the generated examples using an LLM

        Inputs:
        - llm_name: string, name of the LLM to use
        - dataloader: Dataloader, the dataloader object with the original real data
        - serializer: Serializer, the serializer object to transform the data to text
        - give_dataset_info: bool, whether to give the dataset information in the prompt, default=True
        - only_weakness: bool, whether the feedback should list only the weaknesses of the examined data (True) or both the strengths and weaknesses (False), default=False
        - temperature: float, temperature to use in the LLM, default=0.7
        - max_new_tokens: int, maximum number of tokens to generate, default=2048
        - do_sample: bool, whether to sample the tokens or not, default=True
        - seed: int, seed to use in the LLM, default=42
        """
        self.feedback_type = "llm"
        self.llm_name = llm_name
        self.dataloader = dataloader
        self.serializer = serializer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.seed = seed
        self.llm_handler = LLMHandler(llm_name)
        self.prompt_history = []
        self.give_dataset_info = give_dataset_info
        self.only_weakness = only_weakness

        if give_dataset_info:
            data_info = dataset_info_template.format(dataset_name=self.dataloader.dname,
                                                    n_examples=len(self.dataloader.data),
                                                    n_features=len(self.dataloader.data.columns),
                                                    features_info=self.dataloader.get_features_information())
            if only_weakness:
                system_prompt = system_prompt_feedback_weakness.format(dataset_info=data_info)
            else:
                system_prompt = system_prompt_feedback.format(dataset_info=data_info)
        else:
            if only_weakness:
                system_prompt = system_prompt_feedback_no_info_weakness
            else:
                system_prompt = system_prompt_feedback_no_info
        self.prompt_history.append({"role": "system", "content": system_prompt})

        print(f"\n--DEBUG: system prompt feedback:\n{system_prompt}", flush=True)
    

    def get_feedback(self, gen_examples: pd.DataFrame, add_to_history: bool = False, original_examples: pd.DataFrame = None) -> str:
        """
        Get the feedback on the generated examples using an LLM. Returns the feedback as a string

        Inputs:
        - gen_examples: pd.DataFrame, the generated examples
        - add_to_history: bool, whether to add the generated examples to the prompt history, default=False
        """
        # copy the prompt history if do not want to save the conversation
        prompt_history = [{"role": item["role"], "content": item["content"]} for item in self.prompt_history]

        # transform generated data to text
        gen_examples_str = self.serializer.serialize_examples(gen_examples)

        # get and format prompt
        if len(prompt_history) == 1 and original_examples is not None:
            if type(original_examples) == str:
                original_examples_str = original_examples
            else:
                original_examples_str = self.serializer.serialize_examples(original_examples)
            prompt = first_prompt_get_feedback.format(dataset_name=self.dataloader.dname,
                                                      generated_data=gen_examples_str,
                                                      original_data=original_examples_str)
        else:
            prompt = prompt_get_feedback.format(dataset_name=self.dataloader.dname,
                                                generated_data=gen_examples_str)
        prompt_history.append({"role": "user", "content": prompt})

        print(f"\n--DEBUG: prompt feedback:\n{prompt}", flush=True)

        # generate feedback
        feedback = self.llm_handler.generate_text(prompt_history,
                                                  max_new_tokens=self.max_new_tokens,
                                                  temperature=self.temperature,
                                                  do_sample=self.do_sample,
                                                  seed=self.seed
                                                  )

        if add_to_history:
            self.prompt_history = prompt_history
            self.prompt_history.append({"role": "assistant", "content": feedback})

        return feedback
    

    def reset_prompt_history(self):
        """
        Resets the prompt history for the feedback model.
        """
        self.prompt_history = []
        if self.give_dataset_info:
            data_info = dataset_info_template.format(dataset_name=self.dataloader.dname,
                                                        n_examples=len(self.dataloader.data),
                                                        n_features=len(self.dataloader.data.columns),
                                                        features_info=self.dataloader.get_features_information())
            system_prompt = system_prompt_feedback.format(dataset_info=data_info)
        else:
            system_prompt = system_prompt_feedback_no_info
        self.prompt_history.append({"role": "system", "content": system_prompt})






