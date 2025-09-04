#################################################################################################
#
# Implements the different models based on the feedback loop with LLMs to generated data
#
#################################################################################################


import pandas as pd

from feedback_maker import Feedback
from dataloader import Dataloader
from serializer import Serializer
from llm_handler import LLMHandler
from prompt_templates import *



class GenLoopModel:

    def generate(self, n_examples: int):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError



class SynthLoop(GenLoopModel):

    def __init__(self,
                 llm_name: str,
                 dataloader: Dataloader,
                 serializer: Serializer,
                 temperature: float = 0.7,
                 max_new_tokens: int = 8192,
                 do_sample: bool = True,
                 random_seed: int = 42
                 ):
        """
        Creates a SynthLoop model to generate data using a LLM and a feedback loop

        Inputs:
        - llm_name: str, the name of the LLM used to generate the data. It will be used to load both the LLM and the tokenizer
        - dataloader: Dataloader, the dataloader to use to load the data
        - serializer: Serializer, the serializer to use to serialize the data
        - temperature: float, the temperature to use for the LLM
        - max_new_tokens: int, the maximum number of tokens to generate at each iteration
        - random_seed: int, the random seed to use
        """
        self.llm_name = llm_name
        self.dataloader = dataloader
        self.serializer = serializer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.seed = random_seed
        self.llm_handler = LLMHandler(llm_name)
        self.prompt_history = []    # store the conversation history with the roles system, user, assistant


    def generate(self,
                 feedback: Feedback,
                 n_examples: int = 100,
                 n_iter_max: int = 5,
                 n_shots: int = 5,
                 give_dataset_info: bool = True,
                 few_shots_feedback: bool = False,
                 epic_few_shots: bool = False,
                 verbose: bool = False
                 ) -> pd.DataFrame:
        """
        Generates tabular data examples using the feedback loop process.

        Inputs:
        - feedback_maker: Feedback, the feedback maker to use to generate the feedback
        - n_examples: int, the number of examples to generate, default=100
        - n_iter_max: int, the maximum number of iterations of the feedback loop to perform
        - n_shots: int, the number of few-shots examples to put in the prompt
        - give_dataset_info: bool, whether to put a description for all features of the dataset in the prompt or not
        - few_shots_feedback: bool, whether to use the few-shots examples in the feedback or not, default=False
        - epic_few_shots: bool, whether to use the few-shots format from the EPIC paper or not, default=False
        - verbose: bool, whether to show the print with information or not, default=False
        """
        synth_data = pd.DataFrame(columns=self.dataloader.data.columns) # store the generated data

        i_full_run = 0

        while len(synth_data) < n_examples:
            
            if give_dataset_info:
            # init prompt history with system prompt
                data_info = dataset_info_template.format(dataset_name=self.dataloader.dname,
                                                         n_examples=len(self.dataloader.data),
                                                         n_features=len(self.dataloader.data.columns),
                                                         features_info=self.dataloader.get_features_information())
                # system_prompt = system_prompt_gen.format(dataset_info=data_info, format_example=self.serializer.get_format())
                system_prompt = system_prompt_gen.format(dataset_info=data_info)
            else:
                system_prompt = system_prompt_gen_no_info

            self.prompt_history = [{"role": "system", "content": system_prompt}]

            print(f"\n--DEBUG: system prompt generation:\n{system_prompt}", flush=True)
            
            # get the few-shot examples for the whole loop
            if epic_few_shots:
                # do not consider a system prompt and first user prompt contains only the few shots
                self.prompt_history = [{"role": "system", "content": "You must generate new data following the format of the given few shots."}]
                first_prompt = self.get_epic_few_shots(n_shots=n_shots, n_batches=4)
            else:
                few_shot_examples = pd.DataFrame()
                for label in self.dataloader.data[self.dataloader.target].unique():
                    few_shot_examples_label = self.dataloader.data[self.dataloader.data[self.dataloader.target] == label].sample(n_shots)
                    few_shot_examples = pd.concat([few_shot_examples, few_shot_examples_label], ignore_index=True)
                # EPIC -> better results if do not shuffle the classes
                few_shot_examples_str = self.serializer.serialize_examples(examples=few_shot_examples)

                # get first prompt and format it
                first_prompt = first_prompt_gen.format(dataset_name=self.dataloader.dname,
                                                       few_shots=few_shot_examples_str)
            
            self.prompt_history.append({"role": "user", "content": first_prompt})

            print(f"\n--DEBUG: first prompt generation:\n{first_prompt}", flush=True)

            if feedback.feedback_type == "llm":
                feedback.reset_prompt_history()

            iter_count = 0
            n_failed_iter = 0
            n_failed_iter_max = 3
            while iter_count < n_iter_max:

                # generate tabular data as text using the LLM
                synth_examples_str = self.llm_handler.generate_text(self.prompt_history,        # TODO: make sure we only get the generated examples here
                                                                    self.max_new_tokens,
                                                                    self.temperature,
                                                                    self.do_sample,
                                                                    self.seed)
                print(f"\n--DEBUG: generated examples (raw text):\n{synth_examples_str}", flush=True)

                # deserialize the generated data (get only valid examples)
                synth_examples = self.serializer.deserialize_examples(synth_examples_str, verbose=verbose)
                if synth_examples is None:
                    print(f"\n--DEBUG: WARNING: no valid examples generated, starting again iteration", flush=True)
                    n_failed_iter += 1
                    if n_failed_iter >= n_failed_iter_max:
                        print(f"\n--DEBUG: WARNING: too many failed iterations ({n_failed_iter}), restarting full round of generation with fresh history", flush=True)
                        break
                    continue
                
                print(f"\n--DEBUG: generated examples (deserialized):\n{synth_examples}", flush=True)

                # add the answer from the LLM in the history
                # do like that to consider only the valid examples
                n_samples = 30 if len(synth_examples) > 30 else len(synth_examples)
                sub_synth_examples = synth_examples.sample(n_samples)

                self.prompt_history.append({"role": "assistant", "content": self.serializer.serialize_examples(sub_synth_examples)})

                # get the feedback on the generated data
                if few_shots_feedback:
                    if epic_few_shots:
                        feedback_str = feedback.get_feedback(sub_synth_examples, add_to_history=True, original_examples=first_prompt)
                    else:
                        feedback_str = feedback.get_feedback(sub_synth_examples, add_to_history=True, original_examples=few_shot_examples)
                else:
                    feedback_str = feedback.get_feedback(sub_synth_examples, add_to_history=True)
                # feedback_str = feedback.get_feedback(synth_examples.sample(n_samples), add_to_history=True)
                prompt_feedback = prompt_give_feedback.format(dataset_name=self.dataloader.dname,
                                                              feedback=feedback_str)
                self.prompt_history.append({"role": "user", "content": prompt_feedback})

                print(f"\n--DEBUG: feedback:\n{prompt_feedback}", flush=True)

                synth_examples.to_csv(f"synthloop_{self.dataloader.dname}_{i_full_run}_iter_{iter_count}_{self.llm_name}_{self.serializer.serializer_type}_noInfo{not give_dataset_info}_weaknessOnly{feedback.only_weakness}_temp{self.temperature}_shots{n_shots}_shotsFb{few_shots_feedback}_epic{epic_few_shots}_feat{self.serializer.features_order_type}.csv", index=False)
                iter_count += 1

            if n_failed_iter >= n_failed_iter_max:
                continue

            synth_data = pd.concat([synth_data, synth_examples], ignore_index=True)

            print(f"\n--DEBUG: generated data count: {len(synth_data)} / {n_examples}", flush=True)

            i_full_run += 1
                    
        if len(synth_data) > n_examples:
            synth_data = synth_data.sample(n_examples)
        return synth_data


    def get_epic_few_shots(self, n_shots: int, n_batches: int = 4) -> str:
        # get number of few shots per batches
        n_shots_per_batch = n_shots // n_batches
        output = ""
        for i in range(n_batches):
            few_shot_examples = pd.DataFrame()
            for label in self.dataloader.data[self.dataloader.target].unique():
                few_shot_examples_label = self.dataloader.data[self.dataloader.data[self.dataloader.target] == label].sample(n_shots_per_batch)
                few_shot_examples = pd.concat([few_shot_examples, few_shot_examples_label], ignore_index=True)
            few_shot_examples_str = self.serializer.serialize_examples(examples=few_shot_examples)
            output += few_shot_examples_str + "\n"
        output += ",".join([feat for feat in self.serializer.features_order])
        return output




    def __str__(self):
        s = f"Model: SynthLoop for dataset {self.dataloader.dname}\n"
        s += f"Generator LLM: {self.llm_name}\n"
        s += f"Temperature: {self.temperature}, Max_new_tokens: {self.max_new_tokens}, Do_sample: {self.do_sample}, Seed: {self.seed}\n"
        s += f"Serializer:\n{str(self.serializer)}"
        return s



class ReducedLoop(GenLoopModel):

    def __init__(self,
                 llm_name: str,
                 dataloader: Dataloader,
                 serializer: Serializer,
                 temperature: float = 0.7,
                 max_new_tokens: int = 8192,
                 do_sample: bool = True,
                 random_seed: int = 42
                 ):
        """
        Creates a SynthLoop model to generate data using a LLM and a feedback loop

        Inputs:
        - llm_name: str, the name of the LLM used to generate the data. It will be used to load both the LLM and the tokenizer
        - dataloader: Dataloader, the dataloader to use to load the data
        - serializer: Serializer, the serializer to use to serialize the data
        - temperature: float, the temperature to use for the LLM
        - max_new_tokens: int, the maximum number of tokens to generate at each iteration
        - random_seed: int, the random seed to use
        """
        self.llm_name = llm_name
        self.dataloader = dataloader
        self.serializer = serializer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.seed = random_seed
        self.llm_handler = LLMHandler(llm_name)
        self.prompt_history = []    # store the conversation history with the roles system, user, assistant


    def generate(self,
                 feedback: Feedback,
                 n_examples: int = 100,
                 n_iter_max: int = 5,
                 n_shots: int = 5,
                 remove_duplicates: bool = True,
                 give_dataset_info: bool = True,
                 few_shots_feedback: bool = False,
                 verbose: bool = False
                 ) -> pd.DataFrame:
        """
        Generates tabular data examples using the feedback loop process.

        Inputs:
        - feedback_maker: Feedback, the feedback maker to use to generate the feedback
        - n_examples: int, the number of examples to generate, default=100
        - n_iter_max: int, the maximum number of iterations of the feedback loop to perform
        - n_shots: int, the number of few-shots examples to put in the prompt
        - remove_duplicates: bool: whether to remove duplicates in the generated data or not, default=True
        - give_dataset_info: bool, whether to put a description for all features of the dataset in the prompt or not
        - few_shots_feedback: bool, whether to use the few-shots examples in the feedback or not, default=False
        - verbose: bool, whether to show the print with information or not, default=False
        """
        synth_data = pd.DataFrame(columns=self.dataloader.data.columns) # store the generated data

        # init prompt history with system prompt
        if give_dataset_info:
            data_info = dataset_info_template.format(dataset_name=self.dataloader.dname,
                                                    n_examples=len(self.dataloader.data),
                                                    n_features=len(self.dataloader.data.columns),
                                                    features_info=self.dataloader.get_features_information())
            # system_prompt = system_prompt_gen.format(dataset_info=data_info, format_example=self.serializer.get_format())
            system_prompt = system_prompt_gen.format(dataset_info=data_info)
        else:
            system_prompt = system_prompt_gen_no_info
        self.prompt_history = [{"role": "system", "content": system_prompt}]

        # get the few-shot examples for the whole loop
        few_shot_examples = pd.DataFrame()
        for label in self.dataloader.data[self.dataloader.target].unique():
            few_shot_examples_label = self.dataloader.data[self.dataloader.data[self.dataloader.target] == label].sample(n_shots)
            few_shot_examples = pd.concat([few_shot_examples, few_shot_examples_label], ignore_index=True)
        few_shot_examples = few_shot_examples.sample(frac=1)
        few_shot_examples_str = self.serializer.serialize_examples(examples=few_shot_examples)

        # get first prompt and format it
        first_prompt = first_prompt_gen.format(dataset_name=self.dataloader.dname,
                                                few_shots=few_shot_examples_str)
        self.prompt_history.append({"role": "user", "content": first_prompt})

        print(len(self.prompt_history))
        print(system_prompt, flush=True)
        print(first_prompt, flush=True)

        if feedback.feedback_type == "llm":
            feedback.reset_prompt_history()

        iter_count = 0
        while iter_count < n_iter_max - 1:  # -1 because we want to generate the last data with the same prompt history
            synth_examples_str = self.llm_handler.generate_text(self.prompt_history,        # TODO: make sure we only get the generated examples here
                                                                self.max_new_tokens,
                                                                self.temperature,
                                                                self.do_sample,
                                                                self.seed)
            print(synth_examples_str, flush=True)
            synth_examples = self.serializer.deserialize_examples(synth_examples_str, verbose=verbose)
            if synth_examples is None:
                continue

            if remove_duplicates:
                synth_examples = synth_examples.drop_duplicates()
                
            # add the answer from the LLM in the history
            # do like that to consider only the valid examples
            n_samples = 30 if len(synth_examples) > 30 else len(synth_examples)
            sub_synth_examples = synth_examples.sample(n_samples)

            self.prompt_history.append({"role": "assistant", "content": self.serializer.serialize_examples(sub_synth_examples)})

            # get the feedback on the generated data
            if few_shots_feedback:
                feedback_str = feedback.get_feedback(sub_synth_examples, add_to_history=True, original_examples=few_shot_examples)
            else:
                feedback_str = feedback.get_feedback(sub_synth_examples, add_to_history=True)
            # feedback_str = feedback.get_feedback(synth_examples.sample(n_samples), add_to_history=True)
            prompt_feedback = prompt_give_feedback.format(dataset_name=self.dataloader.dname,
                                                        feedback=feedback_str)
            self.prompt_history.append({"role": "user", "content": prompt_feedback})

            synth_examples.to_csv(f"gen_data_{self.dataloader.dname}_iter_{iter_count}_{self.llm_name}_reduced.csv", index=False)

            iter_count += 1
        
        # generate the data for the last iteration using the same prompt history (and do not update it)
        synth_data = pd.DataFrame(columns=self.dataloader.data.columns) # store the generated data
        while len(synth_data) < n_examples:
            synth_examples_str = self.llm_handler.generate_text(self.prompt_history,        # TODO: make sure we only get the generated examples here
                                                                self.max_new_tokens,
                                                                self.temperature,
                                                                self.do_sample,
                                                                self.seed)
            print(synth_examples_str, flush=True)
            synth_examples = self.serializer.deserialize_examples(synth_examples_str, verbose=verbose)
            if synth_examples is None:
                continue

            synth_data = pd.concat([synth_data, synth_examples], ignore_index=True)

            if remove_duplicates:
                synth_data = synth_data.drop_duplicates()

            if verbose:
                print(f"generated data: {len(synth_data)} / {n_examples}", flush=True)

        if len(synth_data) > n_examples:
            synth_data = synth_data.sample(n_examples)
        return synth_data


    def get_epic_few_shots(self, n_shots: int, n_batches: int = 4) -> str:
        # get number of few shots per batches
        n_shots_per_batch = n_shots // n_batches
        output = ""
        for i in range(n_batches):
            few_shot_examples = pd.DataFrame()
            for label in self.dataloader.data[self.dataloader.target].unique():
                few_shot_examples_label = self.dataloader.data[self.dataloader.data[self.dataloader.target] == label].sample(n_shots_per_batch)
                few_shot_examples = pd.concat([few_shot_examples, few_shot_examples_label], ignore_index=True)
            few_shot_examples_str = self.serializer.serialize_examples(examples=few_shot_examples)
            output += few_shot_examples_str + "\n"
        output += ",".join([feat for feat in self.serializer.features_order])
        return output



class PromptRefine(GenLoopModel):
    
    def __init__(self,
                 llm_name: str,
                 dataloader: Dataloader,
                 serializer: Serializer,
                 temperature: float = 0.7,
                 max_new_tokens: int = 8192,
                 do_sample: bool = True,
                 random_seed: int = 42
                 ):
        """
        Creates a PromptRefine model to generate data using a LLM and a feedback loop

        Inputs:
        - llm_name: str, the name of the LLM used to generate the data. It will be used to load both the LLM and the tokenizer
        - dataloader: Dataloader, the dataloader to use to load the data
        - serializer: Serializer, the serializer to use to serialize the data
        - temperature: float, the temperature to use for the LLM
        - max_new_tokens: int, the maximum number of tokens to generate at each iteration
        - random_seed: int, the random seed to use
        """
        self.llm_name = llm_name
        self.dataloader = dataloader
        self.serializer = serializer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.seed = random_seed
        self.llm_handler = LLMHandler(llm_name)
        self.prompt_history = []    # store the conversation history with the roles system, user, assistant
        self.__refined = False # whether or not the prompt has already been refined
        self.refined_prompt = ""
        self.gen_history_str = ""
    

    def __first_loop(self,
                     feedback: Feedback,
                     n_iter_max: int = 5,
                     n_shots: int = 5,
                     give_dataset_info: bool = True,
                     few_shots_feedback: bool = False,
                     epic_few_shots: bool = False,
                     verbose: bool = False):
        if give_dataset_info:
        # init prompt history with system prompt
            data_info = dataset_info_template.format(dataset_name=self.dataloader.dname,
                                                     n_examples=len(self.dataloader.data),
                                                     n_features=len(self.dataloader.data.columns),
                                                     features_info=self.dataloader.get_features_information())
            # system_prompt = system_prompt_gen.format(dataset_info=data_info, format_example=self.serializer.get_format())
            system_prompt = system_prompt_gen.format(dataset_info=data_info)
        else:
            system_prompt = system_prompt_gen_no_info

        self.prompt_history = [{"role": "system", "content": system_prompt}]

        print(f"\n--DEBUG: system prompt generation:\n{system_prompt}", flush=True)

        # get the few-shot examples for the whole loop
        if epic_few_shots:
            # do not consider a system prompt and first user prompt contains only the few shots
            self.prompt_history = [{"role": "system", "content": "You must generate new data following the format of the given few shots."}]
            first_prompt = self.get_epic_few_shots(n_shots=n_shots, n_batches=4)
        else:
            few_shot_examples = pd.DataFrame()
            for label in self.dataloader.data[self.dataloader.target].unique():
                few_shot_examples_label = self.dataloader.data[self.dataloader.data[self.dataloader.target] == label].sample(n_shots)
                few_shot_examples = pd.concat([few_shot_examples, few_shot_examples_label], ignore_index=True)
            # EPIC -> better results if do not shuffle the classes
            few_shot_examples_str = self.serializer.serialize_examples(examples=few_shot_examples)

            # get first prompt and format it
            first_prompt = first_prompt_gen.format(dataset_name=self.dataloader.dname,
                                                    few_shots=few_shot_examples_str)
        
        self.prompt_history.append({"role": "user", "content": first_prompt})

        if feedback.feedback_type == "llm":
            feedback.reset_prompt_history()

        iter_count = 0
        n_failed_iter = 0
        n_failed_iter_max = 3
        while iter_count < n_iter_max:
            # generate tabular data as text using the LLM
            synth_examples_str = self.llm_handler.generate_text(self.prompt_history,        # TODO: make sure we only get the generated examples here
                                                                self.max_new_tokens,
                                                                self.temperature,
                                                                self.do_sample,
                                                                self.seed)
            print(f"\n--DEBUG: generated examples (raw text):\n{synth_examples_str}", flush=True)

            # deserialize the generated data (get only valid examples)
            synth_examples = self.serializer.deserialize_examples(synth_examples_str, verbose=verbose)
            if synth_examples is None:
                print(f"\n--DEBUG: WARNING: no valid examples generated, starting again iteration", flush=True)
                n_failed_iter += 1
                if n_failed_iter >= n_failed_iter_max:
                    print(f"\n--DEBUG: WARNING: too many failed iterations ({n_failed_iter}), restarting full round of generation with fresh history", flush=True)
                    iter_count = 0
                continue
            
            print(f"\n--DEBUG: generated examples (deserialized):\n{synth_examples}", flush=True)

            # add the answer from the LLM in the history
            # do like that to consider only the valid examples
            n_samples = 30 if len(synth_examples) > 30 else len(synth_examples)
            sub_synth_examples = synth_examples.sample(n_samples)

            self.prompt_history.append({"role": "assistant", "content": self.serializer.serialize_examples(sub_synth_examples)})

            # get the feedback on the generated data
            if few_shots_feedback:
                if epic_few_shots:
                    feedback_str = feedback.get_feedback(sub_synth_examples, add_to_history=True, original_examples=first_prompt)
                else:
                    feedback_str = feedback.get_feedback(sub_synth_examples, add_to_history=True, original_examples=few_shot_examples)
            else:
                feedback_str = feedback.get_feedback(sub_synth_examples, add_to_history=True)
            # feedback_str = feedback.get_feedback(synth_examples.sample(n_samples), add_to_history=True)
            prompt_feedback = prompt_give_feedback.format(dataset_name=self.dataloader.dname,
                                                          feedback=feedback_str)
            self.prompt_history.append({"role": "user", "content": prompt_feedback})

            print(f"\n--DEBUG: feedback:\n{prompt_feedback}", flush=True)

            iter_count += 1

        # --- HERE: first loop is over --
        # ask LLM to generate a summary prompt to be used to generate new data
        self.__get_summary_prompt(verbose=verbose)
        

    def __get_summary_prompt(self, verbose: bool = False):
        # get system prompt for summary
        system_prompt = system_prompt_summary
        prompt_history_summary = [{"role": "system", "content": system_prompt}]

        # transform history to a single text
        if self.gen_history_str == "":
            full_text = ""
            for pr in self.prompt_history:
                full_text += f"{pr['role'].capitalize()}: {pr['content']}\n\n"
            self.gen_history_str = full_text
        prompt_summary = prompt_get_summary.format(dataset_name=self.dataloader.dname,
                                                   conversation=self.gen_history_str)
        prompt_history_summary.append({"role": "user", "content": prompt_summary})
        self.prompt_history = prompt_history_summary

        print(f"\n--DEBUG: system prompt for summary generation:\n{system_prompt}", flush=True)
        # print(f"\n--DEBUG: user prompt for summary generation:\n{prompt_summary}", flush=True)

        # generate the prompt summary
        self.refined_prompt = self.llm_handler.generate_text(self.prompt_history,
                                                             self.max_new_tokens,
                                                             self.temperature,
                                                             self.do_sample,
                                                             self.seed)
        self.__refined = True

        if verbose:
            print(f"Refined prompt:\n{self.refined_prompt}", flush=True)

    
    def generate(self,
                 feedback: Feedback,
                 n_examples: int = 100,
                 n_iter_max: int = 5,
                 n_shots: int = 5,
                 give_dataset_info: bool = True,
                 few_shots_feedback: bool = False,
                 epic_few_shots: bool = False,
                 verbose: bool = False
                 ) -> pd.DataFrame:
        """
        Generates tabular data examples using the feedback loop process.

        Inputs:
        - feedback_maker: Feedback, the feedback maker to use to generate the feedback
        - n_examples: int, the number of examples to generate, default=100
        - n_iter_max: int, the maximum number of iterations of the feedback loop to perform
        - n_shots: int, the number of few-shots examples to put in the prompt
        - give_dataset_info: bool, whether to put a description for all features of the dataset in the prompt or not
        - few_shots_feedback: bool, whether to use the few-shots examples in the feedback or not, default=False
        - epic_few_shots: bool, whether to use the few-shots format from the EPIC paper or not, default=False
        - verbose: bool, whether to show the print with information or not, default=False
        """
        if not self.__refined:
            self.__first_loop(feedback, n_iter_max, n_shots, give_dataset_info, few_shots_feedback, epic_few_shots, verbose)
        
        synth_data = pd.DataFrame(columns=self.dataloader.data.columns)
        n_fail = 0
        n_fail_max = 10
        while len(synth_data) < n_examples:

            # try putting few shot examples in the prompt if there is the flag 'few_shots'
            few_shot_examples = pd.DataFrame()
            for label in self.dataloader.data[self.dataloader.target].unique():
                few_shot_examples_label = self.dataloader.data[self.dataloader.data[self.dataloader.target] == label].sample(n_shots)
                few_shot_examples = pd.concat([few_shot_examples, few_shot_examples_label], ignore_index=True)
            few_shot_examples_str = self.serializer.serialize_examples(examples=few_shot_examples)

            try:
                prompt = self.refined_prompt.format(few_shots=few_shot_examples_str)
            except KeyError:
                prompt = self.refined_prompt

            self.prompt_history = [{"role" : "system", "content": "You must only generate data following the format and indications given in the prompt. Do not generare anything else."},
                                   {"role": "user", "content": prompt}]

            # print(f"\n--DEBUG: user prompt generation:\n{prompt}", flush=True)

            synth_examples_str = self.llm_handler.generate_text(self.prompt_history,
                                                                self.max_new_tokens,
                                                                self.temperature,
                                                                self.do_sample,
                                                                self.seed)
            
            print(f"\n--DEBUG: generated examples (raw text):\n{synth_examples_str}", flush=True)
            
            synth_examples = self.serializer.deserialize_examples(synth_examples_str, verbose=verbose)
            if synth_examples is None:
                n_fail += 1
                if n_fail >= n_fail_max:
                    # recompute a summary prompt
                    self.__get_summary_prompt(verbose=verbose)
                    n_fail = 0
                continue
            else:
                n_fail = 0

            synth_data = pd.concat([synth_data, synth_examples], ignore_index=True)
            # synth_data = synth_data.drop_duplicates()

            if verbose:
                print(f"generated data count: {len(synth_data)} / {n_examples}", flush=True)

        if len(synth_data) > n_examples:
            synth_data = synth_data.sample(n_examples)
        
        return synth_data
            

    def get_epic_few_shots(self, n_shots: int, n_batches: int = 4) -> str:
        # get number of few shots per batches
        n_shots_per_batch = n_shots // n_batches
        output = ""
        for i in range(n_batches):
            few_shot_examples = pd.DataFrame()
            for label in self.dataloader.data[self.dataloader.target].unique():
                few_shot_examples_label = self.dataloader.data[self.dataloader.data[self.dataloader.target] == label].sample(n_shots_per_batch)
                few_shot_examples = pd.concat([few_shot_examples, few_shot_examples_label], ignore_index=True)
            few_shot_examples_str = self.serializer.serialize_examples(examples=few_shot_examples)
            output += few_shot_examples_str + "\n"
        output += ",".join([feat for feat in self.serializer.features_order])
        return output