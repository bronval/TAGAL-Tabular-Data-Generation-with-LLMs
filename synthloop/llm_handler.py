

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json

from llms_infos import llms_infos



# convenient class to handle the different ways to use LLMs (openai, huggingface, ...)
class LLMHandler:

    def __init__(self,
                 llm_name: str):
        self.llm_name = llm_name
        try:
            llm_info = llms_infos[llm_name]
        except KeyError:
            raise ValueError(f"Unknown LLM name {llm_name}. Possible names are {llms_infos.keys()}.")
        self.llm = None
        self.tokenizer = None
        self.pipeline = None
        self.client = None
        self.source = llm_info["source"]
        self.need_tokenizer = llm_info["need_tokenizer"]

        # check the source and download the model if needed
        if self.source == "huggingface":
            self.llm = AutoModelForCausalLM.from_pretrained(llm_info["model_name"], device_map="cuda", trust_remote_code=True, torch_dtype="auto")
            if self.need_tokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(llm_info["model_name"], trust_remote_code=True)
            self.pipeline = pipeline("text-generation", model=self.llm, tokenizer=self.tokenizer, device_map="cuda")

        elif self.source == "openai":
            # read the openai api key from file 'openai_key.txt'
            with open("openai_key.txt", "r") as f:
                api_key = f.read().strip()
            self.client = OpenAI(api_key=api_key)
            self.model_name = llm_info["model_name"]

        elif self.source == "deepseek":
            # read the deepseek api key from file 'deepseek_key.txt'
            with open("deepseek_key.txt", "r") as f:
                api_key = f.read().strip()
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            self.model_name = llm_info["model_name"]
            
        else:
            sources = [item["source"] for item in llms_infos.values()]
            raise ValueError(f"Unknown source {self.source} for LLM. Possible sources are {sources}.")
    

    def generate_text(self,
                      prompt_history: list[dict],
                      max_new_tokens: int,
                      temperature: float,
                      do_sample: bool,
                      seed: int
                      ) -> str:
        if self.source == "huggingface":
            gen_args = {"do_sample": do_sample,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        #"seed": seed
                        }
            
            # print total number of tokens using self.tokenizer
            # total_tokens = 0
            # for item in prompt_history:
            #     total_tokens += len(self.tokenizer(item["content"])["input_ids"])
            # print(f"\n--DEBUG: prompt tokens: {total_tokens}", flush=True)
            print(f"\n--DEBUG: prompt history size: {len(prompt_history)}", flush=True)

            try:
                out = self.pipeline(prompt_history, **gen_args)[0]["generated_text"][-1]["content"]
            except Exception as e:
                print(out)
                raise e
            return out
        
        elif self.source == "openai":
            out = self.client.chat.completions.create(
                messages=prompt_history,
                model=self.model_name,
                max_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed
            ).choices[0].message.content
            return out

        elif self.source == "deepseek":
            counter = 0
            while counter < 3:
                try:
                    out = self.client.chat.completions.create(
                        messages=prompt_history,
                        model=self.model_name,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        seed=seed
                    ).choices[0].message.content
                    return out
                except json.decoder.JSONDecodeError as e:
                    print(e)
                    print(f"error in deepseek API, retrying... ({counter+1}/3)")
                    counter += 1
                    max_new_tokens = max_new_tokens // 2
                    print(f"reducing max_new_tokens to {max_new_tokens}")
            raise ValueError("Deepseek API did not return a valid response.")

        else:
            raise ValueError(f"Unknown source {self.source} for LLM. Possible sources are 'huggingface', 'openai'.")


