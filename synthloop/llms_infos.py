#################################################################################################
#
# Contains the required information to load a LLM model and its tokenizer by name
#
#################################################################################################

llms_infos = {

    ## FROM HUGGINGFACE ##

    "llama3.1": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "need_tokenizer": True,
        "source": "huggingface"
    },

    "deepseekr1": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "need_tokenizer": True,
        "source": "huggingface"
    },

    ## FROM OPENAI ##

    "gpt4o": {
        "model_name": "gpt-4o-2024-11-20",
        "need_tokenizer": False,
        "source": "openai"
    },

    "gpt4o-mini": {
        "model_name": "gpt-4o-mini-2024-07-18",
        "need_tokenizer": False,
        "source": "openai"
    },

    ## FROM DEEPSEEK ##

    "deepseek-v3": {
        "model_name": "deepseek-chat",
        "need_tokenizer": False,
        "source": "deepseek"
    }

    ## FROM GEMINI ##

    ## FORM CLAUDE ##



}



