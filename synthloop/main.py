#################################################################################################
#
# Main file to launch the code with the arguments given as input
#
#################################################################################################

import argparse
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.datasets_infos import datasets_infos
from llms_infos import llms_infos
from dataloader import Dataloader
from llm_handler import LLMHandler
from serializer import Serializer, SentenceSerializer, CSVSerializer
from models import SynthLoop, PromptRefine, ReducedLoop
from feedback_maker import LLMFeedback, UserFeedback


def parse_args():
    parser = argparse.ArgumentParser(description="SynthLoop")
    parser.add_argument("--verbose", action="store_true", help="Whether to show the print statements")

    # dataset and models
    all_dnames = datasets_infos.keys()
    all_llms = llms_infos.keys()
    parser.add_argument("--dataset", type=str, help=f"Name of the dataset for which we generate data. Possible values: {all_dnames}", required=True, choices=all_dnames)  #nargs=1
    parser.add_argument("--parent_directory", type=str, help="Path to the directory where the data is stored", default="data")
    parser.add_argument("--gen_model", type=str, help=f"Name of the LLM to use for the data generation. Possible values: {all_llms}", default="gpt4o", choices=all_llms)
    parser.add_argument("--critic_model", type=str, help=f"Name of the LLM to use for the feedback. Possible values: {all_llms}")  # if None, use the same as gen_model
    parser.add_argument("--no_data_info", action="store_true", help="Whether to give the dataset information in the prompt")
    parser.add_argument("--fb_only_weakness", action="store_true", help="Whether the feedback should list only the weaknesses of the examined data (True) or both the strengths and weaknesses (False)")


    # serialization
    parser.add_argument("--serialization", type=str, help="Type of serialization to use", default="csv", choices=["sentence", "csv"])
    parser.add_argument("--specifier", type=str, help="Serialization: specifier to use (mind the spaces)", default="the ")
    parser.add_argument("--value_sep", type=str, help="Serialization: separator between feature and value (mind the spaces)", default=" is ")
    parser.add_argument("--feat_sep", type=str, help="Serialization: separator between features (mind the spaces)", default=", ")
    parser.add_argument("--end_of_line", type=str, help="Serialization: end of line character (mind the spaces)", default="\n")

    # synthloop parameters
    parser.add_argument("--max_iter", type=int, help="Maximum number of iterations for the synthloop", default=3)
    parser.add_argument("--n_examples", type=int, help="Number of examples to generate", default=100)
    parser.add_argument("--gen_method", type=str, help="Method to use for the generation", default="synthloop", choices=["synthloop", "promptrefine", "reduced"])
    parser.add_argument("--critic_method", type=str, help="Type of feedback to use", default="llm", choices=["llm", "human"])
    parser.add_argument("--n_shots", type=int, help="Number of original examples to put in the prompt for generation", default=5)
    
    # llm parameters
    parser.add_argument("--gen_max_new_tokens", type=int, help="Maximum number of tokens to generate for the generation part", default=16384)
    parser.add_argument("--gen_temperature", type=float, help="Temperature for the generation part", default=0.7)
    parser.add_argument("--gen_do_sample", type=bool, help="Whether to sample the generation part", default=True)
    parser.add_argument("--gen_seed", type=int, help="Seed for the generation part", default=None)
    parser.add_argument("--critic_max_new_tokens", type=int, help="Maximum number of tokens to generate for the critic part", default=2048)
    parser.add_argument("--critic_temperature", type=float, help="Temperature for the critic part", default=0.7)
    parser.add_argument("--critic_do_sample", type=bool, help="Whether to sample the critic part", default=True)
    parser.add_argument("--critic_seed", type=int, help="Seed for the critic part", default=42)

    parser.add_argument("--feat_order", type=str, help="Order of the features in the serialization", default="original", choices=["original", "cat_first", "num_first"])
    parser.add_argument("--few_shots_feedback", action="store_true", help="Whether to use few shots feedback or not")
    parser.add_argument("--epic_few_shots", action="store_true", help="Whether to use the epic format for few shots and first prompt or not")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    print("\n--DEBUG: args values", flush=True)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}", flush=True)

    # get dataset and models
    print("\n--DEBUG: loading datasets and model...", flush=True)
    dataloader = Dataloader(args.dataset, args.parent_directory, test_size=0.2)
    gen_llm_name = args.gen_model
    if args.critic_model is None:
        critic_llm_name = gen_llm_name
    else:
        critic_llm_name = args.critic_model
    
    # serialization
    print("\n--DEBUG: creating serializer...", flush=True)
    if args.serialization == "csv":
        serializer = CSVSerializer(dataloader, feat_order=args.feat_order)
    elif args.serialization == "sentence":
        serializer = SentenceSerializer(dataloader, args.specifier, args.value_sep, args.feat_sep, args.end_of_line)
    else:
        raise ValueError(f"Unknown serialization method {args.serialization}. Possible values are ['csv', 'sentence']")

    # synthloop models
    print("\n--DEBUG: creating models...", flush=True)

    if args.gen_model == "deepseek-v3":
        args.gen_max_new_tokens = min(args.gen_max_new_tokens, 2048) #8192, 4096

    if args.gen_method == "synthloop":
        gen_method = SynthLoop(gen_llm_name,
                               dataloader,
                               serializer,
                               args.gen_temperature,
                               args.gen_max_new_tokens,
                               args.gen_do_sample,
                               args.gen_seed)
    elif args.gen_method == "promptrefine":
        gen_method = PromptRefine(gen_llm_name,
                                  dataloader,
                                  serializer,
                                  args.gen_temperature,
                                  args.gen_max_new_tokens,
                                  args.gen_do_sample,
                                  args.gen_seed)
    elif args.gen_method == "reduced":
        gen_method = ReducedLoop(gen_llm_name,
                                 dataloader,
                                 serializer,
                                 args.gen_temperature,
                                 args.gen_max_new_tokens,
                                 args.gen_do_sample,
                                 args.gen_seed)
    else:
        raise ValueError(f"Unknown generation method {args.gen_method}. Possible values are ['synthloop', 'promptrefine']")
    
    if args.critic_method == "llm":
        critic_method = LLMFeedback(args.critic_model,
                                    dataloader,
                                    serializer,
                                    not args.no_data_info,
                                    args.fb_only_weakness,
                                    args.critic_temperature,
                                    args.critic_max_new_tokens,
                                    args.critic_do_sample,
                                    args.critic_seed)
    elif args.critic_method == "human":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown critic method {args.critic_method}. Possible values are ['llm', 'human']")
    
    # generate the data
    print("\n--DEBUG: generating data...", flush=True)
    if args.gen_method == "reduced":
        generated_data = gen_method.generate(critic_method,
                                             args.n_examples,
                                             args.max_iter,
                                             args.n_shots,
                                             remove_duplicates=False,
                                             give_dataset_info= not args.no_data_info,
                                             few_shots_feedback=args.few_shots_feedback,
                                             verbose=args.verbose,
                                             )
    else:
        generated_data = gen_method.generate(critic_method,
                                             args.n_examples,
                                             args.max_iter,
                                             args.n_shots,
                                             give_dataset_info= not args.no_data_info,
                                             few_shots_feedback=args.few_shots_feedback,
                                             epic_few_shots=args.epic_few_shots,
                                             verbose=args.verbose
                                             )
    
    print(generated_data)

    filename = f"generated_examples/{args.gen_method}_{args.dataset}_{gen_llm_name}_{args.max_iter}iters_{args.serialization}_noInfo{args.no_data_info}_weaknessOnly{args.fb_only_weakness}_temp{args.gen_temperature}_shots{args.n_shots}_shotFb{args.few_shots_feedback}_order{args.feat_order}_epic{args.epic_few_shots}.csv"

    generated_data.to_csv(filename, index=False)
    print(f"saved generated at {filename}")





