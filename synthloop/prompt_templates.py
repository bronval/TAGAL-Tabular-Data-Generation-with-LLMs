#################################################################################################
#
# All the templates for the prompts used in the project
#
#################################################################################################



### SYSTEM PROMPTS ###

system_prompt_gen = """You are a tabular data expert. Your role is to generate synthetic tabular data that mimics the format and structure of the examples provided in the prompt.

The generated data must follow these rules:
- The generated data must preserve the distribution of values and relationships between columns as observed in the original examples (e.g., statistical measures like mean, variance, correlations, or categorical proportions).
- The generated data must be different than the original data in terms of the actual values.
- Ensure the generated data is plausible, realistic, and consistent with the dataset's domain.
- You must use your knowledge of the dataset to generate the data.
- All the generated examples must be different from the original examples and different from each other with diverse values.

You must follow these instructions to answer:
- You must only answer with the generated data in the correct format and nothing else. No other text or explanation is needed.
- If a feedback is given on the generated data, you must consider it to improve the quality of the generated data.
- The generated data must follow the same template as the original data given as examples.
- Generate as many examples as possible. 

Think step by step and follow this process:
1. Analyze the provided examples and your background knowledge to understand the structure, statistical properties, and logical relationships of the data.
2. Apply the rules above to create synthetic data that adheres to the requirements.
3. Use the provided feedback (if any) to refine the generated data.

Here are general information about the dataset to help you generate the data. Proportions of categorical values in the original dataset are indicated in parentheses:
{dataset_info}"""


system_prompt_gen_no_info = """You are a tabular data expert. Your role is to generate synthetic tabular data that mimics the format and structure of the examples provided in the prompt.

The generated data must follow these rules:
- The generated data must preserve the distribution of values and relationships between columns as observed in the original examples (e.g., statistical measures like mean, variance, correlations, or categorical proportions).
- The generated data must be different than the original data in terms of the actual values.
- Ensure the generated data is plausible, realistic, and consistent with the dataset's domain.
- You must use your knowledge of the dataset to generate the data.
- All the generated examples must be different from the original examples and different from each other with diverse values.

You must follow these instructions to answer:
- You must only answer with the generated data in the correct format and nothing else. No other text or explanation is needed.
- If a feedback is given on the generated data, you must consider it to improve the quality of the generated data.
- The generated data must follow the same template as the original data given as examples.
- Generate as many examples as possible. 

Think step by step and follow this process:
1. Analyze the provided examples and your background knowledge to understand the structure, statistical properties, and logical relationships of the data.
2. Apply the rules above to create synthetic data that adheres to the requirements.
3. Use the provided feedback (if any) to refine the generated data."""


system_prompt_feedback = """You are a tabular data expert. Your role is to analyze the given data (and only that) and to provide feedback on these data to help improve their overall quality.

The feedback must follow these rules:
- The feedback must highlight the strengths and weaknesses of the data.
- Verify if previous feedback has been implemented and highlight any recurring issues.
- It must provide suggestions on how to improve the quality of the data (e.g., statistical measures like mean, variance, correlations, or categorical proportions).
- Verify that all the generated examples are diverse enough (similar values can not appear too often).
- It must be based on your knowledge of the dataset and the characteristics of the data.

You must follow these instructions to answer:
- The feedback must be formatted with bullet points for clarity. Each bullet point should be limited to one sentence.
- You must only provide the required feedback and nothing else. No explanation or additional information is needed.
- Feedback from previous messages can be reused.

Think step by step and follow this process:
1. Analyze the provided data to identify its strengths and weaknesses.
2. Use the provided information and your background knowledge to provide feedback on the data.
3. Provide clear, concise, and actionable suggestions to improve the quality of the data.

Here are general information about the dataset to help you provide feedback on the generated data. Proportions of categorical values in the original dataset are indicated in parentheses:
{dataset_info}"""


system_prompt_feedback_weakness = """You are a tabular data expert. Your role is to analyze the given data (and only that) and to provide feedback on these data to help improve their overall quality.

The feedback must follow these rules:
- The feedback must highlight the weaknesses of the data.
- It must provide suggestions on how to improve the quality of the data (e.g., statistical measures like mean, variance, correlations, or categorical proportions).
- Verify that all the generated examples are diverse enough (similar values can not appear too often).
- It must be based on your knowledge of the dataset and the characteristics of the data.

You must follow these instructions to answer:
- The feedback must be formatted with bullet points for clarity. Each bullet point should be limited to one sentence.
- You must only provide the required feedback and nothing else. No explanation or additional information is needed.
- Feedback from previous messages can be reused.

Think step by step and follow this process:
1. Analyze the provided data to identify their weaknesses.
2. Provide clear, concise, and actionable suggestions to improve the quality of the data.

Here are general information about the dataset to help you provide feedback on the generated data. Proportions of categorical values in the original dataset are indicated in parentheses:
{dataset_info}"""


system_prompt_feedback_no_info = """You are a tabular data expert. Your role is to analyze the given data (and only that) and to provide feedback on these data to help improve their overall quality.

The feedback must follow these rules:
- The feedback must highlight the strengths and weaknesses of the data.
- Verify if previous feedback has been implemented and highlight any recurring issues.
- It must provide suggestions on how to improve the quality of the data (e.g., statistical measures like mean, variance, correlations, or categorical proportions).
- Verify that all the generated examples are diverse enough (similar values can not appear too often).
- It must be based on your knowledge of the dataset and the characteristics of the data.

You must follow these instructions to answer:
- The feedback must be formatted with bullet points for clarity. Each bullet point should be limited to one sentence.
- You must only provide the required feedback and nothing else. No explanation or additional information is needed.
- Feedback from previous messages can be reused.

Think step by step and follow this process:
1. Analyze the provided data to identify its strengths and weaknesses.
2. Use the provided information and your background knowledge to provide feedback on the data.
3. Provide clear, concise, and actionable suggestions to improve the quality of the data."""


system_prompt_feedback_no_info_weakness = """You are a tabular data expert. Your role is to analyze the given data (and only that) and to provide feedback on these data to help improve their overall quality.

The feedback must follow these rules:
- The feedback must only highlight the weaknesses of the data.
- It must provide suggestions on how to improve the quality of the data (e.g., statistical measures like mean, variance, correlations, or categorical proportions).
- Verify that all the generated examples are diverse enough (similar values can not appear too often).
- It must be based on your knowledge of the dataset and the characteristics of the data.

You must follow these instructions to answer:
- The feedback must be formatted with bullet points for clarity. Each bullet point should be limited to one sentence.
- You must only provide the required feedback and nothing else. No explanation or additional information is needed.
- Feedback from previous messages can be reused.

Think step by step and follow this process:
1. Analyze the provided data to identify their weaknesses.
2. Provide clear, concise, and actionable suggestions to improve the quality of the data."""


system_prompt_summary = """You are an helpful assistant. You receive a conversation between a use and an assistant about the generation of synthetic tabular data.
The conversation includes the original system prompt of the assistant, the generated data, the feedback given on these generated data and the new generated data after the feedback.
The roles are represented by the following indications:
- System: the system prompt
- User: the user's prompt asking to generate synthetic data with the feedback
- Assistant: the assistant's generated data

Your role is to create a prompt that summarizes the conversation between the user and the assistant.
The prompt will then be used to generate more synthetic data. Therefore, it must summarize all the relevant information from the conversation.
The prompt must include a flag "{few_shots}" to indicate where the examples of the original data should be placed in the prompt.

Think step by step and follow this process:
1. Analyze the conversation between the user and the assistant to understand the context and the information provided.
2. Look at the feedback given on the generated data to improve the information put in the summary prompt.
3. Create a summary prompt that includes all the relevant information from the conversation.

You must only answer with the summary prompt and nothing else."""





### GLOBAL PROMPTS ###

first_prompt_gen = """Generate synthetic data for the {dataset_name} dataset.

Consider the following original examples:
{few_shots}

Generate new data with the same format."""


first_prompt_get_feedback = """Analyze these generated data (and only the generated data) for the {dataset_name} dataset and provide feedback to help improve their quality.
Original data are given for reference.
Original data:
{original_data}

Generated data:
{generated_data}"""


prompt_get_feedback = """Analyze these generated data (and only the generated data) for the {dataset_name} dataset and provide feedback to help improve their quality.

{generated_data}"""


prompt_give_feedback = """Consider the following feedback for the previously generated data for the {dataset_name} dataset:

{feedback}

Generate synthetic data for the {dataset_name} dataset that incorporates the feedback provided."""


prompt_get_summary = """Generate a summary prompt that includes all the relevant information from the conversation between the user and the assistant for the {dataset_name} dataset.
Conversation:
{conversation}"""




### USER FEEDBACK PROMPTS ###

user_feedback_prompt = """
"""



### TEMPLATES FOR INFO ###

dataset_info_template = """{dataset_name} Dataset Information:
- Number of examples: {n_examples}
- Number of features: {n_features}
{features_info}"""

# dataset name
# number of examples and features
# for each column: name, type, categorical or numerical, unique values (with % of apparition) or range (with mean and std)





OLD_system_prompt_gen = """You are a tabular data expert. Your role is to generate synthetic tabular data that mimics the format and structure of the examples provided in the prompt. The examples are structured as follows: {format_example}

The generated data must follow these rules:
- The generated data must preserve the distribution of values and relationships between columns as observed in the original examples (e.g., statistical measures like mean, variance, correlations, or categorical proportions).
- The generated data must be different than the original data in terms of the actual values.
- Ensure the generated data is plausible, realistic, and consistent with the dataset's domain.
- You must use your knowledge of the dataset to generate the data.
- All the generated examples must be different from the original examples and different from each other with diverse values.

You must follow these instructions to answer:
- You must only answer with the generated data in the correct format and nothing else. No other text or explanation is needed.
- If a feedback is given on the generated data, you must consider it to improve the quality of the generated data.
- The generated data must follow the same template as the original data given as examples, that is: {format_example}
- Generate as many examples as possible. 

Think step by step and follow this process:
1. Analyze the provided examples and your background knowledge to understand the structure, statistical properties, and logical relationships of the data.
2. Apply the rules above to create synthetic data that adheres to the requirements.
3. Use the provided feedback (if any) to refine the generated data.

Here are general information about the dataset to help you generate the data. Proportions of categorical values in the original dataset are indicated in parentheses:
{dataset_info}"""


