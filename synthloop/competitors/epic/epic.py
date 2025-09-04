
## compare to EPIC model
## from https://github.com/seharanul17/synthetic-tabular-LLM

import numpy as np
import pandas as pd
from io import StringIO

import os
from dotenv import load_dotenv
import pandas as pd
import string
import random
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from synthloop.dataloader import Dataloader
from synthloop.llm_handler import LLMHandler
from synthloop.competitors.competitor import Competitor



class EPIC(Competitor):

    def __init__(self,
                 llm: str,
                 dataloader: Dataloader,
                 ):
        self.dataloader = dataloader
        self.llm = llm
        self.target = self.dataloader.target
        self.unique_categorical_features = get_unique_features(dataloader.data, dataloader.cat_features + [dataloader.target])
        self.data = dataloader.data
        self.NAME_COLS = ','.join(self.data.columns) + '\n'
        self.output_parser = StrOutputParser()
        self.llm_handler = LLMHandler(llm)

        cat_idx = []
        for i, c in enumerate(self.data.columns):
            if c in self.dataloader.cat_features:
                cat_idx.append(i)

        self.params = {
            "model": llm,
            "DATA_NAME": dataloader.dname,
            "N_CLASS":2,
            "N_SAMPLES_PER_CLASS":15,
            "N_SET":4,
            "USE_RANDOM_WORD":True,
            "N_BATCH":1,
            "N_TARGET_SAMPLES":1000,
        }

        if self.params["USE_RANDOM_WORD"]:
            self.mapper, self.mapper_r, self.unique_categorical_features = make_random_categorical_values(self.unique_categorical_features)
            for c in self.mapper:
                self.data[c] = self.data[c].map(lambda x: self.mapper[c][x])

        if self.llm_handler.source == "huggingface":
            pipe = pipeline("text-generation", model=self.llm_handler.llm, tokenizer=self.llm_handler.tokenizer, device_map="cuda", max_length=16384) #8192
            llm = HuggingFacePipeline(pipeline=pipe)
        else:
            ...

        self.initial_prompt = {
            "adult": """Age: the age of the individual in years,
Workclass: the type of employment or work sector,
fnlwgt: a weighting factor to adjust for census over- or under-sampling,
Education: the highest level of education achieved,
Education-Num: numeric representation of the education level,
Marital-Status: marital state of the individual,
Occupation: type of job or occupation,
Relationship: the individual's relationship status,
Race: the individual's race,
Sex: the individual's gender,
Capital-Gain: income received from investment gains,
Capital-Loss: income lost from investment losses,
Hours-per-week: number of hours worked per week,
Native-Country: the country of birth of the individual,
Income: binary classification indicating if income exceeds $50K/year.\n\n""",

            "bank": """1 - age (numeric)
2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services") 
3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
4 - education (categorical: "unknown","secondary","primary","tertiary")
5 - default: has credit in default? (binary: "yes","no")
6 - balance: average yearly balance, in euros (numeric) 
7 - housing: has housing loan? (binary: "yes","no")
8 - loan: has personal loan? (binary: "yes","no")
# related with the last contact of the current campaign:
9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
10 - day: last contact day of the month (numeric)
11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
12 - duration: last contact duration, in seconds (numeric)
# other attributes:
13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
15 - previous: number of contacts performed before this campaign and for this client (numeric)
16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")""",

            "thyroid": """Age, Gender, Smoking, Hx Smoking, Hx Radiothreapy, Thyroid Function, Physical Examination, Adenopathy, Pathology, Focality, Risk, T, N, M, Stage, Response, Recurred""",
        
            "german": """Age,Sex,Job,Housing,Saving accounts,Checking account,Credit amount,Duration,Purpose,Risk""",

            "sick": """age,sex,on_thyroxine,query_on_thyroxine,on_antithyroid_medication,sick,pregnant,thyroid_surgery,I131_treatment,query_hypothyroid,query_hyperthyroid,lithium,goitre,tumor,hypopituitary,psych,TSH_measured,TSH,T3_measured,T3,TT4_measured,TT4,T4U_measured,T4U,FTI_measured,FTI,referral_source,Class""",

            "travel": """Age,FrequentFlyer,AnnualIncomeClass,ServicesOpted,AccountSyncedToSocialMedia,BookedHotelOrNot,ChurnOrNot""",

        }

        self.N_CLASS = self.params["N_CLASS"]
        self.N_SAMPLES_PER_CLASS = self.params["N_SAMPLES_PER_CLASS"]
        self.N_SET = self.params["N_SET"]
        self.N_BATCH = self.params["N_BATCH"]
        self.N_SAMPLES_TOTAL = self.N_SAMPLES_PER_CLASS * self.N_SET * self.N_BATCH

        self.numbering = ["A", "B", "C", "D"]
        self.prompt = get_prompt_conclass(self.initial_prompt[self.dataloader.dname],
                                          self.numbering,
                                          self.N_SAMPLES_PER_CLASS, self.N_CLASS, self.N_SET, self.NAME_COLS)
        
        template1 = self.prompt
        self.template1_prompt = PromptTemplate.from_template(template1)

        self.llm1 = (
            self.template1_prompt | llm | self.output_parser
        )

        self.final_prompt, _ = make_final_prompt(self.unique_categorical_features, self.target, self.data, self.template1_prompt,
                                                 self.N_SAMPLES_TOTAL, self.N_BATCH, self.N_SAMPLES_PER_CLASS, self.N_SET, self.NAME_COLS, self.N_CLASS)


    def generate(self, n_examples: int):
        input_df_all = pd.DataFrame()
        synthetic_df_all = pd.DataFrame()
        text_results = []

        columns1 = self.data.columns
        columns2 = list(self.data.columns)

        err = []

        while len(synthetic_df_all) < n_examples:
            final_prompt, input_batch = make_final_prompt(self.unique_categorical_features, self.target, self.data, self.template1_prompt,
                                                          self.N_SAMPLES_TOTAL, self.N_BATCH, self.N_SAMPLES_PER_CLASS, self.N_SET, self.NAME_COLS, self.N_CLASS)
            inter_text = self.llm1.batch(input_batch)
            # inter_text = self.llm1.invoke(input_batch)
            print(len(inter_text), flush=True)
            for i in range(len(inter_text)):
                try:
                    text_results.append(final_prompt[i].text + inter_text[i])
                    input_df = parse_prompt2df(final_prompt[i].text, split=self.NAME_COLS, inital_prompt=self.initial_prompt[self.dataloader.dname], col_name=columns1)
                    result_df = parse_result(inter_text[i], self.NAME_COLS, columns2, self.dataloader.cat_features + [self.dataloader.target], self.unique_categorical_features)
                    
                    input_df_all = pd.concat([input_df_all, input_df], axis=0)
                    synthetic_df_all = pd.concat([synthetic_df_all, result_df], axis=0)
                except Exception as e:
                    err.append(inter_text[i])
                    print(err, flush=True)
            print('Number of Generated Samples:', len(synthetic_df_all),'/', n_examples)

        synthetic_df_all_r = synthetic_df_all.copy()
        if self.params["USE_RANDOM_WORD"]:
            for c in self.mapper_r:
                input_df_all[c] = input_df_all[c].map(lambda x: self.mapper_r[c][x] if x in self.mapper_r[c] else x)
            for c in self.mapper_r:
                synthetic_df_all_r[c] = synthetic_df_all_r[c].map(lambda x: self.mapper_r[c][x] if x in self.mapper_r[c] else x)
        
        return synthetic_df_all_r





def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    first = ''.join(random.choice(string.ascii_uppercase) for _ in range(1))
    left = ''.join(random.choice(chars) for _ in range(size-1))
    return first+left


def make_random_categorical_values(unique_categorical_features):
    mapper = {}
    mapper_r = {}
    new_unique_categorical_features = {}
    for c in unique_categorical_features:
        mapper[c] ={}
        mapper_r[c]={}
        new_unique_categorical_features[c] = []

        for v in unique_categorical_features[c]:
            a = id_generator(3)
            new_unique_categorical_features[c].append(a)

            mapper[c][v] = a
            mapper_r[c][a] = v
    return mapper, mapper_r, new_unique_categorical_features








def get_prompt_conclass(inital_prompt, numbering, n_samples_per_class,nclass,nset, name_cols):
    prompt=""
    for i in range(nset):
        prompt+=name_cols
        for j in range(nclass):
            prompt+=f'{numbering[j]}.\n'
            for k in range(n_samples_per_class):
                prompt +='{'+f'v{i*(n_samples_per_class*nclass)+j*n_samples_per_class+k}'+'}'
            prompt += f'\n'
        prompt += f'\n'  
    prompt+=name_cols
    
    prompt = inital_prompt+prompt
    return prompt
    

def filtering_categorical(result_df, categorical_features, unique_features):
    org_df = result_df.copy()
    shape_before = org_df.shape
    
    for column in categorical_features:
        if column=='Target':
            result_df = result_df[result_df[column].map(lambda x: int(x) in unique_features[column])]
        else:
            result_df = result_df[result_df[column].map(lambda x: x in unique_features[column])]
        
    if shape_before!=result_df.shape:
        for column in categorical_features:
            filtered = org_df[org_df[column].map(lambda x: x not in unique_features[column])]
    return result_df
    

def parse_prompt2df(one_prompt, split, inital_prompt, col_name):
    one_prompt = one_prompt.replace(inital_prompt, '')
    input_prompt_data = one_prompt.split(split)
    input_prompt_data = [x for x in input_prompt_data if x]
    input_prompt_data = '\n'.join(input_prompt_data)
    input_df = pd.read_csv(StringIO(input_prompt_data), sep=",", header=None, names=col_name)
    input_df = input_df.dropna()
    return input_df


def parse_result(one_prompt, name_cols, col_name, categorical_features, unique_features, filter_flag=True):
    one_prompt = one_prompt.replace(name_cols, '')
    result_df = pd.read_csv(StringIO(one_prompt), sep=",", header=None, names=col_name)
    result_df = result_df.dropna()
    if filter_flag:
        result_df = filtering_categorical(result_df, categorical_features, unique_features)
    return result_df
    

def get_unique_features(data, categorical_features):
    unique_features={}
    for column in categorical_features:
        try:
            unique_features[column] = sorted(data[column].unique())
        except:
            unique_features[column] = data[column].unique()
    return unique_features


def get_sampleidx_from_data(unique_features, target, n_samples_total, n_batch, n_samples_per_class, nset, name_cols, data):
    # input sampling
    unique_classes = unique_features[target]
    random_idx_batch_list=[]
    target_df_list=[]
    for c in unique_classes:
        target_df=data[data[target]==c]
        if len(target_df) < n_samples_total:
            replace_flag=True
        else:
            replace_flag=False
        random_idx_batch = np.random.choice(len(target_df), n_samples_total, replace=replace_flag)
        random_idx_batch = random_idx_batch.reshape(n_batch,nset,1,n_samples_per_class)
        random_idx_batch_list.append(random_idx_batch)
        target_df_list.append(target_df)
    random_idx_batch_list = np.concatenate(random_idx_batch_list, axis=2)
    return random_idx_batch_list, target_df_list


def get_input_from_idx(target_df_list, random_idx_batch_list, data, n_batch, n_samples_per_class, nset, nclass ):
    fv_cols = ('{},'*len(data.columns))[:-1] + '\n' 
    # input selection 
    inputs_batch = []
    for batch_idx in range(n_batch):
        inputs = {}
        for i in range(nset): #5
            for j in range(nclass): #2
                target_df = target_df_list[j]
                for k in range(n_samples_per_class): #3
                    idx = random_idx_batch_list[batch_idx, i,j,k]
                    inputs[f'v{i*(n_samples_per_class*nclass)+j*n_samples_per_class+k}']=fv_cols.format(
                        *target_df.iloc[idx].values
                    )
        inputs_batch.append(inputs)
    return inputs_batch
    

def make_final_prompt(unique_categorical_features, TARGET, data, template1_prompt,
                      N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS):
    
    random_idx_batch_list, target_df_list = get_sampleidx_from_data(unique_categorical_features, TARGET, 
                                                                    N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, data)
    inputs_batch = get_input_from_idx(target_df_list, random_idx_batch_list, data, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, N_CLASS)
    final_prompt = template1_prompt.batch(inputs_batch)
    return final_prompt, inputs_batch


def useThis(one_prompt):
    char = one_prompt[0]
    if char.isdigit() and int(char) in [0,1,2,3,4]:
        return True, int(char)
    else:
        return False, None

