# Team : Outliers
# Team Member 1 : Soumya Chowdary Daruru (115361470)
# Team Member 2 : Abishek Vanam (115077012)
# Team Member 3 : Susrutha Kanisetty (116065245)


# General Description:
# This file is used as a baseline where we directly give the input
# to Gemma model without any information retrieval system.

# NLP Concepts Used:
# 1) Syntax: Input Prompt to the generater is tokenized which uses tokenization of words from "1. Syntax"
# 2) Semantics: Query is compared to the text chunks of Information source to find relevant chunks. This comparison is done using contextual embeddings of both the query and text chunks. This is from "2. Semantics"
# 3) Language Modeling: The context and the query is passed to the Auto Generative model Gemma that generates the final response. "3. Generative Language Modeling"
# 4) Applications: We first perform Information retrieval, augment the prompt and finally generate the response. This is part of "4. Quetion Answering"


# Code Execution : Google Colab with T4 GPU

import torch
import requests
import fitz
from tqdm.auto import tqdm
import random
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer
import textwrap
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
import numpy as np
from transformers import BitsAndBytesConfig
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from transformers import BitsAndBytesConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


class Baseline1:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.the_embedding_model = None
        self.tokenizer = None
        # self.use_quantization_config = None
        # self.model_id = None
        self.llm_model = None
        self.eval_data = None
        self.unique_questions = None
        self.eval_ans_df = None
        self.use_quantization_config = False
        self.model_id = "google/gemma-2b-it"

    
    def embedding_model(self):
        self.the_embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=self.device) # choose the device to load the model to
        print("\ncompleted embedding model ")
    
    def hugging_face_login(self):
        login()
        print("\ncompleted hugging face login ")
    
    def llm_model_initialization(self):
        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"
        print(f"[INFO] Using attention implementation: {attn_implementation}")
        model_id = self.model_id
        print(f"[INFO] Using model_id: {model_id}")

        # Instantiate tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

        # Instantiate the model
        self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                        torch_dtype=torch.float16, # datatype to use, we want float16
                                                        quantization_config = None,
                                                        low_cpu_mem_usage=False, # use full memory
                                                        attn_implementation=attn_implementation)

        
        self.llm_model.to("cuda")
        
        print("\ncompleted initializing LLM Model ")

    def answer_generation(self, query):
        dialogue_template = [{"role": "user", "content": query}]
        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                            tokenize=False, # keep as raw text (not tokenized)
                                            add_generation_prompt=True)
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        # print(f"Model input (tokenized):\n{input_ids}\n")

        outputs = self.llm_model.generate(**input_ids, max_new_tokens=256)

        outputs_decoded = self.tokenizer.decode(outputs[0])
        # print(f"Input text: {query}\n")
        # print(f"Output text:\n{outputs_decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')}")
        output = outputs_decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')
        return output
    
    def read_eval_data(self):
        file_path = 'final_evaluation_data.csv'
        
        self.eval_data = pd.read_csv(file_path)
        self.eval_data = self.eval_data.drop(columns=['question_paraphrase'])
        self.eval_data = self.eval_data.dropna(subset=['answer'])
        
        self.unique_questions = self.eval_data.drop_duplicates(subset='question_id')
        self.unique_questions = self.unique_questions.drop(columns=['answer'])
        
        self.unique_questions['generated_answer'] = self.unique_questions['question_summary'].apply(self.answer_generation)
        self.eval_ans_df = self.eval_data.merge(self.unique_questions[['question_id', 'generated_answer']], on='question_id')
        self.eval_ans_df.to_csv('Generated_Ans_Baseline_1.csv', index=False, escapechar='\\')
        print("\n Completed generating answers")


def main():
    rag = Baseline1()
    rag.embedding_model()
    rag.hugging_face_login()
    rag.llm_model_initialization()
    rag.read_eval_data()

if __name__ == "__main__":
    main()
