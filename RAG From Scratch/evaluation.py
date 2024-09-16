# Team : Outliers
# Team Member 1 : Soumya Chowdary Daruru (115361470)
# Team Member 2 : Abishek Vanam (115077012)
# Team Member 3 : Susrutha Kanisetty (116065245)

# General Description:
# This file evaluates Cosine similarity, BLEU scores and Rouge scores for 
# Baseline 1, RAG with TF-IDF, RAG with Sentence Transformer Embeddings and RAG with Bert Embeddings

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
from rouge import Rouge

# from nltk.translate.bleu_score import sentence_bleu
# from nltk.tokenize import word_tokenize
# import sys
# import nltk
# nltk.download('punkt')

import sacrebleu

# Check if a GPU is available and use it, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

class Evaluation:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.the_embedding_model = None
        self.data_to_eval = None
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='google/gemma-2b-it')

    def embedding_model(self):
        self.the_embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=self.device) # choose the device to load the model to
        print("\ncompleted embedding model ")

    def read_the_data(self, file_path):
        # path = "short_final_evaluation_data.csv"
        # file_path = "Generated_Ans_Baseline.csv"
        self.data_to_eval = pd.read_csv(file_path)
        print("Completed reading the data")
    
    def semantic_similarity(self,type):
        self.data_to_eval['gt_embeddings'] = self.data_to_eval['answer'].apply(lambda x: self.the_embedding_model.encode(x, convert_to_tensor=True))
        self.data_to_eval['gen_embeddings'] = self.data_to_eval['generated_answer'].apply(lambda x: self.the_embedding_model.encode(x, convert_to_tensor=True))

        # Calculate cosine similarity
        self.data_to_eval['cosine_similarity'] = self.data_to_eval.apply(lambda row: util.pytorch_cos_sim(row['gt_embeddings'], row['gen_embeddings']).item(), axis=1)

        # Optionally, save or return the results
        filename = f'Evaluation_with_Similarity_{type}.csv'
        self.data_to_eval.to_csv(filename, index=False)
        # return self.eval_ans_df
        # print(self.data_to_eval.head())
        print("Completed generating similarity scores")

    def calculate_blue_scores(self,type):
        self.data_to_eval['bleu_score'] = self.data_to_eval.apply(self.blue_score, axis=1)

        # Write the results to a CSV file
        filename = f'Evaluation_with_blue_score_{type}.csv'
        self.data_to_eval.to_csv(filename, index=False)
        print("\n Completed Blue Score calculation")

    def blue_score(self, row):
        # try:
        #     reference = [self.tokenizer(row['answer'].lower())]
        #     candidate = self.tokenizer(row['generated_answer'].lower())
        #     score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        # except TypeError as e:
        #     print(f"Error calculating BLEU score: {e}")
        #     score = 0
        #     sys.exit()
        # return score

        reference = self.tokenizer.tokenize(row['answer'].lower())
        candidate = self.tokenizer.tokenize(row['generated_answer'].lower())

        # Convert tokens back to string for BLEU calculation
        reference_text = [' '.join(reference)]
        candidate_text = ' '.join(candidate)

        # Compute BLEU score
        bleu = sacrebleu.corpus_bleu([candidate_text], [reference_text])
        return bleu.score
    
    def calculate_rouge_scores(self,type):
        rouge = Rouge()
        # Applying the ROUGE calculation for each row
        self.data_to_eval['rouge_scores'] = self.data_to_eval.apply(
            lambda row: rouge.get_scores(row['generated_answer'], row['answer'], avg=True), axis=1
        )
        # Write the results to a CSV file
        filename = f'Evaluation_with_rouge_score_{type}.csv'
        self.data_to_eval.to_csv(filename, index=False)
        print("\n Completed Rouge Score calculation")
  

def main():
    evaluation = Evaluation()
    evaluation.embedding_model()

    #Calculate scores for Baseline 1
    file_path = "Generated_Ans_Baseline_1.csv" 
    evaluation.read_the_data(file_path)
    evaluation.semantic_similarity("baseline")
    evaluation.calculate_blue_scores("baseline")
    evaluation.calculate_rouge_scores("baseline")

    #Calculate Scores for RAG using Sentence Transformer embeddings
    file_path = "Generated_Ans_RAG.csv" 
    evaluation.read_the_data(file_path)
    evaluation.semantic_similarity("RAG")
    evaluation.calculate_blue_scores("RAG")
    evaluation.calculate_rouge_scores("RAG")

    #Calculate Scores for RAG using TF-IDF Vectors
    file_path = "Generated_Ans_TFIDF.csv" 
    evaluation.read_the_data(file_path)
    evaluation.semantic_similarity("TFIDF")
    evaluation.calculate_blue_scores("TFIDF")
    evaluation.calculate_rouge_scores("TFIDF")

    #Calculate Scores for RAG using BERT Embeddings
    file_path = "Generated_Ans_Bert_RAG.csv" 
    evaluation.read_the_data(file_path)
    evaluation.semantic_similarity("Bert")
    evaluation.calculate_blue_scores("Bert")
    evaluation.calculate_rouge_scores("Bert")


if __name__ == "__main__":
    main()

