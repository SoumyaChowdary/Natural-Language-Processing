# Team : Outliers
# Team Member 1 : Soumya Chowdary Daruru (115361470)
# Team Member 2 : Abishek Vanam (115077012)
# Team Member 3 : Susrutha Kanisetty (116065245)

# General Description:
# This file implements RAG using TF-IDF vectors(Use TFIDF_helper.py for generating TF-IDF vectors) for getting similar text chunks from document

# NLP Concepts Used:
# 1) Syntax: Input Prompt to the generater is tokenized which uses tokenization of words from "1. Syntax"
# 2) Semantics: Query is compared to the text chunks of Information source to find relevant chunks. This comparison is done using contextual embeddings of both the query and text chunks. This is from "2. Semantics"
# 3) Language Modeling: The context and the query is passed to the Auto Generative model Gemma that generates the final response. "3. Generative Language Modeling"
# 4) Applications: We first perform Information retrieval, augment the prompt and finally generate the response. This is part of "4. Quetion Answering"


# Code Execution : Google Colab with T4 GPU



import torch
import fitz
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from huggingface_hub import login
import re
from transformers import BitsAndBytesConfig
import pickle

class RAGBaseLine2:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.pages_and_chunks = None
        self.tokenizer = None
        self.llm_model = None
        self.model_id = "google/gemma-2b-it"
        self.use_quantization_config = False

    def load_tfidf_model(self):
        # Load your precomputed TF-IDF model and matrix if already saved
        
        with open("tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open("tfidf_matrix.pkl", "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        self.pages_and_chunks = pd.read_csv("text_chunks.csv")

    
    # This performs a cosine similarity between the input query and our information source document.
    def retrieve_relevant_resources(self, query, n_resources_to_return=5):
        query_vector = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = cosine_sim.argsort()[-n_resources_to_return:][::-1]
        return [(index, self.pages_and_chunks.iloc[index]['text'], cosine_sim[index]) for index in top_indices]

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

    def ask(self, query, temperature=0.7, max_new_tokens=512):

        context_items = self.retrieve_relevant_resources(query)
        prompt_text = "User query: {}\n\n".format(query) +"Use the following additional information retrived from local information storage to enhance the answers. Make sure the answers are explanatory."+ "\n\n".join([item[1] for item in context_items]) + " Answer:"
        
        # Tokenize the prompt
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")

        # Generate an output of tokens
        outputs = self.llm_model.generate(**input_ids,
                                    temperature=temperature,
                                    do_sample=True,
                                    max_new_tokens=max_new_tokens)

        # Turn the output tokens into text
        output_text = self.tokenizer.decode(outputs[0])
        # Define the regular expression pattern to find text between "Answer:" and "<eos>"
        # pattern = r"**Answer:** (.*?)<eos>"
        # pattern = r"Answer: (.*?)<eos>"
        pattern = r"Answer:\s*(.*)"



        # Use re.search to find the first occurrence in the text
        match = re.search(pattern, output_text, re.DOTALL)

        # Extract the matched text if found
        extracted_text = match.group(1) if match else "No match found"
        if(extracted_text=="No match found"):
            print("******************"+query)
        
        epattern = r"<eos>"

        replaced_text = re.sub(epattern, "", extracted_text)
        return replaced_text
    
    def read_and_generate(self):
        file_path = 'final_evaluation_data.csv'
        #Reading and Preprocessing
        self.eval_data = pd.read_csv(file_path)
        self.eval_data = self.eval_data.drop(columns=['question_paraphrase'])
        self.eval_data = self.eval_data.dropna(subset=['answer'])
        #Extracting the questions
        self.unique_questions = self.eval_data.drop_duplicates(subset='question_id')
        self.unique_questions = self.unique_questions.drop(columns=['answer'])
        #Generating the answers
        self.unique_questions['generated_answer'] = self.unique_questions['question_summary'].apply(
            lambda x: self.ask(query=x, temperature=0.7, max_new_tokens=512))
        #Merging the generated answers back to the original DataFrame
        self.eval_ans_df = self.eval_data.merge(self.unique_questions[['question_id', 'generated_answer']], on='question_id')
        
        self.eval_ans_df.to_csv('Generated_Ans_TFIDF.csv', index=False, escapechar='\\')
        print("\n Completed generating answers")

def main():
    rag = RAGBaseLine2()
    rag.load_tfidf_model()
    rag.hugging_face_login()
    rag.llm_model_initialization()
    rag.read_and_generate()

if __name__ == "__main__":
    main()
