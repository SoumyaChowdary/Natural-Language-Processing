# Team : Outliers
# Team Member 1 : Soumya Chowdary Daruru (115361470)
# Team Member 2 : Abishek Vanam (115077012)
# Team Member 3 : Susrutha Kanisetty (116065245)

# General Description:
# This file implements RAG using BERT embeddings(generated using bert_embeddings.py file) for getting similar text chunks from document

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
from transformers import BertModel, BertTokenizer
from torch.nn.functional import cosine_similarity
from transformers import BitsAndBytesConfig

# Check if a GPU is available and use it, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


class RAGUsingBert:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.the_embedding_model = None
        self.bert_tokenizer = None
        self.pages_and_chunks = None
        self.embeddings = None
        self.tokenizer = None
        self.llm_model = None
        self.use_quantization_config = False
        self.model_id = "google/gemma-2b-it"
    
    def reading_the_embeddings(self):
        # Import texts and embedding df
        text_chunks_and_embedding_df = pd.read_csv("BERT_text_chunks_and_embeddings_df.csv")
        # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        # Convert texts and embedding df to list of dicts
        self.pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
        # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
        self.embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
        print("\n completed reading embeddings")
    
    def embedding_model(self):
        # Initialize the BERT model specifically for encoding embeddings
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.the_embedding_model = BertModel.from_pretrained('bert-base-uncased')
        self.the_embedding_model.to(self.device)  # Move the model to the appropriate device
        print("\nBERT model loaded for embeddings.")
   
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
   
    def format_prompt(self, query: str, context_items: list[dict]) -> str:
        
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
        base_prompt = """User query: {query}. Use the following additional information retrived from local information storage to enhance the answers. Make sure the answers are explanatory.
{context}. Answer:"""

        # Update base prompt with context items and query
        base_prompt = base_prompt.format(context=context, query=query)

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
            "content": base_prompt}
        ]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
        return prompt

    def retrieve_relevant_resources(self, query: str, n_resources_to_return: int=5, print_time: bool=True):
        """
        Embeds a query with BERT model and returns top k scores and indices from embeddings.
        """
        # Tokenize the query
        inputs = self.bert_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Generate embeddings using the BERT model
        with torch.no_grad():
            outputs = self.the_embedding_model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling over the sequence dimension

        # Calculate cosine similarities
        start_time = timer()
        dot_scores = cosine_similarity(query_embedding, self.embeddings)  # Assuming self.embeddings is already on the same device
        end_time = timer()

        if print_time:
            print(f"[INFO] Time taken to get scores on {len(self.embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

        # Get top k scores and their indices
        scores, indices = torch.topk(dot_scores, k=n_resources_to_return, dim=0)

        return scores, indices

    def ask(self, query, temperature, max_new_tokens,return_answer_only=True, format_answer_text=True ):
        """
        Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
        """

        # Get just the scores and indices of top related results
        scores, indices = self.retrieve_relevant_resources(query=query)

        # Create a list of context items
        context_items = [self.pages_and_chunks[i] for i in indices]

        # Add score to context item
        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu() # return score back to CPU

        # Format the prompt with context items
        prompt = self.format_prompt(query=query, context_items=context_items)

        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate an output of tokens
        outputs = self.llm_model.generate(**input_ids,
                                    temperature=temperature,
                                    do_sample=True,
                                    max_new_tokens=max_new_tokens)

        # Turn the output tokens into text
        output_text = self.tokenizer.decode(outputs[0])

        if format_answer_text:
            # Replace special tokens and unnecessary help message
            output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

        # Only return the answer without the context items
        if return_answer_only:
            return output_text

        return output_text, context_items
    
    def read_and_generate(self):
        file_path = 'short_final_evaluation_data.csv'
        #Reading and Preprocessing
        self.eval_data = pd.read_csv(file_path)
        self.eval_data = self.eval_data.drop(columns=['question_paraphrase'])
        self.eval_data = self.eval_data.dropna(subset=['answer'])
        #Extracting the questions
        self.unique_questions = self.eval_data.drop_duplicates(subset='question_id')
        self.unique_questions = self.unique_questions.drop(columns=['answer'])
        #Generating the answers
        self.unique_questions['generated_answer'] = self.unique_questions['question_summary'].apply(
            lambda x: self.ask(query=x, temperature=0.7, max_new_tokens=512, return_answer_only=True))
        #Merging the generated answers back to the original DataFrame
        self.eval_ans_df = self.eval_data.merge(self.unique_questions[['question_id', 'generated_answer']], on='question_id')
        
        self.eval_ans_df.to_csv('Generated_Ans_Bert_RAG.csv', index=False, escapechar='\\')
        print("\n Completed generating answers")

    
def main():
    rag = RAGUsingBert()
    rag.reading_the_embeddings()
    rag.embedding_model()
    rag.hugging_face_login()
    rag.llm_model_initialization()
    rag.read_and_generate()

if __name__ == "__main__":
    main()
