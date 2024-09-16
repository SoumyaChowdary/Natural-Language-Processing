# Team : Outliers
# Team Member 1 : Soumya Chowdary Daruru (115361470)
# Team Member 2 : Abishek Vanam (115077012)
# Team Member 3 : Susrutha Kanisetty (116065245)

# General Description:
# This file generates the embeddings using BERT and saves into a file.

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
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers import util, SentenceTransformer
import numpy as np
from transformers import BertModel, BertTokenizer

# Check if a GPU is available and use it, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


class CreateBertEmbeddings:
    def __init__(self):
        self.pages_and_texts = []
        self.pages_and_chunks = []
        self.pages_and_chunks_over_min_token_len = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
    def read_file(self):
        pdf_path = "output.pdf"
        doc = fitz.open(pdf_path)  
        self.pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc)):  
            # if page_number >= 2000:  
            #     break
            text = page.get_text()  
            text = text.replace("\n", " ").strip()
            self.pages_and_texts.append({"page_number": page_number,
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split(" ")),
                                    "page_sentence_count_raw": len(text.split(". ")),
                                    "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                    "text": text})
        # return self.pages_and_texts
        print("\nCompleted Reading the file")
            
    def pages_to_sentences(self):
        nlp = English()

        # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/
        nlp.add_pipe("sentencizer")
        for item in tqdm(self.pages_and_texts):
            item["sentences"] = list(nlp(item["text"]).sents)

            # Make sure all sentences are strings
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            # Count the sentences
            item["page_sentence_count_spacy"] = len(item["sentences"])
        print("\n Completed pages to sentences")
            
    def sentences_to_chunks(self):
        # Define split size to turn groups of sentences into chunks
        num_sentence_chunk_size = 10

        # Create a function that recursively splits a list into desired sizes
        def split_list(input_list: list,
                    slice_size: int) -> list[list[str]]:
            """
            Splits the input_list into sublists of size slice_size (or as close as possible).

            For example, a list of 17 sentences would be split into two lists of [[10], [7]]
            """
            return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

        # Loop through pages and texts and split sentences into chunks
        for item in tqdm(self.pages_and_texts):
            item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                                slice_size=num_sentence_chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])
        print("\n Completed pages to chunks")

    def split_chunks(self):
        # Split each chunk into its own item
        pages_and_chunks = []
        for item in tqdm(self.pages_and_texts):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

                pages_and_chunks.append(chunk_dict)
        
        df = pd.DataFrame(pages_and_chunks)
        min_token_length = 30
        self.pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
        print("\nCompleted split chunks")

    def create_embeddings(self):
        for item in tqdm(self.pages_and_chunks_over_min_token_len):
            inputs = self.tokenizer(item['sentence_chunk'], return_tensors='pt', padding=True, truncation=True).to(self.model.device)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Pool the outputs to a single vector (mean pooling)
            item['embedding'] = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def save_the_embeddings(self):
        text_chunks_and_embeddings_df = pd.DataFrame(self.pages_and_chunks_over_min_token_len)
        embeddings_df_save_path = "BERT_text_chunks_and_embeddings_df.csv"
        text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False, escapechar='\\')
        print("\n Completed saving the embeddings")
   

def main():
    helper = CreateBertEmbeddings()
    helper.read_file()
    helper.pages_to_sentences()
    helper.sentences_to_chunks()
    helper.split_chunks()
    helper.create_embeddings()
    helper.save_the_embeddings()


if __name__ == "__main__":
    main()
