# Team : Outliers
# Team Member 1 : Soumya Chowdary Daruru (115361470)
# Team Member 2 : Abishek Vanam (115077012)
# Team Member 3 : Susrutha Kanisetty (116065245)


# General Description:
# This file is a helper for generating the TF-IDF vectors that 
# are used for finding similar passages from information source.

# NLP Concepts Used:
# 1) Syntax: Input Prompt to the generater is tokenized which uses tokenization of words from "1. Syntax"
# 2) Semantics: Query is compared to the text chunks of Information source to find relevant chunks. This comparison is done using contextual embeddings of both the query and text chunks. This is from "2. Semantics"
# 3) Language Modeling: The context and the query is passed to the Auto Generative model Gemma that generates the final response. "3. Generative Language Modeling"
# 4) Applications: We first perform Information retrieval, augment the prompt and finally generate the response. This is part of "4. Quetion Answering"


# Code Execution : Google Colab with T4 GPU


import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
nltk.download('punkt')

class TfidfHelper:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def read_pdf(self):
        # Open the PDF file
        doc = fitz.open(self.pdf_path)
        # Extract text from each page and process it
        for page in doc:
            text = page.get_text("text")
            # Use nltk to split text into sentences
            sentences = nltk.tokenize.sent_tokenize(text)
            # Group sentences into chunks of 10
            for i in range(0, len(sentences), 10):
                chunk = " ".join(sentences[i:i+10])
                self.documents.append(chunk)
        doc.close()
        print("Completed reading and preprocessing PDF into sentence chunks.")

    def compute_tfidf(self):
        # Fit and transform the documents to create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        print("TF-IDF computation completed.")

    def save_tfidf_results(self):
        # Save the TF-IDF vectorizer and matrix
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open("tfidf_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        print("Saved TF-IDF model and matrix.")

    def save_documents_csv(self):
        # Create a DataFrame with documents and their indices
        df = pd.DataFrame(self.documents, columns=['text'])
        df['chunk_number'] = df.index  # Add chunk number column
        df.to_csv("text_chunks.csv", index=False)
        print("Saved text documents as CSV.")

def main():
    pdf_path = "output.pdf"  # Specify your PDF file path here
    tfidf_helper = TfidfHelper(pdf_path)
    tfidf_helper.read_pdf()
    tfidf_helper.compute_tfidf()
    tfidf_helper.save_tfidf_results()
    tfidf_helper.save_documents_csv()

if __name__ == "__main__":
    main()
