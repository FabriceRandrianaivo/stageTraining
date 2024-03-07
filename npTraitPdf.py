from PyPDF2 import PdfReader
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertTokenizer, BertForQuestionAnswering
import fitz  # PyMuPDF
import torch

# Télécharger les ressources nécessaires pour NLTK (vous pouvez le faire une fois)
nltk.download('punkt')
nltk.download('stopwords')

# Fonction pour extraire le texte d'un fichier PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Fonction pour obtenir une réponse à partir d'un PDF
def get_answer_from_pdf(pdf_path, question):
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Tokenization avec NLTK
    tokens = word_tokenize(pdf_text.lower())
    
    # Supprimer les mots vides (stopwords)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Utiliser la similarité cosinus pour trouver la phrase la plus similaire à la question
    question_tokens = word_tokenize(question.lower())
    question_vectorizer = TfidfVectorizer().fit(question_tokens + tokens)
    question_vector = question_vectorizer.transform([question])
    
    answer_vectorizer = TfidfVectorizer().fit(tokens)
    tokens_vector = answer_vectorizer.transform([' '.join(tokens)])
    
    similarity = cosine_similarity(question_vector, tokens_vector)
    
    # Trouver l'indice de la phrase la plus similaire
    max_similarity_index = np.argmax(similarity)
    
    # Extraire la réponse (par exemple, une phrase autour de l'indice)
    start_index = max(0, max_similarity_index - 100)
    end_index = min(len(tokens), max_similarity_index + 100)
    
    answer_tokens = tokens[start_index:end_index]
    answer = ' '.join(answer_tokens)
    
    return answer

# Chemin vers le fichier PDF contenant les réponses
pdf_path_with_answers = './assets/French01.pdf'
# Poser une question
question = input("Posez votre question : ")
# Obtenir la réponse à partir du PDF
answer = get_answer_from_pdf(pdf_path_with_answers, question)
# Afficher la réponse
print("Réponse :", answer)




