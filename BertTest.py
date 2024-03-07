# from transformers import pipeline, AutoTokenizer, AutoModel
# from pdfminer.high_level import extract_text

# token = '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'

# # Charger le tokenizer et le modèle BERT pour la réponse à une question
# tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large", token=token)
# model = AutoModel.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large", token=token)

# # Créer le pipeline pour remplir les masques
# fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# # Fonction pour obtenir une réponse à partir d'un fichier PDF en utilisant MiniLMv2
# def get_answer_from_pdf_minilmv2(pdf_path, question):
#     pdf_text = extract_text(pdf_path)

#     # Diviser le texte en morceaux (chunks) de la longueur maximale spécifiée
#     max_length = 512
#     chunks = [pdf_text[i:i + max_length] for i in range(0, len(pdf_text), max_length)]

#     # Initialiser une liste pour stocker les réponses des chunks
#     answers = []

#     # Traiter chaque chunk séparément
#     for chunk in chunks:
#         # Remplacer la question avec un masque
#         masked_question = question.replace("color", fill_mask.tokenizer.mask_token)

#         # Utiliser le modèle MiniLMv2 pour remplir le masque
#         result = fill_mask(masked_question, targets=[chunk])
        
#         # Ajouter la réponse du chunk à la liste
#         answers.append(result[0]["sequence"])

#     # Concaténer les réponses des chunks
#     final_answer = ' '.join(answers)

#     return final_answer

# # Chemin vers le fichier PDF contenant les réponses
# pdf_path_with_answers = 'assets/French01.pdf'

# # Poser une question
# question = input("Posez votre question : ")

# # Obtenir la réponse en utilisant MiniLMv2
# answer = get_answer_from_pdf_minilmv2(pdf_path_with_answers, question)

# # Afficher la réponse
# print("Réponse :", answer)


from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from pdfminer.high_level import extract_text

token = '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'

# Charger le tokenizer et le modèle BERT multilingue pour la réponse à une question
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", token=token)
model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased", token=token)

# Créer le pipeline pour remplir les masques
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Fonction pour obtenir une réponse à partir d'un fichier PDF en utilisant BERT multilingue
def get_answer_from_pdf_bert(pdf_path, question, max_responses=3):
    pdf_text = extract_text(pdf_path)

    # Diviser le texte en morceaux (chunks) de la longueur maximale spécifiée
    max_length = 512
    chunks = [pdf_text[i:i + max_length] for i in range(0, len(pdf_text), max_length)]

    # Initialiser une liste pour stocker les réponses des chunks
    answers = []

    # Traiter chaque chunk séparément
    for chunk in chunks:
        # Remplacer la question avec un masque
        masked_question = question.replace(question.split()[0], fill_mask.tokenizer.mask_token)

        # Utiliser le modèle BERT multilingue pour remplir le masque
        results = fill_mask(masked_question, targets=[chunk])

        # Limiter le nombre de réponses
        results = results[:max_responses]

        # Ajouter la réponse du chunk à la liste 
        answers.extend([result["sequence"] for result in results])

    # Concaténer les réponses des chunks
    final_answer = ' '.join(answers)

    return final_answer

# Chemin vers le fichier PDF contenant les réponses
pdf_path_with_answers = 'assets/Portrait.pdf'

# Poser une question
question = input("Posez votre question : ")

# Obtenir la réponse en utilisant BERT multilingue avec une limite de réponses
answer = get_answer_from_pdf_bert(pdf_path_with_answers, question, max_responses=3)

# Afficher la réponse
print("Réponse :", answer)
