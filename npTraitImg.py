import numpy as np
import matplotlib.pyplot as plt
from PIL import Image




def getImageTraitement(Image:str):
    image_path = "./img/{Image}.jpg"
    image = Image.open(image_path).convert("L")  
    image_array = np.array(image)
    plt.imshow(image_array)
    plt.title("Image originale en couleur")
    plt.show()
    # Convertir l'image en une matrice NumPy
    image_matrix = np.array(image)
    print(image_matrix)
    return image_matrix
    # Afficher la matrice
    # print("Matrice représentant l'image en couleur :"
    # print(image_matrix)




# plt.imshow(image_array, cmap="gray")
# plt.title("Image originale en niveaux de gris")
# plt.show()


# kernel = np.ones((3, 3)) / 9.0
# filtered_image = np.convolve(image_array, kernel, mode="same")

# plt.imshow(filtered_image, cmap="yellow")
# plt.title("Image filtrée avec un filtre de moyenne")
# plt.show()

# ---------------------------------------------------------------



# # Obtenir les dimensions de l'image
# hauteur, largeur, canaux = image_array.shape
# print(f"Dimensions de l'image : {hauteur} x {largeur} x {canaux}")

# # Effectuer des opérations sur l'image
# # Par exemple, convertir l'image en niveaux de gris en moyennant les canaux de couleur
# gray_image = np.mean(image_array, axis=2)

# # Afficher l'image en niveaux de gris
# plt.imshow(gray_image, cmap="gray")
# plt.title("Image en niveaux de gris")
# plt.show()

# # Appliquer un filtre de moyenne sur chaque canal de couleur
# kernel = np.ones((3, 3)) / 9.0
# filtered_image = np.zeros_like(image_array)

# for canal in range(canaux):
#     filtered_image[:, :, canal] = np.convolve(image_array[:, :, canal], kernel, mode="same")

# # Afficher l'image filtrée
# plt.imshow(filtered_image)
# plt.title("Image filtrée avec un filtre de moyenne sur chaque canal")
# plt.show()



# ------------------------------------


# # Afficher les dimensions de la matrice
# print("Dimensions de la matrice représentant l'image en couleur :")
# print(image_matrix.shape)



