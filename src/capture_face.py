"""
Este arquivo contem um script que captura uma imagens da webcam e salva em um arquivo .jpg
contido na pasta "images".
"""

import os
import cv2
import numpy as np
import json
from keras.applications import MobileNetV3Large
from keras.applications.mobilenet_v3 import preprocess_input

def create_user(func):
    def wrapper():
        user_name = input("Digite o nome do usuário: ")
        cpf = input("Digite os números do CPF: ")
        embedding_path = func(user_name=user_name)
        record_user = {
            "user_name": user_name,
            "cpf": cpf,
            "embedding": embedding_path
        }

        create_user_path = os.path.join(f"data/{user_name}.txt")
        with open(create_user_path, "w") as file:
            # grava o conteúdo de record_user no arquivo
            file.write(str(record_user))

    return wrapper

@create_user
def capture_face(user_name):
    # Carregar MobileNetV3
    model = MobileNetV3Large(weights="imagenet", input_shape=(224, 224, 3), include_top=False, pooling="avg")

    # Captura da webcam
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    if ret:
        print("Imagem capturada. Gerando embedding...")

        # Pré-processamento da imagem
        img = cv2.resize(frame, (224, 224))  # Redimensiona para o tamanho esperado
        img = np.expand_dims(img, axis=0)  # Adiciona dimensão do batch
        img = preprocess_input(img)  # Normaliza

        # Gera embedding
        embedding = model.predict(img)
        embedding_list = embedding.tolist()  # Converte para lista para salvar no JSON
        print("Embedding gerado com sucesso.")

        # Salvar embedding em JSON
        embedding_path = os.path.join("embedding", f"{user_name}_embedding.json")
        embedding_data = {"embedding": embedding_list}
        with open(embedding_path, "w") as file:
            json.dump(embedding_data, file, indent=4)

        print("✅ Embedding salvo em 'embedding'.")
    else:
        print("❌ Erro ao capturar imagem da webcam!")

    return embedding_path


if __name__ == "__main__":
    capture_face()
