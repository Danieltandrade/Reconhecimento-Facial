"""
Módulo de captura de faces.

Este módulo captura imagens de rostos e gera embeddings para os rostos capturados.
"""

import os
import cv2
import numpy as np
import json
from keras.api.applications import MobileNetV3Large
from keras.api.applications.mobilenet_v3 import preprocess_input

from src.users import create_user

@create_user
def capture_face(user_name: str) -> str:
    """
    Captura imagens de rostos e gera embeddings para os rostos capturados.

    Args:
        user_name (str): Nome do usuário.

    Returns:
        str: Caminho para o arquivo de embedding gerado.
    """
    # carregando haarcascade do diretorio raiz do projeto
    HAARCASCADE_FRONTALFACE_DEFAULT = os.path.join(os.path.dirname('Reconhecimento-Facial'), 'files', 'haarcascade_frontalface_default.xml')
    print(HAARCASCADE_FRONTALFACE_DEFAULT)
    model = MobileNetV3Large(weights="imagenet", input_shape=(224, 224, 3), include_top=False, pooling="avg")

    # Captura da webcam
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Converte a imagem para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplica um filtro de detecção de face
        faces = cv2.CascadeClassifier(HAARCASCADE_FRONTALFACE_DEFAULT).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Se encontrar um rosto, desenha uma delimitação ao redor dele
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Exibe a imagem com a delimitação
        cv2.imshow('Face Detection', frame)

        # Se pressionar a tecla 'c', captura a imagem e salva a embedding
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Gera a embedding
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

            break

    cam.release()
    cv2.destroyAllWindows()

    return embedding_path
