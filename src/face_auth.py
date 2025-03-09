"""

"""

import cv2
import json
import numpy as np
import time
from keras.api.applications import MobileNetV3Large
from keras.api.applications.mobilenet_v3 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

def authenticate_face(embedding_path: str) -> None:
    # Carregar o modelo de detecção de face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Carregar o modelo de embedding
    model = MobileNetV3Large(weights="imagenet", input_shape=(224, 224, 3), include_top=False, pooling="avg")

    # Carregar a embedding criada anteriormente
    with open(embedding_path, "r") as file:
        embedding_data = json.load(file)
    embedding = np.array(embedding_data["embedding"])

    # Função para extrair a face da imagem
    def extract_face(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return img[y:y+h, x:x+w]
        else:
            return None

    # Função para calcular a similaridade entre a face detectada e a embedding
    def calculate_similarity(face):
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        face_embedding = model.predict(face)
        similarity = cosine_similarity(embedding, face_embedding)
        return similarity

    # Iniciar a captura de vídeo
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    authenticated = False

    while True:
        ret, frame = cap.read()
        if ret:
            face = extract_face(frame)
            if face is not None:
                similarity = calculate_similarity(face)
                if similarity > 0.67:
                    authenticated = True
                    break
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        if time.time() - start_time > 7:
            break

    cap.release()
    cv2.destroyAllWindows()

    if authenticated:
        print("Autenticado com sucesso!")
        exit(0)
    else:
        print("Autenticação falhou.")
        exit(1)
