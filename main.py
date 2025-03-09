"""
Módulo principal do programa.

Este módulo oferece uma interface de linha de comando para capturar imagens de rostos e autenticar imagens de rostos com base em embeddings gerados anteriormente.

Funcionalidades:
- Capturar imagens de rostos e gerar embeddings para os rostos capturados.
- Autenticar imagens de rostos com base em embeddings gerados anteriormente.

Dependências:
- src.capture_faces: Módulo para capturar imagens de rostos e gerar embeddings.
- src.face_auth: Módulo para autenticar imagens de rostos com base em embeddings.
"""

from src import capture_face
from src import authenticate_face

def main():
    """
    Função principal do programa.
    """
    while True:
        print("\nEscolha uma opção:")
        choice = input("'c' - Capturar imagem, 'a' - Autenticar imagem ou 'q' - Sair: ").lower()
        
        match choice:
            case 'c':
                capture_face()
            case 'a':
                user = input("Digite o nome do usuário para autenticar: ")
                embedding_path = f"embedding/{user}_embedding.json"
                authenticate_face(embedding_path)
            case 'q':
                print("Saindo...")
                break

if __name__ == '__main__':
    main()
