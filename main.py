"""
"""

from src import capture_face
from src import authenticate_face

def main():

    while True:
        print("BEM VINDO!!!")
        print("Escolha uma opção:")
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
