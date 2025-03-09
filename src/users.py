"""
Este arquivo contem um script que cria um novo usuário
"""
import os

def create_user(func: callable) -> callable:
    """
    Função decoradora que cria um novo usuário
    
    Args:
        func (callable): Função que cria a embedding do usuário

    Returns:
        callable: Função que cria um novo usuário
    """
    def wrapper() -> None:
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
