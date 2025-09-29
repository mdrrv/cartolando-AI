def fazer_perguntas():
    """Faz as perguntas iniciais ao usuário para definir as restrições."""
    
    cartoletas = float(input("0. Quantas cartoletas você tem disponíveis? "))
    
    print("1. Qual o foco da sua escalação?")
    print("   - [1] Maior Pontuação")
    print("   - [2] Valorização (Ainda não implementado)")
    foco = int(input("   Escolha uma opção: "))
    
    usar_tudo = input("2. Deseja tentar usar todas as cartoletas? (s/n) ").lower() == 's'
    
    return {
        "cartoletas": cartoletas,
        "foco": foco,
        "usar_tudo": usar_tudo
    }
