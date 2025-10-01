import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus

def otimizar_escalacao(df_jogadores: pd.DataFrame, cartolas: float, engine, usar_tudo: bool = False, objective_column: str = 'pontuacao_prevista'):
    """
    Otimiza a escalação do time com base em uma coluna objetivo (pontuação ou valorização),
    respeitando as restrições de custo, posições e clubes.

    Retorna o time escalado, a formação tática e o valor total do objetivo.
    """
    print(f"\nIniciando otimização com foco em: {objective_column}...")

    # --- Adicionar Nomes dos Clubes e Partida ---
    try:
        clubes_df = pd.read_sql("SELECT id, nome FROM dados_base.clubes", engine)
        clubes_map = clubes_df.set_index('id')['nome'].to_dict()

        df_jogadores['clube_nome'] = df_jogadores['clube_id'].map(clubes_map)
        df_jogadores['adversario_nome'] = df_jogadores['adversario_id'].map(clubes_map)

        def formatar_partida(row):
            if row['joga_em_casa'] == 1:
                return f"{row['clube_nome']} x {row['adversario_nome']}"
            else:
                return f"{row['adversario_nome']} x {row['clube_nome']}"
        df_jogadores['partida'] = df_jogadores.apply(formatar_partida, axis=1)

    except Exception as e:
        print(f"Aviso: Não foi possível adicionar informações de clube/partida. Verifique a tabela 'dados_base.clubes'. Erro: {e}")
        df_jogadores['clube_nome'] = df_jogadores['clube_id']
        df_jogadores['partida'] = 'N/A'
    # ---

    # Mapeamento de ID de posição para nome
    posicao_map = {1: 'Goleiro', 2: 'Lateral', 3: 'Zagueiro', 4: 'Meia', 5: 'Atacante', 6: 'Técnico'}
    df_jogadores['posicao'] = df_jogadores['posicao_id'].map(posicao_map)

    # Criar o problema de otimização
    prob = LpProblem("OtimizacaoCartola", LpMaximize)

    # Criar variáveis de decisão (se um jogador é escolhido ou não)
    jogadores_vars = LpVariable.dicts("Jogador", df_jogadores.index, cat='Binary')

    # Adicionar a função objetivo (maximizar a coluna objetivo)
    prob += lpSum(df_jogadores.loc[i, objective_column] * jogadores_vars[i] for i in df_jogadores.index), f"Total_{objective_column}"

    # --- Adicionar as restrições ---
    # 1. Custo total do time não pode exceder as cartolas
    prob += lpSum(df_jogadores.loc[i, 'preco_num'] * jogadores_vars[i] for i in df_jogadores.index) <= cartolas, "Custo_Total"
    # if usar_tudo:
    #     # Se o usuário quiser usar o máximo de cartoletas, adicionamos uma restrição para o custo ser próximo do total
    #     prob += lpSum(df_jogadores.loc[i, 'preco_num'] * jogadores_vars[i] for i in df_jogadores.index) >= cartolas * 0.98, "Custo_Minimo"

    # 2. Exatamente 11 jogadores + 1 técnico
    prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'posicao_id'] != 6) == 11, "Total_Jogadores_Linha"
    prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'posicao_id'] == 6) == 1, "Total_Tecnicos"
    
    # 3. Restrições de posição para garantir uma formação válida (ex: 4-4-2, 4-3-3, etc.)
    # Mínimo e máximo de jogadores por posição (excluindo técnico)
    prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'posicao_id'] == 1) == 1, "Num_Goleiros"
    prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'posicao_id'] in [2, 3]) >= 3, "Min_Defensores"
    prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'posicao_id'] in [2, 3]) <= 5, "Max_Defensores"
    prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'posicao_id'] == 4) >= 3, "Min_Meias"
    prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'posicao_id'] == 4) <= 5, "Max_Meias"
    prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'posicao_id'] == 5) >= 1, "Min_Atacantes"
    prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'posicao_id'] == 5) <= 3, "Max_Atacantes"

    # Restrição de no máximo 5 jogadores por clube foi REMOVIDA conforme solicitado.
    # for clube_id in df_jogadores['clube_id'].unique():
    #     prob += lpSum(jogadores_vars[i] for i in df_jogadores.index if df_jogadores.loc[i, 'clube_id'] == clube_id) <= 5, f"Max_Jogadores_Clube_{clube_id}"

    # Resolver o problema
    prob.solve()

    # --- Extrair e exibir os resultados ---
    if LpStatus[prob.status] == 'Optimal':
        time_escalado = []
        for i in df_jogadores.index:
            if jogadores_vars[i].varValue == 1:
                time_escalado.append(df_jogadores.loc[i])
        
        df_time = pd.DataFrame(time_escalado)
        
        # Ordenar por Posição (G, Z, L, M, A, T)
        df_time['posicao_id'] = pd.Categorical(
            df_time['posicao_id'],
            categories=[1, 3, 2, 4, 5, 6],
            ordered=True
        )
        df_time = df_time.sort_values('posicao_id')

        # Determinar a formação
        num_def = len(df_time[df_time['posicao_id'].isin([2, 3])])
        num_mei = len(df_time[df_time['posicao_id'] == 4])
        num_ata = len(df_time[df_time['posicao_id'] == 5])
        formacao = f"{num_def}-{num_mei}-{num_ata}"

        custo_total = df_time['preco_num'].sum()
        objetivo_total = df_time[objective_column].sum()

        # Eleger o capitão (maior pontuação prevista, APENAS se o foco for pontuação)
        if objective_column == 'pontuacao_prevista':
            capitao_idx = df_time['pontuacao_prevista'].idxmax()
            pontuacao_capitao = df_time.loc[capitao_idx, 'pontuacao_prevista']
            objetivo_total_final = objetivo_total + (pontuacao_capitao * 0.5)
        else:
            objetivo_total_final = objetivo_total


        # Adiciona uma linha de total ao DataFrame para exibição
        display_cols = ['apelido', 'posicao', 'clube_nome', 'partida', 'preco_num', objective_column]
        df_display = df_time[display_cols].reset_index(drop=True)
        df_display['preco_num'] = df_display['preco_num'].map(lambda x: f"{x:.2f}")
        df_display[objective_column] = df_display[objective_column].map(lambda x: f"{x:.2f}")
        
        # Marcar o capitão no DataFrame de exibição (APENAS se o foco for pontuação)
        if objective_column == 'pontuacao_prevista':
            display_capitao_idx = df_display[df_display['apelido'] == df_time.loc[capitao_idx, 'apelido']].index[0]
            pontuacao_capitao = df_time.loc[capitao_idx, 'pontuacao_prevista']
            pontuacao_capitao_final = pontuacao_capitao * 1.5
            df_display.loc[display_capitao_idx, 'apelido'] += ' (C)'
            df_display.loc[display_capitao_idx, objective_column] = f"{pontuacao_capitao:.2f} ({pontuacao_capitao_final:.2f})"

        if objective_column == 'pontuacao_prevista':
            objetivo_total_display = f"{objetivo_total:.2f} ({objetivo_total_final:.2f})"
        else:
            objetivo_total_display = f"{objetivo_total_final:.2f}"

        total_row = pd.DataFrame([{
            'apelido': '--- TOTAL ---',
            'posicao': '',
            'clube_nome': '',
            'partida': '',
            'preco_num': f"{custo_total:.2f}",
            objective_column: objetivo_total_display
        }])
        df_display = pd.concat([df_display, total_row], ignore_index=True)

        print(f"\n--- Time Otimizado ---")
        print(f"Formação: {formacao}")
        if objective_column == 'pontuacao_prevista':
            print(f"Pontuação Prevista (com Capitão): {objetivo_total_final:.2f}")
        else:
            print(f"Valorização Prevista: {objetivo_total_final:.2f} C$")
        print(f"Custo Total: C$ {custo_total:.2f}")
        print(f"\nJogadores Escalados:")
        print(df_display.to_string()) # Usar to_string() para melhor formatação

        # --- Seleção do Banco de Reservas ---
        print(f"\n--- Banco de Reservas Sugerido ---")
        df_nao_escalados = df_jogadores[~df_jogadores.index.isin(df_time.index)].copy()
        banco_reservas = []
        posicoes_reserva = {1: 'Goleiro', 2: 'Lateral', 3: 'Zagueiro', 4: 'Meia', 5: 'Atacante'} # Exclui Técnico

        for pos_id, pos_nome in posicoes_reserva.items():
            jogadores_posicao = df_nao_escalados[df_nao_escalados['posicao_id'] == pos_id]
            if not jogadores_posicao.empty:
                # Seleciona o mais barato da posição
                reserva = jogadores_posicao.sort_values(by='preco_num').iloc[0]
                banco_reservas.append(reserva)
            else:
                banco_reservas.append(pd.Series({'apelido': f'N/A ({pos_nome})', 'posicao': pos_nome, 'preco_num': 0.0, objective_column: 0.0}))

        if banco_reservas:
            df_banco = pd.DataFrame(banco_reservas)
            df_banco['posicao_id'] = pd.Categorical(
                df_banco['posicao_id'],
                categories=[1, 3, 2, 4, 5], # Ordem das posições no banco
                ordered=True
            )
            df_banco = df_banco.sort_values('posicao_id')
            
            # Adiciona uma linha de total ao DataFrame do banco
            custo_total_banco = df_banco['preco_num'].sum()
            objetivo_total_banco = df_banco[objective_column].sum()
            total_banco_row = pd.DataFrame([{
                'apelido': '--- TOTAL BANCO ---',
                'posicao': '',
                'clube_nome': '',
                'partida': '',
                'preco_num': custo_total_banco,
                objective_column: objetivo_total_banco
            }])
            df_banco_display = pd.concat([df_banco[display_cols[:6]], total_banco_row], ignore_index=True)
            print(df_banco_display.to_string())
        else:
            print("Nenhum jogador disponível para o banco de reservas.")
        # --- Fim Seleção Banco de Reservas ---

        return df_time, formacao, objetivo_total_final, df_banco
    else:
        print(f"\nNão foi possível encontrar uma escalação ótima com as restrições definidas.")
        return None, None, None, None
