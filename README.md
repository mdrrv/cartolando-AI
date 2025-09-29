# Cartolando-AI: Assistente Inteligente para Escalação no Cartola FC

Cartolando-AI é um projeto de ciência de dados que utiliza uma abordagem híbrida de machine learning e regras de negócio para ajudar você a escalar o melhor time possível no Cartola FC. Ele permite focar em dois objetivos distintos: **maximizar a pontuação** na rodada ou **maximizar a valorização** (ganho de cartoletas).

Ele busca dados históricos, calcula métricas avançadas de desempenho, treina modelos de previsão e utiliza otimização matemática para encontrar a escalação ideal para o seu objetivo, sempre respeitando seu orçamento.

## 🚀 Como Funciona: O Pipeline de Dados e IA

O projeto funciona em um pipeline sequencial, onde cada etapa alimenta a próxima. O fluxo é orquestrado pelo arquivo principal `app.py`.

### 1. Coleta de Dados (`src/data/data_fetcher.py`)

- **Fonte:** Conecta-se diretamente à API oficial do Cartola FC.
- **Processo:** Baixa e armazena de forma incremental todos os dados relevantes:
    - **Histórico de Pontuação:** Todas as pontuações, scouts e **variações de preço** de cada jogador em todas as rodadas passadas.
    - **Partidas:** Informações sobre todos os jogos de cada rodada.
    - **Mercado:** Dados dos jogadores disponíveis para a rodada atual.
- **Armazenamento:** Os dados são salvos em um banco de dados **PostgreSQL**.

### 2. Engenharia de Features (`src/features/feature_builder.py`)

Esta é a etapa onde transformamos dados brutos em inteligência para os modelos.

- **Processo:** Utiliza a biblioteca **Pandas** em Python para calcular features a partir do histórico.
- **Cálculos Realizados:**
    - **Métricas Avançadas:** Calcula estatísticas mais inteligentes do que os scouts puros (ex: `taxa_conversao`, `balanco_duelos`, etc.).
    - **Médias Móveis:** Para cada scout e métrica avançada, calcula a média dos **últimos 3 jogos** de cada atleta, refletindo a fase atual do jogador.
- **Resultado:** A função gera dois conjuntos de dados (DataFrames):
    - `df_train`: Um grande dataset com o histórico de todos os jogadores, contendo as features e os alvos para o treinamento (`pontuacao` e `variacao_num`).
    - `df_predict`: Um dataset menor, com os jogadores da rodada atual e suas features, pronto para a previsão.

#### Tabela de Equivalência de Features

A tabela abaixo detalha as métricas avançadas (features) criadas pelo sistema e como elas são calculadas a partir dos scouts básicos do Cartola FC.

| Métrica Avançada (Feature)      | Fórmula (Scouts do Cartola FC)                                 |
| ------------------------------- | -------------------------------------------------------------- |
| **--- Ataque e Finalização ---**    |                                                                |
| `participacoes_gol`             | `G` (Gol) + `A` (Assistência)                                  |
| `total_finalizacoes`            | `G` (Gol) + `FD` (Finalização Defendida) + `FF` (Finalização pra Fora) + `FT` (Finalização na Trave) |
| `finalizacoes_alvo`             | `G` (Gol) + `FD` (Finalização Defendida)                       |
| `pontaria`                      | `finalizacoes_alvo` / `total_finalizacoes`                     |
| `taxa_conversao`                | `G` (Gol) / `total_finalizacoes`                               |
| `criacao_oportunidades`         | `A` (Assistência) + `FS` (Falta Sofrida) + `PS` (Pênalti Sofrido) |
| **--- Defesa ---**                  |                                                                |
| `total_defesas`                 | `DD` (Defesa Difícil) + `DP` (Defesa de Pênalti)               |
| `eficiencia_defensiva`          | `total_defesas` / (`G` (Gol) + `GS` (Gol Sofrido))             |
| `balanco_duelos`                | `DS` (Desarme) - `FC` (Falta Cometida)                         |
| **--- Erros e Disciplina ---**      |                                                                |
| `erros_capitais`                | `PI` (Passe Incompleto) + `GC` (Gol Contra) + `PP` (Pênalti Perdido) |
| `indice_indisciplina`           | `FC` (Falta Cometida) + (`CA` (Cartão Amarelo) * 2) + (`CV` (Cartão Vermelho) * 5) |


### 3. Treinamento e Previsão (`src/models/predict.py`)

O projeto utiliza o **XGBoost** e treina conjuntos de modelos separados para cada objetivo. Para aumentar a precisão, cada conjunto contém um modelo especialista para jogos **em casa** e outro para jogos **visitantes**.

- **Modelos de Pontuação:** Treinados para prever a `pontuacao` de cada atleta.
- **Modelos de Valorização:** Treinados para prever a `variacao_num` (a valorização em cartoletas). Estes modelos são usados a partir da Rodada 2.

### 4. Otimização da Escalação (`src/models/optimization.py`)

Esta é a etapa final, que utiliza a biblioteca **PuLP** para resolver o problema de Programação Linear.

- **Abordagem Híbrida para Valorização:**
    - **Rodada 1:** Utiliza uma regra de negócio específica. A "pontuação" a ser otimizada é calculada como: `Pontuação Prevista - (0.46 * Preço)`.
    - **Rodada 2 em diante:** Utiliza a `valorizacao_prevista` gerada pelo modelo de machine learning.
- **Objetivo Flexível:** O otimizador encontra a combinação de 12 jogadores que maximiza o objetivo escolhido (`pontuacao_prevista` ou `valorizacao_prevista`).
- **Restrições:** A escalação ideal respeita as regras do Cartola (orçamento, posições válidas, etc.).
- **Capitão:** A lógica de eleger um capitão e aplicar seu bônus de pontuação é ativada **apenas** quando o foco da escalação é "Maior Pontuação".

## 🏗️ Estrutura do Projeto

```
/
├── app.py                  # Ponto de entrada principal da aplicação
├── requirements.txt        # Dependências do Python
├── README.md               # Esta documentação
└── src/
    ├── data/
    │   ├── data_fetcher.py # Módulo para buscar e salvar dados da API
    │   └── scouts_manual.py# Dicionário de scouts
    ├── features/
    │   └── feature_builder.py # Módulo para engenharia de features
    ├── models/
    │   ├── predict.py      # Módulo para treinamento e previsão
    │   └── optimization.py # Módulo para otimização da escalação
    └── ui/
        └── cli.py          # Módulo para a interface de linha de comando
```

## ⚙️ Como Executar

### Pré-requisitos
- Python 3.10+
- Um servidor PostgreSQL em execução.

### 1. Instalação
Clone o repositório e instale as dependências:
```bash
git clone <url_do_repositorio>
cd cartolando
pip install -r requirements.txt
```

### 2. Variáveis de Ambiente
Crie um arquivo chamado `.env` na raiz do projeto e preencha com suas credenciais do banco de dados:
```
DB_USER=seu_usuario
DB_PASSWORD=sua_senha
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cartola_db
```

### 3. Execução
Para rodar a aplicação, basta executar o `app.py`:
```bash
python app.py
```
O programa fará as perguntas iniciais e guiará você pelo processo, permitindo escolher entre focar em pontuação ou valorização.
