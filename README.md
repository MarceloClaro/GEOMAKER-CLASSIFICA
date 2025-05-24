# Analisador de Imagens com Deep Learning - Frontend React e App Streamlit

Este repositório contém um projeto para análise de imagens usando Deep Learning, dividido em duas partes principais:

1.  **`frontend/`**: Uma aplicação React interativa (existente) que simula e permite a exploração de conceitos de classificação e clusterização de imagens. Ideal para demonstrações e design de UI.
    *   **Chat com IA - Marcelo Claro**: Inclui um chat interativo com um assistente de IA (Marcelo Claro) que utiliza a API Gemini. Marcelo Claro pode:
        *   Perguntar sobre o tipo de classificação que está sendo realizada para contextualizar suas respostas.
        *   Analisar os resultados da simulação do modelo.
        *   Simular a consulta a "agentes de pesquisa especializados" para perguntas complexas, exibindo um log dessa atividade simulada.
        *   As respostas do Marcelo Claro podem ser vocalizadas em Português do Brasil usando a API de Síntese de Fala do navegador.
2.  **`streamlit_app/`**: Uma aplicação Python usando Streamlit, destinada a demonstrar e fornecer uma estrutura para a implementação de processamento real de Machine Learning (treinamento de modelos, inferência, XAI) e uma simulação rica de análise avançada com agentes e chat interativo.
    *   **Chat com IA - Marcelo Claro (Streamlit Edition)**: Um chat totalmente funcional integrado com a API Gemini.
        *   **Contextualização Automática**: Tenta carregar e utilizar os resultados (simulados) gerados pela seção "Configurações e Upload" do app Streamlit para fornecer respostas mais relevantes.
        *   **Interação Proativa**: Marcelo Claro pergunta sobre o "tipo de classificação" para personalizar a análise, similar ao app React.
        *   **Simulação de Agentes de Pesquisa**: Para perguntas complexas, simula a ativação de agentes especializados e exibe um log dessa atividade.
        *   **Vocalização de Respostas da IA**: As respostas da IA no chat do Streamlit podem ser ouvidas usando a biblioteca `gTTS` (Google Text-to-Speech), que gera um áudio para ser tocado diretamente no navegador.

## Estrutura do Projeto

*   **`frontend/`**: Contém a aplicação React.
    *   `index.html`, `index.tsx`, `App.tsx`, `components/`, `services/`, `types.ts`, `constants.ts`, `metadata.json`
*   **`streamlit_app/`**: Contém a aplicação Streamlit para o backend e processamento de ML.
    *   `app.py`: Ponto de entrada principal da aplicação Streamlit, incluindo toda a lógica do Chat com IA.
    *   `utils/`: Módulos auxiliares para a lógica de ML e IA.
        *   `data_loader.py`: Esqueleto para carregar e pré-processar dados.
        *   `model_trainer.py`: Esqueleto para treinamento e avaliação de modelos.
        *   `explainability.py`: Esqueleto para interpretabilidade (ex: CAM).
        *   `gemini_utils.py`: Funções para interagir com a API Gemini (Python SDK).
        *   `crewai_handler.py`: Esqueleto para definir e orquestrar agentes CrewAI (simulado).
        *   `mcp_tools_sim.py`: Esqueleto para ferramentas simuladas inspiradas no MCP que os agentes CrewAI podem usar.
        *   `tts_utils.py`: Utilitário para converter texto em fala usando `gTTS`.
    *   `assets/`: Para quaisquer arquivos estáticos que o app Streamlit possa precisar.
*   **`README.md`**: Este arquivo.
*   **`.gitignore`**: Especifica arquivos e pastas a serem ignorados pelo Git.
*   **`requirements.txt`**: Lista as dependências Python para a aplicação Streamlit e o backend, incluindo `google-generativeai`, `crewai` e `gTTS`.

## Configuração e Execução

### Frontend React (Demonstração)

(Instruções permanecem as mesmas)
... Para que o Chat com IA (Marcelo Claro) e outras funcionalidades da API Gemini funcionem, defina a variável de ambiente `API_KEY` ...

### Streamlit App (Processamento Real de ML e Simulação de Agentes)

1.  **Crie um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```
2.  **Instale as Dependências:**
    Navegue até a pasta raiz do repositório e execute:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure a Chave da API Gemini:**
    Defina a variável de ambiente `API_KEY` com sua chave da API Google Gemini.
    ```bash
    export API_KEY="SUA_CHAVE_API_AQUI" # Linux/macOS
    # set API_KEY="SUA_CHAVE_API_AQUI" # Windows Command Prompt
    # $env:API_KEY="SUA_CHAVE_API_AQUI" # Windows PowerShell
    ```
    A aplicação Streamlit verificará esta chave. Sem ela, as funcionalidades do Chat com IA não operarão.
4.  **Execute a Aplicação Streamlit:**
    Navegue até a pasta raiz do repositório e execute:
    ```bash
    streamlit run streamlit_app/app.py
    ```
    Isso deve abrir a aplicação Streamlit no seu navegador. Explore a seção "Configurações e Upload" para simular a geração de resultados, e então interaja com "Chat com IA - Marcelo Claro".

## Desenvolvimento

### Backend de Machine Learning (em `streamlit_app/utils/`)

Os arquivos em `streamlit_app/utils/` (`data_loader.py`, `model_trainer.py`, `explainability.py`) contêm esqueletos de funções. Você precisará implementar a lógica de Machine Learning real usando PyTorch ou TensorFlow dentro dessas funções. Após o processamento simulado na aba "Configurações e Upload", os dados de resultados (mock) são armazenados em `st.session_state` para serem usados pelo Chat com IA.

### Análise Avançada com Agentes (CrewAI & MCP)

*   **`crewai_handler.py`**: Contém um esqueleto para definir agentes e tarefas usando o framework CrewAI. A execução da "crew" é simulada para mostrar o fluxo de interação entre agentes e ferramentas.
*   **`mcp_tools_sim.py`**: Define ferramentas simuladas (ex: busca na web, leitura de arquivos) que os agentes CrewAI utilizariam.
*   **Importante:** A implementação de um sistema MCP real com servidores Dockerizados e a integração completa com agentes CrewAI é um trabalho de backend significativo. A presente estrutura no Streamlit serve como uma **demonstração conceitual e um ponto de partida para tal desenvolvimento**.

A funcionalidade TTS no Streamlit com `gTTS` requer conexão com a internet. A simulação de agentes e o chat contextual visam fornecer uma experiência rica e demonstrar o potencial de assistentes de IA mais avançados.
```