
# App link - https://factcheck-ai.streamlit.app/

# 🛡️ FactCheck AI

**FactCheck AI** is an automated web agent that acts as a gatekeeper between your drafts and the publish button. It ingests PDF documents, extracts statistical claims and dates, and cross-references them against live web data to flag inaccuracies.

## 🚀 Features

* **PDF Ingestion:** Upload any PDF report or draft.
* **Smart Extraction:** Identifies hard facts (numbers, dates, financial figures) while ignoring fluff.
* **Live Verification:** Uses Tavily Search API to find real-time sources (no hallucinated training data).
* **LLM Adjudication:** An AI Judge evaluates the evidence vs. the claim to render a verdict (Verified/Inaccurate/False).
* **Scoring:** specific "Truth Score" for the document.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **LLM:** Google Gemini 2.0 Flash (via OpenRouter)
* **Search:** Tavily Search API

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sarthakdahiya8/FactCheck-AI.git
    cd FactCheck-AI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Secrets:**
    Create a `.streamlit/secrets.toml` file and add your keys:
    ```toml
    OPENROUTER_API_KEY = "sk-..."
    TAVILY_API_KEY = "tvly-..."
    ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```


