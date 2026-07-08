# 🤖 AI Customer Support Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Function%20Calling-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Search-orange)](https://www.trychroma.com/)

An autonomous AI customer service agent that combines **Retrieval-Augmented Generation (RAG)** with **LLM tool calling** to handle product inquiries, customer authentication, and policy-aware responses.

## Architecture

```mermaid
flowchart LR
    User([Customer Query]) --> Agent[LLM Agent]
    Agent -->|Semantic Search| RAG[(ChromaDB\nVector Store)]
    Agent -->|Structured Filter| CSV[(Product CSV\nPandas)]
    Agent -->|Lookup / Auth| DB[(Customer DB\nmongomock)]
    Agent -->|Arithmetic| Calc[Safe Calculator]
    Agent -->|Policy Check| Policies[policies.txt]
    RAG --> Agent
    CSV --> Agent
    DB --> Agent
    Calc --> Agent
    Policies --> Agent
    Agent --> Response([Response])
```

**ReAct Loop**: The agent uses OpenAI function calling in a loop (capped at 15 iterations). It reasons about which tool to use, executes it, observes the result, and repeats until it has a final answer.

## Features

- **Semantic Product Search** — ChromaDB + `all-MiniLM-L6-v2` embeddings for natural language queries
- **Structured Filtering** — Pandas-based filtering by price, brand, category, availability
- **Customer Authentication** — Email/password verification before exposing private data
- **Policy Enforcement** — Premier discounts (5%), recommendation limits, exact no-match responses
- **Safe Calculator** — AST-based math evaluation (no `eval()`)
- **Error Recovery** — Content filter errors are caught and fed back to the LLM to self-correct

## Project Structure

| File | Description |
|------|-------------|
| `agent.py` | Core agent — tool definitions, OpenAI loop, system prompt |
| `customer_store.py` | Loads customer data into in-memory MongoDB (mongomock) |
| `fetch_product_data.py` | Downloads & curates 787 products from Kaggle's Walmart dataset |
| `prepare_dataset.py` | Validates all required files are present and correct |
| `test_agent.py` | 10 sample queries exercising all agent capabilities |
| `prompts.md` | System prompt documentation and design notes |
| `policies.txt` | 7 company policies the agent must enforce |
| `customers.json` | 50 synthetic customer records |
| `example_queries.txt` | Sample queries for reference |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate product data

Downloads the Walmart product dataset from Kaggle (no account required):

```bash
python fetch_product_data.py
```

### 3. Verify setup

```bash
python prepare_dataset.py
```

### 4. Configure environment

Copy the example env file and fill in your values:

```bash
cp .env.example .env
# Edit .env with your API key and endpoint
```

Required variables (see [`.env.example`](.env.example)):
| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `OPENAI_BASE_URL` | API endpoint URL |
| `OPENAI_MODEL` | Model name (default: `gpt-4.1-mini`) |

## Usage

```python
from agent import answer_question

response = answer_question("What scales do you have under $100?")
print(response)
```

Or run the full test suite:

```bash
python test_agent.py
```

## Design Decisions

- **Dual search strategy**: Semantic search (RAG) for natural language needs + structured Pandas filtering for hard constraints (price, brand). These can be combined in a single query.
- **Safe arithmetic**: A whitelist-based AST evaluator handles all math — the LLM never does calculations "in its head".
- **Lazy initialization**: ChromaDB index is built on first call and persisted, avoiding re-embedding on subsequent runs.
- **mongomock**: Customer data uses an in-memory MongoDB mock — same pymongo API, zero infrastructure.

## Technologies

- **Python 3.10+**
- **OpenAI API** — GPT-4.1 Mini with function calling
- **ChromaDB** — Vector similarity search
- **Sentence Transformers** — `all-MiniLM-L6-v2` embeddings
- **mongomock** — In-memory MongoDB
- **Pandas** — Structured data filtering

> **Note**: All customer data in this project is entirely fictional and generated for demonstration purposes.
