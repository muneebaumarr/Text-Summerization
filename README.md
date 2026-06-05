# Text Summarization with LangChain & Groq

A professional AI-powered PDF summarization app built with LangChain and Groq.

---

## Features

- Upload any PDF and generate a structured summary
- Choose **summary length**: short, medium, or long
- Choose **output format**: paragraph or bullet points
- Choose **writing style**: technical, executive, academic, or simple
- Map-reduce pipeline handles large documents without hitting token limits
- Download the final summary as a PDF

---

## Project Structure

```
Text Summerization/
├── app.py            # Streamlit web app
├── requirements.txt  # Python dependencies
├── .env              # API keys (not committed to version control)
└── README.md
```

---

## Setup

### 1. Clone / navigate to the project

```bash
cd "Text Summerization"
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

---

## Usage

### Web App

```bash
streamlit run app.py
```

1. Open the browser at `http://localhost:8501`
2. Upload a PDF file
3. Select your preferred length, format, and writing style from the sidebar
4. Click **Generate Summary**
5. Download the result as a PDF

---

## How It Works

The web app uses a **map-reduce** summarization strategy:

1. **Map** — each chunk of the PDF is summarized independently
2. **Hierarchical Reduce** — summaries are merged in batches of 5 to stay within the model's token-per-minute limit
3. **Output** — the final summary is rendered in the app and available as a downloadable PDF

This approach handles large documents without hitting Groq's TPM rate limits.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web UI |
| `langchain-groq` | Groq LLM integration |
| `langchain-core` | Prompt templates, output parsers |
| `langchain-community` | PDF document loader |
| `langchain-text-splitters` | Chunk documents |
| `reportlab` | Generate downloadable PDFs |
| `python-dotenv` | Load environment variables |
| `pypdf` | Parse PDF files |

**Model:** `qwen/qwen3-32b` via Groq
