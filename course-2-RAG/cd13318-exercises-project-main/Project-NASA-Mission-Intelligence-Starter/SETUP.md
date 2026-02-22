# NASA Mission Intelligence RAG System - Setup Guide

## Prerequisites
- Python 3.8+
- OpenAI API key (get one from https://platform.openai.com/account/api-keys)

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Your OpenAI API Key
You can provide your API key in two ways:

**Option A: Command-line argument (recommended for VOC workspace)**
```bash
python embedding_pipeline.py --openai-key YOUR_OPENAI_API_KEY --data-path data_text
```

**Option B: Environment variable (optional)**
Create a `.env` file in the project directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Project

### Step 1: Process Documents and Create Embeddings
This step processes all text documents in the `data_text` folder and creates a ChromaDB collection with embeddings.

```bash
python embedding_pipeline.py --openai-key YOUR_OPENAI_API_KEY --data-path data_text
```

**Important Parameters:**
- `--openai-key`: Your OpenAI API key (starts with `sk-`)
- `--data-path`: Path to the folder containing text files (default: `data_text`)
- `--chunk-size`: Size of text chunks (default: 500)
- `--chunk-overlap`: Overlap between chunks (default: 100)
- `--update-mode`: How to handle existing documents: `skip`, `update`, or `replace` (default: `skip`)

**Example with custom settings:**
```bash
python embedding_pipeline.py \
  --openai-key YOUR_OPENAI_API_KEY \
  --data-path data_text \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --update-mode replace
```

**Expected Output:**
You should see logs indicating:
- Number of files found (apollo_11: 6 files, apollo_13: 3 files, challenger: 3 files)
- Processing progress for each file
- Total chunks created and documents added to the collection

### Step 2: Launch the Chat Application
After processing the documents, start the Streamlit chat interface:

```bash
streamlit run chat.py
```

Or if you need to specify the OpenAI key:
```bash
streamlit run chat.py -- --openai-key YOUR_OPENAI_API_KEY
```

**The chat interface will open in your browser** (usually at http://localhost:8501)

### Step 3: Test with Sample Questions
Try asking questions like:
- "What happened during the Apollo 11 moon landing?"
- "Tell me about the Apollo 13 mission problems"
- "What caused the Challenger disaster?"
- "Who were the crew members of Apollo 11?"
- "How did Apollo 13 crew survive?"

## File Descriptions

### Core Python Files
1. **embedding_pipeline.py**: Processes text documents and creates ChromaDB embeddings
2. **rag_client.py**: Handles retrieval of relevant documents from ChromaDB
3. **llm_client.py**: Manages communication with OpenAI's LLM
4. **ragas_evaluator.py**: Evaluates RAG system responses using RAGAS metrics
5. **chat.py**: Streamlit chat interface with real-time evaluation

### Data Structure
```
data_text/
├── apollo11/          # Apollo 11 mission documents
├── apollo13/          # Apollo 13 mission documents
└── challenger/        # Challenger mission documents
```

## Troubleshooting

### SQLite3 Version Error
If you see: `RuntimeError: Your system has an unsupported version of sqlite3`

**Solution:** The required `pysqlite3-binary` package is already added to `requirements.txt`. Make sure you've run:
```bash
pip install -r requirements.txt
```

### API Key Errors
If you see: `Error code: 401 - Incorrect API key provided`

**Solution:** 
- Make sure your API key starts with `sk-` (not `voc-`)
- Get a valid key from https://platform.openai.com/account/api-keys
- Use the correct format in the command line

### No Files Processed (All Zeros)
If the embedding pipeline shows 0 files processed:

**Solution:**
- Check that the `data_text` folder exists and contains subdirectories
- Make sure you're using `--data-path data_text` in the command
- Verify that text files exist in `data_text/apollo11/`, `data_text/apollo13/`, and `data_text/challenger/`

## Workflow Summary

1. **Install dependencies** → `pip install -r requirements.txt`
2. **Run embedding pipeline** → `python embedding_pipeline.py --openai-key YOUR_KEY --data-path data_text`
3. **Launch chat app** → `streamlit run chat.py`
4. **Test with questions** → Ask about NASA missions in the chat interface
5. **Review evaluation scores** → Check RAGAS metrics displayed after each response

## Project Submission Checklist

- [ ] All TODO items completed in all Python files
- [ ] Embedding pipeline runs successfully
- [ ] Chat application works and responds to questions
- [ ] RAGAS evaluation scores are displayed
- [ ] `evaluation_dataset.txt` created with sample questions and expected answers
- [ ] All code is clean and well-documented
- [ ] Ready to zip for submission

## Support
For issues or questions, refer to:
- ChromaDB docs: https://docs.trychroma.com/
- OpenAI API docs: https://platform.openai.com/docs
- RAGAS docs: https://docs.ragas.io/
