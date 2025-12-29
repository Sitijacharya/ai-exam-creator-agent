# 🎓 AI Exam Creator - RAG-Powered Question Generator

An intelligent exam generation system that leverages **Retrieval-Augmented Generation (RAG)** to create high-quality, context-aware assessment questions from textbook materials. Built with OpenAI GPT models, ChromaDB vector store, and advanced RAGAS evaluation metrics.

---

## ✨ Features

### 🔍 **Smart RAG System**
- **Vector-based retrieval** using ChromaDB for efficient document search
- **Cross-encoder reranking** for improved context relevance
- **Hybrid generation**: Uses textbook context when available, falls back to LLM knowledge otherwise
- **Relevance threshold control** to ensure quality retrieval

### 📚 **Flexible Exam Generation**
- **Multiple question types**: MCQ, Short Answer, Long Answer, True/False
- **Difficulty levels**: Easy, Medium, Hard
- **Weighted distribution**: Allocate questions based on syllabus credit hours
- **Automatic scoring**: Configurable marks per question

### 📊 **Advanced Evaluation (RAGAS Framework)**
- **Faithfulness** (40%): Detects hallucinations, ensures grounding in source material
- **Answer Relevance** (30%): Verifies questions test the intended topics
- **Context Precision** (20%): Measures retrieval quality
- **Context Recall** (10%): Assesses context sufficiency
- **Overall RAGAS Score**: Industry-standard RAG quality metric (A-F grading)

### 🎨 **Interactive UI**
- **Streamlit-based interface** with real-time feedback
- **Document upload**: Add PDFs to knowledge base
- **Syllabus parser**: Extract topics and weights automatically
- **Live evaluation dashboard**: Visualize question quality metrics
- **Export options**: Download exam papers in multiple formats

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE (Streamlit)              │
│  - Upload PDFs    - Define syllabus    - Configure exam         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    EXAM GENERATION ENGINE                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  Retriever   │──▶│  Reranker    │──▶│  Generator   │        │
│  │(Vector Search)│   │(Cross-Encoder)│   │  (GPT-4)     │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   KNOWLEDGE BASE (ChromaDB)                      │
│  - Embedded PDF chunks    - Metadata    - Similarity search     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    EVALUATION LAYER (RAGAS)                      │
│  - Faithfulness check    - Relevance scoring    - Quality report│
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- **Python 3.9+**
- **OpenAI API Key**
- **Git**

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/sitijacharya-cloud/ai-exam-generator.git
cd ai-exam-generator
```

2. **Create virtual environment**
```bash
python -m venv examcreator
source examcreator/bin/activate  # On Windows: examcreator\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "LLM_MODEL=gpt-4o-mini" >> .env
echo "TEMPERATURE=0.3" >> .env
echo "CHUNK_SIZE=1000" >> .env
echo "CHUNK_OVERLAP=200" >> .env
echo "TOP_K_CONTEXTS=3" >> .env
echo "RELEVANCE_THRESHOLD=0.5" >> .env
```

5. **Set up knowledge base** (Optional - for initial setup)
```bash
# Add PDF files to knowledge-base/books/
python scripts/setup_knowledge_base.py
```

---

## 🚀 Usage

### Starting the Application
```bash
streamlit run ui/app.py
```

The app will open at `http://localhost:8501`

### Workflow

#### 1️⃣ **Upload Documents**
- Navigate to the sidebar
- Upload PDF textbooks/reference materials
- System automatically processes and indexes content

#### 2️⃣ **Define Syllabus**
Enter syllabus with credit hours (optional):
```
Unit 1: Introduction to Programming (3 credits)
Unit 2: Data Structures (4 credits)
Unit 3: Algorithms (5 credits)
```

Or manually add topics with weights.

#### 3️⃣ **Configure Exam**
- **Exam Title**: e.g., "Midterm Exam - CS101"
- **Instructions**: Custom instructions for students
- **Question Types**: Select MCQ, Short Answer, etc.
- **Difficulty**: Choose Easy, Medium, Hard
- **Marks**: Set marks per question

#### 4️⃣ **Generate Exam**
Click **"Generate Exam"** button. The system will:
1. Retrieve relevant contexts from vector store
2. Rerank contexts using cross-encoder
3. Generate questions using GPT-4
4. Validate and format output

#### 5️⃣ **Evaluate Quality**
Navigate to **"RAG Evaluation"** tab to:
- View RAGAS scores (A-F grading)
- Check faithfulness (hallucination detection)
- Analyze context relevance
- Review detailed metrics per question

#### 6️⃣ **Export**
Download exam in preferred format (JSON, PDF, DOCX)

---

## 📂 Project Structure

```
exam-creator/
├── config/                     # Configuration
│   ├── __init__.py
│   └── settings.py            # Environment variables, API keys
│
├── data/                       # Data processing
│   ├── __init__.py
│   ├── loader.py              # PDF loading utilities
│   └── processor.py           # Text chunking and preprocessing
│
├── vectorstore/                # Vector database
│   ├── __init__.py
│   ├── embeddings.py          # OpenAI embedding manager
│   └── store.py               # ChromaDB integration
│
├── generation/                 # Question generation
│   ├── __init__.py
│   ├── retriever.py           # Context retrieval with reranking
│   ├── reranker.py            # Cross-encoder reranker
│   └── question_generator.py  # LLM-based question generation
│
├── evaluation/                 # Quality assessment
│   ├── __init__.py
│   └── rag_evaluator.py       # RAGAS metrics evaluation
│
├── models/                     # Data models
│   ├── __init__.py
│   └── schemas.py             # Pydantic models for questions, configs
│
├── ui/                         # User interface
│   ├── __init__.py
│   └── app.py                 # Streamlit application
│
├── scripts/                    # Utility scripts
│   └── setup_knowledge_base.py # Initial KB setup
│
├── knowledge-base/             # Source documents
│   └── books/                 # PDF textbooks
│
├── chroma_db/                  # Vector store (auto-generated)
│
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
└── README.md                   # This file
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | **Required** |
| `LLM_MODEL` | OpenAI model for generation | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `TEMPERATURE` | LLM temperature (0-1) | `0.3` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_CONTEXTS` | Contexts to retrieve | `3` |
| `RELEVANCE_THRESHOLD` | Min relevance score (0-1) | `0.5` |

### Relevance Threshold Guide
- **0.3-0.4**: Lenient (broader matches)
- **0.5-0.6**: Moderate (recommended)
- **0.7-0.8**: Strict (only highly relevant)

---

## 🧪 Testing

Run tests with pytest:

```bash
# Test reranker
pytest test_ranker.py -v

# Test setup
pytest test_setup.py -v

# Test all
pytest -v
```

---

## 📊 Evaluation Metrics

### RAGAS Score Breakdown

| Component | Weight | Description |
|-----------|--------|-------------|
| **Faithfulness** | 40% | No hallucinations, grounded in context |
| **Answer Relevance** | 30% | On-topic, tests intended knowledge |
| **Context Precision** | 20% | Retrieved contexts are relevant |
| **Context Recall** | 10% | Sufficient context retrieved |

### Score Interpretation

| Grade | Score Range | Status |
|-------|-------------|--------|
| **A** | ≥ 80% | Excellent quality |
| **B** | 70-79% | Good quality |
| **C** | 60-69% | Fair, minor issues |
| **D** | 50-59% | Needs improvement |
| **F** | < 50% | Critical issues |

---

## 🔧 Advanced Features

### Cross-Encoder Reranking

Improves retrieval quality by rescoring contexts:
```python
# In generation/reranker.py
reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked_contexts = reranker.rerank(query, contexts, top_k=5)
```

### Weighted Question Distribution

Automatically distribute questions by syllabus weights:
```python
# Topics with credit hours
topics = [
    ("Introduction", 3),  # 3 credits
    ("Advanced Topics", 5)  # 5 credits
]
# System allocates questions proportionally
```

### Custom Evaluation

Extend the evaluator for custom metrics:
```python
from evaluation.rag_evaluator import RAGEvaluator

evaluator = RAGEvaluator()
result = evaluator.evaluate_faithfulness(question)
print(f"Faithfulness: {result['score']:.2f}")
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. ChromaDB connection errors**
```bash
# Clear and rebuild database
rm -rf chroma_db/
python scripts/setup_knowledge_base.py
```

**2. Out of memory during embedding**
```bash
# Reduce chunk size in .env
CHUNK_SIZE=500
```

**3. Low RAGAS scores**
- Check `RELEVANCE_THRESHOLD` (try lowering to 0.4)
- Ensure PDFs are high-quality and relevant
- Verify OpenAI API key has GPT-4 access

**4. Reranker not loading**
```bash
pip install sentence-transformers --upgrade
```

---

## 🚦 Roadmap

- [ ] Support for multiple languages
- [ ] Diagram/image question generation
- [ ] Integration with LMS platforms (Moodle, Canvas)
- [ ] Fine-tuned models for specific domains
- [ ] Real-time collaboration features
- [ ] Mobile app support

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Siti Jacharya**
- GitHub: [@sitijacharya-cloud](https://github.com/sitijacharya-cloud)

---

## 🙏 Acknowledgments

- **OpenAI** for GPT models and embeddings
- **ChromaDB** for vector database
- **Streamlit** for rapid UI development
- **RAGAS Framework** for evaluation methodology
- **Sentence Transformers** for cross-encoder reranking


