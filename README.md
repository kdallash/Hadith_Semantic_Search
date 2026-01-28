---
title: Hadith Semantic Search
emoji: 📚
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Hadith Semantic Search Project

## Overview

This project implements an AI-powered semantic search engine for Hadith (Islamic traditions). Unlike traditional keyword-based search tools that match exact words, this system understands the **meaning** behind queries and returns relevant Hadiths even when different wording is used.

The project uses advanced natural language processing (NLP) techniques including:
- **Semantic embeddings** using multilingual sentence transformers
- **BM25 ranking** for keyword relevance
- **Hybrid search** combining semantic and keyword approaches
- **Anchor-based retrieval** for improved accuracy
- **FAISS** for efficient similarity search

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Semantic Understanding**: Retrieves Hadiths based on meaning, not just exact word matches
- **Multilingual Support**: Works with Arabic text using multilingual models
- **Hybrid Search**: Combines semantic similarity with BM25 keyword matching for optimal results
- **Anchor-based Enhancement**: Uses subject-based anchors to improve retrieval accuracy
- **Web Interface**: Gradio-based interface for easy interaction
- **Efficient Search**: Uses FAISS for fast similarity search on large datasets
- **Evaluation Metrics**: Includes Precision@K and Recall@K for performance measurement

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

## Dataset

The project uses the `hadith_by_book.csv` dataset containing:
- **Hadith text** (matn_text)
- **Subject classifications** (main_subj)
- **Reference URLs** (xref_url)
- **Book metadata**

### Data Processing Steps

1. **Loading**: Import data from CSV
2. **Cleaning**: Remove duplicate entries and unnecessary columns
3. **Preprocessing**: Remove Arabic diacritics (tashkeel) for better matching
4. **Analysis**: Visualize text length distribution and subject categories

## Project Structure

```
hadith-semantic-search/
│
├── hadith.ipynb              # Main Jupyter notebook
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── app.py                    # Gradio web application
├── retrieval.py              # Search retrieval functions
├── utils.py                  # Utility functions
│
├── data/                     # Data directory
│   ├── hadith_embeddings.npy # Pre-computed embeddings
│   ├── bm25.pkl             # BM25 model
│   └── anchor_index.faiss   # Anchor embeddings index
│
└── hadith_by_book.csv       # Dataset
```

## Methodology

### 1. Text Preprocessing

- Remove Arabic diacritics (tashkeel) to normalize text
- Clean special characters while preserving Arabic script
- Tokenize text for BM25 processing

### 2. Embedding Generation

Uses **paraphrase-multilingual-MiniLM-L12-v2** model to create 384-dimensional embeddings that capture semantic meaning of Hadith text.

### 3. Search Approaches

#### a) Pure Semantic Search (FAISS)
- Encodes query into embedding
- Uses FAISS IndexFlatIP for cosine similarity search
- Returns top-K most similar Hadiths

#### b) Hybrid Search (BM25 + Semantic)
1. **BM25 Retrieval**: Get top-50 candidates using keyword matching
2. **Semantic Re-ranking**: Re-rank candidates using semantic similarity
3. **Score Fusion**: Combine BM25 and semantic scores with weighted average (alpha=0.8)

#### c) Enhanced Hybrid Search with Anchors
1. **Anchor Creation**: Create subject-based anchors from main topics
2. **Query-Anchor Matching**: Find relevant subject anchors for query
3. **Candidate Expansion**: Include Hadiths from relevant subjects
4. **Hybrid Scoring**: Combine BM25, semantic, and anchor signals

### 4. Evaluation

Performance measured using:
- **Precision@K**: Proportion of relevant results in top-K
- **Recall@K**: Proportion of all relevant Hadiths retrieved in top-K

Test queries cover various topics:
- Importance of intention in deeds
- Virtues of prayer
- Rights of neighbors
- Seeking knowledge
- Charity and giving


### Example Queries

```python
# Example 1: Query about intention
query = "ما هو الحديث الذي يشرح أهمية النية وأثرها في قبول الأعمال عند الله"

# Example 2: Query about charity
query = "فضل الصدقة والإنفاق في سبيل الله"

# Example 3: Query about knowledge
query = "أهمية طلب العلم وفضل العالم"
```

## Evaluation

The project includes a comprehensive evaluation framework:

### Evaluation Queries

5 carefully crafted queries with known relevant Hadith IDs:
1. **Intention (Niyyah)**: Importance of intention in accepting deeds
2. **Prayer virtues**: Excellence of prayer and its rewards
3. **Neighbor rights**: Rights and treatment of neighbors
4. **Seeking knowledge**: Importance and virtue of knowledge
5. **Charity**: Giving in the path of Allah

### Metrics

- **Precision@5**: Accuracy of top 5 results
- **Recall@5**: Coverage of relevant results in top 5
- **Average scores** across all queries

### Results Comparison

| Method | Precision@5 | Recall@5 |
|--------|-------------|----------|
| Pure Semantic (FAISS) | ~0.03| ~0.03 |
| Hybrid (BM25 + Semantic) | ~0.17 | ~0.17 |
| Enhanced (with Anchors) | ~0.79 | ~0.80 |

## Deployment

The project includes deployment-ready files:

### Files Created

1. **app.py**: Main Gradio application
2. **retrieval.py**: Core search functions
3. **utils.py**: Preprocessing utilities
4. **requirements.txt**: Dependencies

### Deployment Steps

1. Ensure all data files are in the `data/` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python app.py`
4. For production, consider using:
   - Docker containers
   - Cloud platforms (AWS, GCP, Azure)
   - Gradio Spaces for easy hosting

## Technologies Used

### Core Libraries

- **sentence-transformers**: Multilingual semantic embeddings
- **transformers**: Hugging Face transformer models
- **torch**: PyTorch deep learning framework
- **faiss-cpu**: Fast similarity search and clustering
- **rank-bm25**: BM25 ranking algorithm

### Data & Analysis

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualization

### Web Interface

- **gradio**: Interactive web interface
- **scikit-learn**: Machine learning utilities

## Results

### Key Findings

1. **Hybrid approach outperforms** pure semantic or keyword-only search
2. **Anchor-based enhancement** improves precision for subject-specific queries
3. **Arabic text preprocessing** (removing diacritics) improves matching
4. **Multilingual models** effectively capture Arabic semantic meaning

### Performance Insights

- Average query time: ~0.1-0.5 seconds
- Index size: Efficient for datasets up to 100K+ Hadiths
- Embedding dimension: 384 (balanced between accuracy and speed)

## Future Improvements

1. **Cross-encoder Re-ranking**: Add a second-stage cross-encoder for final ranking
2. **Query Expansion**: Automatically expand queries with synonyms
3. **Multi-language Support**: Add English and other language interfaces
4. **Advanced Filtering**: Filter by book, narrator, or authenticity grade
5. **Feedback Loop**: Incorporate user feedback to improve rankings
6. **GPU Acceleration**: Use FAISS GPU for faster search on large datasets
7. **Context Window**: Show surrounding Hadiths for better understanding
8. **Citation Network**: Leverage hadith-to-hadith references


### Areas for Contribution

- Improving Arabic text preprocessing
- Adding new evaluation queries
- Optimizing search algorithms
- Enhancing the web interface
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Sentence Transformers** team for multilingual models
- **FAISS** developers for efficient similarity search
- Hadith dataset providers
- Islamic scholars for categorization and verification

## Contact

For questions, suggestions, or collaboration:
- Open an issue on GitHub
- Contact: [Your Email]

---

**Note**: This is an educational project for demonstrating semantic search techniques on Islamic texts. For religious guidance, always consult qualified Islamic scholars.
