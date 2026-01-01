# Fake Political News Detection Using BERT & RoBERTa

An AI-powered system that uses advanced Natural Language Processing and transformer models to detect fake political news with high accuracy.

## 📋 Overview

This project addresses the critical challenge of misinformation in political news by developing a deep learning-based classification system. Using RoBERTa (a robustly optimized BERT variant) combined with Named Entity Recognition and political jargon analysis, the model achieves 88.45% accuracy in distinguishing real from fake political news articles.

## 🎯 Key Features

- **Advanced Transformer Models**: Utilizes RoBERTa-base with custom classification layer
- **Named Entity Recognition**: Extracts political entities (politicians, organizations, locations) for contextual analysis
- **Political Jargon Detection**: Custom feature engineering to identify political-specific terminology
- **High Accuracy**: 88.45% average accuracy across 5-fold cross-validation
- **Comprehensive Evaluation**: Includes precision, recall, F1-score, and ROC-AUC metrics

## 🏗️ Architecture

The system follows a multi-layered approach:

1. **Input Processing**: Text cleaning, normalization, and tokenization using RoBERTa tokenizer
2. **Feature Extraction**: 
   - Contextual embeddings from RoBERTa (768-dimensional vectors)
   - NER features using SpaCy's `en_core_web_sm` model
   - Political jargon count features
3. **Classification**: Custom classifier combining RoBERTa output with engineered features
4. **Training**: 5-fold cross-validation with class balancing and early stopping

## 📊 Results

| Model | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score (Fake) | ROC-AUC |
|-------|----------|------------------|---------------|-----------------|---------|
| Baseline BERT | 82.55% | 0.72 | 0.87 | 0.79 | 0.8957 |
| **RoBERTa (Optimized)** | **88.45%** | **0.84** | **0.90** | **0.86** | **0.9260** |

### Performance Across Folds
- Fold 1: 90.09%
- Fold 2: 91.47%
- Fold 3: 87.68%
- Fold 4: 86.26%
- Fold 5: 86.73%

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Deep Learning**: PyTorch
- **Transformers**: Hugging Face Transformers (RoBERTa, BERT)
- **NLP**: SpaCy
- **Data Processing**: Pandas, NumPy
- **Evaluation**: scikit-learn
- **Development**: Google Colab, M1 MacBook with MPS acceleration

## 📁 Dataset

The project uses the **FakeNewsNet** dataset:
- **Total Articles**: 1,056 political news articles
- **Real News**: 624 articles (60%)
- **Fake News**: 432 articles (40%)
- **Sources**: News websites (90.8%), Twitter (8.4%), Facebook (0.75%)


## 📈 Model Training

The model was trained with the following configuration:
- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with weight decay
- **Epochs**: Up to 10 with early stopping (patience=2)
- **Validation Split**: 10% of training data
- **Cross-Validation**: 5-fold stratified

## 🔍 Key Findings

1. **Feature Engineering Impact**: Adding NER and political jargon features improved accuracy by 5.9%
2. **Model Selection**: RoBERTa outperformed BERT due to enhanced pre-training (30B vs 3.3B words)
3. **False Positives**: Real news with emotional/dramatic language sometimes misclassified
4. **Limitations**: Model struggles with satirical content and lacks true external context awareness


## ⚖️ Ethics & Limitations

**Ethical Considerations**:
- Model should be used as a reference tool, not definitive fact-checker
- Transparent communication of limitations to prevent misuse
- Potential for false positives affecting credible journalism

**Limitations**:
- Limited to textual features only (no propagation analysis)
- Computational constraints (trained with 128 tokens vs full 512)
- Dataset size (1,000 articles) may limit generalization
- No access to external verification databases

## 📄 License

This project is for academic and research purposes.


## 🙏 Acknowledgments

- FakeNewsNet dataset providers
- Hugging Face for transformer models
- SpaCy for NLP tools
- Research papers that guided this work (Devlin et al., 2019; Lample et al., 2016)

