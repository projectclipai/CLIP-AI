# CLIPAI - Sentiment Analysis using OpenAI CLIP

A sentiment analysis project that leverages OpenAI's CLIP model to classify Amazon product reviews into positive, neutral, and negative sentiments using zero-shot learning.

## Overview

This project demonstrates how to use CLIP's multimodal capabilities for text-based sentiment analysis without traditional training data. By computing semantic similarity between review text and predefined sentiment prompts, the system can accurately classify customer sentiments.

## Key Features

- **Zero-shot sentiment classification** using CLIP ViT-B/32 model
- **Large-scale processing** of 16,074 Amazon product reviews
- **Semantic similarity analysis** with cosine similarity metrics
- **Comparative analysis** between CLIP predictions and star ratings
- **Data visualization** with comprehensive charts and graphs
- **Text preprocessing** pipeline for clean data analysis

## Dataset

The project analyzes Amazon product reviews containing:
- Product information and brand details
- Customer ratings (1-5 stars)
- Review text content
- Review engagement metrics
- **Total**: 16,074 reviews across various product categories

## Methodology

1. **Data Preprocessing**: Text cleaning, normalization, and preparation
2. **Model Setup**: Loading CLIP ViT-B/32 with predefined sentiment prompts
3. **Embedding Generation**: Converting text to high-dimensional vectors
4. **Similarity Computation**: Calculating cosine similarity between reviews and sentiment prompts
5. **Classification**: Assigning sentiment labels based on highest similarity scores
6. **Analysis**: Comparing results with traditional star ratings

## Results

### Sentiment Distribution
- **Negative**: 7,315 reviews (45.5%)
- **Neutral**: 6,120 reviews (38.1%)
- **Positive**: 2,639 reviews (16.4%)

### Key Insights
- Discovered significant mismatches between star ratings and actual text sentiment
- 66% of reviews showed sentiment-rating discrepancies
- CLIP model revealed more nuanced sentiment patterns than simple star ratings

## Technical Specifications

- **Model**: OpenAI CLIP ViT-B/32
- **Framework**: PyTorch
- **Embedding Dimension**: 512
- **Processing**: Batch processing (64 reviews per batch)
- **Similarity Metric**: Cosine similarity
- **Environment**: Google Colab with GPU acceleration

## Visualizations

The project includes comprehensive data visualizations:
- Sentiment distribution pie charts
- Rating vs sentiment comparison bar charts
- Mismatch analysis visualizations
- Statistical summary charts

## Applications

- **E-commerce Analytics**: Product review sentiment monitoring
- **Market Research**: Customer opinion analysis
- **Brand Management**: Reputation tracking and analysis
- **Recommendation Systems**: Enhanced product suggestions
- **Customer Service**: Automated feedback categorization

## Requirements

- Python 3.7+
- PyTorch
- OpenAI CLIP
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab (recommended)

## File Structure

```
vignan/
├── CLIPAI (1).ipynb                        # Main analysis notebook
├── README.md                               # Project documentation
└── final_review_sentiment_analysis.csv    # Results output
```

## Future Enhancements

- Multi-language sentiment analysis support
- Aspect-based sentiment classification
- Real-time sentiment monitoring dashboard
- Integration with additional NLP models
- Fine-tuning on domain-specific datasets

## Academic Context

This project demonstrates the practical application of state-of-the-art multimodal AI models for natural language processing tasks, showcasing the versatility of CLIP beyond its traditional image-text applications.

---

*Developed as part of a final year project exploring innovative applications of transformer-based models in sentiment analysis.*