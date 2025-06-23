# Book Recommendation System - Data Preprocessing

This document outlines the comprehensive data preprocessing pipeline for the book recommendation system using the 7K Books with Metadata dataset.

## Overview

The preprocessing pipeline transforms raw book data into a clean, feature-rich dataset suitable for machine learning models, particularly for semantic book recommendation systems.

## Dataset Information

- **Source**: 7K Books with Metadata from Kaggle
- **Original Columns**: 12 columns including ISBN, title, authors, categories, description, ratings, etc.
- **Target**: Clean dataset with engineered features for recommendation algorithms

## Preprocessing Pipeline

### 1. Data Loading and Exploration
- Load the CSV dataset
- Display basic information (shape, data types, missing values)
- Show sample data for initial assessment

### 2. Missing Value Handling
- **Text fields**: Fill empty strings for subtitle and description
- **Categorical fields**: Fill 'Unknown' for missing authors and categories
- **Numerical fields**: Use median values for missing published_year, average_rating, num_pages, ratings_count

### 3. Data Cleaning and Validation
- **Remove duplicates**: Eliminate duplicate book entries
- **Text cleaning**: 
  - Convert to lowercase
  - Remove special characters and extra whitespace
  - Remove numbers from text fields
- **Data validation**:
  - Published year: 1800-2024 range
  - Average rating: 0-5 range
  - Number of pages: 1-5000 range
  - Ratings count: Non-negative values

### 4. Feature Engineering

#### Text-Based Features
- **Text length features**: title_length, subtitle_length, description_length
- **Preprocessed text**: description_processed (lemmatized, stopwords removed)

#### Author Features
- **num_authors**: Count of authors per book
- **primary_author**: First author listed
- **author_list**: List of all authors

#### Category Features
- **primary_category**: Main category
- **num_categories**: Number of categories assigned
- **category_list**: List of all categories

#### Sentiment Analysis Features
- **textblob_polarity**: Sentiment polarity (-1 to 1)
- **textblob_subjectivity**: Sentiment subjectivity (0 to 1)
- **vader_compound**: VADER compound sentiment score
- **vader_positive**: VADER positive sentiment score

#### Temporal Features
- **publication_decade**: Decade of publication (e.g., 1990, 2000, 2010)

#### Rating Features
- **rating_category**: Categorical rating (Poor, Fair, Good, Very Good, Excellent)
- **popularity_score**: Combined rating and popularity metric

### 5. Feature Normalization
- **Min-max normalization**: Applied to numerical features
- **Features normalized**: title_length, subtitle_length, description_length, num_authors, num_categories, num_pages, ratings_count

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Preprocessing Pipeline
```python
from preprocessing_script import BookDataPreprocessor

# Initialize preprocessor
preprocessor = BookDataPreprocessor()

# Run full preprocessing pipeline
processed_df, summary = preprocessor.run_full_preprocessing(
    input_path='data/books.csv',
    output_path='data/books_processed.csv'
)
```

### Individual Steps
You can also run individual preprocessing steps:

```python
# Load data
df = preprocessor.load_data('data/books.csv')

# Handle missing values
df = preprocessor.handle_missing_values(df)

# Clean and validate
df = preprocessor.clean_and_validate_data(df)

# Create features
df = preprocessor.create_features(df)

# Normalize features
df = preprocessor.normalize_features(df)
```

## Output Features

The processed dataset includes the following new features:

### Original Features (Cleaned)
- isbn13, isbn10, title, subtitle, authors, categories
- thumbnail, description, published_year, average_rating
- num_pages, ratings_count

### Engineered Features
- **Text Features**: title_length, subtitle_length, description_length, description_processed
- **Author Features**: num_authors, primary_author
- **Category Features**: primary_category, num_categories
- **Sentiment Features**: textblob_polarity, textblob_subjectivity, vader_compound, vader_positive
- **Temporal Features**: publication_decade
- **Rating Features**: rating_category, popularity_score
- **Normalized Features**: *_normalized versions of numerical features

## Quality Metrics

The preprocessing pipeline provides:
- **Data quality report**: Missing values, duplicates, data types
- **Summary statistics**: Total books, unique authors/categories, publication range
- **Validation checks**: Data range validation, outlier detection

## Benefits for Recommendation Systems

1. **Semantic Analysis**: Preprocessed text enables better semantic understanding
2. **Feature Richness**: Multiple engineered features capture different aspects of books
3. **Data Quality**: Clean, validated data improves model performance
4. **Scalability**: Normalized features work well with various ML algorithms
5. **Interpretability**: Categorical features and sentiment scores provide interpretable insights

## Next Steps

After preprocessing, the dataset is ready for:
- **Embedding Generation**: Using preprocessed text for semantic embeddings
- **Feature Selection**: Choosing relevant features for recommendation models
- **Model Training**: Training recommendation algorithms
- **Evaluation**: Testing recommendation quality

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- nltk: Natural language processing
- textblob: Sentiment analysis
- vaderSentiment: Advanced sentiment analysis
- scikit-learn: Machine learning utilities
- matplotlib/seaborn/plotly: Visualization
- wordcloud: Text visualization 