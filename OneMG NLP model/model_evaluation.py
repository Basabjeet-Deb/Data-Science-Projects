"""
Comprehensive Model Evaluation for Medicine Recommendation System
Calculates: Confusion Matrix, Accuracy, Precision, Recall, F1-Score, and more
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import os

# Setup NLTK
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Dataset/onemg.csv')
if df.isnull().sum().any():
    df.fillna('', inplace=True)

print(f"Loaded {len(df)} medicines")

# Preprocessing functions
def remove_stopwords(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(filtered_words)
    return text

# Preprocess data
print("Preprocessing data...")
df['Uses_cleaned'] = df['Uses'].apply(remove_stopwords)
df['Uses_tokens'] = df['Uses_cleaned'].apply(lambda x: word_tokenize(x) if isinstance(x, str) else [])
df['Uses_lemmatized'] = df['Uses_tokens'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x] if isinstance(x, list) else x)
df['Uses_for_tfidf'] = df['Uses_lemmatized'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

# TF-IDF Vectorization
print("Creating TF-IDF matrix...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=200,
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 3),
    sublinear_tf=True,
    use_idf=True,
    smooth_idf=True,
    analyzer='word',
    token_pattern=r'\w{2,}',
    lowercase=True
)

tfidf_matrix = tfidf_vectorizer.fit_transform(df['Uses_for_tfidf'])
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

# Recommendation function
def recommend_medicines(user_problem, top_n=5):
    user_cleaned = remove_stopwords(user_problem)
    user_tokens = word_tokenize(user_cleaned) if isinstance(user_cleaned, str) else []
    user_lemmatized = [lemmatizer.lemmatize(token) for token in user_tokens]
    user_text = ' '.join(user_lemmatized)
    
    if not user_text.strip():
        return pd.DataFrame()
    
    user_tfidf = tfidf_vectorizer.transform([user_text])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)[0]
    
    top_indices = similarities.argsort()[::-1][:top_n]
    top_similarities = similarities[top_indices]
    
    # Normalize scores
    max_sim = top_similarities.max()
    if max_sim > 0.5:
        normalized_scores = (top_similarities / max_sim * 100)
    elif max_sim > 0.1:
        normalized_scores = (top_similarities / max_sim * 85)
    elif max_sim > 0:
        normalized_scores = (top_similarities / max_sim * 60) + 30
    else:
        normalized_scores = np.array([20, 15, 10, 8, 5][:len(top_similarities)])
    
    normalized_scores = np.clip(normalized_scores, 5, 100)
    
    recommendations = pd.DataFrame({
        'Drug_Name': df['Drug_Name'].iloc[top_indices].values,
        'Uses': df['Uses'].iloc[top_indices].values,
        'Similarity_Score': similarities[top_indices],
        'Accuracy_Score': normalized_scores
    })
    
    return recommendations

# Create test dataset with ground truth
print("\n" + "="*80)
print("CREATING TEST DATASET")
print("="*80)

test_cases = [
    # Format: (query, expected_category, expected_keywords)
    ("fever and pain", "pain_relief", ["fever", "pain"]),
    ("headache", "pain_relief", ["headache", "pain"]),
    ("stomach pain", "digestive", ["stomach", "gastric", "acid"]),
    ("acidity", "digestive", ["acid", "reflux", "gastric"]),
    ("high blood pressure", "cardiovascular", ["hypertension", "blood pressure"]),
    ("diabetes", "metabolic", ["diabetes", "blood sugar"]),
    ("anxiety", "mental_health", ["anxiety", "stress"]),
    ("cough and cold", "respiratory", ["cough", "cold", "flu"]),
    ("asthma", "respiratory", ["asthma", "breathing"]),
    ("allergy", "immune", ["allergy", "allergic"]),
]

# Evaluate model
print("\n" + "="*80)
print("EVALUATING MODEL")
print("="*80)

y_true = []
y_pred = []
detailed_results = []

for query, expected_category, expected_keywords in test_cases:
    recommendations = recommend_medicines(query, top_n=5)
    
    if len(recommendations) > 0:
        # Check if top recommendation contains expected keywords
        top_result = recommendations.iloc[0]
        uses_text = str(top_result['Uses']).lower()
        
        # Check if any expected keyword is in the result
        match_found = any(keyword.lower() in uses_text for keyword in expected_keywords)
        
        y_true.append(1)  # Expected: relevant result
        y_pred.append(1 if match_found else 0)  # Predicted: relevant or not
        
        detailed_results.append({
            'Query': query,
            'Expected_Category': expected_category,
            'Top_Drug': top_result['Drug_Name'],
            'Accuracy_Score': top_result['Accuracy_Score'],
            'Match_Found': match_found
        })
    else:
        y_true.append(1)
        y_pred.append(0)
        detailed_results.append({
            'Query': query,
            'Expected_Category': expected_category,
            'Top_Drug': 'No results',
            'Accuracy_Score': 0,
            'Match_Found': False
        })

# Calculate metrics
print("\n" + "="*80)
print("PERFORMANCE METRICS")
print("="*80)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\n📊 Overall Metrics:")
print(f"   Accuracy:  {accuracy:.2%}")
print(f"   Precision: {precision:.2%}")
print(f"   Recall:    {recall:.2%}")
print(f"   F1-Score:  {f1:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(f"\n📈 Confusion Matrix:")
print(f"   True Positives (TP):  {cm[1][1]}")
print(f"   False Positives (FP): {cm[0][1]}")
print(f"   True Negatives (TN):  {cm[0][0]}")
print(f"   False Negatives (FN): {cm[1][0]}")

# Detailed results
print(f"\n📋 Detailed Results:")
print("-"*80)
results_df = pd.DataFrame(detailed_results)
for idx, row in results_df.iterrows():
    status = "✓" if row['Match_Found'] else "✗"
    print(f"{status} Query: '{row['Query']}'")
    print(f"   Top Result: {row['Top_Drug']}")
    print(f"   Accuracy: {row['Accuracy_Score']:.1f}%")
    print(f"   Category: {row['Expected_Category']}")
    print("-"*80)

# Classification Report
print(f"\n📑 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Not Relevant', 'Relevant']))

# Visualize Confusion Matrix
print("\n📊 Generating confusion matrix visualization...")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Relevant', 'Relevant'],
            yticklabels=['Not Relevant', 'Relevant'])
plt.title('Confusion Matrix - Medicine Recommendation System', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion matrix to 'confusion_matrix.png'")

# Visualize Metrics
print("\n📊 Generating metrics visualization...")
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics.keys(), metrics.values(), color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
plt.ylim(0, 1.0)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig('model_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved metrics visualization to 'model_metrics.png'")

# Summary
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)
print(f"\n✓ Total test cases: {len(test_cases)}")
print(f"✓ Correct predictions: {cm[1][1]}")
print(f"✓ Incorrect predictions: {cm[1][0]}")
print(f"✓ Overall accuracy: {accuracy:.2%}")
print(f"\n✓ Visualizations saved:")
print(f"   - confusion_matrix.png")
print(f"   - model_metrics.png")
print("\n" + "="*80)
