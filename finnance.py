# ==============================================================================
# MODULE: finnance.py
# DESCRIPTION: Unified Financial Document Intelligence, Revenue Forecasting,
#              and Machine Learning Auditing Dashboard. Enhanced with advanced
#              burn-rate forecasting and statistical anomaly risk detection layers.
# ==============================================================================

# --- STEP 0: REQUIREMENT ASSURANCE ---

import io
import os
import re
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
from PIL import Image

# Natural Language Processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Machine Learning Core
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support
from xgboost import XGBRegressor

import gradio as gr
from wordcloud import WordCloud

# Set modern plotting aesthetics
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# --- SERVICE CORE INTERFACE FOR STREAMLIT ---
class FinanceService:
    """Core Service class mapped directly to backend AIVORA Streamlit modules."""
    def __init__(self):
        self.metrics = {
            "revenue": 0.0, 
            "burn_rate": 0.0, 
            "margin": 0.0, 
            "estimated_runway_months": 0.0,
            "anomaly_count": 0
        }
        self.chart = None
        self.anomalies_df = pd.DataFrame()

    def process_ledger(self, file_path):
        """
        Parses CSV or Excel ledger files, computes real financial indices, executes
        statistical anomaly detection, and generates predictive cash visualizations.
        """
        try:
            # Read files dynamically based on file extensions
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Normalize column text signatures
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            # Auto-detect vital column names
            amount_col = next((c for c in ['amount', 'value', 'total', 'sales', 'revenue'] if c in df.columns), None)
            type_col = next((c for c in ['type', 'category', 'direction', 'transaction_type', 'item'] if c in df.columns), None)
            date_col = next((c for c in ['date', 'timestamp', 'period', 'ds'] if c in df.columns), None)

            if not amount_col:
                # Fallback to the first numeric column discovered
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    amount_col = num_cols[0]
                else:
                    raise ValueError("Ledger parsing error: Unable to identify numeric transactional amount values.")

            # Standardize transaction data type formats
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0.0)
            
            # Differentiate Revenue flows from Operating Expenses
            if type_col:
                rev_mask = df[type_col].astype(str).str.lower().str.contains('rev|inc|sales|credit|gain', regex=True)
                exp_mask = df[type_col].astype(str).str.lower().str.contains('exp|burn|spend|debit|cost|loss', regex=True)
                
                gross_revenue = df[rev_mask][amount_col].sum()
                total_expenses = df[exp_mask][amount_col].abs().sum()
                
                # If everything maps to positive numbers but flags distinct labels
                if total_expenses == 0 and gross_revenue == 0:
                    gross_revenue = df[df[amount_col] > 0][amount_col].sum()
                    total_expenses = df[df[amount_col] < 0][amount_col].abs().sum()
            else:
                # Alternative mathematical signs baseline strategy
                gross_revenue = df[df[amount_col] > 0][amount_col].sum()
                total_expenses = df[df[amount_col] < 0][amount_col].abs().sum()

            # Dynamic timeline duration tracking
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col]).sort_values(by=date_col)
                
                if len(df) > 1:
                    timespan_days = (df[date_col].max() - df[date_col].min()).days
                    months_elapsed = max(1.0, timespan_days / 30.4)
                    monthly_burn_rate = total_expenses / months_elapsed
                else:
                    months_elapsed = 1.0
                    monthly_burn_rate = total_expenses
            else:
                months_elapsed = 12.0
                monthly_burn_rate = total_expenses / 12.0  # Annual normalization estimate

            net_margin_profit = gross_revenue - total_expenses
            operational_margin_pct = (net_margin_profit / gross_revenue * 100) if gross_revenue > 0 else 0.0

            # --- FINANCIAL INTELLIGENCE SPRINT: ANOMALY DETECTION (Z-SCORE) ---
            anomaly_count = 0
            if type_col and len(df) > 5:
                # Isolate operating expense rows safely
                exp_rows = df[df[amount_col].abs() > 0].copy()
                grouped_std = exp_rows.groupby(type_col)[amount_col].transform('std').fillna(0)
                grouped_mean = exp_rows.groupby(type_col)[amount_col].transform('mean')
                
                # Calculate absolute distance metrics
                z_scores = np.where(grouped_std > 0, (exp_rows[amount_col] - grouped_mean) / grouped_std, 0)
                exp_rows['z_score'] = np.abs(z_scores)
                
                self.anomalies_df = exp_rows[exp_rows['z_score'] > 2.2].sort_values(by='z_score', ascending=False)
                anomaly_count = len(self.anomalies_df)

            # --- FINANCIAL INTELLIGENCE SPRINT: ML CASH RUNWAY FORECASTING ---
            estimated_runway = 999.0  # Default value indicating structural safety
            net_monthly_burn = monthly_burn_rate - (gross_revenue / months_elapsed)
            
            if net_monthly_burn > 0:
                current_cash_pool = max(10000.0, gross_revenue * 0.4) # Target baseline scaling assumption
                estimated_runway = current_cash_pool / net_monthly_burn

            self.metrics = {
                "revenue": float(gross_revenue),
                "burn_rate": float(monthly_burn_rate),
                "margin": float(operational_margin_pct),
                "estimated_runway_months": float(round(estimated_runway, 1)),
                "anomaly_count": int(anomaly_count)
            }

            # Generate dynamic data visualization panels
            fig, ax = plt.subplots(figsize=(11, 4))
            if date_col and len(df) > 1:
                # Reconstruct cumulative cash position runway maps
                df['net_flow'] = df.apply(lambda row: row[amount_col] if (not type_col or 'exp' not in str(row[type_col]).lower()) else -abs(row[amount_col]), axis=1)
                # If numbers already carry correct polarity signs
                if (df[amount_col] < 0).any():
                    df['net_flow'] = df[amount_col]
                    
                df['cumulative_cash_runway'] = df['net_flow'].cumsum()
                
                # Core plot line
                ax.plot(df[date_col], df['cumulative_cash_runway'], color='#1abc9c', linewidth=2.5, marker='o', markersize=4, label='Net Liquidity Vector')
                
                # Predictive Trend Modeling Extrapolation Array
                try:
                    df['date_delta_idx'] = (df[date_col] - df[date_col].min()).dt.days
                    X_trend = df[['date_delta_idx']].values
                    y_trend = df['cumulative_cash_runway'].values
                    
                    reg_model = LinearRegression().fit(X_trend, y_trend)
                    future_days = np.array([[df['date_delta_idx'].max() + 30], [df['date_delta_idx'].max() + 60]])
                    future_preds = reg_model.predict(future_days)
                    
                    future_dates = [df[date_col].max() + timedelta(days=30), df[date_col].max() + timedelta(days=60)]
                    ax.plot(future_dates, future_preds, color='#e67e22', linestyle=':', linewidth=2, label='ML Capital Projection Trend')
                except Exception:
                    pass

                ax.fill_between(df[date_col], df['cumulative_cash_runway'], color='#1abc9c', alpha=0.15)
                ax.set_title("Operational Cash Flow Accumulation & Predictive Model", fontsize=11, fontweight="bold", pad=12)
                ax.set_ylabel("Capital Units ($)", fontsize=9)
                ax.tick_params(axis='x', rotation=15)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(loc="upper left")
            else:
                # Visual backup distribution layout charts
                chart_data = df.groupby(type_col if type_col else df.columns[0])[amount_col].sum().sort_values()
                chart_data.plot(kind='barh', ax=ax, color=['#e74c3c' if x < 0 else '#2ecc71' for x in chart_data], edgecolor='black', alpha=0.8)
                ax.set_title("Fiscal Volume Breakdown Array", fontsize=11, fontweight="bold")
            
            plt.tight_layout()
            self.chart = fig
            return self.metrics, self.chart

        except Exception as e:
            return {"error": str(e), "revenue": 0.0, "burn_rate": 0.0, "margin": 0.0, "estimated_runway_months": 0.0, "anomaly_count": 0}, None

# --- 1. NLTK INITIALIZATION & SEED TEXTS ---
print("📥 Step 1/5: Loading internal linguistic corpora...")
nltk.download('movie_reviews', quiet=True)  # Used as baseline sentiment weights
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import movie_reviews, stopwords
stop_words_list = list(stopwords.words('english'))

# --- 2. TRAIN TEXT CLASSIFIER SUITE FOR THE LEADERBOARD ---
print("⚙️ Step 2/5: Initializing Multi-Model Sentiment Suite...")
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Shuffle the dataset array before validation slicing to resolve the single-class ValueError crash
random.seed(42)
random.shuffle(documents)

train_texts, train_labels = zip(*documents)

# Downsample slightly for ultra-fast compilation while maintaining variance
vectorizer = TfidfVectorizer(stop_words=stop_words_list, max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(train_texts[:1000])
y_train_vec = np.array([1 if label == 'pos' else 0 for label in train_labels[:1000]])

# Core text classification engine
sentiment_classifier = LogisticRegression(max_iter=1000)
sentiment_classifier.fit(X_train_vec, y_train_vec)

def generate_leaderboard_data():
    """Generates the Model Performance Leaderboard tracking database."""
    models = {
        "Logistic Regression": LogisticRegression(),
        "Linear SVM": LinearSVC(dual=False),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    leaderboard_rows = []
    split = 800
    X_tr, X_val = X_train_vec[:split], X_train_vec[split:]
    y_tr, y_val = y_train_vec[:split], y_train_vec[split:]
    
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds) * 100
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average='binary', zero_division=0)
        
        leaderboard_rows.append({
            "Model Name": name,
            "Accuracy": f"{acc:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%"
        })
    return pd.DataFrame(leaderboard_rows)

# --- 3. COMPONENT ONE: FINANCIAL TEXT INTEL PIPELINE ---
def generate_text_dashboard(text, top_keywords, pos_prob, neg_prob):
    fig = plt.figure(figsize=(15, 9), facecolor='#f8f9fa')
    ax_sent = plt.subplot2grid((2, 2), (0, 0))
    ax_keys = plt.subplot2grid((2, 2), (0, 1))
    ax_cloud = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    categories = ['Bearish / Negative', 'Bullish / Positive']
    probs = [neg_prob * 100, pos_prob * 100]
    bars = ax_sent.barh(categories, probs, color=['#e74c3c', '#2ecc71'], height=0.4, edgecolor='black', alpha=0.85)
    ax_sent.set_xlim(0, 100)
    ax_sent.set_xlabel('Confidence (%)', fontweight='bold')
    ax_sent.set_title('Document Sentiment Allocation Profile', fontsize=12, fontweight='bold', pad=10)
    for bar in bars:
        width = bar.get_width()
        ax_sent.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontweight='bold')

    words = [kw[0] for kw in top_keywords][::-1]
    scores = [kw[1] for kw in top_keywords][::-1]
    ax_keys.barh(words, scores, color='#34495e', edgecolor='black', alpha=0.85)
    ax_keys.set_xlabel('Localized TF-IDF Relevance Density', fontweight='bold')
    ax_keys.set_title('Top Extracted Financial Keywords', fontsize=12, fontweight='bold', pad=10)

    try:
        clean_text = " ".join([w.lower() for w in word_tokenize(text) if w.isalnum() and w.lower() not in stop_words_list])
        wordcloud = WordCloud(width=1000, height=350, background_color='white', colormap='plasma', max_words=75).generate(clean_text)
        ax_cloud.imshow(wordcloud, interpolation='bilinear')
        ax_cloud.axis('off')
        ax_cloud.set_title('Visual Word Frequency Mapping', fontsize=12, fontweight='bold', pad=10)
    except Exception:
        ax_cloud.text(0.5, 0.5, "Insufficient parsing context for visual map.", ha='center', va='center', fontsize=12)
        ax_cloud.axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def analyze_document_finance(uploaded_file, pasted_text, summary_ratio):
    if uploaded_file is not None:
        try:
            with open(uploaded_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            filename = os.path.basename(uploaded_file.name)
        except Exception as e:
            return f"Extraction Error: {str(e)}", None, None
    elif pasted_text and pasted_text.strip():
        text = pasted_text.strip()
        filename = "pasted_transcript.txt"
    else:
        return "⚠️ Action Required: Provide an analysis transcript or upload a file.", None, None

    if len(text) < 15:
         return "⚠️ Document context insufficient to perform ML evaluations.", None, None

    raw_sentences = sent_tokenize(text)
    words = [w.lower() for w in word_tokenize(text) if w.isalnum()]
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]

    if not sentences:
        return "⚠️ Error parsing structural sentence boundaries.", None, None

    vectorized_text = vectorizer.transform([text])
    prob = sentiment_classifier.predict_proba(vectorized_text)[0]
    neg_p, pos_p = prob[0], prob[1]

    sentiment_lbl = "Bullish / Positive 🟢" if pos_p > 0.55 else ("Bearish / Negative 🔴" if neg_p > 0.55 else "Neutral / Balanced 🟡")

    doc_vectorizer = TfidfVectorizer(stop_words=stop_words_list)
    try:
        tfidf_matrix = doc_vectorizer.fit_transform(sentences)
        feature_names = doc_vectorizer.get_feature_names_out()
        mean_weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        sorted_indices = np.argsort(mean_weights)[::-1]
        top_keywords = [(feature_names[i], float(mean_weights[i])) for i in sorted_indices[:10]]
    except Exception:
        word_freq = Counter([w for w in words if w not in stop_words_list])
        top_keywords = [(w, float(f)) for w, f in word_freq.most_common(10)]

    sentence_scores = {}
    for idx, sentence in enumerate(sentences):
        sentence_vec = doc_vectorizer.transform([sentence])
        sentence_scores[idx] = sentence_vec.sum()

    target_count = max(3, int(len(sentences) * summary_ratio))
    target_count = min(target_count, len(sentences))
    top_indices = sorted(sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:target_count])
    extractive_summary = "\n\n".join([sentences[i] for i in top_indices])

    report = f"======================================================================\n"
    report += f"                   FINANCIAL TEXT ANALYSIS REPORT                     \n"
    report += f"======================================================================\n\n"
    report += f"📊 CORPORATE METRICS LOG:\n"
    report += f"  • Target Entity Source : {filename}\n"
    report += f"  • Document Character Volume : {len(text):,}\n"
    report += f"  • Computed Token Count : {len(words):,}\n"
    report += f"  • Structural Sentence Count : {len(raw_sentences):,}\n\n"
    report += f"🧠 ALGORITHMIC SENTIMENT VECTOR:\n"
    report += f"  • Dominant Market Outlook : {sentiment_lbl}\n"
    report += f"  • Positive Coefficient : {pos_p * 100:.2f}%\n"
    report += f"  • Negative Coefficient : {neg_p * 100:.2f}%\n\n"
    report += f"📝 EXTRACTIVE EXECUTIVE COMPRESSION:\n"
    report += f"----------------------------------------------------------------------\n"
    report += f"{extractive_summary}\n"
    report += f"----------------------------------------------------------------------\n"

    output_path = f"Finance_Intel_Report_{os.path.splitext(filename)[0]}.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    chart_img = generate_text_dashboard(text, top_keywords, pos_p, neg_p)
    return report, chart_img, output_path

# --- 4. COMPONENT TWO: PERFORMANCE & REVENUE FORECASTING ENGINE ---
def generate_synthetic_finance_data(num_days):
    np.random.seed(42)
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    trend = np.linspace(5000, 12000, num_days) 
    weekly = np.array([800 if d.weekday() >= 5 else -300 for d in dates])
    noise = np.random.normal(0, 400, num_days)
    
    revenue = trend + weekly + noise
    revenue = np.clip(revenue, 1000, None)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': revenue,
        'price': np.random.uniform(19, 25, num_days),
        'promotion': np.random.binomial(1, 0.1, num_days),
        'temperature': np.random.normal(20, 5, num_days),
        'competitor_price': np.random.uniform(18, 26, num_days)
    })
    return df

def parse_uploaded_csv(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    except Exception as e:
        raise ValueError(f"Incompatible format: {str(e)}")
        
    col_mapping = {c: c.lower().strip() for c in df.columns}
    date_col, sales_col = None, None
    for orig, norm in col_mapping.items():
        if norm in ['date', 'ds', 'timestamp', 'time', 'period']: date_col = orig
        if norm in ['sales', 'y', 'revenue', 'turnover', 'amount']: sales_col = orig

    if not date_col or not sales_col:
        raise ValueError("Missing explicit 'date' or 'revenue/sales' structure.")

    df = df.rename(columns={date_col: 'date', sales_col: 'sales'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
    df = df.dropna(subset=['date', 'sales']).sort_values('date').reset_index(drop=True)

    for feat, def_val in [('price', 20.0), ('promotion', 0), ('temperature', 20.0), ('competitor_price', 20.0)]:
        if feat not in df.columns: df[feat] = def_val
    return df[['date', 'sales', 'price', 'promotion', 'temperature', 'competitor_price']]

def engineer_finance_features(df):
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    for lag in [1, 3, 7]:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
    for w in [3, 7]:
        df[f'rolling_mean_{w}'] = df['sales_lag_1'].rolling(window=w).mean()
    return df.dropna().reset_index(drop=True)

def execute_forecast_pipeline(df, test_days):
    test_days = min(test_days, int(len(df) * 0.3))
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()

    features = [c for c in df.columns if c not in ['date', 'sales']]
    X_train, y_train = train_df[features], train_df['sales']
    X_test, y_test = test_df[features], test_df['sales']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)

    def eval_model(yt, yp):
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        return mae, rmse, r2, (0.6 * mae + 0.4 * rmse)

    rf_mae, rf_rmse, rf_r2, rf_score = eval_model(y_test, rf_pred)
    xgb_mae, xgb_rmse, xgb_r2, xgb_score = eval_model(y_test, xgb_pred)

    use_xgb = xgb_score < rf_score
    champ_name = "XGBoost Engine" if use_xgb else "Random Forest Regressor"
    champ_model = xgb if use_xgb else rf
    champ_preds = xgb_pred if use_xgb else rf_pred

    mean_rev = np.mean(champ_preds)
    std_rev = np.std(champ_preds) if np.std(champ_preds) > 0 else 1.0
    safety_buffer = 1.65 * std_rev * np.sqrt(3)
    trigger_point = (mean_rev * 3) + safety_buffer

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    
    axes[0, 0].plot(test_df['date'], y_test, label="Observed Realized Revenue", color='black', linewidth=2, linestyle=':')
    axes[0, 0].plot(test_df['date'], rf_pred, label=f"Random Forest (Score: {rf_score:.1f})", alpha=0.8)
    axes[0, 0].plot(test_df['date'], xgb_pred, label=f"XGBoost (Score: {xgb_score:.1f})", alpha=0.8)
    axes[0, 0].set_title("Forecast Horizon Accuracy Trajectory", fontsize=11, fontweight="bold")
    axes[0, 0].tick_params(axis='x', rotation=20)
    axes[0, 0].legend()

    axes[0, 1].plot(test_df['date'], champ_preds, label="Projected Cash Volume Flow", color='#2980b9', linewidth=2)
    axes[0, 1].axhline(trigger_point + (mean_rev * 7), color='#27ae60', linestyle='--', label="Optimal Reserve Target")
    axes[0, 1].axhline(trigger_point, color='#f39c12', linestyle='--', label="Liquidity Trigger Warning")
    axes[0, 1].axhline(safety_buffer, color='#c0392b', linestyle='--', label="Critical Liquidity Floor")
    axes[0, 1].set_title("Capital Capitalization Risk & Runway Policy Simulation", fontsize=11, fontweight="bold")
    axes[0, 1].tick_params(axis='x', rotation=20)
    axes[0, 1].legend()

    residuals = y_test - champ_preds
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='teal', bins=10)
    axes[1, 0].axvline(0, color='red', linestyle='--')
    axes[1, 0].set_title("Mathematical Deviation Spread (Residual Distributions)", fontsize=11, fontweight="bold")

    importances = champ_model.feature_importances_
    idx = np.argsort(importances)[-5:]
    axes[1, 1].barh(range(len(idx)), importances[idx], color='#8e44ad', align='center')
    axes[1, 1].set_yticks(range(len(idx)))
    axes[1, 1].set_yticklabels([features[i] for i in idx])
    axes[1, 1].set_title(f"Top Asset Valuation Driver Matrix ({champ_name})", fontsize=11, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130)
    buf.seek(0)
    chart_out = Image.open(buf)
    plt.close(fig)

    log = f"""=== FINANCIAL FORECAST SYSTEM LOGS ===
Timestamp Vector: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Operational Champion: {champ_name.upper()}

--- STATISTICAL MODEL PERFORMANCE MATRIX ---
[Random Forest Regressor]:
  • MAE: {rf_mae:.2f} |  RMSE: {rf_rmse:.2f} |  R² Correlation: {rf_r2:.4f}
[XGBoost Engine]:
  • MAE: {xgb_mae:.2f} |  RMSE: {xgb_rmse:.2f} |  R² Correlation: {xgb_r2:.4f}

--- RISK CAPACITY MANAGEMENT METRICS ---
  • Safe Tactical Liquidity Floor: {safety_buffer:.2f} Capital Units
  • Capital Stream Threshold Trigger: {trigger_point:.2f} Capital Units
  • Systemic Optimization Range: {trigger_point + (mean_rev * 7):.2f} Capital Units
========================================"""
    return log, chart_out

def routing_forecast_pipeline(source_mode, file_obj, days_generation, horizon_window):
    try:
        if source_mode == "Upload Custom Financial CSV":
            if file_obj is None: return "❌ Input Error: Source CSV required.", None
            raw_data = parse_uploaded_csv(file_obj.name)
        else:
            raw_data = generate_synthetic_finance_data(int(days_generation))

        if len(raw_data) < (int(horizon_window) + 10):
            return f"❌ Insufficient Dataset Depth: Row structure count ({len(raw_data)}) must exceed evaluation target.", None

        engineered_df = engineer_finance_features(raw_data)
        logs, output_plot = execute_forecast_pipeline(engineered_df, int(horizon_window))
        return logs, output_plot
    except Exception as e:
        return f"❌ Pipeline Execution Interrupted:\n{str(e)}", None

# --- 5. UNIFIED GRADIO INTERFACE BLUEPRINT ASSEMBLY ---
def start_standalone_gradio_app():
    print("🎨 Step 3/5: Constructing Integrated UX Interface Blueprint...")
    
    with gr.Blocks() as financial_app:
        gr.Markdown(
            """
            # 📈 Advanced Financial Machine Learning Dashboard (`finnance`)
            Enterprise executive workspace combining classical NLP classification, operational predictive regression forecasting pipelines, and localized system training benchmarks.
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("📰 Financial Document Intelligence"):
                gr.Markdown("### Corporate Report Parsing & Semantic Extractor")
                with gr.Row():
                    with gr.Column(scale=1):
                        doc_file = gr.File(label="Upload Financial Audit / Earnings .txt file", file_types=[".txt"])
                        doc_pasted = gr.Textbox(label="Or execute raw transcript analytical inputs directly", placeholder="Paste earnings logs...", lines=6)
                        doc_slider = gr.Slider(minimum=0.1, maximum=0.5, step=0.05, value=0.20, label="Summary Extraction Density Ratio")
                        doc_run = gr.Button("Execute Document Diagnostics", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("📊 Semantic Analysis Charts"):
                                doc_chart = gr.Image(label="Linguistic Analytics Dashboard Visualization")
                            with gr.TabItem("📋 Structural Metrics Report"):
                                doc_report = gr.Textbox(label="Formatted Text Analysis Metric Summary", lines=16)
                        doc_export = gr.File(label="Download Formatted Structural File (.txt)")

                doc_run.click(
                    fn=analyze_document_finance,
                    inputs=[doc_file, doc_pasted, doc_slider],
                    outputs=[doc_report, doc_chart, doc_export]
                )

            with gr.TabItem("📊 Predictive Revenue Projections"):
                gr.Markdown("### Asset Valuation & Horizon Flow Projections")
                with gr.Row():
                    with gr.Column(scale=1):
                        forecast_mode = gr.Radio(choices=["Simulate Automated Operational Framework Data", "Upload Custom Financial CSV"], value="Simulate Automated Operational Framework Data", label="Ingestion Selection Engine")
                        forecast_file = gr.File(label="Asset Valuation Data Stream Ingestion", file_types=[".csv"], visible=False)
                        forecast_days = gr.Slider(minimum=180, maximum=730, step=30, value=365, label="Synthetic Target Lifecycle Spans (Days)")
                        forecast_horizon = gr.Slider(minimum=7, maximum=60, step=1, value=30, label="Evaluation Validation Forecasting Window")
                        forecast_run = gr.Button("Execute Optimization Run Pipelines", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("📉 Multi-Perspective Modeling Panels"):
                                forecast_chart = gr.Image(label="Algorithmic Trajectory Evaluations")
                            with gr.TabItem("📋 Production Deployment Logs"):
                                forecast_logs = gr.Textbox(label="Model Execution Diagnostic Logs", lines=16)

                def toggle_forecast_inputs(selection):
                    if selection == "Upload Custom Financial CSV":
                        return gr.update(visible=True), gr.update(visible=False)
                    return gr.update(visible=False), gr.update(visible=True)

                forecast_mode.change(fn=toggle_forecast_inputs, inputs=[forecast_mode], outputs=[forecast_file, forecast_days])
                
                # [FIXED]: Swapped 'horizon_window' out for the correct layout slider variable instance 'forecast_horizon'
                forecast_run.click(
                    fn=routing_forecast_pipeline,
                    inputs=[forecast_mode, forecast_file, forecast_days, forecast_horizon],
                    outputs=[forecast_logs, forecast_chart]
                )

            with gr.TabItem("🏆 Algorithmic Core Leaderboard"):
                gr.Markdown("### System Validation Benchmarks & Audit Logs")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            """
                            **Framework Control Audit**
                            This panel tracks the model registry. Click below to verify classification parity across core modeling engines.
                            """
                        )
                        audit_btn = gr.Button("Refresh Pipeline Registries", variant="secondary")
                        audit_status = gr.HTML("<div style='color: #2ecc71; font-weight: bold;'>✓ Model Registry Connection Active</div>")
                    with gr.Column(scale=2):
                        leaderboard_view = gr.DataFrame(value=generate_leaderboard_data(), label="Active Model Performance Benchmarking Metrics", interactive=False)
                
                audit_btn.click(fn=generate_leaderboard_data, inputs=[], outputs=[leaderboard_view])

    print("🚀 Step 4/5: Deploying Combined Gradio App Engine Context inside your Workspace Environment...")
    financial_app.launch(theme='soft', share=True, debug=False)

# --- 6. LAUNCH ENVIRONMENT RUNTIME CONTROL ---
if __name__ == "__main__":
    start_standalone_gradio_app()