import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import base64

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved Light Medical Theme CSS
st.markdown("""
<style>

/* ---------- Fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;600&display=swap');

/* ---------- Main Background ---------- */
.main {
    background: #ffffff;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #f9e6ee;
    border-right: 1px solid #eed3dd;
}

/* ---------- Headers ---------- */
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #7a1f3d;
}

h1 {
    font-size: 2.4rem !important;
    text-align: center;
    padding: 1rem 0;
    color: #c2185b;
    text-shadow: none;
}

/* ---------- Body Text ---------- */
p, div, label {
    font-family: 'Inter', sans-serif;
    color: #2f2f2f;
}

/* ---------- Cards ---------- */
.info-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.4rem;
    border: 1px solid #f0d6df;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin: 1rem 0;
    width: 100%;
}

/* ---------- Metric Cards ---------- */
.metric-card {
    background: #fff7fa;
    border-radius: 10px;
    padding: 1rem;
    border: 1px solid #f0d6df;
    text-align: center;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: #f48fb1;
    color: ffffff;
    border-radius: 8px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background: #ec407a;
}

/* ---------- File Uploader ---------- */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #e4b8c7;
    border-radius: 10px;
}

/* ---------- Select Box ---------- */
.stSelectbox > div > div {
    background-color: white;
    border: 1px solid #e4b8c7;
    border-radius: 8px;
}

/* ---------- Alerts ---------- */
.stAlert {
    background-color: #fff6f9;
    border-left: 4px solid #d81b60;
}

/* ---------- Ribbon ---------- */
.ribbon {
    background: #f06292;
    color: white;
    padding: 0.4rem 1.6rem;
    border-radius: 20px;
    display: inline-block;
    font-weight: 600;
    margin: 0.8rem 0;
}

/* ---------- Divider ---------- */
hr {
    border: none;
    height: 1px;
    background: #f0d6df;
    margin: 1.5rem 0;
}
            
            /* ---------- Sidebar Info Card ---------- */
.sidebar-card {
    background: #ffffff;
    border-left: 4px solid #d81b60;
    padding: 0.9rem 1rem;
    border-radius: 10px;
    margin-top: 0.6rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

.sidebar-card h4 {
    margin: 0 0 0.4rem 0;
    color: #8b0045;
    font-size: 1rem;
}

.sidebar-card p {
    margin: 0;
    font-size: 0.9rem;
    color: #3a3a3a;
    line-height: 1.5;
}

/* ---------- Expander Header ---------- */
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #7a1f3d;
    font-size: 1.05rem;
}

</style>
""", unsafe_allow_html=True)


# Helper function to load models
@st.cache_resource
def load_model(model_name):
    """Load a trained model from pickle file"""
    model_path = Path("model") / f"{model_name}.pkl"
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None

# Helper function to calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC Score': matthews_corrcoef(y_true, y_pred)
    }
    
    # Calculate AUC if probability predictions are available
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multi-class
                metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['AUC Score'] = 0.0
    else:
        metrics['AUC Score'] = 0.0
    
    return metrics

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    """Create an interactive confusion matrix using Plotly"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels
    labels = sorted(np.unique(y_true))
    label_names = [f'Class {i}' for i in labels]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=label_names,
        y=label_names,
        colorscale=[[0, '#fff0f5'], [0.5, '#ffb3d9'], [1, '#ff1493']],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16, "color": "#2f2f2f"},
        hoverongaps=False,
        colorbar=dict(
            title="Count",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=cm.max()//5 if cm.max() > 5 else 1
        )
    ))
    
    fig.update_layout(
        title={
            'text': 'Confusion Matrix',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#8b0045', 'family': 'Playfair Display'}
        },
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Lato', color='#4a0026'),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig

# Function to display classification report
def display_classification_report(y_true, y_pred):
    """Display classification report as a formatted dataframe"""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Convert to dataframe
    df_report = pd.DataFrame(report).transpose()
    
    # Format the dataframe
    df_report = df_report.round(3)
    
    # Style the dataframe
    styled_df = df_report.style.background_gradient(
        cmap='RdPu', 
        subset=['precision', 'recall', 'f1-score']
    ).format(precision=3)
    
    return styled_df

# Main app
def main():
    # Header with ribbon
    st.markdown('<h1>🎗️ Breast Cancer Classification System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<center><div class="ribbon">Empowering Early Detection Through Machine Learning</div></center>', 
        unsafe_allow_html=True
    )
    
    # Introduction section
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>About This Application</h3>

        <p style='font-size: 1.1rem; line-height: 1.8;'>
        Welcome to <strong>Breast Cancer Classification System</strong>. This application leverages 
        state-of-the-art machine learning algorithms to predict breast cancer diagnosis with high accuracy.
        </p>

        <p style='font-size: 1.1rem; line-height: 1.8;'>
        This system compares <strong>six different ML models</strong> to provide comprehensive diagnostic insights, 
        helping healthcare professionals make informed decisions.
        </p>

        <!-- Spacer to match right card height -->
        <div style="height: 18px;"></div>

        </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>Dataset Information</h3>

        <ul style='font-size: 1rem; line-height: 1.8;'>
        <li><strong>Domain:</strong> Medical Diagnosis</li>
        <li><strong>Task:</strong> Binary Classification</li>
        <li><strong>Features:</strong> Clinical measurements</li>
        <li><strong>Classes:</strong> Malignant / Benign</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    
    # Sidebar
    st.sidebar.markdown(
        '<h2 class="sidebar-heading">Model Configuration</h2>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    
    # Model selection
    model_options = {
        'Logistic Regression': 'logistic',
        'Decision Tree Classifier': 'decision_tree',
        'K-Nearest Neighbors': 'knn',
        'Naive Bayes': 'naive_bayes',
        'Random Forest (Ensemble)': 'random_forest',
        'XGBoost (Ensemble)': 'xgboost'
    }
    
    selected_model_name = st.sidebar.selectbox(
        '-> Select ML Model',
        options=list(model_options.keys()),
        help="Choose a machine learning model for classification"
    )
    
    selected_model_file = model_options[selected_model_name]
    
    # Model descriptions
    model_descriptions = {
        'Logistic Regression': 'A linear model for binary classification using logistic function. Fast and interpretable.',
        'Decision Tree Classifier': 'Tree-based model that makes decisions through hierarchical splits. Easy to visualize.',
        'K-Nearest Neighbors': 'Instance-based learning algorithm that classifies based on nearest training examples.',
        'Naive Bayes': 'Probabilistic classifier based on Bayes theorem with strong independence assumptions.',
        'Random Forest (Ensemble)': 'Ensemble of decision trees using bagging. Reduces overfitting and improves accuracy.',
        'XGBoost (Ensemble)': 'Gradient boosting ensemble method. Highly efficient and often achieves best performance.'
    }
    
    # st.sidebar.info(f"**{selected_model_name}**\n\n{model_descriptions[selected_model_name]}")
    st.sidebar.markdown(f"""
    <div class="sidebar-card">
        <h4>{selected_model_name}</h4>
        <p>{model_descriptions[selected_model_name]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Test Data")
    
    # Download sample data button
    sample_data_path = Path("data/test_data.csv")
    if sample_data_path.exists():
        with open(sample_data_path, 'rb') as f:
            st.sidebar.download_button(
                label="⬇️ Download Sample Test Data",
                data=f,
                file_name="test_data.csv",
                mime="text/csv",
                help="Download a sample CSV file to test the application"
            )
    else:
        st.sidebar.warning("Sample test data not found at data/test_data.csv")
    
    st.sidebar.markdown("---")
    
    # File uploader
    st.markdown("---")
    st.markdown("### 📤 Upload Test Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing test data",
        type=['csv'],
        help="Upload your test dataset in CSV format. Make sure it contains the same features as the training data."
    )
    
    if uploaded_file is not None:
        try:
            # Load the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} samples from the uploaded file!")
            
            # Display dataset preview
            with st.expander("⬇️ Preview Dataset", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 rows of {len(df)} total samples")
            
            # Check if target column exists
            if 'target' not in df.columns and 'label' not in df.columns and 'diagnosis' not in df.columns:
                st.error("⚠️ Target column not found! Please ensure your CSV has a column named 'target', 'label', or 'diagnosis'.")
                return
            
            # Determine target column
            if 'target' in df.columns:
                target_col = 'target'
            elif 'label' in df.columns:
                target_col = 'label'
            else:
                target_col = 'diagnosis'
            
            # Separate features and target
            y_true = df[target_col]
            X_test = df.drop(columns=[target_col])
            
            st.markdown("---")
            
            # Load and predict with selected model
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Run Prediction & Evaluate", use_container_width=True):
                    with st.spinner(f'Loading {selected_model_name} model and making predictions...'):
                        # Load model
                        model = load_model(selected_model_file)
                        
                        if model is not None:
                            try:
                                # Make predictions
                                y_pred = model.predict(X_test)
                                
                                # Get probability predictions if available
                                try:
                                    y_pred_proba = model.predict_proba(X_test)
                                except:
                                    y_pred_proba = None
                                
                                # Calculate metrics
                                metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
                                
                                st.markdown("---")
                                st.markdown(f"## 📊 Model Performance: {selected_model_name}")
                                st.markdown("---")
                                
                                # Display metrics in cards
                                st.markdown("### 🎯 Evaluation Metrics")
                                
                                cols = st.columns(3)
                                metric_items = list(metrics.items())
                                
                                for idx, (metric_name, metric_value) in enumerate(metric_items):
                                    with cols[idx % 3]:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4 style='color: #8b0045; margin-bottom: 0.5rem;'>{metric_name}</h4>
                                            <p style='font-size: 2rem; font-weight: bold; color: #c71585; margin: 0;'>
                                                {metric_value:.4f}
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                st.markdown("---")
                                
                                # Confusion Matrix and Classification Report
                                # ---------- Classification Report (Top) ----------
                                st.markdown("### 📋 Classification Report")

                                styled_report = display_classification_report(y_true, y_pred)

                                st.dataframe(
                                    styled_report,
                                    use_container_width=True
                                )

                                st.markdown("---")


                                # ---------- Confusion Matrix (Bottom) ----------
                                st.markdown("### 🔄 Confusion Matrix")

                                cm_fig = plot_confusion_matrix(y_true, y_pred)

                                # Center the square matrix
                                c1, c2, c3 = st.columns([1, 2, 1])

                                with c2:
                                    st.plotly_chart(
                                        cm_fig,
                                        use_container_width=False
                                    )

                                
                                # Summary
                                st.markdown("---")
                                st.markdown("### ✨ Performance Summary")
                                
                                best_metric = max(metrics, key=metrics.get)
                                st.success(f"""
                                    **Best Performing Metric:** {best_metric} ({metrics[best_metric]:.4f})
                                    
                                    The {selected_model_name} model has been evaluated on {len(y_true)} samples.
                                    Overall accuracy: **{metrics['Accuracy']:.2%}**
                                """)
                                
                            except Exception as e:
                                st.error(f"❌ Error during prediction: {str(e)}")
                        else:
                            st.error("❌ Failed to load the model. Please check if the model file exists in the 'model' folder.")
            
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to begin classification and evaluation.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #8b0045; padding: 2rem 0;'>
        <p style='font-size: 0.9rem;'>
            🎗️ <strong>Breast Cancer Classification System</strong> | 
            Developed for ML Assignment 2 | 
            M.Tech AIML - BITS Pilani
        </p>
        <p style='font-size: 0.8rem; color: #c71585;'>
            Early detection saves lives. This tool is for educational purposes only.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()