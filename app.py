from sys import byteorder
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="IRIS Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load all models
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_names = [
        'logistic_regression',
        'decision_tree',
        'naive_bayes',
        'bagging',
        'xgboost',
        'adaboost',
        'mlp'
    ]

    for model_name in model_names:
        try:
            with open(f'models/{model_name}_model.pkl', 'rb') as f:
                models[model_name] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Model {model_name} not found. Please train the model first.")

    return models

@st.cache_data
def load_dataset():
    """Load the cleaned IRIS dataset"""
    try:
        df = pd.read_csv('models/cleaned_iris_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please run the data preparation notebook first.")
        return None

def create_3d_scatter_plot(df, predictions=None, feature_combo=(0, 2, 3)):
    """Create interactive 3D scatter plot"""
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    selected_features = [feature_names[i] for i in feature_combo]

    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    df_plot = df.copy()
    df_plot['species_name'] = df_plot['species'].map(species_map)

    fig = go.Figure()

    colors = {'Setosa': 'red', 'Versicolor': 'blue', 'Virginica': 'green'}

    for species in ['Setosa', 'Versicolor', 'Virginica']:
        df_species = df_plot[df_plot['species_name'] == species]
        fig.add_trace(go.Scatter3d(
            x=df_species[selected_features[0]],
            y=df_species[selected_features[1]],
            z=df_species[selected_features[2]],
            mode='markers',
            name=species,
            marker=dict(
                size=6,
                color=colors[species],
                opacity=0.7,
                line=dict(color='white', width=0.5)
            )
        ))

    fig.update_layout(
        title='3D Visualization of IRIS Dataset',
        scene=dict(
            xaxis_title=selected_features[0].replace('_', ' ').title(),
            yaxis_title=selected_features[1].replace('_', ' ').title(),
            zaxis_title=selected_features[2].replace('_', ' ').title(),
        ),
        height=600,
        showlegend=True
    )

    return fig

def predict_species(features, model_data):
    """Make prediction using the selected model"""
    scaler = model_data['scaler']
    model = model_data['model']

    # Scale features
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    return prediction, probabilities

def get_all_predictions(features, models):
    """Get predictions from all models"""
    results = {}
    for model_name, model_data in models.items():
        if model_data:
            pred, proba = predict_species(features, model_data)
            results[model_name] = {
                'prediction': pred,
                'probabilities': proba
            }
    return results

# Main App
def main():
    st.markdown('<div class="main-header">🌸 IRIS Flower Classification System 🌸</div>', unsafe_allow_html=True)

    # Load resources
    models = load_models()
    df = load_dataset()

    if df is None:
        st.stop()

    # Species mapping
    species_names = ['Setosa', 'Versicolor', 'Virginica']

    # Create tabs
    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Data Exploration"])

    # ===== TAB 1: PREDICTION =====
    with tab1:
        st.markdown('<div class="sub-header">Model Prediction Interface</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 🎯 Input Features")

            # Feature inputs using sliders
            with st.container(border = True):
                sepal_length = st.slider("Sepal Length (cm)",min_value=4.0,max_value=8.0,value=5.8,
                                        step=0.1,help="Length of the sepal in centimeters")

            with st.container(border = True):
                sepal_width = st.slider("Sepal Width (cm)",min_value=2.0,max_value=4.5,value=3.0,
                step=0.1,help="Width of the sepal in centimeters")

            with st.container(border = True):
                petal_length = st.slider("Petal Length (cm)",min_value=1.0,max_value=7.0,value=4.0,
                step=0.1,help="Length of the petal in centimeters")

            with st.container(border = True):
                petal_width = st.slider("Petal Width (cm)",min_value=0.1,max_value=2.5,value=1.2,
                step=0.1,help="Width of the petal in centimeters")

            st.markdown("---")

            # Model selection
            st.markdown("### 🤖 Select Model")
            model_display_names = {
                'logistic_regression': 'Logistic Regression',
                'decision_tree': 'Decision Tree',
                'naive_bayes': 'Naive Bayes',
                'bagging': 'Bagging Classifier',
                'xgboost': 'XGBoost',
                'adaboost': 'AdaBoost',
                'mlp': 'MLP Neural Network'
            }

            selected_model_key = st.selectbox(
                "Choose a classification model",
                options=list(models.keys()),
                format_func=lambda x: model_display_names.get(x, x)
            )

            st.markdown("---")

            # Prediction button
            predict_button = st.button("🔮 Predict Species", type="primary", use_container_width=True)
            reset_button = st.button("🔄 Reset", use_container_width=True)

        with col2:
            if predict_button:
                features = [sepal_length, sepal_width, petal_length, petal_width]

                # Get predictions from all models
                all_results = get_all_predictions(features, models)

                # Selected model prediction
                if selected_model_key in all_results:
                    result = all_results[selected_model_key]
                    predicted_class = result['prediction']
                    probabilities = result['probabilities']

                    # Display main prediction
                    st.markdown("### 🎯 Prediction Result")
                    st.success(f"**{species_names[predicted_class]}**".upper())

                    # Display probabilities
                    st.markdown("### 📊 Prediction Confidence")
                    for i, species in enumerate(species_names):
                        st.metric(
                            label=species,
                            value=f"{probabilities[i]:.2%}",
                            delta=None
                        )

                    # Probability bar chart
                    fig_proba = go.Figure(data=[
                        go.Bar(
                            x=species_names,
                            y=probabilities,
                            text=[f'{p:.2%}' for p in probabilities],
                            textposition='auto',
                            marker_color=['#ff7f0e' if i == predicted_class else '#1f77b4' 
                                        for i in range(len(species_names))]
                        )
                    ])
                    fig_proba.update_layout(
                        title=f'Prediction Probabilities - {model_display_names[selected_model_key]}',
                        xaxis_title='Species',
                        yaxis_title='Probability',
                        yaxis_range=[0, 1],
                        height=300
                    )
                    st.plotly_chart(fig_proba, use_container_width=True)

                    # Comparison across all models
                    st.markdown("### 🔬 Comparison Across All Models")
                    comparison_data = []
                    for model_name, result in all_results.items():
                        pred = result['prediction']
                        proba = result['probabilities']
                        comparison_data.append({
                            'Model': model_display_names[model_name],
                            'Prediction': species_names[pred],
                            'Setosa': f'{proba[0]:.4f}',
                            'Versicolor': f'{proba[1]:.4f}',
                            'Virginica': f'{proba[2]:.4f}',
                            'Confidence': f'{proba[pred]:.4f}'
                        })

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                    # Model performance metrics
                    st.markdown("### 📈 Model Performance on Test Set")
                    if selected_model_key in models and models[selected_model_key]:
                        model_data = models[selected_model_key]

                        # Display test accuracy
                        if 'test_accuracy' in model_data:
                            st.info(f"**Test Accuracy: {model_data['test_accuracy']:.2%}**")

                        # Classification report
                        st.markdown("#### Classification Report")
                        X_test = df.drop('species', axis=1)
                        y_test = df['species']

                        scaler = model_data['scaler']
                        model = model_data['model']

                        X_test_scaled = scaler.transform(X_test)
                        y_pred = model.predict(X_test_scaled)

                        report = classification_report(y_test, y_pred, 
                                                     target_names=species_names,
                                                     output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4), use_container_width=True)

                        # Confusion Matrix
                        st.markdown("#### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)

                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=species_names,
                            y=species_names,
                            colorscale='Blues',
                            text=cm,
                            texttemplate='%{text}',
                            textfont={"size": 16}
                        ))
                        fig_cm.update_layout(
                            title='Confusion Matrix',
                            xaxis_title='Predicted Label',
                            yaxis_title='True Label',
                            height=400
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                    # 3D Visualization
                    st.markdown("### 🎨 3D Cluster Visualization")
                    fig_3d = create_3d_scatter_plot(df)
                    st.plotly_chart(fig_3d, use_container_width=True)

            elif reset_button:
                st.rerun()

            else:
                st.info("👈 Adjust the feature sliders and select a model, then click 'Predict Species' to see results!")

    # ===== TAB 2: DATA EXPLORATION =====
    with tab2:
        st.markdown('<div class="sub-header">Data Exploration & Visualization</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📊 Plot Configuration")

            plot_type = st.selectbox(
                "Select Plot Type",
                options=[
                    "Scatter Plot",
                    "Box Plot",
                    "Violin Plot",
                    "Histogram",
                    "Pair Plot (2 features)",
                    "3D Scatter Plot",
                    "Correlation Heatmap"
                ]
            )

            feature_options = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

            if plot_type in ["Scatter Plot", "Pair Plot (2 features)"]:
                x_feature = st.selectbox("X-axis Feature", feature_options, index=0)
                y_feature = st.selectbox("Y-axis Feature", feature_options, index=2)
            elif plot_type in ["Box Plot", "Violin Plot", "Histogram"]:
                selected_feature = st.selectbox("Select Feature", feature_options)
            elif plot_type == "3D Scatter Plot":
                x_feature = st.selectbox("X-axis Feature", feature_options, index=0)
                y_feature = st.selectbox("Y-axis Feature", feature_options, index=2)
                z_feature = st.selectbox("Z-axis Feature", feature_options, index=3)

            explore_button = st.button("📊 Explore Data", type="primary", use_container_width=True)
            reset_explore = st.button("🔄 Reset Exploration", use_container_width=True)

        with col2:
            if explore_button:
                species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
                df_plot = df.copy()
                df_plot['species_name'] = df_plot['species'].map(species_map)

                if plot_type == "Scatter Plot":
                    fig = px.scatter(
                        df_plot,
                        x=x_feature,
                        y=y_feature,
                        color='species_name',
                        title=f'{x_feature.replace("_", " ").title()} vs {y_feature.replace("_", " ").title()}',
                        labels={
                            x_feature: x_feature.replace('_', ' ').title(),
                            y_feature: y_feature.replace('_', ' ').title(),
                            'species_name': 'Species'
                        },
                        color_discrete_map={'Setosa': 'red', 'Versicolor': 'blue', 'Virginica': 'green'}
                    )
                    fig.update_traces(marker=dict(size=10, opacity=0.7))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Box Plot":
                    fig = px.box(
                        df_plot,
                        x='species_name',
                        y=selected_feature,
                        color='species_name',
                        title=f'Box Plot: {selected_feature.replace("_", " ").title()} by Species',
                        labels={
                            'species_name': 'Species',
                            selected_feature: selected_feature.replace('_', ' ').title()
                        },
                        color_discrete_map={'Setosa': 'red', 'Versicolor': 'blue', 'Virginica': 'green'}
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Violin Plot":
                    fig = px.violin(
                        df_plot,
                        x='species_name',
                        y=selected_feature,
                        color='species_name',
                        box=True,
                        title=f'Violin Plot: {selected_feature.replace("_", " ").title()} by Species',
                        labels={
                            'species_name': 'Species',
                            selected_feature: selected_feature.replace('_', ' ').title()
                        },
                        color_discrete_map={'Setosa': 'red', 'Versicolor': 'blue', 'Virginica': 'green'}
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Histogram":
                    fig = px.histogram(
                        df_plot,
                        x=selected_feature,
                        color='species_name',
                        marginal='box',
                        title=f'Distribution: {selected_feature.replace("_", " ").title()}',
                        labels={
                            selected_feature: selected_feature.replace('_', ' ').title(),
                            'species_name': 'Species'
                        },
                        color_discrete_map={'Setosa': 'red', 'Versicolor': 'blue', 'Virginica': 'green'}
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Pair Plot (2 features)":
                    fig = px.scatter(
                        df_plot,
                        x=x_feature,
                        y=y_feature,
                        color='species_name',
                        marginal_x='box',
                        marginal_y='violin',
                        title=f'Pair Plot: {x_feature.replace("_", " ").title()} vs {y_feature.replace("_", " ").title()}',
                        labels={
                            x_feature: x_feature.replace('_', ' ').title(),
                            y_feature: y_feature.replace('_', ' ').title(),
                            'species_name': 'Species'
                        },
                        color_discrete_map={'Setosa': 'red', 'Versicolor': 'blue', 'Virginica': 'green'}
                    )
                    fig.update_traces(marker=dict(size=10, opacity=0.7))
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "3D Scatter Plot":
                    fig = px.scatter_3d(
                        df_plot,
                        x=x_feature,
                        y=y_feature,
                        z=z_feature,
                        color='species_name',
                        title=f'3D Scatter: {x_feature}, {y_feature}, {z_feature}',
                        labels={
                            x_feature: x_feature.replace('_', ' ').title(),
                            y_feature: y_feature.replace('_', ' ').title(),
                            z_feature: z_feature.replace('_', ' ').title(),
                            'species_name': 'Species'
                        },
                        color_discrete_map={'Setosa': 'red', 'Versicolor': 'blue', 'Virginica': 'green'}
                    )
                    fig.update_traces(marker=dict(size=5, opacity=0.7))
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Correlation Heatmap":
                    corr_matrix = df[feature_options].corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=[f.replace('_', ' ').title() for f in corr_matrix.columns],
                        y=[f.replace('_', ' ').title() for f in corr_matrix.index],
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.values.round(2),
                        texttemplate='%{text}',
                        textfont={"size": 14}
                    ))
                    fig.update_layout(
                        title='Feature Correlation Heatmap',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Display dataset statistics
                st.markdown("### 📊 Dataset Statistics")
                st.dataframe(df.describe(), use_container_width=True)

            elif reset_explore:
                st.rerun()

            else:
                st.info("👈 Select a plot type and features, then click 'Explore Data' to visualize!")

    # Sidebar with information
    with st.sidebar:
        st.markdown("## 📖 About")
        st.info("""
        This application demonstrates IRIS flower classification using multiple machine learning models.

        **Features:**
        - 7 different ML models
        - Interactive predictions
        - Comprehensive visualizations
        - Model comparison
        - Performance metrics
        """)

        st.markdown("## 📊 Dataset Info")
        if df is not None:
            st.metric("Total Samples", len(df))
            st.metric("Features", 4)
            st.metric("Classes", 3)

            st.markdown("### Class Distribution")
            species_counts = df['species'].value_counts().sort_index()
            for i, count in enumerate(species_counts):
                st.metric(species_names[i], count)

        st.markdown("---")
        st.markdown("### 🌸 IRIS Species")
        st.write("**Setosa**: Small petals, distinct")
        st.write("**Versicolor**: Medium petals")
        st.write("**Virginica**: Large petals")

if __name__ == "__main__":
    main()
