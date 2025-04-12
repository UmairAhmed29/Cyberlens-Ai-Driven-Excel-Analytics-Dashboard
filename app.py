import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(layout="wide", page_title="CyberLens: AI-Driven Excel Analytics Dashboard", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --neon-blue: #00f3ff;
        --neon-pink: #ff00c8;
        --neon-purple: #9600ff;
        --dark-bg: #0a0a0f;
        --darker-bg: #050508;
        --text-color: #ffffff;
    }
    
    /* Override Streamlit's background with dark cyberpunk theme */
    .stApp {
        background-color: var(--dark-bg);
        background-image: 
            linear-gradient(0deg, transparent 24%, rgba(0, 243, 255, 0.05) 25%, rgba(0, 243, 255, 0.05) 26%, transparent 27%, transparent 74%, rgba(0, 243, 255, 0.05) 75%, rgba(0, 243, 255, 0.05) 76%, transparent 77%, transparent),
            linear-gradient(90deg, transparent 24%, rgba(0, 243, 255, 0.05) 25%, rgba(0, 243, 255, 0.05) 26%, transparent 27%, transparent 74%, rgba(0, 243, 255, 0.05) 75%, rgba(0, 243, 255, 0.05) 76%, transparent 77%, transparent);
        background-size: 50px 50px;
    }
    
    /* Glowing elements */
    .glow-text {
        color: var(--neon-blue);
        text-shadow: 0 0 5px var(--neon-blue), 0 0 10px var(--neon-blue), 0 0 15px var(--neon-blue);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    /* Cyberpunk animations */
    @keyframes glow {
        0% {
            text-shadow: 0 0 5px var(--neon-blue), 0 0 10px var(--neon-blue);
        }
        100% {
            text-shadow: 0 0 10px var(--neon-blue), 0 0 20px var(--neon-blue), 0 0 30px var(--neon-blue);
        }
    }
    
    @keyframes scanline {
        0% {
            transform: translateY(-100%);
        }
        100% {
            transform: translateY(100%);
        }
    }
    
    @keyframes flicker {
        0%, 19.999%, 22%, 62.999%, 64%, 64.999%, 70%, 100% {
            opacity: 0.99;
        }
        20%, 21.999%, 63%, 63.999%, 65%, 69.999% {
            opacity: 0.4;
        }
    }
    
    /* Scanline effect */
    .scanline {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100px;
        background: linear-gradient(to bottom, 
            rgba(0, 243, 255, 0) 0%,
            rgba(0, 243, 255, 0.1) 50%,
            rgba(0, 243, 255, 0) 100%);
        opacity: 0.1;
        z-index: 9999;
        pointer-events: none;
        animation: scanline 8s linear infinite;
    }
    
    /* Blinking cursor */
    .cursor {
        display: inline-block;
        width: 10px;
        height: 20px;
        background-color: var(--neon-blue);
        animation: blink 1s step-end infinite;
        margin-left: 5px;
    }
    
    @keyframes blink {
        0%, 49% {
            opacity: 1;
        }
        50%, 100% {
            opacity: 0;
        }
    }
    
    /* Main header with cyberpunk style */
    .main-header {
        font-size: 2.5rem;
        color: var(--neon-blue);
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 3px;
        animation: flicker 5s linear infinite;
        margin-bottom: 20px;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 5px var(--neon-blue), 0 0 10px var(--neon-blue);
    }
    
    /* Other UI elements */
    .sub-header {
        font-size: 1.5rem;
        color: var(--neon-blue);
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 3px var(--neon-blue);
    }
    
    .card {
        border-radius: 5px;
        background-color: rgba(10, 10, 15, 0.7);
        padding: 20px;
        margin-bottom: 10px;
        border: 1px solid rgba(0, 243, 255, 0.2);
    }
    
    .insight-card {
        background-color: #212529;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid var(--neon-blue);
        box-shadow: 0 0 5px rgba(0, 243, 255, 0.2);
        color: #ffffff;
    }
    
    .insight-card b {
        color: #4dabf7;
        font-size: 1.1rem;
    }
    
    /* Streamlit element styling */
    .stButton button {
        background-color: var(--darker-bg);
        color: var(--neon-blue);
        border: 1px solid var(--neon-blue);
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: var(--neon-blue);
        color: var(--darker-bg);
        box-shadow: 0 0 10px var(--neon-blue);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: var(--darker-bg);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'date_columns' not in st.session_state:
    st.session_state.date_columns = []
if 'insights' not in st.session_state:
    st.session_state.insights = []

def get_data_types(df):
    """Identify column types in the dataframe"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = []
    
    # Try to convert object columns to datetime
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
            categorical_cols.remove(col)
        except:
            pass
    
    return numeric_cols, categorical_cols, date_cols

def generate_insights(df, numeric_cols, categorical_cols):
    """Generate automated insights from the data"""
    insights = []
    
    # Basic statistics
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            mean_val = df[col].mean()
            max_val = df[col].max()
            min_val = df[col].min()
            insights.append(f"The average {col} is {mean_val:.2f}, ranging from {min_val:.2f} to {max_val:.2f}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing.items():
            if count > 0:
                insights.append(f"Column '{col}' has {count} missing values ({count/len(df)*100:.1f}%)")
    
    # Correlations between numeric columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.7:
                    strength = "strong positive" if corr_val > 0 else "strong negative"
                    insights.append(f"There is a {strength} correlation ({corr_val:.2f}) between {col1} and {col2}")
    
    # Distribution of categorical columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if df[col].nunique() < 10:  # Only for columns with reasonable number of categories
                top_category = df[col].value_counts().idxmax()
                top_percentage = df[col].value_counts().max() / len(df) * 100
                insights.append(f"The most common {col} is '{top_category}' ({top_percentage:.1f}%)")
    
    return insights

def create_visualization(df, viz_type, x_col, y_col=None, color_col=None, size_col=None):
    """Create visualization based on user selection"""
    fig = None
    
    if viz_type == "Bar Chart":
        if y_col:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                         title=f"{y_col} by {x_col}")
        else:
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, 'count']
            fig = px.bar(counts, x=x_col, y='count', title=f"Count of {x_col}")
            
    elif viz_type == "Line Chart":
        fig = px.line(df, x=x_col, y=y_col, color=color_col, 
                     title=f"{y_col} over {x_col}")
        
    elif viz_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                        title=f"{y_col} vs {x_col}")
        
    elif viz_type == "Pie Chart":
        counts = df[x_col].value_counts().reset_index()
        counts.columns = [x_col, 'count']
        fig = px.pie(counts, values='count', names=x_col, title=f"Distribution of {x_col}")
        
    elif viz_type == "Histogram":
        fig = px.histogram(df, x=x_col, color=color_col, 
                          title=f"Distribution of {x_col}")
        
    elif viz_type == "Box Plot":
        fig = px.box(df, x=x_col, y=y_col, color=color_col,
                    title=f"Box Plot of {y_col} by {x_col}")
        
    elif viz_type == "Heatmap":
        if len(df) > 1000:
            sample_df = df.sample(1000)
        else:
            sample_df = df
        corr_matrix = sample_df.select_dtypes(include=['number']).corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Heatmap")
    
    if fig:
        fig.update_layout(height=500)
        return fig
    return None

def perform_clustering(df, columns, n_clusters=3):
    """Perform K-means clustering on selected columns"""
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])
    
    # Apply PCA if more than 2 dimensions
    if len(columns) > 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
    else:
        reduced_data = scaled_data
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Create a new dataframe with the results
    result_df = pd.DataFrame({
        'x': reduced_data[:, 0],
        'y': reduced_data[:, 1],
        'cluster': clusters
    })
    
    # Create the plot
    fig = px.scatter(result_df, x='x', y='y', color='cluster', 
                    title=f"K-means Clustering (k={n_clusters})")
    
    return fig, clusters

# Main application layout
st.markdown("<h1 class='main-header'>CyberLens: AI-Driven Excel Analytics Dashboard</h1>", unsafe_allow_html=True)

# Scanline effect
st.markdown("<div class='scanline'></div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Upload Data</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.session_state.numeric_columns, st.session_state.categorical_columns, st.session_state.date_columns = get_data_types(df)
            st.session_state.insights = generate_insights(df, st.session_state.numeric_columns, st.session_state.categorical_columns)
            st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    if st.session_state.df is not None:
        st.markdown("<h2 class='sub-header'>Dashboard Settings</h2>", unsafe_allow_html=True)
        dashboard_tabs = st.radio("Select Dashboard Mode", 
                                ["Quick Analysis", "Custom Visualizations", "AI Insights"])

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Data Overview Tab
    expander = st.expander("Data Overview", expanded=True)
    with expander:
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
        with col2:
            st.write(f"**Numeric columns:** {len(st.session_state.numeric_columns)}")
            st.write(f"**Categorical columns:** {len(st.session_state.categorical_columns)}")
    
    # Dashboard content based on selected tab
    if dashboard_tabs == "Quick Analysis":
        st.markdown("<h2 class='sub-header'>Quick Analysis</h2>", unsafe_allow_html=True)
        
        # Automatic visualizations based on data types
        col1, col2 = st.columns(2)
        
        # Numeric distributions
        if len(st.session_state.numeric_columns) > 0:
            with col1:
                st.subheader("Numeric Distributions")
                selected_num_col = st.selectbox("Select column", st.session_state.numeric_columns)
                fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Categorical distributions
        if len(st.session_state.categorical_columns) > 0:
            with col2:
                st.subheader("Categorical Distributions")
                selected_cat_col = st.selectbox("Select column", st.session_state.categorical_columns)
                counts = df[selected_cat_col].value_counts().head(10).reset_index()
                counts.columns = [selected_cat_col, 'count']
                fig = px.bar(counts, x=selected_cat_col, y='count', title=f"Top 10 {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Correlation heatmap for numeric columns
        if len(st.session_state.numeric_columns) > 1:
            st.subheader("Correlation Heatmap")
            corr_matrix = df[st.session_state.numeric_columns].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Time series if date columns exist
        if len(st.session_state.date_columns) > 0 and len(st.session_state.numeric_columns) > 0:
            st.subheader("Time Series Analysis")
            col1, col2 = st.columns(2)
            with col1:
                selected_date_col = st.selectbox("Select date column", st.session_state.date_columns)
            with col2:
                selected_value_col = st.selectbox("Select value column", st.session_state.numeric_columns)
            
            # Convert to datetime and sort
            df_ts = df.copy()
            df_ts[selected_date_col] = pd.to_datetime(df_ts[selected_date_col])
            df_ts = df_ts.sort_values(selected_date_col)
            
            fig = px.line(df_ts, x=selected_date_col, y=selected_value_col, 
                         title=f"{selected_value_col} over time")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif dashboard_tabs == "Custom Visualizations":
        st.markdown("<h2 class='sub-header'>Custom Visualizations</h2>", unsafe_allow_html=True)
        
        # Visualization creator
        viz_type = st.selectbox("Select Visualization Type", 
                               ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", 
                                "Histogram", "Box Plot", "Heatmap"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            all_columns = df.columns.tolist()
            x_col = st.selectbox("X-axis", all_columns)
        
        with col2:
            y_col = None
            if viz_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"]:
                y_options = ["None"] + st.session_state.numeric_columns
                y_selection = st.selectbox("Y-axis", y_options)
                if y_selection != "None":
                    y_col = y_selection
        
        with col3:
            color_col = None
            if viz_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot"]:
                color_options = ["None"] + st.session_state.categorical_columns
                color_selection = st.selectbox("Color", color_options)
                if color_selection != "None":
                    color_col = color_selection
        
        size_col = None
        if viz_type == "Scatter Plot":
            size_options = ["None"] + st.session_state.numeric_columns
            size_selection = st.selectbox("Size", size_options)
            if size_selection != "None":
                size_col = size_selection
        
        # Create and display visualization
        fig = create_visualization(df, viz_type, x_col, y_col, color_col, size_col)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        if st.button("Export Visualization"):
            buffer = BytesIO()
            fig.write_image(buffer, format="png")
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="visualization.png">Download Image</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Advanced analytics section
        st.markdown("<h3 class='sub-header'>Advanced Analytics</h3>", unsafe_allow_html=True)
        
        analytics_type = st.selectbox("Select Analysis Type", ["Clustering"])
        
        if analytics_type == "Clustering":
            st.write("Select columns for clustering:")
            cluster_columns = st.multiselect("Columns", st.session_state.numeric_columns)
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            
            if len(cluster_columns) >= 2 and st.button("Perform Clustering"):
                with st.spinner("Performing clustering..."):
                    cluster_fig, clusters = perform_clustering(df, cluster_columns, n_clusters)
                    st.plotly_chart(cluster_fig, use_container_width=True)
                    
                    # Add cluster labels to dataframe
                    df_with_clusters = df.copy()
                    df_with_clusters['Cluster'] = clusters
                    
                    # Show cluster statistics
                    st.write("Cluster Statistics:")
                    for i in range(n_clusters):
                        st.write(f"**Cluster {i}:** {(clusters == i).sum()} data points")
                    
                    # Show sample from each cluster
                    st.write("Sample from each cluster:")
                    st.dataframe(df_with_clusters.groupby('Cluster').apply(lambda x: x.sample(min(3, len(x)))).reset_index(drop=True))
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif dashboard_tabs == "AI Insights":
        st.markdown("<h2 class='sub-header'>AI Insights</h2>", unsafe_allow_html=True)
        
        # Display automated insights
        if st.session_state.insights:
            st.subheader("Automated Data Insights")
            for i, insight in enumerate(st.session_state.insights):
                insight_text = insight
                # Add icons based on insight type
                if "average" in insight.lower():
                    icon = "📊"
                elif "missing" in insight.lower():
                    icon = "⚠️"
                elif "correlation" in insight.lower():
                    icon = "🔄"
                elif "most common" in insight.lower():
                    icon = "🔝"
                else:
                    icon = "💡"
                
                st.markdown(f"<div class='insight-card'><b>{icon} Insight {i+1}:</b> {insight_text}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No automated insights available. Upload data to generate insights.")
        
        # Custom insight generation
        st.subheader("Generate Custom Insights")
        
        insight_type = st.selectbox("Select Insight Type", 
                                   ["Column Analysis", "Relationship Analysis", "Outlier Detection"])
        
        if insight_type == "Column Analysis":
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_col = st.selectbox("Select column to analyze", df.columns.tolist())
                analyze_button = st.button("Generate Column Insights", type="primary")
            
            with col2:
                st.markdown("""
                <div style="background-color: #212529; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: #ffffff;">
                    <p><b style="color: #4dabf7;">Column Analysis</b> examines the distribution, statistics, and patterns within a single column.</p>
                    <p>For numeric columns, you'll get statistics and distribution visualization.</p>
                    <p>For categorical columns, you'll see category distribution and frequency analysis.</p>
                </div>
                """, unsafe_allow_html=True)
            
            if analyze_button:
                with st.spinner("Analyzing..."):
                    if selected_col in st.session_state.numeric_columns:
                        # Numeric column analysis
                        stats = df[selected_col].describe()
                        
                        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                        st.write(f"**📈 Basic Statistics for {selected_col}:**")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean", f"{stats['mean']:.2f}")
                        col2.metric("Median", f"{stats['50%']:.2f}")
                        col3.metric("Min", f"{stats['min']:.2f}")
                        col4.metric("Max", f"{stats['max']:.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Distribution plot
                        fig = px.histogram(df, x=selected_col, marginal="box", 
                                          title=f"Distribution of {selected_col}")
                        fig.update_layout(
                            height=400,
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Outlier analysis
                        q1 = stats['25%']
                        q3 = stats['75%']
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                        
                        if len(outliers) > 0:
                            st.markdown("<div class='insight-card' style='background-color: #ffebee; padding: 15px; border-radius: 5px; margin: 15px 0;'>", unsafe_allow_html=True)
                            st.write(f"**Outliers detected:** {len(outliers)} values outside the range [{lower_bound:.2f}, {upper_bound:.2f}]")
                            st.dataframe(outliers.head(10), use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    elif selected_col in st.session_state.categorical_columns:
                        # Categorical column analysis
                        value_counts = df[selected_col].value_counts()
                        
                        st.markdown("<div class='insight-card' style='background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 15px 0;'>", unsafe_allow_html=True)
                        st.write(f"**Category Distribution for {selected_col}:**")
                        st.write(f"- Number of unique values: {df[selected_col].nunique()}")
                        st.write(f"- Most common value: '{value_counts.index[0]}' ({value_counts.iloc[0]} occurrences)")
                        st.write(f"- Least common value: '{value_counts.index[-1]}' ({value_counts.iloc[-1]} occurrences)")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Bar chart of top categories
                        top_n = min(10, len(value_counts))
                        fig = px.bar(x=value_counts.index[:top_n], y=value_counts.values[:top_n],
                                    labels={'x': selected_col, 'y': 'Count'},
                                    title=f"Top {top_n} categories in {selected_col}")
                        fig.update_layout(
                            height=400,
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        elif insight_type == "Relationship Analysis":
            col1, col2 = st.columns([1, 2])
            with col1:
                first_col = st.selectbox("First column", df.columns.tolist())
                second_col = st.selectbox("Second column", [c for c in df.columns if c != first_col])
                analyze_button = st.button("Analyze Relationship", type="primary")
            
            with col2:
                st.markdown("""
                <div style="background-color: #212529; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: #ffffff;">
                    <p><b style="color: #4dabf7;">Relationship Analysis</b> examines how two columns relate to each other.</p>
                    <p>For numeric-numeric relationships, you'll get correlation analysis and scatter plots.</p>
                    <p>For categorical-numeric relationships, you'll see group comparisons and box plots.</p>
                    <p>For categorical-categorical relationships, you'll get contingency tables and heatmaps.</p>
                </div>
                """, unsafe_allow_html=True)
            
            if analyze_button:
                with st.spinner("Analyzing relationship..."):
                    # Check column types and perform appropriate analysis
                    if first_col in st.session_state.numeric_columns and second_col in st.session_state.numeric_columns:
                        # Numeric vs Numeric: Scatter plot and correlation
                        corr = df[first_col].corr(df[second_col])
                        
                        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                        st.write(f"**🔄 Correlation between {first_col} and {second_col}:** {corr:.4f}")
                        
                        # Additional insights based on correlation strength
                        if abs(corr) > 0.7:
                            st.write(f"There is a strong {'positive' if corr > 0 else 'negative'} correlation between these variables.")
                        elif abs(corr) > 0.3:
                            st.write(f"There is a moderate {'positive' if corr > 0 else 'negative'} correlation between these variables.")
                        else:
                            st.write("There is a weak correlation between these variables.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        fig = px.scatter(df, x=first_col, y=second_col, trendline="ols",
                                        title=f"Relationship between {first_col} and {second_col}")
                        fig.update_layout(
                            height=400,
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif (first_col in st.session_state.categorical_columns and second_col in st.session_state.numeric_columns) or \
                         (second_col in st.session_state.categorical_columns and first_col in st.session_state.numeric_columns):
                        # Ensure cat_col is the categorical one and num_col is the numeric one
                        if first_col in st.session_state.categorical_columns:
                            cat_col, num_col = first_col, second_col
                        else:
                            cat_col, num_col = second_col, first_col
                            
                        # Categorical vs Numeric: Box plot and ANOVA
                        fig = px.box(df, x=cat_col, y=num_col,
                                    title=f"Distribution of {num_col} by {cat_col}")
                        fig.update_layout(
                            height=400,
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Group statistics
                        group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std']).reset_index()
                        
                        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                        st.write(f"**🔍 Group Statistics:**")
                        st.dataframe(group_stats, use_container_width=True)
                        
                        # Check if there are significant differences between groups
                        if df[cat_col].nunique() <= 10:  # Only for a reasonable number of categories
                            max_mean = group_stats['mean'].max()
                            min_mean = group_stats['mean'].min()
                            max_group = group_stats.loc[group_stats['mean'].idxmax(), cat_col]
                            min_group = group_stats.loc[group_stats['mean'].idxmin(), cat_col]
                            
                            st.write(f"The highest average {num_col} is in the '{max_group}' group ({max_mean:.2f}).")
                            st.write(f"The lowest average {num_col} is in the '{min_group}' group ({min_mean:.2f}).")
                            
                            if (max_mean - min_mean) / group_stats['std'].mean() > 2:
                                st.write("There appears to be a significant difference between groups.")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    elif first_col in st.session_state.categorical_columns and second_col in st.session_state.categorical_columns:
                        # Categorical vs Categorical: Contingency table and chi-square
                        contingency = pd.crosstab(df[first_col], df[second_col])
                        
                        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                        st.write(f"**📊 Contingency Table:**")
                        st.dataframe(contingency, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Visualization
                        fig = px.imshow(contingency, text_auto=True, aspect="auto",
                                      title=f"Relationship between {first_col} and {second_col}")
                        fig.update_layout(
                            height=400,
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate proportions
                        prop_table = contingency.div(contingency.sum(axis=1), axis=0)
                        
                        # Check for strong associations
                        if prop_table.max().max() > 0.8:
                            st.info("There appears to be a strong association between these categorical variables.")
        
        elif insight_type == "Outlier Detection":
            if len(st.session_state.numeric_columns) > 0:
                col1, col2 = st.columns([1, 2])
                with col1:
                    selected_cols = st.multiselect("Select columns for outlier detection", 
                                                  st.session_state.numeric_columns)
                    outlier_button = st.button("Detect Outliers", type="primary")
                
                with col2:
                    st.markdown("""
                    <div style="background-color: #212529; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: #ffffff;">
                        <p><b style="color: #4dabf7;">Outlier Detection</b> identifies data points that significantly differ from the majority of the data.</p>
                        <p>This analysis uses the Interquartile Range (IQR) method to identify outliers.</p>
                        <p>Select multiple columns to find rows with outliers across different variables.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if selected_cols and outlier_button:
                    with st.spinner("Detecting outliers..."):
                        outlier_results = pd.DataFrame(index=df.index)
                        
                        for col in selected_cols:
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            outlier_results[f"{col}_outlier"] = ((df[col] < lower_bound) | (df[col] > upper_bound))
                        
                        # Add a column for total outliers across selected columns
                        outlier_results['total_outliers'] = outlier_results.sum(axis=1)
                        
                        # Merge with original data
                        result_df = pd.concat([df, outlier_results['total_outliers']], axis=1)
                        
                        # Display summary
                        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                        st.write(f"**⚠️ Outlier Summary:**")
                        st.write(f"- Total rows with outliers: {(outlier_results['total_outliers'] > 0).sum()}")
                        st.write(f"- Percentage of data with outliers: {(outlier_results['total_outliers'] > 0).sum() / len(df) * 100:.2f}%")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display outlier counts by column
                        outlier_counts = outlier_results.iloc[:, :-1].sum()
                        
                        fig = px.bar(x=outlier_counts.index, y=outlier_counts.values,
                                    labels={'x': 'Column', 'y': 'Number of Outliers'},
                                    title="Outliers by Column")
                        fig.update_layout(
                            height=400,
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display top outlier rows
                        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                        st.write("**📋 Top rows with multiple outliers:**")
                        st.dataframe(result_df.sort_values('total_outliers', ascending=False).head(10), use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No numeric columns available for outlier detection.")
        
        st.markdown("</div>", unsafe_allow_html=True)
else:
    # Display welcome message when no data is loaded
    st.markdown("""
    <div style="text-align: center; padding: 50px; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(circle at center, rgba(0,243,255,0.1) 0%, rgba(10,10,15,0) 70%); pointer-events: none;"></div>
        <h2 class="glow-text" style="margin-bottom: 10px; font-family: 'Courier New', monospace; text-transform: uppercase; letter-spacing: 2px;">Welcome to CyberLens</h2>
        <p style="color: var(--neon-blue); font-family: 'Courier New', monospace; margin-bottom: 30px; font-size: 1.2rem; opacity: 0.8;">Transform Your Excel Data into Actionable Intelligence</p>
        <div style="font-family: 'Courier New', monospace; color: #ccc; margin-bottom: 30px; font-size: 1.1rem;">
            <p>> SYSTEM INITIALIZED<span class="cursor"></span></p>
            <p>> AWAITING INPUT: UPLOAD EXCEL FILE TO PROCEED</p>
            <p>> AI ANALYSIS MODULES STANDING BY</p>
        </div>
        <div style="margin: 40px auto; max-width: 600px; padding: 20px; background-color: rgba(10,10,15,0.7); border: 1px solid rgba(0,243,255,0.3); border-radius: 5px;">
            <h3 style="color: var(--neon-blue); margin-bottom: 20px; font-family: 'Courier New', monospace;">FEATURES:</h3>
            <ul style="text-align: left; color: #ccc; list-style-type: none; padding-left: 10px;">
                <li style="margin-bottom: 10px; border-left: 2px solid var(--neon-blue); padding-left: 15px;">⚡ Automated data analysis and insights</li>
                <li style="margin-bottom: 10px; border-left: 2px solid var(--neon-pink); padding-left: 15px;">⚡ Interactive visualizations</li>
                <li style="margin-bottom: 10px; border-left: 2px solid var(--neon-purple); padding-left: 15px;">⚡ Custom dashboard creation</li>
                <li style="margin-bottom: 10px; border-left: 2px solid var(--neon-blue); padding-left: 15px;">⚡ Advanced analytics including clustering</li>
                <li style="margin-bottom: 10px; border-left: 2px solid var(--neon-pink); padding-left: 15px;">⚡ Export and sharing options</li>
            </ul>
        </div>
        <div style="font-size: 0.8rem; color: #666; margin-top: 40px; font-family: monospace;">
            <p>v1.0.0 | RUNNING ON SECURE CONNECTION | ENCRYPTION ENABLED</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
