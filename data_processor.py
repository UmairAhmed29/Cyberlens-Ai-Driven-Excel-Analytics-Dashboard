import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class DataProcessor:
    """
    Utility class to handle data processing operations for the AI Dashboard
    """
    
    @staticmethod
    def identify_column_types(df):
        """
        Identify column types in the dataframe
        
        Returns:
            tuple: (numeric_cols, categorical_cols, date_cols)
        """
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
    
    @staticmethod
    def generate_data_insights(df, numeric_cols, categorical_cols):
        """
        Generate automated insights from the data
        
        Returns:
            list: List of insight strings
        """
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
    
    @staticmethod
    def perform_clustering(df, columns, n_clusters=3):
        """
        Perform K-means clustering on selected columns
        
        Returns:
            tuple: (plotly_figure, cluster_labels)
        """
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[columns])
        
        # Apply PCA if more than 2 dimensions
        if len(columns) > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(scaled_data)
            explained_variance = pca.explained_variance_ratio_.sum()
        else:
            reduced_data = scaled_data
            explained_variance = 1.0
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Create a new dataframe with the results
        result_df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'cluster': clusters
        })
        
        # Additional cluster analysis
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_points = scaled_data[clusters == i]
            cluster_stats[i] = {
                'count': len(cluster_points),
                'percentage': len(cluster_points) / len(df) * 100,
                'centroid': kmeans.cluster_centers_[i]
            }
        
        return result_df, clusters, cluster_stats, explained_variance
    
    @staticmethod
    def detect_outliers(df, columns, method='iqr', threshold=1.5):
        """
        Detect outliers in specified columns
        
        Args:
            df: DataFrame to analyze
            columns: List of columns to check for outliers
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame: Boolean mask of outliers
        """
        outlier_mask = pd.DataFrame(index=df.index)
        
        if method == 'iqr':
            for col in columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
        elif method == 'zscore':
            for col in columns:
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:  # Avoid division by zero
                    z_scores = (df[col] - mean) / std
                    outlier_mask[col] = abs(z_scores) > threshold
                else:
                    outlier_mask[col] = False
        
        # Add total outliers column
        outlier_mask['total_outliers'] = outlier_mask.sum(axis=1)
        
        return outlier_mask
    
    @staticmethod
    def analyze_time_series(df, date_column, value_column, freq=None):
        """
        Perform time series analysis
        
        Args:
            df: DataFrame to analyze
            date_column: Column containing dates
            value_column: Column containing values to analyze
            freq: Frequency for resampling (e.g., 'D', 'M', 'Y')
            
        Returns:
            DataFrame: Processed time series data
        """
        # Ensure date column is datetime
        df_ts = df.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        
        # Sort by date
        df_ts = df_ts.sort_values(date_column)
        
        # Resample if frequency is specified
        if freq:
            df_ts = df_ts.set_index(date_column)
            df_ts = df_ts[value_column].resample(freq).mean().reset_index()
        
        # Calculate additional metrics
        if len(df_ts) > 1:
            # Calculate percent change
            df_ts['pct_change'] = df_ts[value_column].pct_change()
            
            # Calculate rolling metrics if enough data points
            if len(df_ts) > 7:
                df_ts['rolling_mean_7'] = df_ts[value_column].rolling(window=7).mean()
                
            if len(df_ts) > 30:
                df_ts['rolling_mean_30'] = df_ts[value_column].rolling(window=30).mean()
        
        return df_ts
