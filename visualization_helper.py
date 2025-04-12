import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class VisualizationHelper:
    """
    Utility class to handle visualization creation for the AI Dashboard
    """
    
    @staticmethod
    def create_visualization(df, viz_type, x_col, y_col=None, color_col=None, size_col=None, title=None):
        """
        Create visualization based on user selection
        
        Args:
            df: DataFrame containing the data
            viz_type: Type of visualization to create
            x_col: Column for x-axis
            y_col: Column for y-axis (optional)
            color_col: Column for color encoding (optional)
            size_col: Column for size encoding (optional)
            title: Custom title for the visualization (optional)
            
        Returns:
            plotly.graph_objects.Figure: The created visualization
        """
        fig = None
        
        # Set default title if not provided
        if not title:
            if y_col:
                title = f"{y_col} by {x_col}"
            else:
                title = f"Analysis of {x_col}"
        
        if viz_type == "Bar Chart":
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                             title=title)
            else:
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, 'count']
                fig = px.bar(counts, x=x_col, y='count', title=f"Count of {x_col}")
                
        elif viz_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, 
                         title=title)
            
        elif viz_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                            title=title)
            
        elif viz_type == "Pie Chart":
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, 'count']
            fig = px.pie(counts, values='count', names=x_col, title=f"Distribution of {x_col}")
            
        elif viz_type == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col, 
                              title=f"Distribution of {x_col}")
            
        elif viz_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col,
                        title=title)
            
        elif viz_type == "Heatmap":
            if len(df) > 1000:
                sample_df = df.sample(1000)
            else:
                sample_df = df
            corr_matrix = sample_df.select_dtypes(include=['number']).corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Correlation Heatmap")
                           
        elif viz_type == "Area Chart":
            fig = px.area(df, x=x_col, y=y_col, color=color_col,
                         title=title)
                         
        elif viz_type == "Violin Plot":
            fig = px.violin(df, x=x_col, y=y_col, color=color_col,
                           title=title)
        
        elif viz_type == "Bubble Chart":
            if not size_col:
                size_col = y_col  # Default to y_col if size not specified
            fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                            title=title)
        
        if fig:
            # Apply common layout settings
            fig.update_layout(
                height=500,
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            return fig
        return None
    
    @staticmethod
    def create_time_series_plot(df, date_col, value_col, title=None, include_trend=False):
        """
        Create a time series visualization
        
        Args:
            df: DataFrame containing the data
            date_col: Column containing dates
            value_col: Column containing values to plot
            title: Custom title for the visualization (optional)
            include_trend: Whether to include trend line
            
        Returns:
            plotly.graph_objects.Figure: The created visualization
        """
        # Ensure date column is datetime and sort
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.sort_values(date_col)
        
        # Set default title if not provided
        if not title:
            title = f"{value_col} over time"
        
        if include_trend:
            fig = px.scatter(df_ts, x=date_col, y=value_col, trendline="ols", 
                            title=title)
        else:
            fig = px.line(df_ts, x=date_col, y=value_col, title=title)
        
        # Apply common layout settings
        fig.update_layout(
            height=500,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df, columns=None):
        """
        Create a correlation heatmap for numeric columns
        
        Args:
            df: DataFrame containing the data
            columns: List of columns to include (optional, defaults to all numeric)
            
        Returns:
            plotly.graph_objects.Figure: The created visualization
        """
        # Use specified columns or all numeric columns
        if columns:
            numeric_df = df[columns]
        else:
            numeric_df = df.select_dtypes(include=['number'])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Heatmap")
        
        # Apply common layout settings
        fig.update_layout(
            height=600,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_cluster_visualization(result_df, explained_variance=None):
        """
        Create visualization for clustering results
        
        Args:
            result_df: DataFrame with x, y coordinates and cluster labels
            explained_variance: Explained variance if PCA was used
            
        Returns:
            plotly.graph_objects.Figure: The created visualization
        """
        title = "K-means Clustering"
        if explained_variance:
            title += f" (Explained variance: {explained_variance:.2%})"
            
        fig = px.scatter(result_df, x='x', y='y', color='cluster', 
                        title=title)
        
        # Apply common layout settings
        fig.update_layout(
            height=600,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @staticmethod
    def export_figure_as_image(fig, format="png"):
        """
        Export a plotly figure as an image
        
        Args:
            fig: Plotly figure to export
            format: Image format (png, jpg, svg, pdf)
            
        Returns:
            str: Base64 encoded image data
        """
        buffer = BytesIO()
        fig.write_image(buffer, format=format)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        return b64
    
    @staticmethod
    def create_outlier_visualization(df, outlier_mask, column):
        """
        Create visualization highlighting outliers
        
        Args:
            df: DataFrame containing the data
            outlier_mask: Boolean mask indicating outliers
            column: Column to visualize
            
        Returns:
            plotly.graph_objects.Figure: The created visualization
        """
        # Create a copy of the dataframe with outlier flag
        plot_df = df.copy()
        plot_df['is_outlier'] = outlier_mask[column]
        
        # Create histogram with outliers highlighted
        fig = px.histogram(plot_df, x=column, color='is_outlier',
                          color_discrete_map={True: 'red', False: 'blue'},
                          title=f"Distribution of {column} with Outliers Highlighted")
        
        # Apply common layout settings
        fig.update_layout(
            height=500,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
