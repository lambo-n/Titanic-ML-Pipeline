import plotly.express as px
import os



class AnalyzePipeline:
    def __init__(self, X_train, X_test, y_train, y_test, cleanedDF, targetY):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cleanedDF = cleanedDF
        self.targetY = targetY
        

    
    def violin_plot(self):
        if self.targetY not in self.cleanedDF.columns:
            print(f"Target column '{self.targetY}' not found.")
            return

        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)

        for col in self.cleanedDF.columns:
            if col == self.targetY:
                continue
            
            fig = px.violin(
                self.cleanedDF, y=col, x=self.targetY, box=True, points="all",
                title=f"Violin Plot of {col} grouped by {self.targetY}"
            )
            
            file_path = os.path.join(output_dir, f"{col}_violin_plot.html")
            fig.write_html(file_path)
            print(f"Saved: {file_path}")
            
    def histogram(self):
        if self.targetY not in self.cleanedDF.columns:
            print(f"Target column '{self.targetY}' not found.")
            return

        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)

        for col in self.cleanedDF.columns:
            if col == self.targetY:
                continue
            
            fig = px.histogram(
                self.cleanedDF,
                x=col,
                color=self.targetY,         # color by target variable to compare distributions
                barmode='overlay',          # overlay bars for different target classes
                opacity=0.7,
                nbins=30,
                title=f"Histogram of {col} grouped by {self.targetY}"
            )
            
            file_path = os.path.join(output_dir, f"{col}_histogram.html")
            fig.write_html(file_path)
            print(f"Saved: {file_path}")
            
    def scatter_pairs_matrix(self):
        if self.targetY not in self.cleanedDF.columns:
            print(f"Target column '{self.targetY}' not found.")
            return

        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)

        # Select only numeric predictor columns (exclude target)
        predictors = [col for col in self.cleanedDF.columns if col != self.targetY]

        # Create the scatter matrix
        fig = px.scatter_matrix(
            self.cleanedDF,
            dimensions=predictors,
            color=self.targetY,
            title="Scatter Matrix of Predictors Colored by Target",
            height=800,
            width=800
        )
        
        # Update color axis to show only integer values
        fig.update_layout(
            coloraxis_colorbar=dict(
                tickmode='array',
                ticktext=sorted(self.cleanedDF[self.targetY].unique()),
                tickvals=sorted(self.cleanedDF[self.targetY].unique())
            )
        )

        # Save as HTML
        file_path = os.path.join(output_dir, "scatter_matrix.html")
        fig.write_html(file_path)
        print(f"Saved scatter matrix: {file_path}")
        
    def correlation_heatmap(self):
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)

        # Select only numeric columns (excluding the target, if it's non-numeric)
        numeric_df = self.cleanedDF.select_dtypes(include='number')

        if self.targetY in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=self.targetY)

        # Compute correlation matrix
        corr_matrix = numeric_df.corr()

        # Melt the correlation matrix into long-form
        corr_melted = corr_matrix.reset_index().melt(id_vars='index')
        corr_melted.columns = ['Feature X', 'Feature Y', 'Correlation']

        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap of Numeric Features",
            zmin=-1,
            zmax=1
        )

        # Save as HTML
        file_path = os.path.join(output_dir, "correlation_heatmap.html")
        fig.write_html(file_path)
        print(f"Saved: {file_path}")
        
    

    def run(self):
        self.violin_plot()
        self.histogram()
        self.scatter_pairs_matrix()
        self.correlation_heatmap()
