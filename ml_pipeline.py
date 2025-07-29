import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class ML_Pipeline:
    def __init__(self, X_train, X_test, y_train, y_test, cleanedDF, targetY):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cleanedDF = cleanedDF
        self.targetY = targetY
        
        self.models = []

    def add_model(self, name, model):
        self.models.append((name, model))

    def train_and_evaluate(self):
        results = []

        for name, model in self.models:
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"Evaluating {name}...")
            
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)

            class_count = len(np.unique(self.y_test))

            if class_count == 2:
                auc = roc_auc_score(self.y_test, y_proba[:, 1])
            else:
                y_true_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
                auc = roc_auc_score(y_true_bin, y_proba, multi_class='ovr')

            # Append to results table
            results.append({
                "Model": name,
                "Accuracy": acc,
                "AUC": auc
            })
            
            print(f"Accuracy/AUC saved\n")

        # Convert to DataFrame and store it
        self.results_df = pd.DataFrame(results)
        print("\nModel Evaluation Summary:\n")
        print(self.results_df)
        
    def display_results_table(self):
        if not hasattr(self, 'results_df'):
            print("No results to display. Run training first.")
            return

        df = self.results_df

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(df.columns),
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=[df[col] for col in df.columns],
                        fill_color='lavender',
                        align='left'
                    )
                )
            ]
        )

        fig.update_layout(title="Model Evaluation Summary")
        
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        table_path = os.path.join(output_dir, "model_results_table.html")
        fig.write_html(table_path)
        print(f"[INFO] Results table saved: {table_path}")
            
    def run_pca_projection(self):
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Drop the target column and select only numeric features
        features = self.cleanedDF.drop(columns=self.targetY)
        if not all(features.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))):
            print("All features must be numeric for PCA/t-SNE.")
            return

        labels = self.cleanedDF[self.targetY]

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        pca_df = features.copy()
        pca_df['PC1'] = pca_result[:, 0]
        pca_df['PC2'] = pca_result[:, 1]
        pca_df[self.targetY] = labels

        fig_pca = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color=self.targetY,
            title='2D PCA Projection',
            labels={'color': self.targetY},
            opacity=0.7
        )
        
        # Update color axis to show only integer values
        fig_pca.update_layout(
            coloraxis_colorbar=dict(
                tickmode='array',
                ticktext=sorted(self.cleanedDF[self.targetY].unique()),
                tickvals=sorted(self.cleanedDF[self.targetY].unique())
            )
        )
        
        pca_path = os.path.join(output_dir, "pca_2d_projection.html")
        fig_pca.write_html(pca_path)
        print(f"Saved PCA plot: {pca_path}")

    def run_tsne_projection(self):
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Drop the target column and select only numeric features
        features = self.cleanedDF.drop(columns=self.targetY)
        if not all(features.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))):
            print("All features must be numeric for PCA/t-SNE.")
            return

        labels = self.cleanedDF[self.targetY]
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
        tsne_result = tsne.fit_transform(features)
        tsne_df = features.copy()
        tsne_df['TSNE1'] = tsne_result[:, 0]
        tsne_df['TSNE2'] = tsne_result[:, 1]
        tsne_df[self.targetY] = labels

        fig_tsne = px.scatter(
            tsne_df,
            x='TSNE1',
            y='TSNE2',
            color=self.targetY,
            title='2D t-SNE Projection',
            labels={'color': self.targetY},
            opacity=0.7
        )
        
        # Update color axis to show only integer values
        fig_tsne.update_layout(
            coloraxis_colorbar=dict(
                tickmode='array',
                ticktext=sorted(self.cleanedDF[self.targetY].unique()),
                tickvals=sorted(self.cleanedDF[self.targetY].unique())
            )
        )
        
        tsne_path = os.path.join(output_dir, "tsne_2d_projection.html")
        fig_tsne.write_html(tsne_path)
        print(f"Saved t-SNE plot: {tsne_path}\n")

    def run(self):
        # check for PCA and t-SNE as these output plots rather than accuracy/AUC values
        pca_model = ("PCA Projection", None)
        if pca_model in self.models:
            print(f"\nPCA found")
            self.models.remove(pca_model)
            self.run_pca_projection()
            
        tsne_model = ("t-SNE Clustering", None)
        if tsne_model in self.models:
            print("t-SNE found")
            self.models.remove(tsne_model)
            self.run_tsne_projection()
            
        # train/eval rest of the models
        self.train_and_evaluate()
        self.display_results_table()
        