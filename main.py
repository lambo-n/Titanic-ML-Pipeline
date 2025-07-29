from data_pipeline import DataPipeline
from analyze_pipeline import AnalyzePipeline
from ml_pipeline import ML_Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import os 
import random

#setosa(0) versicolor(1) virginica(2) 

def main():
    # DATA PARAMETERS
    directoryPath = "C:/VSCode/Titanic-ML-Pipeline/datasets/"
    csvList = [f for f in os.listdir(directoryPath) if f.endswith('.csv')]
    #randomCSV = random.choice(csvList)
    randomCSV = "Life Expectancy Data(in).csv"
    
    # ML PARAMETERS
    dt_max_depth = 2                # Decision tree
    forest_num_estimators = 100     # Random Forest
    num_neighbors = 5               # Random forest
    kn_hidden_layer_sizes = (10,)   # KNeighbors
    nn_max_iter = 200               # MLP (NN)
    boosting_num_estimators = 100   # Random Forest
    svm_probability = True          # SVM
    
    remove_files_in_directory("plots")
    remove_files_in_directory("outputs")
    
    # cleans and splits data
    targetPath = os.path.join(directoryPath, randomCSV)
    data_pipeline = DataPipeline(targetPath)
    X_train, X_test, y_train, y_test, cleanedDF, targetY = data_pipeline.run()    
    
    # analyze data and save graphs/visualizations
    analyze_pipeline = AnalyzePipeline(X_train, X_test, y_train, y_test, cleanedDF, targetY)
    analyze_pipeline.run()
    
    # test and evaluate with ML models
    ml_pipeline = ML_Pipeline(X_train, X_test, y_train, y_test, cleanedDF, targetY)

    ml_pipeline.add_model("Decision Tree", DecisionTreeClassifier(max_depth=dt_max_depth))
    ml_pipeline.add_model("Random Forest", RandomForestClassifier(n_estimators=forest_num_estimators))
    ml_pipeline.add_model("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=num_neighbors))
    ml_pipeline.add_model("Neural Network", MLPClassifier(hidden_layer_sizes=kn_hidden_layer_sizes, max_iter=nn_max_iter))
    ml_pipeline.add_model("Gradient Boosting", GradientBoostingClassifier(n_estimators=boosting_num_estimators))
    ml_pipeline.add_model("Support Vector Machine", SVC(probability=svm_probability))  # Needed for AUC
    ml_pipeline.add_model("PCA Projection", None)
    ml_pipeline.add_model("t-SNE Clustering", None)
    ml_pipeline.run()
    
    
def remove_files_in_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    for item_name in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item_name)
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                print(f"Removed file: {item_path}")
            except OSError as e:
                print(f"Error removing file {item_path}: {e}")
                
main()