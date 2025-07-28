from pipeline import Pipeline
import os 
import random

def main():
    directoryPath = "C:/VSCode/Titanic-ML-Pipeline/datasets/"
    csvList = [f for f in os.listdir(directoryPath) if f.endswith('.csv')]
    
    # randomly select csv from the directory
    randomCSV = random.choice(csvList)
    
    # test setosa(0) versicolor(1) virginica(2) 
    randomCSV = "iris(in).csv"
    
    targetPath = os.path.join(directoryPath, randomCSV)
    
    pipeline = Pipeline(targetPath)
    pipeline.run()    
    
main()