import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# order:
# 1. Load data
# 2. Drop unnecessary columns
# 3. Encode categorical variables
# 4. Handle missing values

class Pipeline:
    def __init__(self, csv):
        self.csv = csv
        self.targetY = None
        self.unprocessedDF = None
        self.cleanedDF = None

    def load_data(self):
        import pandas as pd
        # create raw dataframe, set targetY to last column
        self.unprocessedDF = pd.read_csv(self.csv)
        self.targetY = self.unprocessedDF.columns[-1]
        
        # create copy of dataframe for cleaning without target column 
        self.cleanedDF = self.unprocessedDF.copy().drop(columns=self.targetY)


        
    def drop_unnecessary_columns(self):
        # check missing values percentage for all columns
        missingPercentage = self.cleanedDF.isnull().mean() * 100
        highMissingValuesColumns = missingPercentage[missingPercentage > 50].index.tolist()
        
        # analyze numerical columns
        numericalColumns = self.cleanedDF.select_dtypes(include=['int64', 'float64']).columns
        numLowVarianceColumns = []
        for col in numericalColumns:
            if self.cleanedDF[col].std() < 0.05 and col != self.targetY: # low variance threshold of 0.05
                numLowVarianceColumns.append(col)
        
        # analyze categorical columns
        categoricalColumns = self.cleanedDF.select_dtypes(include=['object']).columns
        catLowVarianceColumns = []
        for col in categoricalColumns:
            valueCounts = self.cleanedDF[col].value_counts(normalize=True)
            
            # column has low variance if category appears <5% or >95% of the time
            rare_categories = (valueCounts < 0.05).sum()
            if rare_categories > 0 and col != self.targetY:
                catLowVarianceColumns.append(col)

        print(f"Columns with >50% missing values: {highMissingValuesColumns}")
        print(f"Numerical columns with low variance: {numLowVarianceColumns}")
        print(f"Categorical columns with low variance: {catLowVarianceColumns}")
        
        # drop all suggested columns
        columnsToDrop = highMissingValuesColumns + numLowVarianceColumns + catLowVarianceColumns
        self.cleanedDF.drop(columns=columnsToDrop, inplace=True, errors='ignore')
        print(f"\nColumns after drop: {self.cleanedDF.columns.tolist()}")
        
    
    # assign categorical variables to numerical values
    def encode_categorical_variables(self):
        encoder = OneHotEncoder(sparse_output=False) 
    
        # identify categorical columns from the cleaned dataframe
        categoricalColumns = self.cleanedDF.select_dtypes(include=['object']).columns.tolist()
    
        if categoricalColumns:
            print(f"\nCategorical columns found: {categoricalColumns}")

            # fit and transform categorical columns
            numEncodedData = encoder.fit_transform(self.cleanedDF[categoricalColumns])
            featureNames = encoder.get_feature_names_out(categoricalColumns)
            

            # create encoded dataframe
            tempEncodedDF = pd.DataFrame(
                numEncodedData,
                columns=featureNames,
                index=self.cleanedDF.index
            )

            # Remove original categorical columns and add encoded ones
            numericDF = self.cleanedDF.select_dtypes(exclude=['object'])
            self.cleanedDF = pd.concat([numericDF, tempEncodedDF], axis=1)
            
        print(f"\nColumns after encoding: {self.cleanedDF.columns.tolist()}")
    
    # handle null values in dataframe
    def handle_missing_values(self):
        # get columns with missing values
        columnsMissingValues = self.cleanedDF.columns[self.cleanedDF.isnull().any()].tolist()
        
        if columnsMissingValues:
            print(f"\nColumns with missing values: {columnsMissingValues}")
            
            for column in columnsMissingValues:
                missCount = self.cleanedDF[column].isnull().sum()
                missPercentage = (missCount / len(self.cleanedDF)) * 100
                
                # Calculate mean
                columnMean = self.cleanedDF[column].mean()
                
                # fill na (null) values with mean
                self.cleanedDF = self.cleanedDF.fillna({column: columnMean})
                
                print(f"{column}: {missCount} values ({missPercentage:.1f}%) filled with mean = {columnMean:.2f}")
        else:
            print("\nNo missing values found in the dataset")
        
        # Verify no missing values remain
        if self.cleanedDF.isnull().values.any():
            raise ValueError("Missing values still exist after imputation!")
        
    
    def split_data(self):
        # feature variables done cleaning, add target variable back
        self.cleanedDF = pd.concat([self.cleanedDF, self.unprocessedDF[self.targetY]], axis=1)
        
        # split dataframe into features x and target y
        x = self.cleanedDF.drop(columns=self.targetY)
        y = self.cleanedDF[self.targetY]

        # train test split with 80% training and 20% testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        print("\nSplit:")
        print(f"X_train shape: {self.X_train.shape} (rows, columns)")
        print(f"X_test shape: {self.X_test.shape} (rows, columns)")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        print(f"\nFeature columns: {x.columns.tolist()}")
        print(f"Target column: {self.targetY}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def run(self):
        self.load_data()
        self.drop_unnecessary_columns()
        self.encode_categorical_variables()
        self.handle_missing_values()
        X_train, X_test, y_train, y_test = self.split_data()