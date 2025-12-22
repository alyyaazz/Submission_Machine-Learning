import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess_data(input_path, output_dir):
    df = pd.read_csv(input_path).drop_duplicates()

    # Filter Outliers
    Q1, Q3 = df["bmi"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df["bmi"] >= Q1 - 1.5 * IQR) & (df["bmi"] <= Q3 + 1.5 * IQR)]

    # Split Features & Target
    X = df.drop('charges', axis=1)
    y = df['charges'].map({'yes': 1, 'no': 0})

    num_cols = ['age', 'bmi', 'children']
    cat_cols = ['sex', 'region']

    # Pipeline Automation
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop="first", handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    # Execution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Export
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    joblib.dump(preprocessor, f"{output_dir}/preprocess.pkl")
    pd.DataFrame(X_train_proc).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test_proc).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    return X_train_proc, y_train

if __name__ == "__main__":
    preprocess_data("insurance.csv", "./insurance_preprocessing")