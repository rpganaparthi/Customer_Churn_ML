import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    print(55555555555555, filepath)
    df = pd.read_csv(filepath)
    print(df)
    df.dropna(inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    #cat_cols = df.select_dtypes(include='object').columns
    #df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return train_test_split(X, y, test_size=0.2, random_state=42)
