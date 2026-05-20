import pandas as pd

def to_dataframe(features: dict):
    df = pd.DataFrame([features])
    return df.apply(pd.to_numeric, errors="ignore")

