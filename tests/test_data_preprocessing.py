import pandas as pd

from src.data_preprocessing import clean_data


def test_clean_data_converts_totalcharges_and_fills_missing():

    df = pd.DataFrame({
        "TotalCharges": ["100.5", "200.0", " "]
    })

    cleaned_df = clean_data(df)

    assert cleaned_df["TotalCharges"].dtype != "object"
    assert cleaned_df["TotalCharges"].isna().sum() == 0
    assert cleaned_df["TotalCharges"].iloc[2] == 0