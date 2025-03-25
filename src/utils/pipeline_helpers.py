
import pandas as pd

def filter_by_age(df, max_age):
    """
    Filter a dataframe by age, handling both string and numeric age formats.

    Args:
        df: DataFrame to filter
        max_age: Maximum age to include

    Returns:
        Filtered DataFrame
    """
    if max_age is None or "Age" not in df.columns:
        return df

    filtered_df = df.copy()

    if not pd.api.types.is_numeric_dtype(filtered_df["Age"]):
        # Extract main age number before the dash
        filtered_df["Age_numeric"] = filtered_df["Age"].str.split("-").str[0].astype(int)
        return filtered_df[filtered_df["Age_numeric"] <= max_age].copy()
    else:
        return filtered_df[filtered_df["Age"] <= max_age].copy()
