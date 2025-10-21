import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import random
import requests
import io

def fetch_powerball_data():
    url = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
    print(f"Fetching data from {url} ...")
    resp = requests.get(url)
    resp.raise_for_status()
    csv_text = resp.text
    df = pd.read_csv(io.StringIO(csv_text))
    return df

def preprocess(df):
    # Make column names lowercase for easy matching
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Try to automatically detect the right column
    possible_cols = ["winning_numbers", "winning numbers"]
    match_col = None
    for col in df.columns:
        if col in possible_cols:
            match_col = col
            break

    if match_col:
        # Split the column into separate number columns
        df[['white1','white2','white3','white4','white5','powerball']] = (
            df[match_col]
            .astype(str)
            .str.split(' ', expand=True)
            .iloc[:, :6]
            .astype(int)
        )
    else:
        raise ValueError(f"Could not find a winning numbers column. Found columns: {df.columns.tolist()}")

    return df


def analyze(df):
    white_numbers = pd.concat([
        df["white1"], df["white2"], df["white3"],
        df["white4"], df["white5"]
    ])

    white_freq = white_numbers.value_counts().sort_index()
    red_freq = df["powerball"].value_counts().sort_index()

    print("White Ball Frequencies (1–69):")
    print(white_freq)
    print("\nRed Powerball Frequencies (1–26):")
    print(red_freq)

    expected_white = np.ones(69) * (len(white_numbers) / 69)
    chi_white = chisquare(white_freq.reindex(range(1, 70), fill_value=0), expected_white)
    print(f"\nChi‑Square Test (White Balls): {chi_white}")

    expected_red = np.ones(26) * (len(df) / 26)
    chi_red = chisquare(red_freq.reindex(range(1, 27), fill_value=0), expected_red)
    print(f"Chi‑Square Test (Powerballs): {chi_red}")

    hot_whites = white_freq.nlargest(5).index.tolist()
    cold_whites = white_freq.nsmallest(5).index.tolist()
    hot_reds = red_freq.nlargest(3).index.tolist()
    cold_reds = red_freq.nsmallest(3).index.tolist()

    print(f"\n🔥 Hot White Numbers: {hot_whites}")
    print(f"❄️ Cold White Numbers: {cold_whites}")
    print(f"🔥 Hot Powerballs: {hot_reds}")
    print(f"❄️ Cold Powerballs: {cold_reds}")

    white_pool = list(range(1, 70))
    red_pool = list(range(1, 27))
    white_weights = [white_freq.get(i, 0) + 1 for i in white_pool]
    red_weights = [red_freq.get(i, 0) + 1 for i in red_pool]

    print("\n🎲 Weighted Random Picks (for fun only):")
    for i in range(5):
        white_picks = sorted(random.choices(white_pool, weights=white_weights, k=5))
        red_pick = random.choices(red_pool, weights=red_weights, k=1)[0]
        print(f"Ticket {i+1}: {white_picks} + Powerball {red_pick}")

    plt.figure(figsize=(10, 4))
    plt.bar(white_freq.index, white_freq.values)
    plt.title("White Ball Frequencies")
    plt.xlabel("Number")
    plt.ylabel("Occurrences")
    plt.show()

if __name__ == "__main__":
    df_raw = fetch_powerball_data()
    df = preprocess(df_raw)
    analyze(df)
