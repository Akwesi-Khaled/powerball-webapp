import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import random
import requests
import io

"""
Updated powerball_auto_analysis.py

Features:
- Robust detection of winning-numbers column (handles several common schemas).
- Supports datasets that already have white1..white5 + powerball columns.
- Builds observed and expected arrays so scipy.stats.chisquare won't error.
- analyze() returns a results dict and a matplotlib Figure (so Streamlit or other frontends can display them).
- Clear error messages and safe fallbacks.
"""

def fetch_powerball_data(ny_fallback=True):
    """
    Fetch Powerball historical data.
    Primary source: New York open data CSV.
    If that fails and ny_fallback=True, raises the original exception for inspection.
    Returns a pandas.DataFrame (raw).
    """
    urls_tried = []
    # Primary: NY Open Data (commonly available)
    url_ny = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
    urls_tried.append(url_ny)
    try:
        resp = requests.get(url_ny, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return df
    except Exception as e:
        # surface a helpful error with attempted URLs
        raise RuntimeError(f"Failed to fetch NY dataset from {url_ny}. Error: {e}")

def _detect_and_split_winning_numbers(df):
    """
    Detect how the dataset encodes the winning numbers and return a DataFrame
    that includes integer columns: white1..white5 and powerball.
    Accepted formats:
      1) One column containing all numbers in a string, e.g. "04 08 32 40 48 20"
         (commonly named 'winning_numbers' or 'winning numbers' or similar)
      2) Separate columns already present: white1, white2, white3, white4, white5, powerball
      3) Some datasets have 'winning numbers' with commas, e.g. '04,08,32,40,48 20'
    """
    df_copy = df.copy()
    # normalize columns for detection
    cols_lower = [c.lower() for c in df_copy.columns]
    col_map = {c.lower(): c for c in df_copy.columns}

    # Case A: already have white1..white5 and powerball (common cleaned formats)
    required = ["white1","white2","white3","white4","white5","powerball"]
    if all(name in cols_lower for name in required):
        # map back to original-case column names
        mapped = {name: col_map[name] for name in col_map if name in required}
        # return with standardized lowercase column names
        df_copy = df_copy.rename(columns={mapped[name]: name for name in mapped})
        for name in required:
            df_copy[name] = df_copy[name].astype(int)
        return df_copy

    # Case B: single column containing all numbers
    # Common candidate names
    single_col_candidates = ["winning_numbers", "winning numbers", "winningnumbers", "winning number", "winning_number", "numbers", "winning_numbers_with_powerball", "drawn_numbers"]
    match_col = None
    for candidate in single_col_candidates:
        if candidate in cols_lower:
            match_col = col_map[candidate]
            break

    # If not found by candidate list, try to heuristically find a column that looks like numbers
    if match_col is None:
        # look for a column whose values look like "## ## ## ## ## ##" pattern
        for orig_col in df_copy.columns:
            sample = df_copy[orig_col].dropna().astype(str).head(20).tolist()
            # count how many rows look like they contain 6 numbers (space or comma separated)
            looks_like = 0
            for s in sample:
                # remove extra parentheses or text
                s_clean = s.strip()
                # replace commas with spaces for checking
                s2 = s_clean.replace(',', ' ')
                parts = [p for p in s2.split() if p.strip()!='']
                if len(parts) >= 6 and all(part.strip().lstrip('0').isdigit() for part in parts[:6]):
                    looks_like += 1
            if looks_like >= max(3, len(sample)//4):
                match_col = orig_col
                break

    if match_col is None:
        # give a helpful error listing available columns
        raise ValueError(f"Could not find a column containing the winning numbers. Available columns: {list(df_copy.columns)}")

    # Now we have a match_col: split it into 6 parts
    # Normalize separators: commas or multiple spaces
    s = df_copy[match_col].astype(str).str.replace(',', ' ', regex=False).str.replace(r'\s+', ' ', regex=True).str.strip()
    parts = s.str.split(' ', expand=True)
    if parts.shape[1] < 6:
        # some datasets may combine last two numbers with a dash or other separator; try a looser split
        parts = s.str.replace('-', ' ').str.split(' ', expand=True)
    if parts.shape[1] < 6:
        raise ValueError(f"Could not split '{match_col}' into 6 number columns. Sample values: {df_copy[match_col].astype(str).head(5).tolist()}")

    # take first 6 columns as white1..white5 and powerball
    parts = parts.iloc[:, :6].copy()
    parts.columns = ['white1','white2','white3','white4','white5','powerball']
    # convert to int (strip leading zeros)
    for col in parts.columns:
        parts[col] = parts[col].astype(str).str.lstrip('0').replace('', '0')
        parts[col] = parts[col].astype(int)

    # attach these standardized columns to df_copy (drop original match_col to avoid confusion)
    df_copy = df_copy.reset_index(drop=True).copy()
    df_copy[['white1','white2','white3','white4','white5','powerball']] = parts[['white1','white2','white3','white4','white5','powerball']]

    return df_copy

def preprocess(df):
    """
    Returns a DataFrame that contains at least:
      - white1..white5 (int)
      - powerball (int)
    Keeps original columns as well.
    """
    try:
        return _detect_and_split_winning_numbers(df)
    except Exception as e:
        raise

def analyze(df):
    """
    Run analysis on a preprocessed DataFrame (must have white1..white5 and powerball).
    Returns:
      - results: dict containing frequencies, hot/cold lists, chi-square outputs, weighted picks
      - fig: matplotlib Figure containing the white-ball frequency bar chart
    """
    # Combine white numbers into a single Series
    whites = pd.concat([df['white1'], df['white2'], df['white3'], df['white4'], df['white5']], ignore_index=True)
    # Ensure values are in expected ranges
    whites = whites.astype(int)
    reds = df['powerball'].astype(int)

    # Frequency counts
    white_freq = whites.value_counts().reindex(range(1,70), fill_value=0).sort_index()
    red_freq = reds.value_counts().reindex(range(1,27), fill_value=0).sort_index()

    # Prepare observed arrays
    observed_white = white_freq.values.astype(float)
    observed_red = red_freq.values.astype(float)

    # Prepare expected arrays so sums match exactly observed sums
    expected_white = np.ones_like(observed_white) * (observed_white.sum() / observed_white.size)
    expected_red = np.ones_like(observed_red) * (observed_red.sum() / observed_red.size)

    # Run chi-square tests
    chi_white_stat, chi_white_p = chisquare(f_obs=observed_white, f_exp=expected_white)
    chi_red_stat, chi_red_p = chisquare(f_obs=observed_red, f_exp=expected_red)

    # Hot / Cold
    hot_whites = list(np.argsort(-observed_white)[:5] + 1)  # +1 because indices start at 0
    cold_whites = list(np.argsort(observed_white)[:5] + 1)
    hot_reds = list(np.argsort(-observed_red)[:3] + 1)
    cold_reds = list(np.argsort(observed_red)[:3] + 1)

    # Weighted picks (frequency-based weighting + 1 to avoid zero weight)
    white_pool = list(range(1,70))
    red_pool = list(range(1,27))
    white_weights = (observed_white + 1).tolist()
    red_weights = (observed_red + 1).tolist()

    weighted_tickets = []
    for _ in range(10):
        pick = sorted(random.choices(white_pool, weights=white_weights, k=5))
        pb = random.choices(red_pool, weights=red_weights, k=1)[0]
        weighted_tickets.append({'whites': pick, 'powerball': int(pb)})

    # Build bar chart figure
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(range(1,70), observed_white)
    ax.set_title("White Ball Frequencies (1-69)")
    ax.set_xlabel("White Ball Number")
    ax.set_ylabel("Occurrences")
    plt.tight_layout()

    results = {
        'white_freq_series': white_freq,
        'red_freq_series': red_freq,
        'chi_square_white': {'statistic': float(chi_white_stat), 'pvalue': float(chi_white_p)},
        'chi_square_red': {'statistic': float(chi_red_stat), 'pvalue': float(chi_red_p)},
        'hot_whites': hot_whites,
        'cold_whites': cold_whites,
        'hot_reds': hot_reds,
        'cold_reds': cold_reds,
        'weighted_tickets': weighted_tickets
    }

    return results, fig

def generate_weighted_picks_from_df(df, n=5):
    """
    Convenience wrapper: preprocess (if needed) then analyze and return just picks.
    """
    df2 = preprocess(df)
    results, fig = analyze(df2)
    return results['weighted_tickets']

if __name__ == "__main__":
    # quick local test: fetch and run analysis, print summary
    df_raw = fetch_powerball_data()
    df = preprocess(df_raw)
    results, fig = analyze(df)
    print("Chi-square (white):", results['chi_square_white'])
    print("Chi-square (red):", results['chi_square_red'])
    print("Hot whites:", results['hot_whites'])
    print("Hot reds:", results['hot_reds'])
    print("Sample weighted tickets:")
    for t in results['weighted_tickets'][:5]:
        print(t)
    # save figure to file for quick local check
    fig.savefig("white_freq_preview.png")
