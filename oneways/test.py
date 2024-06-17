import pandas as pd
from one_way_factor_utils import plot_value_and_counts

if __name__ == "__main__":
    # Get stock dataset from plotly
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
    df['condition'] = df['AAPL.Close'] - df['AAPL.Open']
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    n_bins = 12
    df['condition'] = pd.cut(df['condition'], bins=n_bins)
    df['condition'] = df['condition'].map(lambda x: f"{x.left:.2f}-{x.right:.2f}")
    fig = plot_value_and_counts(df, 'AAPL.Close', 'condition', freq="1M", save_dir=".")