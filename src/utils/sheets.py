import pandas as pd

def fetch_public_csv(url: str) -> pd.DataFrame:
    """
    Fetch a Google Sheet that has been published as a CSV.
    In Sheets: File → Share → Publish to web → Link → pick the tab → CSV.
    Example URL (must end with `/pub?output=csv`):
    https://docs.google.com/spreadsheets/d/e/2PACX-1vTJEHDNcBhXWv_acdsuEz_Ch72zBGI2ixmThH6Tq1mqleqx02DUHHg6DD5M2S4hVAklHJXj12FoTIQl/pub?output=csv
    """
    return pd.read_csv(url)
