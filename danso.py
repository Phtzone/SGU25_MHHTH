import requests
import pandas as pd
from pathlib import Path

url = "https://api.worldbank.org/v2/country/VN/indicator/SP.POP.TOTL"
payload = {"format": "json", "per_page": 70}
data = requests.get(url, params=payload, timeout=30).json()[1]
pop = (
    pd.DataFrame(data)
      .loc[lambda d: d['date'].between('2005', '2024')]
      .assign(
          year=lambda d: d['date'].astype(int),
          population=lambda d: d['value'].astype(int)
      )
      .sort_values('year')
      .reset_index(drop=True)
)

base_dir = Path(__file__).parent 
code_dir = base_dir / "population"
code_dir.mkdir(parents=True, exist_ok=True)

csv_path = code_dir / "vietnam_population_2005_2024.csv"
pop[['year', 'population']].to_csv(csv_path, index=False)
print(f"CSV đã được lưu tại: {csv_path}")
