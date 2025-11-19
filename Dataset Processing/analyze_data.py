import pandas as pd
import numpy as np
from datetime import datetime

# Load raw data
raw = pd.read_csv('dataset_dengue_2005_2024.csv')

print('='*100)
print('RAW DATA ANALYSIS - DENGUE DATASET 2005-2024')
print('='*100)
print(f'Shape: {raw.shape}')
print(f'Columns: {list(raw.columns)}')

print(f'\n{"-"*100}')
print('T_RES (TEMPORAL RESOLUTION) BREAKDOWN')
print(f'{"-"*100}')
print(raw['T_res'].value_counts())

print(f'\n{"-"*100}')
print('DETAILED BREAKDOWN BY TEMPORAL RESOLUTION')
print(f'{"-"*100}')

for tres in sorted(raw['T_res'].unique()):
    subset = raw[raw['T_res'].str.lower() == tres.lower()]
    print(f'\n{tres.upper()}:')
    print(f'  Records: {len(subset):,}')
    print(f'  Year range: {int(subset["Year"].min())}-{int(subset["Year"].max())}')
    print(f'  Unique provinces: {subset["adm_1_name"].nunique()}')
    print(f'  Cases total: {subset["dengue_total"].sum():,.0f}')
    print(f'  Cases min/max/mean: {subset["dengue_total"].min():.0f} / {subset["dengue_total"].max():.0f} / {subset["dengue_total"].mean():.1f}')
    
    # Breakdown by year
    yearly = subset.groupby('Year')['dengue_total'].agg(['sum', 'count', 'mean'])
    print(f'  Year distribution:')
    for year in sorted(subset['Year'].unique()):
        year_data = subset[subset['Year'] == year]
        print(f'    {int(year)}: {len(year_data):4d} records, {year_data["dengue_total"].sum():9,.0f} cases')

print(f'\n{"-"*100}')
print('OVERALL STATISTICS')
print(f'{"-"*100}')
print(f'Total cases: {raw["dengue_total"].sum():,.0f}')
print(f'Mean cases per record: {raw["dengue_total"].mean():.1f}')
print(f'Median cases per record: {raw["dengue_total"].median():.1f}')
print(f'Null values: {raw.isnull().sum().sum()} total')
print(f'Unique admin 1 regions: {raw["adm_1_name"].nunique()}')

print(f'\n{"-"*100}')
print('LOAD PROCESSED DATA (if exists)')
print(f'{"-"*100}')

try:
    merged = pd.read_csv('dengue_month_complete_2005_2024.csv')
    print(f'Merged data shape: {merged.shape}')
    print(f'Date range: {merged["YearMonth"].min()} to {merged["YearMonth"].max()}')
    print(f'Total cases in merged: {merged["dengue_total_month"].sum():,.0f}')
    
    national_merged = merged[merged['adm_1_name'] == 'National']
    print(f'National level records: {len(national_merged)}')
    print(f'Completeness: {len(national_merged)/240*100:.1f}% (expected 240 months)')
    
    print(f'\nPeriod statistics:')
    periods = {
        '2005-2019 (Pre-COVID)': (national_merged['Year'] < 2020),
        '2020-2022 (COVID)': ((national_merged['Year'] >= 2020) & (national_merged['Year'] <= 2022)),
        '2023-2024 (Post-COVID)': (national_merged['Year'] >= 2023)
    }
    
    for period_name, mask in periods.items():
        period_data = national_merged[mask]
        if len(period_data) > 0:
            print(f'  {period_name}: {len(period_data)} months, {period_data["dengue_total_month"].sum():,.0f} cases (avg: {period_data["dengue_total_month"].mean():.0f}/month)')
            
except FileNotFoundError:
    print('Merged data file not found')

print(f'\n{"-"*100}')
print('POTENTIAL ISSUES TO CHECK')
print(f'{"-"*100}')

# Check for zeros
zero_count = (raw['dengue_total'] == 0).sum()
print(f'Records with zero cases: {zero_count:,} ({zero_count/len(raw)*100:.1f}%)')

# Check for nulls by column
print(f'\nNull values by column:')
null_counts = raw.isnull().sum()
for col, count in null_counts[null_counts > 0].items():
    print(f'  {col}: {count}')

# Check date parsing
print(f'\nDate column analysis:')
raw['calendar_start_date'] = pd.to_datetime(raw['calendar_start_date'], errors='coerce')
date_nulls = raw['calendar_start_date'].isnull().sum()
print(f'  Valid dates: {len(raw) - date_nulls:,}')
print(f'  Invalid dates: {date_nulls}')

print('\n' + '='*100)
