import pandas as pd
import numpy as np

print('='*100)
print('DETAILED DATA QUALITY ASSESSMENT')
print('='*100)

# Load data
raw = pd.read_csv('dataset_dengue_2005_2024.csv')
merged = pd.read_csv('dengue_month_complete_2005_2024.csv')

print('\n1. DISCREPANCIES BETWEEN RAW AND MERGED DATA')
print('-'*100)

# Month data
month_raw = raw[raw['T_res'] == 'Month']
print(f'\nMonth records (Raw):')
print(f'  Total: {len(month_raw):,}')
print(f'  Total cases: {month_raw["dengue_total"].sum():,.0f}')
print(f'  Records per year:')
for year in sorted(month_raw['Year'].unique()):
    year_data = month_raw[month_raw['Year'] == year]
    print(f'    {int(year)}: {len(year_data):4d} records ({len(year_data)/64:.1f} provinces)')

# Week data
week_raw = raw[raw['T_res'] == 'Week']
print(f'\nWeek records (Raw):')
print(f'  Total: {len(week_raw):,}')
print(f'  Total cases: {week_raw["dengue_total"].sum():,.0f}')
print(f'  Years: {sorted(week_raw["Year"].unique())}')

# Year data
year_raw = raw[raw['T_res'] == 'Year']
print(f'\nYear records (Raw):')
print(f'  Total: {len(year_raw):,}')
print(f'  Total cases: {year_raw["dengue_total"].sum():,.0f}')
print(f'  Years covered: {sorted(year_raw["Year"].unique())}')

print(f'\n{"-"*100}')
print('\n2. DATA COVERAGE ISSUES')
print('-'*100)

# Check month data gaps
print(f'\nMonth data gaps by year:')
for year in sorted(month_raw['Year'].unique()):
    year_data = month_raw[month_raw['Year'] == year]
    expected_records = 64 * 12  # 64 provinces * 12 months
    actual_records = len(year_data)
    coverage = actual_records / expected_records * 100
    print(f'  {int(year)}: {actual_records:4d} records ({coverage:5.1f}% of expected {expected_records})')

print(f'\n{"-"*100}')
print('\n3. ZERO CASES ANALYSIS')
print('-'*100)

zero_month = (month_raw['dengue_total'] == 0).sum()
print(f'\nMonth records with zero cases: {zero_month:,} ({zero_month/len(month_raw)*100:.1f}%)')

zero_week = (week_raw['dengue_total'] == 0).sum()
print(f'Week records with zero cases: {zero_week:,} ({zero_week/len(week_raw)*100:.1f}%)')

print(f'\n{"-"*100}')
print('\n4. MERGED DATA ANALYSIS')
print('-'*100)

# National level analysis
national = merged[merged['adm_1_name'] == 'National'].sort_values('YearMonth')
print(f'\nNational level:')
print(f'  Total records: {len(national)}')
print(f'  Date range: {national["YearMonth"].min()} to {national["YearMonth"].max()}')
print(f'  Expected: 240 months (2005-01 to 2024-12)')
print(f'  Completeness: {len(national)/240*100:.1f}%')

# Check missing months
expected_months = pd.date_range('2005-01', '2024-12', freq='MS')
actual_months = set(pd.to_datetime(national['YearMonth']).dt.to_period('M'))
expected_periods = set([d.to_period('M') for d in expected_months])
missing = expected_periods - actual_months

if missing:
    print(f'  Missing {len(missing)} months:')
    for period in sorted(missing):
        print(f'    {period}')

# Regional breakdown
print(f'\nRegional level (provinces):')
provinces = merged[merged['adm_1_name'] != 'National']
print(f'  Total provinces: {provinces["adm_1_name"].nunique()}')
print(f'  Total records: {len(provinces):,}')

prov_summary = provinces.groupby('adm_1_name').agg({
    'YearMonth': ['min', 'max', 'count'],
    'dengue_total_month': 'sum'
}).round(0)
prov_summary.columns = ['Start', 'End', 'Count', 'Total_Cases']

print(f'\n  Province coverage (sorted by record count):')
prov_summary_sorted = prov_summary.sort_values('Count', ascending=False)
for idx, (prov, row) in enumerate(prov_summary_sorted.iterrows()):
    if idx < 10:  # Show top 10
        print(f'    {prov:25s}: {int(row["Count"]):4d} records, {row["Total_Cases"]:9,.0f} cases')

print(f'\n{"-"*100}')
print('\n5. DATA CONSISTENCY CHECK')
print('-'*100)

# Compare 2016 Year vs Week records
year_2016 = year_raw[year_raw['Year'] == 2016]
week_2016 = week_raw[week_raw['Year'] == 2016]

print(f'\nYear 2016 comparison:')
if len(year_2016) > 0:
    year_2016_total = year_2016['dengue_total'].sum()
    print(f'  Year record: {year_2016_total:,.0f} cases')
else:
    print(f'  Year record: None')

if len(week_2016) > 0:
    week_2016_total = week_2016['dengue_total'].sum()
    print(f'  Week records: {week_2016_total:,.0f} cases ({len(week_2016)} weeks)')
    print(f'  Discrepancy: {abs(year_2016_total - week_2016_total):,.0f} cases')
    print(f'  ⚠️  Week data is {week_2016_total - year_2016_total:,.0f} cases HIGHER than Year')

# COVID period (2020-2022) data sources
print(f'\nCOVID Period (2020-2022) analysis:')
merged_2020_2022 = merged[
    (pd.to_datetime(merged['YearMonth']).dt.year >= 2020) &
    (pd.to_datetime(merged['YearMonth']).dt.year <= 2022)
]
print(f'  Merged records: {len(merged_2020_2022):,}')
print(f'  Total cases: {merged_2020_2022["dengue_total_month"].sum():,.0f}')

# Check which years are primarily from Year records
covid_monthly = month_raw[
    (month_raw['Year'] >= 2020) & 
    (month_raw['Year'] <= 2022)
]
covid_year = year_raw[
    (year_raw['Year'] >= 2020) & 
    (year_raw['Year'] <= 2022)
]
print(f'  Month records for 2020-2022: {len(covid_monthly):,} ({covid_monthly["dengue_total"].sum():,.0f} cases)')
print(f'  Year records for 2020-2022: {len(covid_year):,} ({covid_year["dengue_total"].sum():,.0f} cases)')

print('\n' + '='*100)
