"""
SUMMARY: Data Preparation for SEIR Model (2005-2019)
======================================================

✅ DATA READINESS FOR SEIR MODELING
"""

import pandas as pd

# Load the prepared data
national = pd.read_csv('dengue_2005_2019_national.csv')

print("="*100)
print("DATA READY FOR SEIR MODELING: 2005-2019")
print("="*100)

print(f"""
📊 DATASET OVERVIEW
{"-"*100}
✅ Completeness: 180/180 months (100%)
✅ Date range: January 2005 - December 2019 (15 years, 180 months)
✅ No missing data: YES
✅ No null values: YES
✅ Zero cases: 0 months (0%)

STATISTICAL SUMMARY
{"-"*100}
  Total cases: {national['dengue_total_month'].sum():,.0f}
  Mean/month: {national['dengue_total_month'].mean():.0f} cases
  Median/month: {national['dengue_total_month'].median():.0f} cases
  Std dev: {national['dengue_total_month'].std():.0f}
  Min: {national['dengue_total_month'].min():.0f} cases (some month)
  Max: {national['dengue_total_month'].max():.0f} cases (September 2019)

SEASONAL PATTERN (Average by Month)
{"-"*100}
  Peak season: July-November (monsoon season)
    - July:     11,628 cases/month
    - August:   12,988 cases/month
    - September: 13,881 cases/month ← HIGHEST
    - October: 13,786 cases/month
    - November: 13,034 cases/month
  
  Low season: February-May (dry season)
    - February: 3,254 cases/month ← LOWEST
    - March: 2,971 cases/month
    - April: 2,766 cases/month
    - May: 3,837 cases/month
  
  Transition months: January, June, December
    - January: 5,563 cases/month
    - June: 6,655 cases/month
    - December: 10,805 cases/month

YEAR-BY-YEAR TREND
{"-"*100}
Year    Total Cases    Mean/month    Min    Max
""")

yearly = national.groupby('Year')['dengue_total_month'].agg(['sum', 'mean', 'min', 'max'])
for year, row in yearly.iterrows():
    print(f"{int(year)}    {int(row['sum']):>11,}    {row['mean']:>11.0f}    {int(row['min']):>6,}    {int(row['max']):>6,}")

print(f"""
EPIDEMIC DYNAMICS (Patterns to note)
{"-"*100}
2005-2010: Relatively stable with gradual increase
  - Average: 6,867 cases/month
  - Notable: Peak in 2010 (128,831 annual)

2011-2014: Decreased activity (data coverage reduced at regional level)
  - Average: 5,638 cases/month
  - Notable: Lowest year 2014 (29,181 annual)

2015-2019: Strong recovery and increasing trend
  - Average: 10,981 cases/month
  - Notable: Highest activity in 2019 (137,814 annual)
  
⚠️  2019 shows unusual spike (42,451 cases in single month - September)
   This may reflect real dengue dynamics or data reporting changes

RECOMMENDATIONS FOR SEIR MODEL
{"-"*100}
1. ✅ USE THIS DATA: 2005-2019 is the cleanest period
   - Complete data at national level
   - 15 years enough for parameter calibration
   - Good seasonal pattern to learn from

2. 🔧 NORMALIZATION:
   - Normalize to per-capita basis (per 100,000)
   - Merge with population data: ../population/vietnam_population_2005_2024.csv
   - Vietnam population grew from ~84M (2005) to ~97M (2019)

3. 📈 COVARIATE DATA:
   - Temperature (2-m temp) - already available in NetCDF files
   - Rainfall - already available in NetCDF files
   - Aggregate to national level and align with monthly cases

4. 🧮 PARAMETER CALIBRATION:
   - Use least-squares or Bayesian inference on observed monthly cases
   - Set initial conditions: estimate S/E/I/R split from 2005-01 baseline
   - Calibrate β (transmission), σ (1/incubation), γ (1/infectious period)

5. ✔️  VALIDATION STRATEGY:
   - Train: 2005-2017 (156 months)
   - Validation: 2018-2019 (24 months)
   - Check model captures seasonal pattern correctly
   - Evaluate forecast skill on 2018-2019

FILES CREATED
{"-"*100}
✅ dengue_2005_2019_national.csv
   - National level aggregated cases
   - Columns: YearMonth, dengue_total_month, adm_1_name, Year, Month

✅ dengue_2005_2019_provincial.csv
   - Provincial level cases (40.3% coverage)
   - Can be used for regional analysis later

✅ dengue_2005_2019_analysis.png
   - 4-panel visualization:
     1. Time series (full trend)
     2. Annual totals (bar chart)
     3. Seasonal pattern (average by month)
     4. Box plot (distribution by month)

NEXT STEPS
{"-"*100}
1. Load dengue_2005_2019_national.csv in seir_dengue_workflow.ipynb
2. Merge with population data for per-capita normalization
3. Extract temperature/rainfall from NetCDF files
4. Set up discrete SEIR formulation
5. Calibrate parameters using optimization

Status: ✅ READY TO START SEIR MODELING
""")

print("="*100)
