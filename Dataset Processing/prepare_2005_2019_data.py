import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print('='*100)
print('PREPARING DATA FOR SEIR MODEL: 2005-2019')
print('Processing directly from raw dataset_dengue_2005_2024.csv')
print('='*100)

# Load raw data
script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
raw_file = script_dir / 'dataset_dengue_2005_2024.csv'

if not raw_file.exists():
    raise FileNotFoundError(f"Raw data file not found: {raw_file}")

print(f'\n[INFO] Loading raw data from: {raw_file}')
df = pd.read_csv(raw_file)

# Clean and prepare
df['calendar_start_date'] = pd.to_datetime(df['calendar_start_date'], errors='coerce')
df['T_res'] = df['T_res'].str.strip().str.lower()

# Filter invalid dates
invalid_dates = df['calendar_start_date'].isna().sum()
if invalid_dates > 0:
    print(f'[WARNING] Found {invalid_dates} records with invalid dates - filtering out')
    df = df[df['calendar_start_date'].notna()].copy()

# Handle NaN adm_1_name: Check if it's actually National based on S_res
# If S_res is Admin0, it's likely National; otherwise, it might be missing data
if df['adm_1_name'].isna().any():
    na_count = df['adm_1_name'].isna().sum()
    # Check S_res for records with NaN adm_1_name
    na_records = df[df['adm_1_name'].isna()]
    if 'S_res' in df.columns:
        admin0_count = (na_records['S_res'] == 'Admin0').sum()
        print(f'[INFO] Found {na_count} records with NaN adm_1_name')
        print(f'  {admin0_count} have S_res=Admin0 (likely National)')
        # Fill NaN with 'National' if S_res is Admin0, otherwise keep as is for now
        df.loc[df['adm_1_name'].isna() & (df['S_res'] == 'Admin0'), 'adm_1_name'] = 'National'
    else:
        # If no S_res column, assume NaN = National (original behavior)
        df['adm_1_name'] = df['adm_1_name'].fillna('National')
        print(f'[INFO] Filled {na_count} NaN adm_1_name with "National" (no S_res column to verify)')

print(f'Total raw records: {len(df):,}')
print(f'Date range: {df["calendar_start_date"].min()} to {df["calendar_start_date"].max()}')

# Filter to 2005-2019
df['Year'] = df['calendar_start_date'].dt.year
df_2005_2019 = df[(df['Year'] >= 2005) & (df['Year'] <= 2019)].copy()

print(f'\n1. FILTERED DATA (2005-2019)')
print('-'*100)
print(f'Total records: {len(df_2005_2019):,}')
print(f'T_res distribution:')
print(df_2005_2019['T_res'].value_counts())

# ============================================================================
# PROCESS MONTH RECORDS
# ============================================================================
print(f'\n2. PROCESSING MONTH RECORDS')
print('-'*100)

df_month = df_2005_2019[df_2005_2019['T_res'] == 'month'].copy()
df_month['Month'] = df_month['calendar_start_date'].dt.month
df_month['YearMonth'] = df_month['calendar_start_date'].dt.to_period('M').dt.to_timestamp()

month_summary = (
    df_month
    .groupby(['adm_1_name', 'Year', 'Month', 'YearMonth'], as_index=False)['dengue_total']
    .sum()
    .rename(columns={'dengue_total': 'dengue_total_month'})
)

print(f'Month records: {len(month_summary):,}')
print(f'Unique provinces: {month_summary["adm_1_name"].nunique()}')

# ============================================================================
# PROCESS WEEK RECORDS (aggregate to monthly, only if >= 3 weeks)
# ============================================================================
print(f'\n3. PROCESSING WEEK RECORDS')
print('-'*100)

df_week = df_2005_2019[df_2005_2019['T_res'] == 'week'].copy()
df_week['Month'] = df_week['calendar_start_date'].dt.month
df_week['YearMonth'] = df_week['calendar_start_date'].dt.to_period('M').dt.to_timestamp()

# Aggregate weeks to months
week_agg_raw = (
    df_week
    .groupby(['adm_1_name', 'Year', 'Month', 'YearMonth'], as_index=False)
    .agg({
        'dengue_total': ['sum', 'count']  # sum cases, count weeks
    })
)
week_agg_raw.columns = ['adm_1_name', 'Year', 'Month', 'YearMonth', 'dengue_total_month', 'week_count']

# Filter: only keep months with >= 3 weeks (75% of month)
week_agg = week_agg_raw[week_agg_raw['week_count'] >= 3].copy()
week_agg = week_agg.drop('week_count', axis=1)

print(f'Week records (raw): {len(week_agg_raw):,}')
print(f'Week records (>= 3 weeks): {len(week_agg):,}')
if len(week_agg_raw) > len(week_agg):
    print(f'  Filtered out {len(week_agg_raw) - len(week_agg)} months with < 3 weeks')

# ============================================================================
# PROCESS YEAR RECORDS (distribute to monthly using seasonal pattern)
# ============================================================================
print(f'\n4. PROCESSING YEAR RECORDS')
print('-'*100)

df_year = df_2005_2019[df_2005_2019['T_res'] == 'year'].copy()

if len(df_year) > 0:
    print(f'Year records found: {len(df_year)}')
    print(f'Year records:')
    print(df_year[['Year', 'dengue_total', 'adm_1_name']].to_string())
    
    # Calculate seasonal pattern from Month/Week data (2005-2019)
    historical_combined = pd.concat([month_summary, week_agg], ignore_index=True)
    historical_national = historical_combined[
        (historical_combined['adm_1_name'] == 'National') &
        (historical_combined['Year'] >= 2005) &
        (historical_combined['Year'] <= 2019)
    ]
    
    if len(historical_national) > 0:
        # Calculate monthly weights from historical data
        monthly_avg = historical_national.groupby('Month')['dengue_total_month'].mean()
        monthly_weights = monthly_avg / monthly_avg.sum()
        
        print(f'\nMonthly seasonal pattern (from {len(historical_national)} months of historical data):')
        for month, weight in monthly_weights.items():
            print(f'  Month {month:2d}: {weight:.4f} ({weight*100:.2f}%)')
        
        # Check existing months (to avoid double-counting)
        existing_months = set()
        if len(month_summary) > 0:
            month_national = month_summary[month_summary['adm_1_name'] == 'National']
            for _, row in month_national.iterrows():
                existing_months.add((int(row['Year']), int(row['Month'])))
        
        if len(week_agg) > 0:
            week_national = week_agg[week_agg['adm_1_name'] == 'National']
            for _, row in week_national.iterrows():
                existing_months.add((int(row['Year']), int(row['Month'])))
        
        if len(existing_months) > 0:
            print(f'\n[INFO] Found {len(existing_months)} months with existing Month/Week data')
            print(f'  These will use Month/Week data instead of Year distribution')
        
        # Distribute Year records to monthly
        year_monthly_list = []
        for _, row in df_year.iterrows():
            year = int(row['Year'])
            annual_total = float(row['dengue_total'])
            adm_name = row['adm_1_name']
            
            # Check if year has Month/Week data
            year_month_data = month_summary[
                (month_summary['adm_1_name'] == adm_name) &
                (month_summary['Year'] == year)
            ]
            year_week_data = week_agg[
                (week_agg['adm_1_name'] == adm_name) &
                (week_agg['Year'] == year)
            ]
            
            # Check unique months (not just count, to ensure we have 12 different months)
            unique_months_month = set(year_month_data['Month'].unique()) if len(year_month_data) > 0 else set()
            unique_months_week = set(year_week_data['Month'].unique()) if len(year_week_data) > 0 else set()
            unique_months_combined = unique_months_month | unique_months_week
            
            month_count = len(year_month_data)
            week_count = len(year_week_data)
            
            # If year has complete Month or Week data (12 unique months), skip Year distribution
            if len(unique_months_combined) >= 12:
                print(f'\n[INFO] Year {year} ({adm_name}): Has complete Month/Week data ({len(unique_months_combined)} unique months: {month_count} Month records, {week_count} Week records)')
                print(f'  Skipping Year record distribution (Priority: Month/Week > Year)')
                continue
            
            # Calculate remaining months to distribute
            existing_for_year = set()
            for _, m_row in year_month_data.iterrows():
                existing_for_year.add((int(m_row['Year']), int(m_row['Month'])))
            for _, w_row in year_week_data.iterrows():
                existing_for_year.add((int(w_row['Year']), int(w_row['Month'])))
            
            # IMPORTANT: Merge Month and Week with priority to avoid double-counting
            # If same month has both Month and Week data, only count Month (priority)
            if len(year_month_data) > 0 and len(year_week_data) > 0:
                # Merge with priority: Month > Week
                year_combined = pd.concat([
                    year_month_data.assign(priority=1),
                    year_week_data.assign(priority=2)
                ], ignore_index=True)
                year_combined = (
                    year_combined
                    .sort_values('priority')
                    .drop_duplicates(subset=['Year', 'Month'], keep='first')
                )
                existing_total = year_combined['dengue_total_month'].sum()
            elif len(year_month_data) > 0:
                existing_total = year_month_data['dengue_total_month'].sum()
            elif len(year_week_data) > 0:
                existing_total = year_week_data['dengue_total_month'].sum()
            else:
                existing_total = 0.0
            
            remaining_total = annual_total - existing_total
            
            if remaining_total < 0:
                print(f'\n[WARNING] Year {year} ({adm_name}): Month/Week total ({existing_total:,.0f}) > Year total ({annual_total:,.0f})')
                print(f'  Skipping Year record (using Month/Week data instead)')
                continue
            
            # Distribute remaining total to missing months
            missing_months = [m for m in range(1, 13) if (year, m) not in existing_for_year]
            
            if len(missing_months) > 0:
                # Validate that all missing_months exist in monthly_weights
                missing_in_weights = [m for m in missing_months if m in monthly_weights.index]
                if len(missing_in_weights) != len(missing_months):
                    missing_not_in_weights = set(missing_months) - set(missing_in_weights)
                    print(f'\n[WARNING] Year {year} ({adm_name}): Some months not in seasonal pattern: {missing_not_in_weights}')
                    print(f'  Using uniform distribution for these months as fallback')
                    # Use uniform distribution for months not in weights
                    uniform_weight = 1.0 / len(missing_months)
                    missing_weights = []
                    for m in missing_months:
                        if m in monthly_weights.index:
                            missing_weights.append(monthly_weights[m])
                        else:
                            missing_weights.append(uniform_weight)
                    missing_weights = np.array(missing_weights)
                else:
                    # Use seasonal weights for missing months
                    missing_weights = monthly_weights[missing_months].values
                
                # Normalize weights
                missing_weights = missing_weights / missing_weights.sum()
                
                for i, month in enumerate(missing_months):
                    month_cases = remaining_total * missing_weights[i]
                    year_month = pd.Timestamp(year=year, month=month, day=1)
                    
                    year_monthly_list.append({
                        'adm_1_name': adm_name,
                        'Year': year,
                        'Month': month,
                        'YearMonth': year_month,
                        'dengue_total_month': month_cases
                    })
        
        year_summary = pd.DataFrame(year_monthly_list)
        
        # Validate: Check if distributed total matches annual total (with tolerance)
        if len(year_summary) > 0:
            # Group by year and adm_1_name to validate
            for _, row in df_year.iterrows():
                year = int(row['Year'])
                annual_total = float(row['dengue_total'])
                adm_name = row['adm_1_name']
                
                year_distributed = year_summary[
                    (year_summary['Year'] == year) &
                    (year_summary['adm_1_name'] == adm_name)
                ]
                
                if len(year_distributed) > 0:
                    distributed_total = year_distributed['dengue_total_month'].sum()
                    # Get existing total from Month/Week (already calculated above, but recalc for validation)
                    year_month_data = month_summary[
                        (month_summary['adm_1_name'] == adm_name) &
                        (month_summary['Year'] == year)
                    ]
                    year_week_data = week_agg[
                        (week_agg['adm_1_name'] == adm_name) &
                        (week_agg['Year'] == year)
                    ]
                    
                    if len(year_month_data) > 0 and len(year_week_data) > 0:
                        year_combined = pd.concat([
                            year_month_data.assign(priority=1),
                            year_week_data.assign(priority=2)
                        ], ignore_index=True)
                        year_combined = (
                            year_combined
                            .sort_values('priority')
                            .drop_duplicates(subset=['Year', 'Month'], keep='first')
                        )
                        existing_total = year_combined['dengue_total_month'].sum()
                    elif len(year_month_data) > 0:
                        existing_total = year_month_data['dengue_total_month'].sum()
                    elif len(year_week_data) > 0:
                        existing_total = year_week_data['dengue_total_month'].sum()
                    else:
                        existing_total = 0.0
                    
                    total_after_distribution = existing_total + distributed_total
                    diff = abs(total_after_distribution - annual_total)
                    diff_pct = (diff / annual_total * 100) if annual_total > 0 else 0
                    
                    if diff_pct > 1.0:  # More than 1% difference
                        print(f'\n[WARNING] Year {year} ({adm_name}): Distribution mismatch')
                        print(f'  Annual total: {annual_total:,.0f}')
                        print(f'  Existing (Month/Week): {existing_total:,.0f}')
                        print(f'  Distributed (Year): {distributed_total:,.0f}')
                        print(f'  Total after distribution: {total_after_distribution:,.0f}')
                        print(f'  Difference: {diff:,.0f} ({diff_pct:.2f}%)')
        
        print(f'\n[OK] Year records distributed to: {len(year_summary)} monthly records')
    else:
        print('[WARNING] No historical data for seasonal pattern calculation')
        print('  Year records will not be distributed')
        year_summary = pd.DataFrame(columns=['adm_1_name', 'Year', 'Month', 'YearMonth', 'dengue_total_month'])
else:
    print('No Year records found')
    year_summary = pd.DataFrame(columns=['adm_1_name', 'Year', 'Month', 'YearMonth', 'dengue_total_month'])

# ============================================================================
# MERGE ALL SOURCES (Priority: Month > Week > Year)
# ============================================================================
print(f'\n5. MERGING ALL SOURCES')
print('-'*100)
print(f'  Month records: {len(month_summary):,}')
print(f'  Week records: {len(week_agg):,} (only months with >= 3 weeks)')
print(f'  Year records: {len(year_summary):,} (distributed)')
print(f'\n  Priority: Month > Week > Year')

# Add priority columns
month_summary['source'] = 'month'
month_summary['priority'] = 1

week_agg['source'] = 'week'
week_agg['priority'] = 2

if len(year_summary) > 0:
    year_summary['source'] = 'year'
    year_summary['priority'] = 3
    combined = pd.concat([month_summary, week_agg, year_summary], ignore_index=True)
else:
    combined = pd.concat([month_summary, week_agg], ignore_index=True)

# Merge with priority: keep record with lowest priority (Month=1, Week=2, Year=3)
combined_final = (
    combined
    .sort_values('priority', ascending=True)
    .drop_duplicates(subset=['adm_1_name', 'Year', 'Month', 'YearMonth'], keep='first')
    .drop(['source', 'priority'], axis=1)
    .sort_values(['YearMonth', 'adm_1_name'])
    .reset_index(drop=True)
)

duplicates_removed = len(combined) - len(combined_final)
if duplicates_removed > 0:
    print(f'\n[INFO] Removed {duplicates_removed} duplicate records (Priority applied)')
print(f'  Final records: {len(combined_final):,}')

# ============================================================================
# CREATE NATIONAL LEVEL DATA (2005-2019)
# ============================================================================
print(f'\n6. CREATING NATIONAL LEVEL DATA')
print('-'*100)

# Filter to 2005-2019
data_2005_2019 = combined_final[
    (combined_final['Year'] >= 2005) &
    (combined_final['Year'] <= 2019)
].copy()

# Check for existing National records
national_existing = data_2005_2019[
    data_2005_2019['adm_1_name'] == 'National'
].copy()

print(f'Existing National records: {len(national_existing)}')
if len(national_existing) > 0:
    print(f'  Date range: {national_existing["YearMonth"].min()} to {national_existing["YearMonth"].max()}')
    print(f'  Years: {sorted(national_existing["Year"].unique())}')

# Aggregate provincial data to create National level
provincial_data = data_2005_2019[data_2005_2019['adm_1_name'] != 'National'].copy()

if len(provincial_data) > 0:
    # Check data quality before aggregation
    provinces_by_month = provincial_data.groupby('YearMonth')['adm_1_name'].nunique()
    avg_provinces = provinces_by_month.mean()
    min_provinces = provinces_by_month.min()
    max_provinces = provinces_by_month.max()
    
    print(f'Provincial data quality check:')
    print(f'  Average provinces per month: {avg_provinces:.1f}')
    print(f'  Min/Max provinces: {min_provinces}/{max_provinces}')
    
    if min_provinces < 50:  # Vietnam has ~63 provinces
        low_coverage_months = provinces_by_month[provinces_by_month < 50]
        print(f'  [WARNING] {len(low_coverage_months)} months have < 50 provinces')
        print(f'    These months may have incomplete data')
    
    # --- START FIX: LỌC BỎ THÁNG THIẾU DỮ LIỆU TỈNH ---
    # Chỉ tổng hợp dữ liệu Quốc gia từ các tháng có ít nhất 20 tỉnh báo cáo
    # (Số 19 ca thấp bất thường chắc chắn do < 5 tỉnh báo cáo)
    valid_months = provinces_by_month[provinces_by_month >= 20].index
    provincial_data_clean = provincial_data[provincial_data['YearMonth'].isin(valid_months)]
    
    provincial_agg = (
        provincial_data_clean
        .fillna({'dengue_total_month': 0})
        .groupby(['YearMonth', 'Year', 'Month'], as_index=False)
        .agg({'dengue_total_month': 'sum'})
        .assign(adm_1_name='National')
    )
    # --- END FIX ---
    print(f'Aggregated from provincial data: {len(provincial_agg)} months')
    
    # Combine: use National if exists, otherwise use aggregated
    if len(national_existing) > 0:
        # Merge: National records take priority, fill gaps with aggregated
        # Add priority columns to ensure National is kept first
        national_existing_priority = national_existing.assign(source='national', priority=1)
        provincial_agg_priority = provincial_agg.assign(source='provincial', priority=2)
        national_all = pd.concat([national_existing_priority, provincial_agg_priority], ignore_index=True)
        national = (
            national_all
            .sort_values(['YearMonth', 'priority'])  # Sort by priority to ensure National first
            .drop_duplicates(subset=['YearMonth'], keep='first')  # Keep National (priority=1)
            .drop(['source', 'priority'], axis=1)
            .sort_values('YearMonth')
            .reset_index(drop=True)
        )
        print(f'  Combined: {len(national)} months (National records prioritized)')
    else:
        national = provincial_agg.sort_values('YearMonth').reset_index(drop=True)
        print(f'  Using aggregated provincial data only')
else:
    if len(national_existing) > 0:
        national = national_existing.sort_values('YearMonth').reset_index(drop=True)
    else:
        national = pd.DataFrame(columns=['YearMonth', 'Year', 'Month', 'dengue_total_month', 'adm_1_name'])

print(f'\nNational level summary:')
print(f'  Records: {len(national)}')
print(f'  Expected: 180 months (15 years × 12 months)')
print(f'  Completeness: {len(national)/180*100:.1f}%')
print(f'  Total cases: {national["dengue_total_month"].sum():,.0f}')
print(f'  Mean cases/month: {national["dengue_total_month"].mean():.0f}')
print(f'  Median cases/month: {national["dengue_total_month"].median():.0f}')
print(f'  Min/Max: {national["dengue_total_month"].min():.0f} / {national["dengue_total_month"].max():.0f}')
print(f'  Date range: {national["YearMonth"].min()} to {national["YearMonth"].max()}')

# --- FIX: HANDLE OUTLIERS (e.g. 19 cases in April 2011) ---
# If cases are extremely low (< 50) in a month, treat as missing and interpolate
# This handles the specific issue where April 2011 has 19 cases (likely data error)
low_cases_mask = national['dengue_total_month'] < 50
if low_cases_mask.any():
    print(f'\n[INFO] Found {low_cases_mask.sum()} months with < 50 cases - treating as outliers and interpolating')
    print(national[low_cases_mask][['YearMonth', 'dengue_total_month']])
    
    # Set to NaN
    national.loc[low_cases_mask, 'dengue_total_month'] = np.nan
    
    # Interpolate
    national['dengue_total_month'] = national['dengue_total_month'].interpolate(method='linear')
    
    print(f'[INFO] Outliers interpolated.')
# ---------------------------------------------------------

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================
print(f'\n7. DATA QUALITY CHECK')
print('-'*100)

# Check null values
print(f'Null values:')
print(national[['YearMonth', 'dengue_total_month']].isnull().sum())

# Check zero cases
zero_national = (national['dengue_total_month'] == 0).sum()
zero_prov = (provincial_data['dengue_total_month'] == 0).sum() if len(provincial_data) > 0 else 0
print(f'\nZero cases:')
print(f'  National: {zero_national}/{len(national)} ({zero_national/len(national)*100:.1f}%)')
if len(provincial_data) > 0:
    print(f'  Provincial: {zero_prov}/{len(provincial_data)} ({zero_prov/len(provincial_data)*100:.1f}%)')

# Check missing months
expected_dates = pd.date_range('2005-01', '2019-12', freq='MS')
actual_dates = set(national['YearMonth'].dt.to_period('M'))
expected_periods = set([d.to_period('M') for d in expected_dates])
missing_periods = expected_periods - actual_dates

if missing_periods:
    print(f'\n[WARNING] Missing months at national level: {len(missing_periods)}')
    for period in sorted(missing_periods)[:10]:  # Show first 10
        print(f'  {period}')
    if len(missing_periods) > 10:
        print(f'  ... and {len(missing_periods) - 10} more')
else:
    print(f'\n[OK] No missing months at national level - DATA IS COMPLETE')

# ============================================================================
# YEARLY AND MONTHLY SUMMARY
# ============================================================================
print(f'\n8. YEARLY SUMMARY')
print('-'*100)
yearly_summary = national.groupby('Year').agg({
    'dengue_total_month': ['count', 'sum', 'mean', 'min', 'max']
}).round(0)
yearly_summary.columns = ['Months', 'Total', 'Mean', 'Min', 'Max']
print(yearly_summary)

print(f'\n9. SEASONAL PATTERN (2005-2019 average)')
print('-'*100)
monthly_pattern = national.groupby('Month')['dengue_total_month'].agg(['mean', 'std', 'count'])
monthly_pattern.columns = ['Mean_Cases', 'Std', 'Obs']
print(monthly_pattern.round(1))

# ============================================================================
# SAVE DATA
# ============================================================================
print(f'\n10. SAVING DATA')
print('-'*100)

output_path = script_dir / 'dengue_2005_2019_national.csv'
national.to_csv(output_path, index=False)
print(f'[OK] Saved national data: {output_path}')

output_path_prov = script_dir / 'dengue_2005_2019_provincial.csv'
provincial_data.to_csv(output_path_prov, index=False)
print(f'[OK] Saved provincial data: {output_path_prov}')

# ============================================================================
# VISUALIZATION
# ============================================================================
print(f'\n11. CREATING VISUALIZATION')
print('-'*100)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Full timeline
ax1 = axes[0, 0]
ax1.plot(national['YearMonth'], national['dengue_total_month'], 'o-', linewidth=2, markersize=4, alpha=0.7)
ax1.set_title('Dengue Cases: 2005-2019 (National Level)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cases per Month')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Year')

# Plot 2: Yearly totals
ax2 = axes[0, 1]
yearly_totals = national.groupby('Year')['dengue_total_month'].sum()
ax2.bar(yearly_totals.index, yearly_totals.values, alpha=0.7, color='steelblue')
ax2.set_title('Annual Total Cases', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Cases')
ax2.set_xlabel('Year')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Seasonal pattern
ax3 = axes[1, 0]
monthly_pattern['Mean_Cases'].plot(ax=ax3, kind='bar', color='coral', alpha=0.7)
ax3.set_title('Average Cases by Month (2005-2019)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Mean Cases')
ax3.set_xlabel('Month')
ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Box plot by month
ax4 = axes[1, 1]
month_data = [national[national['Month'] == m]['dengue_total_month'].values for m in range(1, 13)]
ax4.boxplot(month_data, tick_labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax4.set_title('Case Distribution by Month', fontsize=12, fontweight='bold')
ax4.set_ylabel('Cases per Month')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_img = script_dir / 'dengue_2005_2019_analysis.png'
plt.savefig(output_img, dpi=150, bbox_inches='tight')
print(f'[OK] Saved visualization: {output_img}')
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print('\n' + '='*100)
print('SUMMARY FOR SEIR MODELING')
print('='*100)
print(f'''
[OK] Data readiness:
  - Total months: {len(national)} ({len(national)/180*100:.1f}% complete)
  - Total cases: {national["dengue_total_month"].sum():,.0f}
  - Time period: {national["YearMonth"].min().strftime("%B %Y")} to {national["YearMonth"].max().strftime("%B %Y")}
  - Missing months: {len(missing_periods) if missing_periods else 0}
  - Zero-case months: {zero_national}
  
[TARGET] For SEIR Model:
  - Use file: dengue_2005_2019_national.csv
  - Columns: YearMonth, dengue_total_month
  - Seasonal pattern established: [OK] YES
  - Can derive S(t), E(t), I(t), R(t) from cases
  - Temperature/rainfall data available: Check ../population/ and NetCDF files
  
[NOTES]:
  - Data processed directly from raw dataset_dengue_2005_2024.csv
  - Priority: Month > Week > Year
  - Year records distributed using seasonal pattern from historical data
  - National level created from existing National records (if any) or aggregated from provincial data
''')

print('='*100)
