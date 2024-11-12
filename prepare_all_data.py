import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the Excel file
file_path = r'C:\Users\svanb\OneDrive\Python\job_stats\jobseeking_sweden.xlsx'
df = pd.read_excel(file_path)

# Convert 'period' to datetime format and then split into 'month' and 'year'
df['period'] = pd.to_datetime(df['period'], format='%Y-%m')
df['month'] = df['period'].dt.month
df['year'] = df['period'].dt.year

# Ensure the 'amount' column is in integer format
df['amount'] = df['amount'].astype(int)

# Map the 'time_without_job' to numeric values
time_mapping = {
    'Mindre än 6 månader': 1,
    'Mellan 6 och 12 månader': 2,
    'Mellan 12 och 24 månader': 3,
    'Mer än 24 månader': 4
}
df['time_without_job_numeric'] = df['time_without_job'].map(time_mapping)

# Check for NaN values after mapping
nan_count = df['time_without_job_numeric'].isna().sum()
print(f'NaN values in time_without_job_numeric: {nan_count}')

# Handle NaN values (if any)
df['time_without_job_numeric'] = df['time_without_job_numeric'].fillna(df['time_without_job_numeric'].median())

# Prepare the data for modeling
df = df.drop(columns=['period'])

# Set the display option
pd.set_option('display.max_columns', None)

# Display the dataframe
print(df.to_string())

# Define the output file path
output_file_path = r'C:\Users\svanb\OneDrive\Python\job_stats\prepared_unemployment_stats.xlsx'

# Save the DataFrame to an Excel file
df.to_excel(output_file_path, index=False)
print(f"Table saved to: {output_file_path}")