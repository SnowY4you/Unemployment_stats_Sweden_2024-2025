import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Load the Excel file with unemployment statistics
unemployment_stats_file = r'C:\Users\svanb\OneDrive\Python\job_stats\unemployment_stats.xlsx'
df = pd.read_excel(unemployment_stats_file)

# Map the 'time_without_job' to numeric values for easier learning
time_mapping = {
    'Mindre än 6 månader': 1,
    'Mellan 6 och 12 månader': 2,
    'Mellan 12 och 24 månader': 3,
    'Mer än 24 månader': 4
}
df['time_without_job_numeric'] = df['time_without_job'].map(time_mapping)

# Map the 'gender' to numeric values for easier learning
gender_mapping = {
    'Kvinna': 0,
    'Man': 1
}
df['gender_numeric'] = df['gender'].map(gender_mapping)

# Map the 'age' to numeric values for easier learning
age_mapping = {
    '18–19': 18,
    '20–24': 20,
    '25–29': 25,
    '30–39': 30,
    '40–49': 40,
    '50–59': 50,
    '60–64': 60
}
df['age_numeric'] = df['age'].map(age_mapping)

# Map the 'country_of_birth' to numeric values for easier learning
country_mapping = {
    'Sverige': 1,
    'Europa utom Sverige': 2,
    'Utomeuropeisk': 3
}
df['country_numeric'] = df['country_of_birth'].map(country_mapping)

# Define features and target variable
X = df[['year', 'gender_numeric', 'age_numeric', 'country_numeric', 'unemployed_people']]
y = df['time_without_job_numeric']

# Feature Engineering: Example of creating new features
df['unemployment_rate'] = df['unemployed_people'] / df['unemployed_people'].sum()

# Preprocessing pipeline for numeric and categorical features
numeric_features = ['year', 'unemployed_people']
categorical_features = ['gender_numeric', 'age_numeric', 'country_numeric']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and the classifier
# Using an ensemble of HistGradientBoostingClassifier and RandomForestClassifier
ensemble_model = VotingClassifier(estimators=[
    ('hgb', HistGradientBoostingClassifier(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
], voting='soft')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ensemble_model)
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Ensemble Model Accuracy: {accuracy:.2f}')

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'classifier__hgb__max_iter': [100, 200, 300],
    'classifier__hgb__learning_rate': [0.01, 0.05, 0.1],
    'classifier__hgb__max_depth': [3, 5, 7],
    'classifier__rf__n_estimators': [100, 200, 500],
    'classifier__rf__max_features': ['sqrt', 'log2'],
    'classifier__rf__max_depth': [10, 20, 30],
    'classifier__rf__min_samples_split': [2, 5, 10],
    'classifier__rf__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Score: {grid_search.best_score_:.2f}')

# Predict for the next 5 years
future_years = [2025, 2026, 2027, 2028, 2029]
future_data = []

for year in future_years:
    for gender in [0, 1]:  # Numeric values for 'gender'
        for age in [18, 20, 25, 30, 40, 50, 60]:  # Numeric values for 'age'
            for country in [1, 2, 3]:  # Numeric values for 'country_of_birth'
                unemployed_people = np.random.randint(1, 1000)
                future_data.append([year, gender, age, country, unemployed_people])

future_df = pd.DataFrame(future_data, columns=['year', 'gender_numeric', 'age_numeric', 'country_numeric', 'unemployed_people'])

# Encode the future data
X_future = future_df[['year', 'gender_numeric', 'age_numeric', 'country_numeric', 'unemployed_people']]
X_future_encoded = pipeline.named_steps['preprocessor'].transform(X_future)

# Predict the time_without_job for future data
future_df['predicted_time_without_job'] = pipeline.named_steps['classifier'].predict(X_future_encoded)

# Create combined x-axis label for age and country
age_mapping_reverse = {
    18: '18–19',
    20: '20–24',
    25: '25–29',
    30: '30–39',
    40: '40–49',
    50: '50–59',
    60: '60–64'
}
country_mapping_reverse = {
    1: 'Sverige',
    2: 'Europa utom Sverige',
    3: 'Utomeuropeisk'
}
future_df['age_country'] = future_df['age_numeric'].map(age_mapping_reverse).astype(str) + ' - ' + future_df['country_numeric'].map(country_mapping_reverse)

# Create the table
result_table = future_df[['year', 'gender_numeric', 'age_country', 'unemployed_people', 'predicted_time_without_job']]
print(result_table.head())

# Save the table to an Excel file
result_table.to_excel(r'C:\Users\svanb\OneDrive\Python\job_stats\predicted_unemployment_stats_next_5_years.xlsx', index=False)
print("Table saved to: C:\\Users\\svanb\\OneDrive\\Python\\job_stats\\predicted_unemployment_stats_next_5_years.xlsx")
