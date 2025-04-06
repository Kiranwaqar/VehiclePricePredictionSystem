# Necessary imports
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

# Load dataset
df = pd.read_csv('data/cars.csv')
df.info()

# Check for nulls
print("\nMissing Values Before Filling:\n", df.isnull().sum())

# Fill numerical columns with mean
for col in ['Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity']:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical column with mode
df['Drivetrain'] = df['Drivetrain'].fillna(df['Drivetrain'].mode()[0])

# Drop irrelevant or high-cardinality columns
cols_to_drop = ['Location', 'Owner', 'Seller Type', 'Color', 'Make', 'Model', 'Max Power', 'Max Torque', 'Engine']
df.drop(columns=cols_to_drop, inplace=True)

# Log transform Kilometer to normalize
df['Kilometer'] = np.log1p(df['Kilometer'])

# Create new feature: Vehicle Age
current_year = datetime.now().year
df['Vehicle Age'] = current_year - df['Year']
df.drop(columns=['Year'], inplace=True)

# Visualize outliers using boxplots
features_to_plot = ['Price', 'Kilometer', 'Vehicle Age']
plt.figure(figsize=(15, 5))

for i, feature in enumerate(features_to_plot):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(data=df, y=feature)
    plt.title(f'Boxplot of {feature}')

plt.tight_layout()
plt.show()

# Function to cap outliers
def cap_outliers(df, column):
   Q1 = df[column].quantile(0.25)
   Q3 = df[column].quantile(0.75)
   IQR = Q3 - Q1
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
   return df

# Cap outliers in selected features
for feature in features_to_plot:
    df = cap_outliers(df, feature)

# Boxplots after capping outliers
plt.figure(figsize=(15, 5))
for i, feature in enumerate(features_to_plot):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(data=df, y=feature)
    plt.title(f'{feature} (Outliers Removed)')
plt.tight_layout()
plt.show()

# One-hot Encoding for categorical features
categorical_columns = ['Fuel Type', 'Transmission', 'Drivetrain']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

# Ensure only valid columns are kept
numerical_columns = ['Length', 'Width', 'Height', 'Seating Capacity', 
                     'Fuel Tank Capacity', 'Kilometer', 'Vehicle Age', 'Price']
df_combined = df[numerical_columns + [col for col in df.columns if col not in numerical_columns and col != 'Price']]

# Split features and target
X = df_combined.drop('Price', axis=1)
y = np.log1p(df_combined['Price'])  # log1p handles zero prices too

# Split data: train, validation, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Linear Regression model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate on training set
y_train_pred = model.predict(X_train_scaled)
print("Train Set Performance:")
print("MAE:", mean_absolute_error(y_train, y_train_pred))
print("MSE:", mean_squared_error(y_train, y_train_pred))
print("R2 Score:", r2_score(y_train, y_train_pred))

# Evaluate on validation set
y_val_pred = model.predict(X_val_scaled)
print("\nValidation Set Performance:")
print("MAE:", mean_absolute_error(y_val, y_val_pred))
print("MSE:", mean_squared_error(y_val, y_val_pred))
print("R2 Score:", r2_score(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = model.predict(X_test_scaled)
print("\nTest Set Performance:")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R2 Score:", r2_score(y_test, y_test_pred))
