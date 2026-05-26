import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

df = pd.read_csv("taxi_trip_pricing.csv")


print("\n================ DATASET SHAPE ================\n")
print(df.shape)
print("\n================ FIRST 5 ROWS ================\n")
print(df.head().to_string())
print("\n================ COLUMN NAMES ================\n")


print(df.columns)
print("\n================ DATA TYPES ================\n")
print(df.dtypes)
print("\n================ DATASET INFO ================\n")
print(df.info())

print("\n================ BASIC STATISTICS ================\n")
print(df.describe())


print("\n================ MISSING VALUES ================\n")
missing_count = df.isnull().sum()
print(missing_count)

missing_percentage = (missing_count / len(df)) * 100
print(missing_percentage)

missing_df = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percentage
})

missing_df = missing_df[missing_df["Missing_Count"] > 0]
print(missing_df)


print("\n================ DUPLICATES ================\n")
duplicate_count = df.duplicated().sum()
print(f"duplicate_rows: {duplicate_count}")


# ============================================================
# NUMERIC VS CATEGORICAL FEATURES
# ============================================================

numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


categorical_cols = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("\n================ NUMERIC FEATURES ================\n")

print(numeric_cols)

print("\n================ CATEGORICAL FEATURES ================\n")

print(categorical_cols)



print("\n================ CATEGORICAL ANALYSIS ================\n")

for col in categorical_cols:
    print(f"\n------{col}-------\n")
    
    print("unique values: ", df[col].nunique())
    
    
    print(df[col].value_counts())
    
    
# ============================================================
# TARGET VARIABLE ANALYSIS
# ============================================================

target = "Trip_Price"

print("\n================ TARGET ANALYSIS ================\n")

print(df[target].describe())


plt.figure(figsize=(10, 6))

sns.histplot(
    df[target], 
    bins=30,
    kde=True
)
plt.title("trip price distribution")
plt.xlabel("trip price")
plt.ylabel("frequency")

plt.tight_layout()

#plt.show()


# ============================================================
# NUMERIC FEATURE DISTRIBUTIONS
# ============================================================

fig, axes = plt.subplots(
    nrows=(len(numeric_cols) // 3) +1,
    ncols=3,
    figsize=(15, 10)
)

axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    
    sns.histplot(
        df[col],
        bins=30,
        kde=True,
        ax=axes[idx]
    )
    
    axes[idx].set_title(f"{col} distribution")
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])    
    
plt.tight_layout()    
#plt.show()


# ============================================================
# BOXPLOTS FOR OUTLIERS
# ============================================================

fig, axes  = plt.subplots(
    nrows = (len(numeric_cols) // 3) +1,
    ncols = 3,
    figsize=(15,10)
)

axes = axes.flatten()


for idx, col in enumerate(numeric_cols):
    
    sns.boxplot(
        x = df[col],
        ax = axes[idx]
    )
    
    axes[idx].set_title(f"{col}Boxplot")
    
for j in range(idx +1, len(axes)):
    fig.delaxes(axes[j])        
    
plt.tight_layout()
#plt.show()


# ============================================================
# CATEGORICAL FEATURE DISTRIBUTIONS
# ============================================================


fig ,  axes = plt.subplots(
    nrows=(len(categorical_cols) //2) +1,
    ncols=2, 
    figsize=(15,12)
)

axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    
    df[col].value_counts().plot(
        kind = "bar",
        ax = axes[idx]
    )
    
    axes[idx].set_title(f"{col} distribution")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Count")
    
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])
    
plt.tight_layout()
#plt.show()


print("\n================ CORRELATION MATRIX ================\n")


corr_matrix = df[numeric_cols].corr()
print(str(corr_matrix))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm", 
    fmt=".2f"
)

plt.title("correlation heatmap")
plt.tight_layout()
#plt.show()

print("\n================ HIGH CORRELATIONS ================\n")

# Threshold for high correlation
threshold = 0.80

# Compare all numeric columns
for i in range(len(corr_matrix.columns)):

    for j in range(i + 1, len(corr_matrix.columns)):

        corr_value = corr_matrix.iloc[i, j]

        if abs(corr_value) > threshold:

            print(
                f"{corr_matrix.columns[i]} <-> "
                f"{corr_matrix.columns[j]} = "
                f"{corr_value:.2f}"
            )

print("\n================ OUTLIER DETECTION ================\n")

# Function to calculate outliers
def detect_outliers(data, column):

    # First quartile
    Q1 = data[column].quantile(0.25)

    # Third quartile
    Q3 = data[column].quantile(0.75)

    # Interquartile range
    IQR = Q3 - Q1

    # Lower boundary
    lower_bound = Q1 - (1.5 * IQR)

    # Upper boundary
    upper_bound = Q3 + (1.5 * IQR)

    # Detect outliers
    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    return len(outliers)

# Check every numeric feature
for col in numeric_cols:

    outlier_count = detect_outliers(df, col)

    print(f"{col}: {outlier_count} outliers")
    
    
    
# ============================================================
# TARGET RELATIONSHIPS
# ============================================================

# Question:
# Which features most affect Trip_Price?

important_features = [
    "Trip_Distance_km",
    "Trip_Duration_Minutes",
    "Base_Fare"
]

for feature in important_features:

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        x=df[feature],
        y=df[target]
    )

    plt.title(f"{feature} vs Trip Price")

    plt.xlabel(feature)
    plt.ylabel("Trip Price")

    plt.tight_layout()

    #plt.show()
    

quality_report = pd.DataFrame({

  "Feature": df.columns,
  "Data_Type": df.dtypes.values, 
  "Missing_Values": df.isnull().sum().values,
  "Missing Percentage": (
      df.isnull().sum() / len(df) *100
  ).round(2).values, 
  "unique values": df.nunique().values

})

print("\n================ DATA QUALITY REPORT ================\n")

print(quality_report)

# Save report
quality_report.to_csv(
    "data_quality_report.csv",
    index=False
)

print("\nData Quality Report Saved!")




#STEP 2 — DATA CLEANING




#show all columns

pd.set_option('display.max_columns', None)

sns.set_style("whitegrid")

# ============================================================
# LOAD DATASET
# ============================================================

original_shape = df.shape
print("\n================ ORIGINAL DATASET ================\n")


print(f"Original shape: {df.shape}" )

# ============================================================
# STEP 1 — MISSING VALUES ANALYSIS
# ============================================================

print("\n================ MISSING VALUES ANALYSIS ================\n")

missing_count = df.isnull().sum()

missing_percentage = (
    missing_count / len(df)
) * 100

#create missing values report

missing_df = pd.DataFrame({
    "Feature": df.columns,
    "missing_count": missing_count.values,
    "missing_percentage": missing_percentage.values
})

#keep only columns with missing values

missing_df = missing_df[
    missing_df["missing_count"] > 0
]

#sort by highest missing %

missing_df = missing_df.sort_values(
    by="missing_percentage",
    ascending= False
)

print(missing_df)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

#save decisions for documentation

missing_decisions = {}


# -----------------------------
# NUMERIC COLUMNS
# -----------------------------

#get numeric columns

numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns


print("\n================ NUMERIC IMPUTATION ================\n")

for col in numeric_cols:
    
    #calculate the missing percentage
    
    missing_pct = (
        df[col].isnull().sum() / len(df)
    ) * 100
    #skip if no missing value
    if missing_pct == 0:
        continue
    #less than 20% missing ->> fill with median
    elif missing_pct < 20:
        
       median_value = df[col].median()
    
       #fill missing values
       
       df[col] = df[col].fillna(median_value)
       
       print(
           f"{col}:"
           f"filled with median"
           f"({median_value})"
       )
       
       #save decision
       
       missing_decisions[col] = (
           f"median imputation"
           f"({missing_pct: 2f}% missing)"
       )
       
     #more than 20% missing ->>
    else:
        df.drop(columns=col, inplace=True)
        
        print(
            f"✓ {col}: "
            f"Dropped "
            f"({missing_pct:.2f}% missing)"
        )
        #save decision
        missing_decisions[col] = (
            f"dropped"
            f"({missing_pct: .2f}% missing)"
        )
        
# -----------------------------
# CATEGORICAL COLUMNS
# -----------------------------

#get categorical columns

categorical_cols = df.select_dtypes(
    include = ["object"]
).columns

print("\n================ CATEGORICAL IMPUTATION ================\n")

for col in categorical_cols:
    
    #calculate the missing percentage
    
    missing_pct = (
        df[col].isnull().sum() / len(df)
    ) *100
    
    #skip if no missing values
    
    if missing_pct == 0:
        continue
    
    #less than 20% missing
    
    elif missing_pct < 20:
        
        #most common category
        
        mode_value = df[col].mode()[0]
        
        #fill missing values
        
        df[col] = df[col].fillna(mode_value)
        
        print(
            f"✓ {col}: "
            f"Filled with mode "
            f"({mode_value})"
        )

        # Save decision
        missing_decisions[col] = (
            f"Mode Imputation "
            f"({missing_pct:.2f}% missing)"
        )
        
        #more than 20% missing
        
    else:
        df.drop(columns=col, inplace=True)
        
        print(
            f"✓ {col}: "
            f"Dropped "
            f"({missing_pct:.2f}% missing)"
        )
        
        #save decision
        missing_decisions[col] = (
            f"dropped"
            f"({missing_pct: 2f}% missing)"
        )
        
# ============================================================
# VERIFY MISSING VALUES
# ============================================================

print("\n================ REMAINING NULL VALUES ================\n")

remaining_nulls = df.isnull().sum().sum()

print(f"remaining missing values: {remaining_nulls}")


# ============================================================
# STEP 2 — REMOVE DUPLICATES
# ============================================================

# Goal:
# Remove completely identical rows.

print("\n================ DUPLICATE ANALYSIS ================\n")

#count duplicates before removal

duplicate_count = df.duplicated().sum()
print(f"duplicate rows before: {duplicate_count}")


#save row count before removal

rows_before = len(df)

#remove duplicates

df = df.drop_duplicates()

#save row count after removal

rows_after = len(df)

# Calculate removed rows
removed_duplicates = rows_before - rows_after

print(f"Duplicate Rows Removed: {removed_duplicates}")

print(f"Shape After Duplicate Removal: {df.shape}")


# ============================================================
# STEP 3 — FIX DATA TYPES
# ============================================================

# Goal:
# Ensure every column has correct data type.

print("\n================ DATA TYPE FIXES ================\n")

print("current data types: \n")
print(df.dtypes)


#----------------------------
# Convert categorical columns
# to category datatype
# -----------------------------

categorical_cols = df.select_dtypes(
    include=["object"]
).columns

for col in categorical_cols:
    
    df[col] = df[col].astype("category")
    print(f"{col}: converted to category")
    
    
# ============================================================
# VERIFY DATA TYPES
# ============================================================

print("\n================ FINAL DATA TYPES ================\n")

print(df.dtypes)

# ============================================================
# STEP 4 — OUTLIER DETECTION
# ============================================================

# Goal:
# Detect unusual values using IQR method.
# We will mostly KEEP them unless impossible.


print("\n================ OUTLIER DETECTION ================\n")

#function to detect outliers

def detect_outliers_iqr(data, column):
    
    # First quartile
    Q1 = data[column].quantile(0.25)

    # Third quartile
    Q3 = data[column].quantile(0.75)

    # Interquartile range
    IQR = Q3 - Q1

    # Lower bound
    lower_bound = Q1 - (1.5 * IQR)

    # Upper bound
    upper_bound = Q3 + (1.5 * IQR)
    
    
    #find outliers
    
    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]
    
    return outliers, lower_bound, upper_bound

#numeric columns only

numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns

#save the outlier report

outlier_report = {}

#analyse every numeric column

for col in numeric_cols:
    
    outliers, lower, upper = detect_outliers_iqr(
        df,
        col
    )
    
    outlier_count = len(outliers)
    
    #calculate percentage
    
    outlier_pct = (
        outlier_count / len(df)
    ) * 100
    
    #save report
    
    outlier_report[col] = {
        "count": outlier_count,
        "percentage": round(outlier_pct, 2), 
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2)
    }
    
    print(f"\n{col}")
    print(f"outliers: {outlier_count}")
    
    print(
        f"bounds: "
        f"[{lower: 2f}, {upper: .2f}]"
    )
    
    
    
# ============================================================
# OUTLIER VISUALIZATION
# ============================================================

# Goal:
# Visualize outliers with boxplots.


print("\n================ OUTLIER VISUALIZATION ================\n")

# Create subplot layout

fig, axes = plt.subplots(
    nrows=(len(numeric_cols) // 3) + 1,
    ncols=3,
    figsize=(18, 15)
)

#flatten subplot array

axes = axes.flatten()

#plot each numeric column

for idx, col in enumerate(numeric_cols):
    sns.boxplot(
        x = df[col],
        ax = axes[idx]
    )
    axes[idx].set_title(f"{col} boxplot")

#remove empty plots

for j in range(idx + 1, len(axes)):
    
    fig.delaxes(axes[j])

plt.tight_layout()

#save figure

plt.savefig(
    "outliers_boxplot.png", 
    dpi=300
)

#plt.show()

print("✓ Boxplots Saved")

print("\n✓ Most outliers were KEPT")
print("Reason: They appear to be realistic taxi trips")

# ============================================================
# STEP 5 — FINAL DATA QUALITY CHECK
# ============================================================

print("\n================ FINAL DATA QUALITY CHECK ================\n")


#final shape

print(f"original shape: {original_shape}")
print(f"final shape: {df.shape}")

# Remaining missing values

print(
    f"\nremaining missing values:"
    f"{df.isnull().sum().sum()}"
)

#remaining duplicates

print(
    f"remaining duplicates: "
    f"{df.duplicated().sum()}"
)

#data type counts

print("\ndata type counts:\n")
print(df.dtypes.value_counts())


# ============================================================
# CLEANING CHECKLIST
# ============================================================

print("\n================ CLEANING CHECKLIST ================\n")


print(
    f"no  missing values: "
    f"{df.isnull().sum().sum() == 0}"
)

print(
    f"no duplicates: "
    f"{df.duplicated().sum() == 0}"
)

print("✓ Data Types Fixed")

print("✓ Outliers Analyzed")

print("✓ Dataset Ready For Feature Engineering")

# ============================================================
# SAVE CLEANED DATASET
# ============================================================

#save cleaned dataset

df.to_csv(
    "taxi_trip_pricing_cleaned.csv", 
    index=False
)

print("\n✓ Cleaned Dataset Saved")

# ============================================================
# CLEANING DECISIONS LOG
# ============================================================


cleaning_decisions = f"""

#================================================
DATA CLEANING DECISIONS LOG
================================================

1. MISSING VALUES
------------------------------------------------
"""

#add missing value decisions


for  feature, decision in missing_decisions.items():
    
    cleaning_decisions += (
        f"\n- {feature}:{decision}"
    )
#add duplicate info

cleaning_decisions += f"""
2. DUPLICATES
------------------------------------------------
- Duplicate Rows Removed: {removed_duplicates}

3. DATA TYPES
------------------------------------------------
- All object columns converted to category datatype

4. OUTLIERS
------------------------------------------------
- Outliers detected using IQR method
- Most outliers kept because they appear realistic
- Long taxi trips are valid real-world cases

5. FINAL DATASET
------------------------------------------------
- Original Shape: {original_shape}
- Final Shape: {df.shape}
- Missing Values Remaining: {df.isnull().sum().sum()}
- Duplicate Rows Remaining: {df.duplicated().sum()}

NEXT STEP:
Feature Engineering
"""

#print decisions
print(cleaning_decisions)

with open(
    "cleaning_decisions.txt",
    "w"
) as file:

    file.write(cleaning_decisions)

print("✓ Cleaning Decisions Saved")












# ============================================================
# STEP 3 — FEATURE ENGINEERING
# Taxi Fare Prediction Project
# ============================================================

# Goal of Step 3:
# Transform cleaned data into ML-ready features.
# We will:
# 1. Separate features and target
# 2. Encode categorical variables
# 3. Create new useful features
# 4. Scale numeric features
# 5. Remove weak/redundant features
# 6. Save artifacts for deployment



from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import VarianceThreshold


df = pd.read_csv("taxi_trip_pricing_cleaned.csv")
print("\n================ CLEANED DATASET ================\n")

print(df.head())

print(df.shape)














#NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Feature engineering tools
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import VarianceThreshold


# ============================================================
# LOAD CLEANED DATA
# ============================================================

# Load cleaned dataset from Step 2
df = pd.read_csv("taxi_trip_pricing_cleaned.csv")

print("\n================ CLEANED DATASET ================\n")

print(df.head())

print(f"\nDataset Shape: {df.shape}")


# ============================================================
# STEP 1 — SEPARATE FEATURES & TARGET
# ============================================================

#target column

target_column = "Trip_Price"

#features(X)

X = df.drop(target_column, axis =1 )

#target(Y)

y = df[target_column]

print("\n================ FEATURES & TARGET ================\n")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

#numeric columns

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


#categorical columns

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumeric features: \n")
print(numeric_features)
print("\nCategorical Features:\n")
print(categorical_features)

# ============================================================
# STEP 2 — ENCODE CATEGORICAL VARIABLES
# ============================================================

# ML algorithms cannot understand text.
# We must convert categories into numbers.

print("\n================ CATEGORICAL ENCODING ================\n")

#save original feature count

before_encoding_shape=X.shape

# ------------------------------------------------
# TAXI PROJECT ENCODING DECISIONS
# ------------------------------------------------

# Time_of_Day:
# Morning / Afternoon / Evening / Night
# No strict mathematical order
# -> One Hot Encoding

# Day_of_Week:
# Weekday / Weekend
# Binary category
# -> One Hot Encoding

# Traffic_Conditions:
# Low / Medium / High
# Has order
# -> Ordinal Encoding

# Weather:
# Clear / Rain / Snow
# No order
# -> One Hot Encoding

# ============================================================
# ORDINAL ENCODING
# ============================================================

#traffic conditions have natural order

traffic_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

#Apply mapping

if "Traffic_Conditions" in X.columns:
    
    X["Traffic_Conditions"] = (
        X["Traffic_Conditions"]
        .map(traffic_mapping)
    )
    
    print("traffic_conditions encoded as ordinal")
    
    
# ============================================================
# ONE HOT ENCODING
# ============================================================

#Remaining categorical columns

remaining_categorical = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

#one-hot encode all remaining categoricals
X = pd.get_dummies(
    X, 
    columns=remaining_categorical, 
    drop_first=True
)

print("✓ Remaining categorical features one-hot encoded")

print(f"\nShape Before Encoding: {before_encoding_shape}")

print(f"Shape After Encoding: {X.shape}")

# ============================================================
# VERIFY ALL FEATURES ARE NUMERIC
# ============================================================


non_numeric_cols = X.select_dtypes(
    include=["object"]
).shape[1]

print(
    f"\nAll Numeric Features: "
    f"{non_numeric_cols == 0}"
)


# ============================================================
# STEP 3 — CREATE NEW FEATURES
# ============================================================

# Goal:
# Create more informative features from existing data.

print("\n================ NEW FEATURE CREATION ================\n")

# ------------------------------------------------
# FEATURE 1 — PRICE PER MINUTE ESTIMATION
# ------------------------------------------------

#useful relationship feature

X["Distance_Per_Minute"] = (
    X["Trip_Distance_km"] / 
    (X["Trip_Duration_Minutes"] + 1)
)

print("created distance per minute")

# ------------------------------------------------
# FEATURE 2 — IS LONG TRIP
# ------------------------------------------------

#binary feature for long trips

X["Is_Long_Trip"] = (
    X["Trip_Distance_km"] > 30
).astype(int)

print("created is long trip")

# ------------------------------------------------
# FEATURE 3 — COST INTENSITY
# ------------------------------------------------

#interaction feature

X["Cost_Interaction"] = (
    X["Per_Km_Rate"] *
    X["Per_Minute_Rate"]
)
print("created cost interaction")

# ------------------------------------------------
# FEATURE 4 — PASSENGER DENSITY
# ------------------------------------------------

#passenger relative to trip size
X["Passenger_Density"] = (
   X["Passenger_Count"] / 
     (X["Trip_Distance_km"] +1 )
   )

print("✓ Created Passenger_Density")

print(f"\nShape After Feature Creation: {X.shape}")


# ============================================================
# STEP 4 — FEATURE SCALING
# ============================================================

# Goal:
# Put all numeric features on same scale.
# Important for many ML algorithms.

print("\n================ FEATURE SCALING ================\n")

#get numeric columns

numeric_cols = X.select_dtypes(
    include=["int64", "float64", "uint8"]
).columns.tolist()

#create scaler

scaler = StandardScaler()

#fit scaler and transform features

X[numeric_cols]=scaler.fit_transform(
    X[numeric_cols]
)

print("✓ Features scaled using StandardScaler")

# ============================================================
# VERIFY SCALING
# ============================================================

print("\nScaling Verification:\n")

print("Feature Means:\n")

print(X[numeric_cols].mean().head())
print("\nFeature Standard Deviations:\n")
print(X[numeric_cols].std().head())

# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler, 
    "scaler.pkl"
)

print(("\n scaler saved"))

# ============================================================
# STEP 5 — FEATURE SELECTION
# ============================================================

# Goal:
# Keep important features only.
# Remove noisy or weak features.

print("\n================ FEATURE SELECTION ================\n")

# ------------------------------------------------
# REMOVE LOW VARIANCE FEATURES
# ------------------------------------------------

#features with almost no variation are useless

variance_selector = VarianceThreshold(
    threshold=0.01
)

X_variance = variance_selector.fit_transform(X)


#Keep selected column names

variance_selected_columns = X.columns[
    variance_selector.get_support()
]

#convert back to Dataframe

X = pd.DataFrame(
    X_variance,
    columns=variance_selected_columns
)

print("✓ Low variance features removed")

print(f"Shape After Variance Filter: {X.shape}")

# ============================================================
# REMOVE HIGHLY CORRELATED FEATURES
# ============================================================

# Highly correlated features may cause:
# - multicollinearity
# - redundancy

corr_matrix = X.corr().abs()

#upper triangle only

upper_triangle = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape), 
        k = 1
    ).astype(bool)
)

#feature to drop

to_drop = [
    column 
    for column in upper_triangle.columns
    if any(upper_triangle[column] > 0.95)
] 

#drop highly correlated features

X = X.drop(columns=to_drop)

print(
    f"✓ Highly correlated features removed: "
    f"{len(to_drop)}"
)

print(f"Shape After Correlation Filter: {X.shape}")

# ============================================================
# SELECT BEST FEATURES
# ============================================================

# Regression problem -> f_regression

selector = SelectKBest(
    score_func=f_regression,
    k=min(15, X.shape[1])
)

#fit selector
selector.fit(X, y)

#selected feature mask

selected_mask = selector.get_support()

#keep selected features

selected_features = X.columns[
    selected_mask
].tolist()

#feature importance scores
feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

#sort by importance

feature_scores = feature_scores.sort_values(
    by = "Score", 
    ascending = False
)

print("\nTop Feature Scores:\n")

print(feature_scores.head(15))
#keep only selected features
X = X[selected_features]


print(f"\n✓ Final Selected Features: {len(selected_features)}")

print(selected_features)

# ============================================================
# STEP 6 — FINAL VERIFICATION
# ============================================================

print("\n================ FINAL VERIFICATION ================\n")

# ------------------------------------------------
# CHECK 1 — ALL NUMERIC
# ------------------------------------------------

non_numeric = X.select_dtypes(
    include=["object"]
).shape[1]

print(
    f"all numeric features: "
    f"{non_numeric == 0}"
)

# ------------------------------------------------
# CHECK 2 — MISSING VALUES
# ------------------------------------------------

print(f"missing values remaining:" 
    f"{X.isnull().sum().sum()}")

# ------------------------------------------------
# CHECK 3 — FEATURE COUNT
# ------------------------------------------------

print(f"final feature count: "
      f"{X.shape[1]}")


# ------------------------------------------------
# CHECK 4 — DATA TYPES
# ------------------------------------------------

print("\nFeature Data Types:\n")

print(X.dtypes.value_counts())

# ------------------------------------------------
# CHECK 5 — TARGET STATUS
# ------------------------------------------------

print("\nTarget Summary:\n")

print(y.describe())

# ============================================================
# STEP 7 — SAVE ENGINEERED DATA
# ============================================================

print("\n================ SAVING ARTIFACTS ================\n")

# Save engineered features

X.to_csv(
    "X_engineered.csv",
    index= False
    )

#save target 
y.to_csv(
    "y_target.csv",
    index=False
)

#save feature names
feature_names = X.columns.tolist()

joblib.dump(
    feature_names,
    "feature_names.pkl"
)

print("✓ X_engineered.csv saved")

print("✓ y_target.csv saved")

print("✓ feature_names.pkl saved")


feature_log = f"""

================================================
FEATURE ENGINEERING LOG
================================================

1. TARGET
------------------------------------------------
- Target Variable: Trip_Price

2. ENCODING
------------------------------------------------
- Traffic_Conditions:
  Ordinal Encoding
  Low=1, Medium=2, High=3

- Other categorical features:
  One Hot Encoding

3. NEW FEATURES CREATED
------------------------------------------------
- Distance_Per_Minute
- Is_Long_Trip
- Cost_Interaction
- Passenger_Density

4. SCALING
------------------------------------------------
- Method: StandardScaler
- Applied To: All numeric features

5. FEATURE SELECTION
------------------------------------------------
- Low variance features removed
- Highly correlated features removed
- SelectKBest used for importance ranking

6. FINAL STATUS
------------------------------------------------
- All features numeric: YES
- All features scaled: YES
- Missing values: 0
- Ready for Train/Test Split: YES

FINAL FEATURE COUNT:
{X.shape[1]}
"""

# Save log
with open(

    "feature_engineering_log.txt",

    "w"
) as file:

    file.write(feature_log)

print("✓ Feature Engineering Log Saved")






# ============================================================
# STEP 4 — TRAIN / TEST SPLIT
# Taxi Fare Prediction Project
# ============================================================

# Goal of Step 4:
# Split dataset into:
# - Training set (80%)
# - Test set (20%)
#
# Purpose:
# - Train model on train set
# - Evaluate honestly on unseen test set
# - Prevent data leakage
# - Ensure reproducibility

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import json


#train/ test split

from sklearn.model_selection import train_test_split

#optional kfold

from sklearn.model_selection import KFold

# ============================================================
# LOAD FEATURE ENGINEERED DATA
# ============================================================

#load engineered features

X = pd.read_csv("X_engineered.csv")

#load target variable

y = pd.read_csv("y_target.csv").squeeze()



print("\n================ DATA LOADED ================\n")

print(f"feature shape: {X.shape}")
print(f"target shape: {y.shape}")


# ============================================================
# UNDERSTAND THE PROBLEM TYPE
# ============================================================

# Taxi fare prediction predicts numeric prices
# Therefore:
# THIS IS A REGRESSION PROBLEM

print("\n================ PROBLEM TYPE ================\n")

print("✓ Regression Problem Detected")

print("\nTarget Statistics:\n")

print(y.describe())

# ============================================================
# IMPORTANT ML CONCEPT
# ============================================================

# Since this is regression:
# - NO stratification needed
# - We use regular random split
#
# Stratification is mainly for classification problems.

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# 80% -> training
# 20% -> testing

# random_state=42 ensures:
# same split every run (reproducibility)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


print("\n================ SPLIT COMPLETE ================\n")

print(f"Training Samples: {len(X_train)}")

print(f"Testing Samples: {len(X_test)}")

print(f"Total Samples: {len(X)}")


train_pct = (len(X_train) / len(X)) *100
test_pct = (len(X_test) / len( X)) *100

print("\nSplit Percentages:\n")

print(f"train set: {train_pct:2f}%")
print(f"test set: {test_pct:2f}%")

# ============================================================
# VERIFY NO OVERLAP
# ============================================================

# Train and test must NOT share rows

overlap = len(
    set(X_train.index)
    &
    set(X_test.index)
)

print("\n================ DATA LEAKAGE CHECK ================\n")

print(f"overlapping samples: {overlap}")
print(f"no leakege: {overlap == 0}")

# ============================================================
# VERIFY RANDOMIZATION
# ============================================================

print("\n================ RANDOMIZATION CHECK ================\n")

print("first 10 train indices: \n")
print(X_train.index[:10].tolist())
print("\nfirst 10 test indices: \n")
print(X_test.index[:10].tolist())

# ============================================================
# VERIFY TARGET DISTRIBUTION
# ============================================================

# For regression:
# compare train/test means

print("\n================ TARGET DISTRIBUTION ================\n")

train_mean = y_train.mean()
test_mean = y_test.mean()

print(f"train mean price: {train_mean: .2f}")
print(f"test mean price: {test_mean: .2f}")
mean_difference = abs(train_mean - test_mean)

print(f"difference: {mean_difference: .2f}")

similar_distribution = (
    mean_difference < y_train.std() *0.1
    
)

print(f"Similar Distribution: {similar_distribution}")



# ============================================================
# OPTIONAL — K-FOLD CROSS VALIDATION SETUP
# ============================================================

# KFold gives more robust evaluation later

print("\n================ K-FOLD SETUP ================\n")

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state= 42
    )

print("5 fold cross validation created")

# Show fold sizes
fold_number = 1

for train_idx, val_idx in kfold.split(X_train):

    print(

        f"Fold {fold_number}: "

        f"{len(train_idx)} train | "

        f"{len(val_idx)} validation"
    )
    
# ============================================================
# FINAL SPLIT QUALITY VERIFICATION
# ============================================================

print("\n================================================")

print("split quality verification")

print("\n================================================")

# ------------------------------------------------
# CHECK 1 — TRAIN/TEST RATIO
# ------------------------------------------------

ratio = len(X_train) / len(X_test)

print(f"\n train/test ratio: {ratio: .1f}:1")

# ------------------------------------------------
# CHECK 2 — REASONABLE SIZES
# ------------------------------------------------

print("\n✓ Dataset Sizes:")

print(f"train > 100 samples: {len(X_train) > 100}")
print(f"test > 20 samples: {len(X_test) > 20}")


# ------------------------------------------------
# CHECK 3 — NO LEAKAGE
# ------------------------------------------------
print("\n✓ No Data Leakage:")
print(overlap == 0)

# ------------------------------------------------
# CHECK 4 — TARGET DISTRIBUTION
# ------------------------------------------------

print("\n✓ Similar Target Distribution:")

print(similar_distribution)

# ------------------------------------------------
# CHECK 5 — FEATURE COUNTS
# ------------------------------------------------

print("\n✓ Feature Count:")
print(X_train.shape[1])

print("READY FOR STEP 5 — MODEL TRAINING")

print("================================================")

# ============================================================
# SAVE SPLIT DATASETS
# ============================================================

print("\n================ SAVING DATASETS ================\n")

#save trainning features

X_train.to_csv(
    "X_train.csv",
    index=False
)

#save testing features

X_test.to_csv(
    "X_test.csv", 
    index= False
)

#save training target

y_train.to_csv(
    "y_train.csv",
    index=False
)

#save test target

y_test.to_csv(
    "y_test.csv",
    index = False
)


print("✓ X_train.csv saved")

print("✓ X_test.csv saved")

print("✓ y_train.csv saved")

print("✓ y_test.csv saved")

# ============================================================
# SAVE SPLIT METADATA
# ============================================================

split_info= {
    "random_state": 42,

    "test_size": 0.2,

    "train_samples": len(X_train),

    "test_samples": len(X_test),

    "total_features": X.shape[1],

    "problem_type": "Regression",

    "shuffle": True,

    "data_leakage": False,

    "kfold_splits": 5
}

#save JSON metadata

with open(
    "split_info.json", 
    "w"
) as file:
    
    json.dump(
        split_info,
        file,
        indent=4
    )
    
    print("split_info.json saved")
    
    

# ============================================================
# STEP 6: MODEL EVALUATION
# Taxi Trip Price Prediction Project
# ============================================================

# Import libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Regression metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# LOAD TRAIN / TEST DATA
# ============================================================

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

linear_model = joblib.load("linear_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
gb_model = joblib.load("gradient_boosting_model.pkl")

# Store models inside dictionary
models = {
    "LinearRegression": linear_model,
    "RandomForest": rf_model,
    "GradientBoosting": gb_model
}

print("✓ Models loaded successfully")

# ============================================================
# EVALUATE MODELS
# ============================================================

results = {}

print("\n================ MODEL EVALUATION ================\n")

for model_name, model in models.items():

    print(f"\nEvaluating {model_name}...")
    print("-" * 50)

    # ========================================================
    # MAKE PREDICTIONS
    # ========================================================

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # ========================================================
    # TRAIN METRICS
    # ========================================================

    train_mae = mean_absolute_error(y_train, y_train_pred)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )

    train_r2 = r2_score(y_train, y_train_pred)

    # ========================================================
    # TEST METRICS
    # ========================================================

    test_mae = mean_absolute_error(y_test, y_test_pred)

    test_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )

    test_r2 = r2_score(y_test, y_test_pred)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results[model_name] = {
        "Train_MAE": train_mae,
        "Train_RMSE": train_rmse,
        "Train_R2": train_r2,
        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse,
        "Test_R2": test_r2,
        "Predictions": y_test_pred
    }

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nTRAIN RESULTS")

    print("MAE:", round(train_mae, 4))
    print("RMSE:", round(train_rmse, 4))
    print("R²:", round(train_r2, 4))

    print("\nTEST RESULTS")

    print("MAE:", round(test_mae, 4))
    print("RMSE:", round(test_rmse, 4))
    print("R²:", round(test_r2, 4))

    # ========================================================
    # OVERFITTING CHECK
    # ========================================================

    r2_gap = train_r2 - test_r2

    print("\nOVERFITTING CHECK")

    print("Train-Test R² Gap:",
          round(r2_gap, 4))

    if r2_gap > 0.1:
        print("⚠ Model may be overfitting")

    else:
        print("✓ Model generalizes well")

# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame({
    "Model": list(results.keys()),

    "Train_RMSE": [
        results[m]["Train_RMSE"]
        for m in results
    ],

    "Test_RMSE": [
        results[m]["Test_RMSE"]
        for m in results
    ],

    "Train_R2": [
        results[m]["Train_R2"]
        for m in results
    ],

    "Test_R2": [
        results[m]["Test_R2"]
        for m in results
    ]
})

# Sort by best R²
comparison_df = comparison_df.sort_values(
    by="Test_R2",
    ascending=False
)

print("\n================ MODEL COMPARISON ================\n")

print(comparison_df)

# Save comparison table
comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\n✓ Comparison table saved")

# ============================================================
# BEST MODEL
# ============================================================

best_model = comparison_df.iloc[0]["Model"]
best_score = comparison_df.iloc[0]["Test_R2"]

print("\n================ BEST MODEL ================\n")

print("Best model:", best_model)

print("best test r2: ", 
      round(best_score, 4))

# ============================================================
# VISUALIZATION
# ============================================================


plt.figure(figsize=(15, 5))

# ============================================================
# PLOT 1 — ACTUAL VS PREDICTED
# ============================================================
for i, (model_name, model) in enumerate(models.items(), 1):

    plt.subplot(1, 3, i)

    predictions = results[model_name]["Predictions"]

    # Scatter plot
    plt.scatter(
        y_test,
        predictions,
        alpha=0.5
    )

    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.title(model_name)
    
#save figure

plt.tight_layout()
plt.savefig("regression_evaluation.png")
#plt.show()


# ============================================================
# SAVE EVALUATION REPORT
# ============================================================

# ============================================================
# STEP 6: MODEL EVALUATION
# Taxi Trip Price Prediction Project
# ============================================================

# Import libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Regression metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# LOAD TRAIN / TEST DATA
# ============================================================

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

linear_model = joblib.load("linear_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
gb_model = joblib.load("gradient_boosting_model.pkl")

# Store models inside dictionary
models = {
    "LinearRegression": linear_model,
    "RandomForest": rf_model,
    "GradientBoosting": gb_model
}

print("✓ Models loaded successfully")

# ============================================================
# EVALUATE MODELS
# ============================================================

results = {}

print("\n================ MODEL EVALUATION ================\n")

for model_name, model in models.items():

    print(f"\nEvaluating {model_name}...")
    print("-" * 50)

    # ========================================================
    # MAKE PREDICTIONS
    # ========================================================

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # ========================================================
    # TRAIN METRICS
    # ========================================================

    train_mae = mean_absolute_error(y_train, y_train_pred)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )

    train_r2 = r2_score(y_train, y_train_pred)

    # ========================================================
    # TEST METRICS
    # ========================================================

    test_mae = mean_absolute_error(y_test, y_test_pred)

    test_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )

    test_r2 = r2_score(y_test, y_test_pred)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results[model_name] = {
        "Train_MAE": train_mae,
        "Train_RMSE": train_rmse,
        "Train_R2": train_r2,
        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse,
        "Test_R2": test_r2,
        "Predictions": y_test_pred
    }

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nTRAIN RESULTS")

    print("MAE:", round(train_mae, 4))
    print("RMSE:", round(train_rmse, 4))
    print("R²:", round(train_r2, 4))

    print("\nTEST RESULTS")

    print("MAE:", round(test_mae, 4))
    print("RMSE:", round(test_rmse, 4))
    print("R²:", round(test_r2, 4))

    # ========================================================
    # OVERFITTING CHECK
    # ========================================================

    r2_gap = train_r2 - test_r2

    print("\nOVERFITTING CHECK")

    print("Train-Test R² Gap:",
          round(r2_gap, 4))

    if r2_gap > 0.1:
        print("⚠ Model may be overfitting")

    else:
        print("✓ Model generalizes well")

# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame({
    "Model": list(results.keys()),

    "Train_RMSE": [
        results[m]["Train_RMSE"]
        for m in results
    ],

    "Test_RMSE": [
        results[m]["Test_RMSE"]
        for m in results
    ],

    "Train_R2": [
        results[m]["Train_R2"]
        for m in results
    ],

    "Test_R2": [
        results[m]["Test_R2"]
        for m in results
    ]
})

# Sort by best R²
comparison_df = comparison_df.sort_values(
    by="Test_R2",
    ascending=False
)

print("\n================ MODEL COMPARISON ================\n")

print(comparison_df)

# Save comparison table
comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\n✓ Comparison table saved")

# ============================================================
# BEST MODEL
# ============================================================

best_model = comparison_df.iloc[0]["Model"]

best_score = comparison_df.iloc[0]["Test_R2"]

print("\n================ BEST MODEL ================\n")

print("Best Model:", best_model)

print("Best Test R²:",
      round(best_score, 4))

# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(15, 5))

# ============================================================
# PLOT 1 — ACTUAL VS PREDICTED
# ============================================================

for i, (model_name, model) in enumerate(models.items(), 1):

    plt.subplot(1, 3, i)

    predictions = results[model_name]["Predictions"]

    # Scatter plot
    plt.scatter(
        y_test,
        predictions,
        alpha=0.5
    )

    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.title(model_name)

# Save figure
plt.tight_layout()

plt.savefig("regression_evaluation.png")

#plt.show()

print("\n✓ Visualization saved")

# ============================================================
# SAVE EVALUATION REPORT
# ============================================================

report = f"""
MODEL EVALUATION REPORT
=======================

Best Model: {best_model}

Best Test R²: {best_score}

Models Evaluated:
{list(models.keys())}

Taxi Price Prediction Project
"""



# ============================================================
# STEP 7: HYPERPARAMETER TUNING & DEPLOYMENT
# TAXI PRICE PREDICTION PROJECT (REGRESSION)
# BEST MODEL: GradientBoostingRegressor
# ============================================================

import pandas as pd
import numpy as np
import joblib
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# ============================================================
# LOAD BEST MODEL
# ============================================================

best_model_name = "GradientBoosting"



print(f"\n✓ Loaded best model: {best_model_name}")

# ============================================================
# LOAD ARTIFACTS
# ============================================================

feature_names = joblib.load("feature_names.pkl")

print(f"✓ Features loaded: {len(feature_names)}")

# ============================================================
# PROBLEM TYPE
# ============================================================

problem_type = "REGRESSION"

print(f"✓ Problem type: {problem_type}")

# ============================================================
# HYPERPARAMETER GRID
# ============================================================

from sklearn.ensemble import GradientBoostingRegressor

print("\n" + "=" * 80)
print("HYPERPARAMETER SEARCH SPACE")
print("=" * 80)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [2, 3, 4],
    'min_samples_split': [2, 5, 10],
}

for param, values in param_grid.items():
    print(f"{param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])

print(f"\nTotal combinations: {total_combinations}")

# ============================================================
# GRID SEARCH
# ============================================================

from sklearn.model_selection import GridSearchCV, KFold

print("\n" + "=" * 80)
print("RUNNING GRID SEARCH")
print("=" * 80)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

base_model = GradientBoostingRegressor(
    random_state=42
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='r2',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

start_time = time.time()

grid_search.fit(X_train, y_train)

elapsed_time = time.time() - start_time

print(f"\n✓ GridSearch completed in {elapsed_time:.2f} seconds")

# ============================================================
# BEST PARAMETERS
# ============================================================

best_params = grid_search.best_params_

print("\n" + "=" * 80)
print("BEST PARAMETERS")
print("=" * 80)

for param, value in best_params.items():
    print(f"{param}: {value}")

print(f"\nBest CV Score: {grid_search.best_score_:.4f}")

# ============================================================
# GRID SEARCH RESULTS
# ============================================================

results_df = pd.DataFrame(grid_search.cv_results_)

top_results = results_df.nlargest(
    5,
    'mean_test_score'
)

print("\n" + "=" * 80)
print("TOP 5 RESULTS")
print("=" * 80)

print(
    top_results[
        [
            'params',
            'mean_train_score',
            'mean_test_score'
        ]
    ].to_string(index=False)
)

results_df.to_csv(
    "gridsearch_results.csv",
    index=False
)

print("\n✓ gridsearch_results.csv saved")

# ============================================================
# OVERFITTING CHECK
# ============================================================

best_row = results_df.loc[
    results_df['rank_test_score'] == 1
]

train_score = best_row['mean_train_score'].values[0]
test_score = best_row['mean_test_score'].values[0]

gap = train_score - test_score

print("\n" + "=" * 80)
print("CROSS VALIDATION OVERFITTING CHECK")
print("=" * 80)

print(f"Train Score: {train_score:.4f}")
print(f"Validation Score: {test_score:.4f}")
print(f"Gap: {gap:.4f}")

if gap > 0.1:
    print("⚠ Overfitting detected")
else:
    print("✓ No significant overfitting")

# ============================================================
# FINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING FINAL MODEL")
print("=" * 80)

final_model = GradientBoostingRegressor(
    random_state=42,
    **best_params
)

start_time = time.time()

final_model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"✓ Final model trained in {training_time:.2f} seconds")

# ============================================================
# FINAL EVALUATION
# ============================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE")
print("=" * 80)

# Predictions
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

# Metrics
train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)

print("\nTRAIN RESULTS")
print(f"RMSE: {train_rmse:.4f}")

print("\nTEST RESULTS")
print(f"MAE:  {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")

# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("SAVING ARTIFACTS")
print("=" * 80)

joblib.dump(
    final_model,
    "final_model.pkl"
)

print("✓ final_model.pkl saved")

joblib.dump(
    feature_names,
    "final_feature_names.pkl"
)

print("✓ final_feature_names.pkl saved")

# ============================================================
# METADATA
# ============================================================

metadata = {
    "model_type": "GradientBoostingRegressor",
    "problem_type": "REGRESSION",
    "training_date": datetime.now().isoformat(),
    "best_hyperparameters": best_params,
    "metrics": {
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "r2_score": float(test_r2)
    },
    "feature_count": len(feature_names),
    "feature_names": feature_names
}

with open("final_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("✓ final_metadata.json saved")

# ============================================================
# CREATE INFERENCE SCRIPT
# ============================================================

print("\n" + "=" * 80)
print("CREATING INFERENCE SCRIPT")
print("=" * 80)

inference_code = '''
import joblib
import pandas as pd
import json

class TaxiPricePredictor:

    def __init__(
        self,
        model_path="final_model.pkl",
        metadata_path="final_metadata.json"
    ):

        self.model = joblib.load(model_path)

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["feature_names"]

        print("✓ Model loaded successfully")

    def predict(self, data):

        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Correct feature order
        data = data[self.feature_names]

        prediction = self.model.predict(data)

        return {
            "prediction": prediction.tolist()
        }


# Example usage
if __name__ == "__main__":

    predictor = TaxiPricePredictor()

    sample_data = {
        # Put your real features here
    }

    result = predictor.predict(sample_data)

    print(result)
'''


# ============================================================
# DEPLOYMENT GUIDE
# ============================================================

deployment_guide = f"""
DEPLOYMENT GUIDE
================

MODEL:
GradientBoostingRegressor

PROBLEM:
Taxi Price Prediction

FINAL PERFORMANCE:
RMSE: {test_rmse:.4f}
MAE: {test_mae:.4f}
R²: {test_r2:.4f}

FILES:
- final_model.pkl
- final_metadata.json
- final_feature_names.pkl
- inference.py

USAGE:

from inference import TaxiPricePredictor

predictor = TaxiPricePredictor()

result = predictor.predict(data)

"""


# ============================================================
# PROJECT SUMMARY
# ============================================================

summary = f"""
PROJECT COMPLETE
================

FINAL MODEL:
GradientBoostingRegressor

FINAL METRICS:
RMSE: {test_rmse:.4f}
MAE: {test_mae:.4f}
R²: {test_r2:.4f}

BEST PARAMETERS:
{best_params}

FILES CREATED:
✓ final_model.pkl
✓ final_metadata.json
✓ final_feature_names.pkl
✓ inference.py
✓ DEPLOYMENT_GUIDE.txt
✓ PROJECT_SUMMARY.txt
✓ gridsearch_results.csv

STATUS:
✓ READY FOR DEPLOYMENT
"""


print("\n" + "=" * 80)
print(summary)
print("=" * 80)



