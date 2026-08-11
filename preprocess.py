import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/placement_predict_50k Dataset.csv"
TARGET_COLUMN = "PlacementStatus"
RANDOM_STATE = 42


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data/demonstrations", exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

# Save the REAL original shape before any modification
original_shape = df.shape


print("=" * 80)
print("PLACEMENT DATASET - COMPLETE PREPROCESSING")
print("=" * 80)

print("\nOriginal Dataset Shape:")
print(original_shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))


# ============================================================
# 1. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("1. MISSING VALUE ANALYSIS")
print("=" * 80)


missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)


missing_report = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percentage
})


print(missing_report)


missing_report.to_csv(
    "data/demonstrations/missing_value_report.csv"
)


# ============================================================
# 1A. MCAR / MAR / MNAR ANALYSIS
# ============================================================

print("\nMCAR / MAR / MNAR ANALYSIS")
print("-" * 50)

print("""
MCAR = Missing Completely At Random
MAR  = Missing At Random
MNAR = Missing Not At Random

Important:
The exact mechanism cannot be proven from the dataset alone.

Therefore, this program performs missingness-pattern analysis
and reports possible relationships instead of falsely claiming
that a column is definitely MCAR, MAR, or MNAR.
""")


missing_columns = [
    col
    for col in df.columns
    if df[col].isnull().sum() > 0
]


if len(missing_columns) == 0:

    print("No missing values detected.")

else:

    for col in missing_columns:

        missing_flag = df[col].isnull()

        print(f"\nColumn: {col}")

        for other_col in df.columns:

            if other_col == col:
                continue

            if pd.api.types.is_numeric_dtype(
                df[other_col]
            ):

                observed_mean = df.loc[
                    ~missing_flag,
                    other_col
                ].mean()

                missing_mean = df.loc[
                    missing_flag,
                    other_col
                ].mean()

                if (
                    pd.notna(observed_mean)
                    and
                    pd.notna(missing_mean)
                ):

                    difference = abs(
                        observed_mean -
                        missing_mean
                    )

                    if difference > 0:

                        print(
                            f"Possible relationship with "
                            f"{other_col}: "
                            f"observed mean="
                            f"{observed_mean:.2f}, "
                            f"missing-group mean="
                            f"{missing_mean:.2f}"
                        )


# ============================================================
# 2. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("2. DUPLICATE ANALYSIS")
print("=" * 80)


duplicate_count = df.duplicated().sum()


print(
    "Duplicate Rows Before Removal:",
    duplicate_count
)


df = df.drop_duplicates().copy()


print(
    "Shape After Duplicate Removal:",
    df.shape
)
# ============================================================
# 3. IDENTIFY TARGET
# ============================================================

if TARGET_COLUMN not in df.columns:

    raise ValueError(
        f"Target column '{TARGET_COLUMN}' "
        f"was not found in the dataset."
    )


y = df[TARGET_COLUMN].copy()


# ============================================================
# TARGET ENCODING IF TARGET IS CATEGORICAL
# ============================================================

target_encoder = None


if not pd.api.types.is_numeric_dtype(y):

    target_encoder = LabelEncoder()

    y = pd.Series(
        target_encoder.fit_transform(
            y.astype(str)
        ),
        index=df.index,
        name=TARGET_COLUMN
    )

    joblib.dump(
        target_encoder,
        "models/target_label_encoder.pkl"
    )


else:

    y = pd.to_numeric(
        y,
        errors="coerce"
    )


if y.isnull().any():

    raise ValueError(
        "Target column contains missing "
        "or non-convertible values."
    )


# Remove target from X

X = df.drop(
    columns=[TARGET_COLUMN]
).copy()


# ============================================================
# 4. NUMERIC / CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()


categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


print("\nNumeric Features:")
print(numeric_features)


print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 5. CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(6, 5))


sns.countplot(
    x=y
)


plt.title(
    "Placement Status Distribution"
)

plt.xlabel(
    TARGET_COLUMN
)

plt.ylabel(
    "Count"
)


plt.savefig(
    "plots/class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 6. HISTOGRAMS
# ============================================================

print("\nGenerating Histograms...")


for col in numeric_features:

    plt.figure(
        figsize=(6, 4)
    )

    sns.histplot(
        X[col],
        kde=True,
        bins=30
    )

    plt.title(
        f"{col} Distribution"
    )

    plt.savefig(
        f"plots/hist_{col}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. BOXPLOTS
# ============================================================

print("Generating Boxplots...")


for col in numeric_features:

    plt.figure(
        figsize=(6, 4)
    )

    sns.boxplot(
        x=X[col]
    )

    plt.title(
        f"{col} Boxplot"
    )

    plt.savefig(
        f"plots/box_{col}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. CORRELATION MATRIX
# ============================================================

print(
    "Generating Correlation Matrix..."
)


if len(numeric_features) > 0:

    correlation_df = X[
        numeric_features
    ].copy()

    correlation_df[
        TARGET_COLUMN
    ] = y.values

    correlation_matrix = (
        correlation_df.corr()
    )


    plt.figure(
        figsize=(16, 12)
    )


    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )


    plt.title(
        "Numerical Feature Correlation Matrix"
    )


    plt.savefig(
        "plots/correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("9. TRAIN / TEST SPLIT")
print("=" * 80)


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    "Training Shape:",
    X_train.shape
)


print(
    "Testing Shape:",
    X_test.shape
)
# ============================================================
# 10. IQR / TUKEY OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 80)
print("10. IQR / TUKEY OUTLIER DETECTION")
print("=" * 80)


train_numeric = X_train[
    numeric_features
].copy()


outlier_report = []


for column in numeric_features:

    Q1 = train_numeric[
        column
    ].quantile(0.25)


    Q3 = train_numeric[
        column
    ].quantile(0.75)


    IQR = Q3 - Q1


    lower = Q1 - 1.5 * IQR


    upper = Q3 + 1.5 * IQR


    outliers = (

        (train_numeric[column] < lower)

        |

        (train_numeric[column] > upper)

    )


    count = outliers.sum()


    percentage = (
        count /
        len(train_numeric)
    ) * 100


    outlier_report.append({

        "Feature": column,

        "Q1": Q1,

        "Q3": Q3,

        "IQR": IQR,

        "Lower_Bound": lower,

        "Upper_Bound": upper,

        "Outlier_Count": count,

        "Outlier_Percentage": percentage

    })


    print(
        f"{column}: "
        f"{count} outliers "
        f"({percentage:.2f}%)"
    )


outlier_report = pd.DataFrame(
    outlier_report
)


outlier_report.to_csv(
    "data/demonstrations/outlier_report.csv",
    index=False
)


# ============================================================
# 11. OUTLIER REMOVAL
# ============================================================

print("\nRemoving training outliers using IQR...")


X_train_clean = X_train.copy()

y_train_clean = y_train.copy()


# Start with every row considered valid

combined_mask = pd.Series(
    True,
    index=X_train_clean.index
)


for column in numeric_features:

    Q1 = X_train_clean[
        column
    ].quantile(0.25)


    Q3 = X_train_clean[
        column
    ].quantile(0.75)


    IQR = Q3 - Q1


    lower = Q1 - 1.5 * IQR


    upper = Q3 + 1.5 * IQR


    # Missing values are NOT treated as outliers.
    column_mask = (

        X_train_clean[column].isna()

        |

        (
            (X_train_clean[column] >= lower)
            &
            (X_train_clean[column] <= upper)
        )

    )


    combined_mask &= column_mask


# Apply exactly the same mask to X and y

X_train_clean = (
    X_train_clean
    .loc[combined_mask]
    .copy()
)


y_train_clean = (
    y_train_clean
    .loc[combined_mask]
    .copy()
)


print(
    "Training rows before "
    "outlier removal:",
    len(X_train)
)


print(
    "Training rows after "
    "outlier removal:",
    len(X_train_clean)
)


print(
    "Rows removed:",
    len(X_train) -
    len(X_train_clean)
)


print(
    "Test data was NOT modified "
    "during outlier removal."
)
# ============================================================
# 10. IQR / TUKEY OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 80)
print("10. IQR / TUKEY OUTLIER DETECTION")
print("=" * 80)


train_numeric = X_train[
    numeric_features
].copy()


outlier_report = []


for column in numeric_features:

    Q1 = train_numeric[
        column
    ].quantile(0.25)


    Q3 = train_numeric[
        column
    ].quantile(0.75)


    IQR = Q3 - Q1


    lower = Q1 - 1.5 * IQR


    upper = Q3 + 1.5 * IQR


    outliers = (

        (train_numeric[column] < lower)

        |

        (train_numeric[column] > upper)

    )


    count = outliers.sum()


    percentage = (
        count /
        len(train_numeric)
    ) * 100


    outlier_report.append({

        "Feature": column,

        "Q1": Q1,

        "Q3": Q3,

        "IQR": IQR,

        "Lower_Bound": lower,

        "Upper_Bound": upper,

        "Outlier_Count": count,

        "Outlier_Percentage": percentage

    })


    print(
        f"{column}: "
        f"{count} outliers "
        f"({percentage:.2f}%)"
    )


outlier_report = pd.DataFrame(
    outlier_report
)


outlier_report.to_csv(
    "data/demonstrations/outlier_report.csv",
    index=False
)


# ============================================================
# 11. OUTLIER REMOVAL
# ============================================================

print("\nRemoving training outliers using IQR...")


X_train_clean = X_train.copy()

y_train_clean = y_train.copy()


# Start with every row considered valid

combined_mask = pd.Series(
    True,
    index=X_train_clean.index
)


for column in numeric_features:

    Q1 = X_train_clean[
        column
    ].quantile(0.25)


    Q3 = X_train_clean[
        column
    ].quantile(0.75)


    IQR = Q3 - Q1


    lower = Q1 - 1.5 * IQR


    upper = Q3 + 1.5 * IQR


    # Missing values are NOT treated as outliers.
    column_mask = (

        X_train_clean[column].isna()

        |

        (
            (X_train_clean[column] >= lower)
            &
            (X_train_clean[column] <= upper)
        )

    )


    combined_mask &= column_mask


# Apply exactly the same mask to X and y

X_train_clean = (
    X_train_clean
    .loc[combined_mask]
    .copy()
)


y_train_clean = (
    y_train_clean
    .loc[combined_mask]
    .copy()
)


print(
    "Training rows before "
    "outlier removal:",
    len(X_train)
)


print(
    "Training rows after "
    "outlier removal:",
    len(X_train_clean)
)


print(
    "Rows removed:",
    len(X_train) -
    len(X_train_clean)
)


print(
    "Test data was NOT modified "
    "during outlier removal."
)
# ============================================================
# 18. TARGET ENCODING
# ============================================================

print("\n" + "=" * 80)
print("18. TARGET ENCODING - SAMPLE DEMONSTRATION")
print("=" * 80)


print("""
Target Encoding uses the target variable.

It is demonstrated on a SAMPLE only.

It is NOT used in the main preprocessing pipeline.

Reason:
Naive target encoding can cause target leakage if category
statistics are calculated using validation/test information.
""")


if (
    len(categorical_features) > 0
    and len(X_train_clean) > 0
):

    TARGET_SAMPLE_SIZE = min(
        5000,
        len(X_train_clean)
    )


    rng = np.random.RandomState(
        RANDOM_STATE
    )


    target_sample_indices = (
        rng.choice(

            X_train_clean.index,

            size=TARGET_SAMPLE_SIZE,

            replace=False

        )
    )


    target_sample = (
        X_train_clean
        .loc[target_sample_indices]
        .copy()
    )


    target_sample_y = (
        y_train_clean
        .loc[target_sample_indices]
        .copy()
    )


    target_encoded_sample = (
        target_sample.copy()
    )


    global_mean = (
        target_sample_y.mean()
    )


    for column in categorical_features:

        category_means = (

            pd.DataFrame({

                column:
                target_sample[column],

                "target":
                target_sample_y.values

            })

            .groupby(column)["target"]
            .mean()

        )


        target_encoded_sample[column] = (

            target_sample[column]

            .map(category_means)

            .fillna(global_mean)

        )


    target_encoded_sample.to_csv(

        "data/demonstrations/"
        "target_encoded_sample.csv",

        index=False

    )


    print(
        "Target Encoding Sample Size:",
        TARGET_SAMPLE_SIZE
    )


    print(
        "Target Encoding demonstration completed."
    )


# ============================================================
# 19. ROW DELETION
# ============================================================

print("\n" + "=" * 80)
print("19. ROW DELETION - SAMPLE DEMONSTRATION")
print("=" * 80)


print("""
Row deletion is demonstrated on a SAMPLE only.

Deleting every row containing a missing value from a large
real-world dataset can unnecessarily reduce the dataset.
""")


ROW_SAMPLE_SIZE = min(
    5000,
    len(df)
)


row_sample = df.sample(

    ROW_SAMPLE_SIZE,

    random_state=RANDOM_STATE

)


before_rows = len(
    row_sample
)


row_deleted_sample = (
    row_sample.dropna()
)


after_rows = len(
    row_deleted_sample
)


print(
    "Sample used:",
    ROW_SAMPLE_SIZE
)


print(
    "Rows before deletion:",
    before_rows
)


print(
    "Rows after deletion:",
    after_rows
)


print(
    "Rows deleted:",
    before_rows -
    after_rows
)


row_deleted_sample.to_csv(

    "data/demonstrations/"
    "row_deletion_sample.csv",

    index=False

)


# ============================================================
# 20. VARIANCE SHRINKAGE / MISSING INDICATOR
# ============================================================

print("\n" + "=" * 80)
print("20. VARIANCE SHRINKAGE / MISSING INDICATOR")
print("=" * 80)


print("""
Median/mean imputation can reduce the natural variance of
a feature because multiple missing observations receive
the same replacement value.

A missing-indicator feature preserves information about
which values were originally missing.
""")


variance_demo = df.copy()


for column in numeric_features:

    if (
        variance_demo[column]
        .isnull()
        .sum() > 0
    ):

        indicator_column = (
            f"{column}_was_missing"
        )


        variance_demo[
            indicator_column
        ] = (

            variance_demo[column]
            .isnull()
            .astype(int)

        )


        variance_demo[column] = (

            variance_demo[column]
            .fillna(
                variance_demo[column].median()
            )

        )


        print(
            f"Missing indicator created for: "
            f"{column}"
        )


variance_demo.to_csv(

    "data/demonstrations/"
    "variance_missing_indicator.csv",

    index=False

)
# ============================================================
# 21. LOW-VARIANCE FEATURE REMOVAL
# ============================================================

print("\n" + "=" * 80)
print("21. LOW-VARIANCE FEATURE CHECK")
print("=" * 80)


print("""
Features with almost no variation provide little information.

VarianceThreshold is demonstrated here and is also included
in the final numeric preprocessing pipeline.
""")


if len(numeric_features) > 0:

    numeric_demo = (
        X_train_clean[
            numeric_features
        ].copy()
    )


    numeric_demo = (
        numeric_demo.fillna(
            numeric_demo.median()
        )
    )


    variance_selector = (
        VarianceThreshold(
            threshold=0.01
        )
    )


    variance_selector.fit(
        numeric_demo
    )


    selected_features = list(

        numeric_demo.columns[
            variance_selector
            .get_support()
        ]

    )


    removed_features = [

        col

        for col in numeric_demo.columns

        if col not in selected_features

    ]


    print(
        "Features retained:"
    )

    print(
        selected_features
    )


    print(
        "Low-variance features removed:"
    )

    print(
        removed_features
    )


# ============================================================
# 22. STRATEGIC PREPROCESSING SELECTION
# ============================================================

print("\n" + "=" * 80)
print("22. STRATEGIC PREPROCESSING SELECTION")
print("=" * 80)


strategy_table = pd.DataFrame({

    "Feature_Type": [

        "Numeric",

        "Nominal Categorical",

        "Ordinal Categorical",

        "High-Cardinality Categorical",

        "Tree-Based Model",

        "Linear/KNN/SVM Model"

    ],


    "Missing_Value_Method": [

        "Median / Mean",

        "Most Frequent",

        "Most Frequent",

        "Most Frequent / Appropriate",

        "Imputation Required",

        "Imputation Required"

    ],


    "Encoding": [

        "None",

        "One-Hot",

        "Ordinal",

        "Target Encoding with leakage control",

        "Model dependent",

        "One-Hot / appropriate encoding"

    ],


    "Scaling": [

        "Usually required",

        "Usually not required",

        "Depends on model",

        "Depends on model",

        "Usually not required",

        "Usually required"

    ]

})


print(
    strategy_table
)


strategy_table.to_csv(

    "data/demonstrations/"
    "strategic_preprocessing_selection.csv",

    index=False

)
# ============================================================
# 23. COMPLETE LEAK-PROOF PREPROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 80)
print("23. COMPLETE LEAK-PROOF PREPROCESSING PIPELINE")
print("=" * 80)


# ============================================================
# NUMERIC PIPELINE
# ============================================================

numeric_pipeline = Pipeline(

    steps=[

        # ----------------------------------------------------
        # MEDIAN IMPUTATION
        # + MISSING INDICATORS
        # ----------------------------------------------------

        (
            "imputer",

            SimpleImputer(

                strategy="median",

                add_indicator=True

            )
        ),


        # ----------------------------------------------------
        # LOW VARIANCE FEATURE REMOVAL
        # ----------------------------------------------------

        (
            "variance_filter",

            VarianceThreshold(

                threshold=0.01

            )
        ),


        # ----------------------------------------------------
        # Z-SCORE STANDARDIZATION
        # ----------------------------------------------------

        (
            "scaler",

            StandardScaler()

        )

    ]

)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(

    steps=[

        # ----------------------------------------------------
        # MODE / MOST-FREQUENT IMPUTATION
        # ----------------------------------------------------

        (
            "imputer",

            SimpleImputer(

                strategy="most_frequent"

            )

        ),


        # ----------------------------------------------------
        # ONE-HOT ENCODING
        # ----------------------------------------------------

        (
            "onehot",

            OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False

            )

        )

    ]

)


# ============================================================
# COMBINE NUMERIC + CATEGORICAL PIPELINES
# ============================================================

transformers = []


if len(numeric_features) > 0:

    transformers.append(

        (
            "numeric",

            numeric_pipeline,

            numeric_features

        )

    )


if len(categorical_features) > 0:

    transformers.append(

        (
            "categorical",

            categorical_pipeline,

            categorical_features

        )

    )


preprocessor = ColumnTransformer(

    transformers=transformers,

    remainder="drop"

)


# ============================================================
# 24. FIT ONLY ON TRAINING DATA
# ============================================================

print(
    "\nFitting preprocessor "
    "ONLY on training data..."
)


X_train_processed = (

    preprocessor.fit_transform(

        X_train_clean

    )

)


# ============================================================
# 25. TRANSFORM TEST DATA
# ============================================================

print(
    "Transforming test data..."
)


X_test_processed = (

    preprocessor.transform(

        X_test

    )

)


print(
    "No fitting was performed "
    "on test data."
)


# ============================================================
# 26. FEATURE NAMES
# ============================================================

feature_names = (

    preprocessor
    .get_feature_names_out()

)


# ============================================================
# 27. CONVERT TO DATAFRAME
# ============================================================

X_train_processed = pd.DataFrame(

    X_train_processed,

    columns=feature_names,

    index=X_train_clean.index

)


X_test_processed = pd.DataFrame(

    X_test_processed,

    columns=feature_names,

    index=X_test.index

)


print(
    "\nProcessed Training Shape:",
    X_train_processed.shape
)


print(
    "Processed Testing Shape:",
    X_test_processed.shape
)


print(
    "Final Number of Features:",
    len(feature_names)
)


# ============================================================
# 28. SAVE PREPROCESSOR
# ============================================================

joblib.dump(

    preprocessor,

    "models/preprocessor.pkl"

)


print(
    "\nSaved: models/preprocessor.pkl"
)


# ============================================================
# 29. SAVE PROCESSED DATASETS
# ============================================================

X_train_processed.to_csv(

    "data/X_train.csv",

    index=False

)


X_test_processed.to_csv(

    "data/X_test.csv",

    index=False

)


y_train_clean.to_csv(

    "data/y_train.csv",

    index=False,

    header=[TARGET_COLUMN]

)


y_test.to_csv(

    "data/y_test.csv",

    index=False,

    header=[TARGET_COLUMN]

)


# ============================================================
# 30. CREATE FINAL DATASET
# ============================================================

final_train = (
    X_train_processed.copy()
)


final_train[
    TARGET_COLUMN
] = y_train_clean.to_numpy()


final_test = (
    X_test_processed.copy()
)


final_test[
    TARGET_COLUMN
] = y_test.to_numpy()


final_df = pd.concat(

    [
        final_train,
        final_test
    ],

    axis=0

)


final_df.to_csv(

    "data/"
    "placement_predict_final_cleaned.csv",

    index=False

)


# ============================================================
# 31. FIVE-STEP PREPROCESSING ROADMAP
# ============================================================

print("\n" + "=" * 80)
print("FIVE-STEP PREPROCESSING ROADMAP")
print("=" * 80)


print("""
STEP 1 - FEATURE SCALING
    StandardScaler
    MinMaxScaler
    RobustScaler

STEP 2 - CATEGORICAL ENCODING
    One-Hot Encoding
    Ordinal Encoding
    Target Encoding demonstration

STEP 3 - MISSING VALUE HANDLING
    Mean
    Median
    Mode / Most Frequent
    Missing Indicators
    MCAR / MAR / MNAR Analysis
    Row Deletion demonstration

STEP 4 - OUTLIER HANDLING
    IQR / Tukey Fences
    Training-set outlier removal

STEP 5 - FEATURE SELECTION
    VarianceThreshold
    Low-variance feature detection
""")


# ============================================================
# 32. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 80)


print(
    "Original Dataset Shape:",
    original_shape
)


print(
    "After Duplicate Removal:",
    df.shape
)


print(
    "Training Shape:",
    X_train_processed.shape
)


print(
    "Testing Shape:",
    X_test_processed.shape
)


print(
    "Final Number of Features:",
    len(feature_names)
)


print("""
MAIN LEAK-PROOF PIPELINE:

Training Data
      |
      v
IQR Outlier Removal
      |
      v
Numeric Median Imputation
      |
      v
Missing Indicators
      |
      v
Low Variance Filtering
      |
      v
Standard Scaling
      |
      v
Categorical Mode Imputation
      |
      v
One-Hot Encoding
      |
      v
Final Training Features


TEST DATA:

Test Data
      |
      v
Same Fitted Preprocessor
      |
      v
Final Test Features

IMPORTANT:
The preprocessor is FIT only on training data.
The test data is only TRANSFORMED.
""")


# ============================================================
# SAVED FILES
# ============================================================

print("\nSaved Files")
print("-" * 50)

print(
    "plots/class_distribution.png"
)

print(
    "plots/correlation_matrix.png"
)

print(
    "plots/hist_*.png"
)

print(
    "plots/box_*.png"
)

print(
    "data/X_train.csv"
)

print(
    "data/X_test.csv"
)

print(
    "data/y_train.csv"
)

print(
    "data/y_test.csv"
)

print(
    "data/placement_predict_final_cleaned.csv"
)

print(
    "models/preprocessor.pkl"
)

print(
    "models/standard_scaler.pkl"
)

print(
    "models/minmax_scaler.pkl"
)

print(
    "models/robust_scaler.pkl"
)

print(
    "data/demonstrations/"
)


print("\n" + "=" * 80)
print("ALL PREPROCESSING TECHNIQUES COMPLETED")
print("=" * 80)