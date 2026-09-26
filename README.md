# Predicting Sales from Campaign Data

An end-to-end machine-learning case study that predicts product sales from influencer campaign attributes. The project turns messy campaign records into a reusable regression pipeline and explains each decision through short code cells, visualizations, experiments, and observations.

**[Explore the complete notebook](Predicting_Sales_from_Campaign_Data.ipynb)**

The notebook is written as study notes: readers can follow the analysis from raw data to final predictions without needing separate preprocessing scripts or previously saved models. It includes **24 inline visualizations**, a glossary, worked examples, and learning checkpoints.

## Problem statement

Marketing teams need to estimate campaign sales when comparing influencers, planning campaigns, and making inventory decisions. This project investigates how four campaign attributes—followers, engagement rate, advertising spend, and content quality—relate to the number of units sold.

The objectives are to:

- Clean inconsistent formats, missing measurements, mixed units, and extreme values.
- Explore which inputs carry useful predictive information.
- Compare regression approaches using consistent validation folds.
- Tune promising models and evaluate their performance on unseen campaigns.
- Provide a simple function for predicting sales from future messy inputs.

This is a predictive analysis. It does not establish the causal sales lift or financial return from changing campaign spend.

## Results

The selected model is **Huber regression**, with `epsilon=1.2` and `alpha=0.0001`. Huber and Ridge regression are practically tied in development cross-validation; the more flexible models do not improve the validation results.

| Metric | Mean-prediction baseline | Selected Huber model |
|---|---:|---:|
| Holdout RMSE | 2,820.0 units | **2,156.6 units** |
| Holdout MAE | 2,249.6 units | **1,691.7 units** |
| Holdout R² | −0.001 | **0.415** |

The selected model reduces holdout RMSE by approximately **23.5%** compared with the baseline. These results indicate moderate predictive accuracy, with substantial uncertainty for individual campaigns. R² is a measure of explained variation, not a percentage of correct predictions.

### Important validation finding

All **160 training records with originally missing follower counts** fell into the initial holdout. Development and calibration contained none of these records. This imbalance made the holdout harder and limited how well the calibrated intervals transferred to incomplete campaigns.

The original holdout score is retained. A separate availability-balanced analysis of the already selected model reports mean RMSE of **1,999.6 units** and mean R² of **0.492** across 15 folds. This is a **post-selection sensitivity analysis**, not an independent replacement for the holdout result.

The supplied test set has no sales labels, so its RMSE and R² cannot be measured.

## Dataset

The analysis uses two supplied CSV files:

| File | Campaigns | Purpose |
|---|---:|---|
| `messy_train_data.csv` | 8,000 | Model development, calibration, and evaluation |
| `messy_test_data.csv` | 2,000 | Final sales predictions |

| Column | Description | Treatment |
|---|---|---|
| `Followers` | Influencer audience size | Predictor; resolve units and invalid values |
| `EngagementRate (%)` | Engagement in percentage points | Predictor; normalize percent strings |
| `AdSpend (GBP)` | Campaign spending in pounds | Predictor; normalize currency strings |
| `ContentQuality` | Content rating on a 1–10 scale | Predictor; enforce the stated range |
| `Sales (Units)` | Units sold | Target; available only in the training file |
| `ID` | Campaign identifier | Preserve prediction-to-campaign alignment |
| `Timestamp` | Campaign end date | Explore temporal patterns and backtest |
| `Notes` | Administrative text | Exclude from predictive features |

The data includes missing cells, currency and percentage symbols, negative values, extreme corruptions, and follower counts recorded on different scales. Small bare follower values occur in 2.5% of training rows and 10% of test rows, making consistent unit handling especially important.

The CSVs were supplied with the case study; no public download URL or redistribution license is included in the project materials. If the datasets are not distributed with this repository, obtain them from the case-study provider and place them beside the notebook. Publish the supplied datasets only where redistribution is permitted.

## Approach

### 1. Understand and audit the data

Inspect the schema, missing values, target validity, duplicate records, identifier overlap, and date coverage. Preserve the original data and retain all valid sales labels.

### 2. Clean and preprocess inside the pipeline

- Parse formats such as `£5,000`, `3.2%`, and explicit follower suffixes such as `120k`.
- Compare three interpretations of small bare follower counts: keep them, mark them missing, or convert from thousands.
- Treat invalid measurements and extreme follower/spend values as missing.
- Learn generous outlier thresholds from each fitting subset using `Q3 + 10 × IQR`.
- Impute missing inputs with training-fold medians and add missing-value indicators.
- Learn scaling parameters within each training fold.

Bare engagement values remain in percentage points. The selected follower-scale rule is specific to this dataset and should be checked before using a new data source.

### 3. Explore relationships and engineer candidate features

Use distributions, scatterplots, correlation analysis, monthly summaries, and grouped comparisons to investigate sales patterns. Test log transforms, estimated engaged audience, interaction terms, and spend per follower against the original-feature baseline.

### 4. Compare and tune models

The notebook evaluates ten initial approaches on identical five-fold development splits:

1. Mean-prediction baseline
2. Ridge regression with original predictors
3. Ridge regression with engineered predictors
4. Quadratic Ridge regression
5. Spline Ridge regression
6. Huber regression
7. Random Forest regression
8. Extra Trees regression
9. Histogram gradient boosting
10. Log-target Ridge regression

The two strongest non-baseline families undergo grid-search tuning. A fixed equal-weight blend is also evaluated. Selection uses development CV RMSE; the final holdout does not guide tuning.

### 5. Evaluate and investigate reliability

| Labelled subset | Campaigns | Role |
|---|---:|---|
| Development | 5,120 | Exploration, model comparison, and tuning |
| Calibration | 1,280 | Estimate prediction-interval width |
| Holdout | 1,600 | Evaluate the frozen model |

Evaluation includes RMSE, MAE, R², residual analysis, bootstrap score intervals, input-quality subgroup analysis, permutation importance, average model-response plots, chronological backtesting, and formatting/corruption stress tests.

The development-fitted model's nominal 90% prediction intervals achieve **88.4% holdout coverage**. Coverage differs substantially by input completeness: approximately **90.6%** for complete valid inputs and **77.1%** for campaigns requiring imputation. Global coverage should therefore not be treated as a guarantee for an incomplete campaign.

### 6. Refit and predict

After evaluation, refit the selected specification on all 8,000 labelled campaigns and predict the 2,000 test campaigns. Preserve IDs and row order, and display forecasts and input-quality flags directly in the notebook.

The final refit produces point predictions. The earlier calibrated interval belongs to the development-fitted model and is not attached to the refitted model.

## Key findings

- **Followers and advertising spend carry the most predictive information**, followed by content quality and engagement in holdout permutation importance.
- **More complexity did not improve validation accuracy.** Tree ensembles achieved lower training error but higher validation error than the leading additive models.
- **Input completeness matters.** Holdout RMSE was approximately 1,979 units for complete valid inputs and 2,910 units for campaigns requiring imputation.
- **Cleaning assumptions need validation.** Converting ambiguous follower counts from thousands improved development CV compared with keeping them literally or treating them as missing.
- **The best model's tiny CV lead is not a major performance breakthrough.** Huber, Ridge, and their blend are practically tied relative to fold-to-fold variation.

## Run the notebook

The recorded analysis used **Python 3.12**. Package versions are listed in [`requirements.txt`](requirements.txt).

From the repository root, create and activate a fresh environment:

```bash
python3.12 -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the analysis dependencies and a notebook interface:

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
python -m notebook
```

Open `Predicting_Sales_from_Campaign_Data.ipynb`, select the environment's Python kernel, and run the cells from top to bottom. Keep both CSVs in the same folder as the notebook. On Windows, `py -3.12` can be used instead of `python3.12` when creating the environment.

The notebook already contains executed outputs for reading. Rerunning it computes tables, plots, models, and predictions in memory. It does not require or automatically create an `outputs/` directory, exported charts, model files, or helper modules.

### Predict new campaigns

After running the notebook, use its fitted model and prediction function:

```python
new_campaigns = pd.DataFrame({
    "Followers": ["120k"],
    "EngagementRate (%)": ["3.2%"],
    "AdSpend (GBP)": ["£5,000"],
    "ContentQuality": [8],
})

predict_campaign_sales(new_campaigns, final_model)
```

The function returns predicted sales and review flags for imputed inputs or assumed follower scales. All four predictor columns are required; individual cells may be missing. The function applies the fitted pipeline without retraining it on the new campaigns.

## Repository contents

The core project consists of:

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── Predicting_Sales_from_Campaign_Data.ipynb
├── messy_train_data.csv    # When redistribution is permitted
└── messy_test_data.csv     # When redistribution is permitted
```

The approach PDF and `Practice1.ipynb`, `Practice2.ipynb`, and `Practice3.ipynb` informed the scope and explanatory style. They are not runtime dependencies. The current notebook contains all preprocessing and modeling code inline.

## Limitations and next steps

The model assumes the four predictors are available at prediction time. Historical actual spend, engagement, and quality may differ from pre-launch estimates; prospective campaign planning needs validation using genuinely available inputs.

There is no creator identifier for grouped validation, and content-quality ratings may vary across raters. The follower-scale assumption may not transfer to genuinely small creators. Chronological and availability-balanced checks are diagnostic; they do not replace a new prospective evaluation.

Useful next steps are to standardize source units, improve collection of missing reach and spend measurements, gather creator/product identifiers and campaign duration, and evaluate on new campaigns. Randomized campaign experiments are needed before interpreting model associations as causal sales lift or incremental ROI.
