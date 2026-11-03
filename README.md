# Power System Load Type Classification

Machine learning model that predicts the `Load_Type` of an industrial power system (**Light_Load**, **Medium_Load**, **Maximum_Load**) from historical readings.

## Results

Evaluated on the **last month of data (December 2018, 2,976 readings)**, which was never used for training or model selection.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **Hist Gradient Boosting (selected)** | **0.989** | **0.988** |
| Random Forest | 0.989 | 0.987 |
| Logistic Regression (baseline) | 0.730 | 0.571 |

Per-class results for the selected model:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Light_Load | 0.981 | 1.000 | 0.991 |
| Medium_Load | 1.000 | 0.976 | 0.988 |
| Maximum_Load | 1.000 | 0.970 | 0.985 |

## Repository contents

| File | Description |
|---|---|
| `load_type_classification.ipynb` | Full pipeline: cleaning, EDA, features, validation, modelling, evaluation |
| `load_data.csv` | Provided dataset (35,040 readings at 15-minute intervals, Jan-Dec 2018) |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook load_type_classification.ipynb
```

Run all cells. `load_data.csv` must be in the same folder as the notebook (or change `DATA_PATH` in the first data cell).

## Approach

**1. Data cleaning.** The raw file contains several problems, all handled row-wise so nothing leaks from the test set:
- Missing values in most numeric columns.
- About 10% of readings per column are corrupted (genuine values have at most 2 decimals; corrupted ones have long random decimals and are often far off, e.g. usage up to 435 kWh against a real maximum near 153). These are set to missing.
- Power factors above 100% are set to missing.
- `NSM` did not match the timestamps, so it is rebuilt from `Date`.
- One duplicate timestamp is removed.
- Remaining missing values are imputed inside the model pipelines (fitted on training data only); gradient boosting handles them natively.

**2. EDA.** The main finding is that `Load_Type` follows a **time-of-use schedule**: Light overnight and all day Sunday, with Medium and Maximum in fixed windows on weekdays and Saturdays. The Maximum window differs in winter (Nov-Feb), and public holidays behave like Sundays. Usage alone does not separate Medium from Maximum.

**3. Feature engineering.**
- Time of day (sin/cos, 15-minute slot), day of week, weekend flag, winter flag.
- Reactive-power ratios and net reactive power.
- Past-only usage context (previous reading, rolling mean, difference).
- `day_usage_mean`: average usage of the same calendar day, used as a holiday signal.

**4. Validation.**
- Chronological split: the last calendar month is the test set.
- Model selection with `TimeSeriesSplit` (5 folds) on the training data only.
- Class weights handle the class imbalance.
- Metrics: accuracy, precision, recall, F1 (macro and per class), confusion matrix.

## Key findings

- Calendar and time features alone reach 0.857 macro-F1. Adding the readings and same-day context lifts it to 0.988.
- Dropping `day_usage_mean` lowers macro-F1 to 0.973.
- All 33 misclassified readings fall on a single day, Saturday 1 December, which has few winter Saturdays in the training data.

## Limitations and next steps

- Only one test month is used, as the task specifies. Rolling-origin evaluation over several months would be more reliable.
- The first CV folds contain only early-year data, so fold scores vary a lot; the average is still useful for ranking models.
- `day_usage_mean` uses readings from later in the same day. This is fine for classifying recorded data, but a real-time system would need a causal substitute.
- A real holiday calendar and hyperparameter tuning are the most natural improvements.
