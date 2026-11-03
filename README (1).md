# Power System Load Type Classification

Predicts `Load_Type` (Light / Medium / Maximum load) from historical power-system readings.

- `load_type_classification.ipynb` - full pipeline: cleaning, EDA, feature engineering, time-based validation, modelling, evaluation
- `load_data.csv` - provided dataset

**Run:** `pip install pandas numpy scikit-learn matplotlib jupyter`, open the notebook and run all cells.

**Validation:** chronological split; the last calendar month (Dec 2018) is the test set. Models are selected with `TimeSeriesSplit` on the training data only.

**Result on the test month:** accuracy 0.989, macro-F1 0.988 (Hist Gradient Boosting).
