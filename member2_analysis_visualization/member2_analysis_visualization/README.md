# Member 2 — Analysis & Visualization

**Project:** Credit Risk Scoring and Loan Default Analysis

This folder contains Member 2's EDA notebook, figure-generation script, written observations, validated dataset, and nine final visualizations.

## Files

| File / folder | Purpose |
|---|---|
| `final_validated_dataset.csv` | Validated input dataset (32,416 rows, 12 columns) |
| `make_figures.py` | Generates all nine EDA figures and prints the supporting rate tables |
| `member2_eda_visualization.ipynb` | Step-by-step notebook for the same nine figures |
| `EDA_observations.md` | Figure captions, findings, limitations, and data notes |
| `figures/` | Nine generated PNG visualizations |

## Requirements

Install the libraries once:

```bash
pip install pandas numpy matplotlib seaborn
```

## Run the Python script

From the repository root in Windows PowerShell:

```powershell
python member2_analysis_visualization\make_figures.py
```

Or, after opening a terminal in this folder:

```powershell
python make_figures.py
```

The script automatically looks for `final_validated_dataset.csv` beside itself. If the dataset is stored elsewhere, set its full path first:

```powershell
$env:DATA_PATH = "C:\full\path\to\final_validated_dataset.csv"
python member2_analysis_visualization\make_figures.py
```

The nine PNG files are saved in this folder's `figures` directory. The script checks that the required columns exist and that `loan_status` contains only 0 and 1 before creating charts.

## Run the Jupyter notebook

1. Open `member2_eda_visualization.ipynb`.
2. Run **Kernel → Restart & Run All** (or run all cells from top to bottom).
3. Confirm that all nine figures display without errors.

The notebook searches for the CSV in its current folder and in the repository's `member2_analysis_visualization` folder. If the CSV is elsewhere, set `DATA_PATH` in the first code cell to its full path.

## Figure list

1. Loan status distribution
2. Default rate by loan grade
3. Default rate by loan burden (loan amount as a share of income)
4. Prior default and home ownership
5. Default rate by loan intent
6. Applicant income
7. Loan amount and interest rate
8. Age and employment length
9. Spearman correlation heatmap

## Interpretation notes

- The EDA currently treats `loan_status = 1` as default and `loan_status = 0` as repaid. Confirm this target definition with the dataset owner / Member 1 before final modelling.
- The dataset does not state the currency for income or loan amount, so the figures use the raw numeric values and do not add a currency symbol.
- These are descriptive associations in historical data; they do not establish causation.
- Grade and interest rate are lender-assigned variables and may already reflect the lender's risk assessment.
