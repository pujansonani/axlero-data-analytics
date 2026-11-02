# Member 2: Analysis & Visualization

Final EDA figures, short evidence-based observations and figure captions for the
Credit Risk Scoring and Loan Default Analysis project.

## Files
| File | What it is |
|---|---|
| `EDA_observations.md` | Observations, figure captions, limitations and data notes (read this first) |
| `figures/` | 9 final figures (PNG) |
| `member2_eda_visualization.ipynb` | Notebook that creates every figure step by step |
| `make_figures.py` | Same code as a script |

## How to reproduce
```bash
pip install pandas numpy matplotlib seaborn
# from the repository root; point DATA_PATH to the validated dataset
DATA_PATH=path/to/final_validated_dataset.csv python member2_analysis_visualization/make_figures.py
```
On Windows PowerShell:
```powershell
$env:DATA_PATH="path\to\final_validated_dataset.csv"; python member2_analysis_visualization\make_figures.py
```

## Notes
- `loan_status` = 1 is treated as default (confirm with the data member).
- Observations describe patterns (association). They do not claim that any factor causes default.
