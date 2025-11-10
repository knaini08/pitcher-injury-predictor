# Pitcher Injury Predictor

## Overview
This project uses machine learning to analyze MLB Statcast pitch-tracking data and FanGraphs injury logs (2021–2023).  
The goal was to explore whether changes in workload, pitch mix, and velocity could flag pitchers at higher risk of injury.

## Features
- Engineered features such as:
  - Pitch velocity (p95, average)
  - Pitch mix entropy (fastball, breaking, offspeed balance)
  - Workload (total pitches, games pitched, rest days)
  - Spin rate changes
- Gradient Boosting Classifier trained on combined dataset
- Achieved ROC AUC = 0.71

## Results
- Top predictive features:
  - Mix balance across pitch types
  - Workload spikes
  - High-velocity exposure
  - Rest days distribution
- ROC curve demonstrates predictive performance better than random guessing.
- This project is exploratory and not a medical tool, but it shows how public baseball data can reveal patterns for sports medicine.

## Tools & Libraries
- Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook
- Data sources: [Statcast](https://baseballsavant.mlb.com/statcast_search), [FanGraphs]([https://www.fangraphs.com/](https://www.fangraphs.com/roster-resource/injury-report?groupby=team&timeframe=all&season=2023))

## How to Use
1. Clone the repository.
2. Place CSV data files in the `/data` folder (not included here due to size).
3. Open `pitcherinjury.ipynb` in Jupyter Notebook.
4. Run cells to reproduce feature engineering, model training, and evaluation.

## Notes
- Data is publicly available; no private sources were used.
- All feature engineering, model design, and evaluation were done independently.
- This project is educational and exploratory, not predictive medical advice.

