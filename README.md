# Aircraft Engine Remaining Useful Life Prediction

Predicting how many operating cycles a turbofan engine has left before failure,
using machine learning and deep learning on NASA's C-MAPSS dataset (FD001).

## Problem
Fixed maintenance schedules replace healthy parts too early, while unexpected
failures are dangerous and costly. This project predicts Remaining Useful Life
(RUL) from engine sensor data so maintenance can be planned when it is needed.

## Dataset
NASA C-MAPSS turbofan engine simulation data, subset FD001: 100 training engines
run to failure, 100 test engines, 21 sensors.
Download: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
Place train_FD001.txt, test_FD001.txt and RUL_FD001.txt in Colab's /content folder.

## Approach
- Removed 7 constant sensors, kept 14 informative ones
- Capped RUL at 125 cycles (piecewise-linear target)
- Split by engine to avoid data leakage; scaled using training engines only
- Classical models: baseline, Linear Regression, Random Forest, XGBoost
- Deep learning: 1D CNN, LSTM, GRU on 30-cycle sliding windows
- Interpretation with SHAP; risk categories and early failure warnings
- Interactive Gradio interface

## Results (100 test engines)
| Model | MAE | RMSE | NASA score |
|---|---|---|---|
| Naive baseline | [ ] | [ ] | [ ] |
| Linear Regression | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] |
| XGBoost | [ ] | [ ] | [ ] |
| 1D CNN | [ ] | [ ] | [ ] |
| LSTM | [ ] | [ ] | [ ] |
| GRU | [ ] | [ ] | [ ] |

![Predicted vs actual RUL](images/predicted_vs_actual.png)
![SHAP sensor importance](images/shap_importance.png)

## How to run
1. Open a notebook from the notebooks folder in Google Colab.
2. Upload the three FD001 data files.
3. Run all cells.

## Project structure
    notebooks/   Colab notebooks (data + classical models, deep learning, interface)
    images/      result plots
    requirements.txt

## Author
Kiesha Kapoor ·
