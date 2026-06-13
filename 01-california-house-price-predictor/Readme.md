# California House Price Predictor

End-to-end regression pipeline predicting California median house prices
using LightGBM, tracked with MLflow, served via FastAPI, containerized
with Docker.

---

## Results

| Metric | Score |
|---|---|
| RMSE | ~47,000 |
| MAE  | ~32,000 |
| R²   | ~0.836  |

---

## Stack

- **Model**: LightGBM (tuned with Optuna)
- **Experiment tracking**: MLflow
- **Feature engineering**: log transforms, geographic clustering, interaction terms
- **Serving**: FastAPI
- **Containerization**: Docker + Docker Compose

---

## Project Structure
├── notebooks/      EDA, feature engineering, model training
├── src/            Reusable preprocessing + training modules
├── api/            FastAPI inference endpoint
├── mlflow_runs/    Local MLflow experiment store

## How to Run

### Locally
```bash
pip install -r requirements.txt
python src/train.py
mlflow ui                        # view runs at localhost:5000
uvicorn api.main:app --reload    # inference at localhost:8000
```

### Docker
```bash
docker-compose up --build
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98,
       "AveBedrms": 1.02, "Population": 322, "AveOccup": 2.55,
       "Latitude": 37.88, "Longitude": -122.23}'
```

## Key Learnings

- Log-transforming skewed features (MedInc, Population) reduced RMSE by ~8%
- Geographic clustering (lat/lon → cluster label) was the top feature by SHAP importance
- LightGBM with early stopping outperformed XGBoost by ~3% RMSE on this dataset
