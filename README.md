## 🌍 NO2 Satellite Data Downscaling Application

### Project Overview
An AI/ML-powered Streamlit application that converts coarse-resolution satellite NO2 measurements into fine-resolution maps. It leverages standard Python data science libraries, persists uploaded data in a database, visualizes inputs and outputs with interactive charts, and reports model performance metrics.

Primary users: researchers, environmental agencies, and policy makers who need neighborhood-level air quality insights.

Key goals:
- Generate high-resolution NO2 maps from coarse satellite rasters
- Validate results with independent ground station data
- Provide an intuitive, end-to-end workflow in the browser

---

## 🔧 Tech Stack

- Frontend framework: Streamlit (Python-based UI)
- Visualization: Plotly (interactive maps/charts), custom CSS (`styles.css`)
- Geospatial IO: Rasterio (reads GeoTIFFs)
- Data processing: NumPy, Pandas
- Machine learning: scikit-learn (Random Forest, scaling, train/validation split)
- Database/ORM: SQLAlchemy (default SQLite; optional PostgreSQL via `psycopg2-binary`)
- Config: Environment variables (optional), but defaults provided (no `.env` required)
- Python version: 3.11+ (tested on 3.13)

---

## 🧭 Architecture

- `main.py`: Streamlit app
  - Uploads satellite GeoTIFF and ground CSV
  - Handles missing data interpolation
  - Trains model and generates downscaled map
  - Visualizes original and downscaled maps
  - Computes MSE, RMSE, R²; offers CSV download
  - Initializes DB and saves satellite/ground data

- `model.py`: NO2DownscalingModel
  - Prepares spatial features (normalized i/j positions + NO2 value)
  - Trains `RandomForestRegressor`
  - Predicts on a higher-resolution grid (configurable scale factor)

- `utils.py`: Utilities
  - `load_satellite_data` (rasterio), `load_ground_data` (pandas CSV)
  - `handle_missing_data` (interpolation or mean-fill)
  - `create_no2_map` (Plotly imshow)
  - `calculate_metrics` (MSE, RMSE, R²)
  - DB helpers to save satellite pixels and ground rows

- `database.py`: SQLAlchemy models and session
  - Models: `SatelliteData`, `GroundMeasurement`
  - Uses `DATABASE_URL` if set; otherwise defaults to SQLite `sqlite:///./no2_data.db`
  - `init_db()` creates tables on startup

- `styles.css`: Custom styling for Streamlit layout

---

## 📂 Project Structure

```
scale/
├── main.py              # Streamlit app entry
├── model.py             # ML model (Random Forest downscaling)
├── utils.py             # IO, preprocessing, metrics, plotting
├── database.py          # SQLAlchemy models + engine/session
├── styles.css           # UI styles
├── pyproject.toml       # Dependencies
├── README.md            # Documentation (this file)
└── no2_data.db          # SQLite DB (auto-created)
```

---

## ▶️ Running the App

Prerequisites:
- Python 3.11+

Install dependencies:
```bash
pip install numpy pandas plotly psycopg2-binary python-dotenv rasterio scikit-learn sqlalchemy streamlit
```

Run:
```bash
python -m streamlit run main.py
```

Open in a browser:
- Local: http://localhost:8501

Notes:
- No `.env` file is required. The app defaults to SQLite at `sqlite:///./no2_data.db`.
- If you want PostgreSQL, set `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@host/db`).

---

## 📥 Using the App

1) Upload Satellite GeoTIFF
- The app reads band 1 via rasterio and extracts pixel values, transform, and CRS
- Missing values are interpolated by default
- Original map is displayed (Plotly)
- Pixels are saved to DB (`SatelliteData`) with timestamp/lat/lon/value/resolution

2) (Optional) Upload Ground CSV
- Expected columns: `latitude`, `longitude`, `no2_value`, `station_name` (optional)
- Saved to DB as `GroundMeasurement`

3) Train & Downscale
- The model prepares features: normalized row/col + value
- Trains a Random Forest and validates on a holdout split
- Predicts on a higher-resolution grid (default 2x)
- Displays the downscaled map

4) Metrics & Download
- Shows MSE, RMSE, and R²
- Provides a CSV download of the downscaled grid

---

## 🤖 Machine Learning Details

- Algorithm: `RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)`
- Features per pixel:
  - `row_index / rows`, `col_index / cols`, original NO2 value
- Train/validation: `train_test_split(test_size=0.2, random_state=42)`
- Scaling: `StandardScaler` on features
- Prediction grid: generated with meshgrid at `scale_factor` resolution

Limitations & Considerations:
- The current model uses only simple spatial coordinates + value as features. You can extend with topography, land use, meteorology, etc.
- Interpolation strategy for missing data can be swapped as needed.
- Validation uses a simple random split; spatial CV could be more appropriate.

---

## 🗃️ Database Schema

- `SatelliteData`
  - `id`, `timestamp`, `latitude`, `longitude`, `no2_value`, `resolution`, `source`
- `GroundMeasurement`
  - `id`, `timestamp`, `latitude`, `longitude`, `no2_value`, `station_name`, `satellite_data_id`

Default engine: SQLite (file `no2_data.db`). To switch to PostgreSQL, set `DATABASE_URL`.

---

## 📊 Data Sources

Satellite NO2 (daily tropospheric):
- TROPOMI/Sentinel-5P (Swath): https://search.earthdata.nasa.gov/search/granules?p=C2089270961-GES_DISC
- TROPOMI/Sentinel-5P (GEE, gridded GeoTIFF): https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2
- OMI/Aura (gridded): https://search.earthdata.nasa.gov/search/granules?p=C1266136111GES_DISC
- OMI/Aura (alternate): https://measures.gesdisc.eosdis.nasa.gov/data/MINDS/OMI_MINDS_NO2d.1.1/2024/

Ground measurements:
- CPCB (India): https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqmlanding

---

## 🛠️ Troubleshooting

- Streamlit not found: run via `python -m streamlit run main.py`.
- SQLAlchemy error on Python 3.13: ensure SQLAlchemy ≥ 2.0.43.
- `.env` UnicodeDecodeError: the app no longer requires `.env`. Remove it or ensure it is UTF‑8 if you add one.
- "table already exists": harmless; created earlier. SQLite keeps the table.

---

## 🚀 Roadmap

- Add alternative models (XGBoost, CNN)
- Spatial cross-validation
- Additional features (meteorology, land use)
- Export GeoTIFF of downscaled output
- API endpoints for batch processing

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

Thanks to NASA/ESA/CPCB datasets and the Python open-source ecosystem (Streamlit, scikit-learn, Plotly, SQLAlchemy, Rasterio, NumPy, Pandas).

