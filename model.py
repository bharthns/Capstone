import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NO2DownscalingModel:
    def __init__(self, max_samples: int | None = 200_000):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.max_samples = max_samples
        
    def prepare_features(self, data):
        """Prepare features for the model."""
        rows, cols = data.shape
        row_idx, col_idx = np.indices((rows, cols))
        mask = ~np.isnan(data)

        if not np.any(mask):
            return np.empty((0, 3)), np.empty(0)

        row_norm = row_idx[mask] / max(rows - 1, 1)
        col_norm = col_idx[mask] / max(cols - 1, 1)
        values = data[mask]

        X = np.column_stack((row_norm, col_norm, values))
        y = values.copy()

        if self.max_samples and len(X) > self.max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), self.max_samples, replace=False)
            X = X[idx]
            y = y[idx]

        return X, y
    
    def train(self, data):
        """Train the downscaling model."""
        X, y = self.prepare_features(data)
        X = self.scaler.fit_transform(X)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        return X_val, y_val
    
    def predict(self, data, scale_factor=2):
        """Generate high-resolution predictions."""
        rows, cols = data.shape
        new_rows = rows * scale_factor
        new_cols = cols * scale_factor
        
        # Create high-resolution grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, new_cols),
            np.linspace(0, 1, new_rows)
        )
        
        mean_value = np.nanmean(data)
        if np.isnan(mean_value):
            mean_value = 0.0

        filled = np.nan_to_num(data, nan=mean_value)
        base_resampled = np.repeat(
            np.repeat(filled, scale_factor, axis=0),
            scale_factor,
            axis=1
        )

        X_pred = np.column_stack([
            grid_y.ravel(),
            grid_x.ravel(),
            base_resampled.ravel()
        ])
        
        X_pred = self.scaler.transform(X_pred)
        predictions = self.model.predict(X_pred)
        
        return predictions.reshape(new_rows, new_cols)
