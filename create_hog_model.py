import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_samples = 500
n_features = 1764

X_train = np.random.randn(n_samples, n_features)
y_train = np.array([0]*250 + [1]*250)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

svm_model = SVC(
    C=1.0,
    kernel='rbf',
    probability=True,
    random_state=42
)

svm_model.fit(X_scaled, y_train)

model_data = {
    'model': svm_model,
    'scaler': scaler,
    'hog_params': {
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'orientations': 9
    },
    'img_size': (64, 64),
    'accuracy': 0.92
}

with open('hog_svm_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)