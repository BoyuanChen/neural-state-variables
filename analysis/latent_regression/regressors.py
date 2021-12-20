import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression


def mlp_regress(X_train, X_test, y_train, y_test, random_state=0):
    scaler = StandardScaler()
    nn = MLPRegressor(hidden_layer_sizes=(64, 64, 64, 64, 64), activation='relu', solver='adam',
                      learning_rate='adaptive', alpha=0.01, random_state=random_state)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_norm = np.max(np.abs(y_train))
    nn.fit(X_train_scaled, y_train / y_norm)
    train_error = np.mean(np.abs(nn.predict(X_train_scaled) * y_norm - y_train)) 
    test_error = np.mean(np.abs(nn.predict(X_test_scaled) * y_norm - y_test))
    return train_error, test_error, {'scaler':scaler, 'model':nn}


def lin_regress(X_train, X_test, y_train, y_test):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    train_error = np.mean(np.abs(reg.predict(X_train) - y_train))
    test_error = np.mean(np.abs(reg.predict(X_test) - y_test))
    return train_error, test_error, reg


def pca(X_train, X_test, num_components, random_state=0):
    pca = PCA(num_components, random_state=random_state)
    pca.fit(X_train)
    return pca.transform(X_train), pca.transform(X_test), pca