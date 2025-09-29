import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def create_risk_model(input_dim=5, learning_rate=1e-3):
    """
    Creates a simple MLP model to map KPIs to a risk score.
    Input: A vector of KPIs [cpu, ram, storage, snet, sri].
    Output: A single scalar risk score (higher means worse health).
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="kpi_input"),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(1, activation='linear', name="risk_output")
    ], name="RiskModel")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    return model