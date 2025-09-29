# FILE: lstm_risk_model.py
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K

# --- NEW: Custom Asymmetric Loss Function ---
def penalty_weighted_mse(fn_penalty=10.0):
    """
    A custom Mean Squared Error loss function that applies a heavy penalty
    for under-prediction during a failure event (False Negatives).
    """
    def loss(y_true, y_pred):
        # Calculate the error
        error = y_true - y_pred
        
        # Create a penalty weight. If y_true > y_pred (an under-prediction),
        # and y_true is high (a real drift), apply a large penalty.
        is_underprediction = K.cast(error > 0, K.floatx())
        is_real_event = K.cast(y_true > 0.1, K.floatx())
        
        penalty = is_underprediction * is_real_event * (fn_penalty - 1.0) + 1.0
        
        # Return the mean of the penalty-weighted squared error
        return K.mean(penalty * K.square(error), axis=-1)
        
    return loss

def create_lstm_risk_model(timesteps=15, features=7, learning_rate=1e-3):
    """
    MODIFIED: Compiles the model with the new penalty-weighted loss function.
    """
    model = models.Sequential([
        layers.Input(shape=(timesteps, features), name="kpi_sequence_input"),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(24, activation='relu'),
        layers.Dense(1, activation='linear', name="risk_output")
    ], name="LSTM_RiskModel")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        # Use the new custom loss function
        loss=penalty_weighted_mse(fn_penalty=10.0)
    )
    return model