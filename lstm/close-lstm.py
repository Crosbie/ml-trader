import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf 
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight # NEW: Import for class imbalance

import keras
from keras import optimizers
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional # NEW: Import Bidirectional
from keras.callbacks import EarlyStopping

# =ותר===================================================
# SECTION 1: DATA LOADING & FEATURE ENGINEERING
# ========================================================

# Increased historical data
data = yf.download(tickers = 'UNI-USD', start = '2023-11-01' , end = '2024-12-31', interval="1h")

data.columns = data.columns.get_level_values(0)

# Add time-based features
data.reset_index(inplace=True)
data['hour_of_day'] = data['Datetime'].dt.hour
data['day_of_week'] = data['Datetime'].dt.dayofweek

# Add standard technical indicators
data.ta.bbands(append=True)
data.ta.atr(append=True)
data.ta.adx(append=True)
data['RSI'] = ta.rsi(data.Close, length=15)
data['EMAF'] = ta.ema(data.Close, length=20)
data['EMAM'] = ta.ema(data.Close, length=100)
data['EMAS'] = ta.ema(data.Close, length=150)

# Get MACD components
macd = data.ta.macd()
# Rename columns to avoid conflicts if they exist
macd.columns = ['MACD', 'MACDh', 'MACDs']
data = pd.concat([data, macd], axis=1)

# NEW: Advanced Feature Engineering
data['price_vs_emaf'] = (data['Close'] / data['EMAF']) - 1
data['price_vs_emam'] = (data['Close'] / data['EMAM']) - 1
data['MACD_normalized_by_ATR'] = data['MACDh'] / data['ATRr_14']

# Create the classification target variable
data['TargetNextClose'] = data['Close'].shift(-1)
data['TargetDirection'] = (data['TargetNextClose'] > data['Close']).astype(int)

# Data cleaning
data.dropna(inplace=True)
data.reset_index(drop=True, inplace = True)
data.drop(['Datetime', 'TargetNextClose'], axis=1, inplace=True)
if 'Stock Splits' in data.columns:
    data.drop('Stock Splits', axis=1, inplace=True)
if 'Dividends' in data.columns:
    data.drop('Dividends', axis=1, inplace=True)

# Prepare the dataset for scaling
data_set = data.copy()
pd.set_option('display.max_columns', None)
print("Data head with advanced features:")
print(data_set.head())

sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data_set)

# ========================================================
# SECTION 2: DATA PREPARATION FOR LSTM
# ========================================================

backcandles = 60

# Features are all columns EXCEPT the last one
X = np.array([data_set_scaled[i-backcandles:i, :-1].copy() for i in range(backcandles, len(data_set_scaled))])
# Target is the last column
yi = np.array(data_set_scaled[backcandles:,-1])
y = np.reshape(yi,(len(yi),1))

# Split data into train test sets
splitlimit = int(len(X)*0.8)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]

print("\nTraining and Test Set Shapes:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ========================================================
# SECTION 3: MODEL BUILDING & TRAINING
# ========================================================

# NEW: Implement Bidirectional LSTM architecture
lstm_input = Input(shape=(backcandles, X_train.shape[2]), name='lstm_input')

# First Bidirectional LSTM layer with Dropout
x = Bidirectional(LSTM(150, name='first_layer', return_sequences=True))(lstm_input)
x = Dropout(0.2, name='first_dropout')(x)

# Second Bidirectional LSTM layer with Dropout
x = Bidirectional(LSTM(100, name='second_layer'))(x)
x = Dropout(0.2, name='second_dropout')(x)

# Dense layer to interpret the results
x = Dense(50, name='dense_layer', activation='relu')(x)
# Final output layer for classification
output = Dense(1, name='output', activation='sigmoid')(x)

model = Model(inputs=lstm_input, outputs=output)

# Compile the model for classification
adam = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# NEW: Calculate and apply class weights to handle imbalance
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(y_train.flatten()),
                                                  y=y_train.flatten())
model_class_weights = dict(enumerate(class_weights))
print(f"\nCalculated Class Weights: {model_class_weights}")


# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with class weights
history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=100, shuffle=True, validation_split=0.1,
                    callbacks=[early_stopping],
                    class_weight=model_class_weights) # Apply class weights here

# ========================================================
# SECTION 4: PREDICTION & EVALUATION
# ========================================================

# Evaluate the model using classification metrics
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print('\n===============================')
print('     Classification Report     ')
print('===============================')
print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))

print('\n===============================')
print('        Confusion Matrix       ')
print('===============================')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n(Row=Actual, Col=Predicted)")
print("True Negatives (TN): ", cm[0,0])
print("False Positives (FP): ", cm[0,1])
print("False Negatives (FN): ", cm[1,0])
print("True Positives (TP): ", cm[1,1])
print('===============================')

# Plot training history
plt.figure(figsize=(14, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


# ====== BTC Daily for 14 years =========
# mse = 0.00049
# mse = 0.000751

# ====== BTC Hourly for 1 year =========
# mse = 0.000286



# ====== ETH Hourly for 1 year =========
# mse = 0.000328









# using 60 back candles
# ====== ETH Hourly for 1 year =========
# accuracy 99.97%
# mse = 0.0003228




# directional_accuracy = 47.63% including zeros
# directional_accuracy = 47.45% remove zeros

# directional_accuracy = 17.64% remove zeros
# 15.59%


# ============== Refactor to classification
# ETH - 30 candles - 52% accuracy
