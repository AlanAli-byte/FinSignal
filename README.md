# FinSignal – Pattern Recognition for Financial Time Series Forecasting

## Student Information

**Name:** Alan Ali

**Register Number:** TCR24CS008

**Course:** Pattern Recognition

---

## Project Description

This project applies time–frequency signal processing and deep learning to predict stock prices from financial time series data.

Financial data such as stock prices, revenue, and market indices are treated as non-stationary signals. The Short-Time Fourier Transform (STFT) is applied to convert these signals into 2D spectrograms, which are then used as input to a Convolutional Neural Network (CNN) for regression-based price forecasting.

The project includes an interactive Streamlit web interface for data collection, signal visualization, model training, and result analysis.

---

## Problem Definition

Given a multivariate financial signal:

```
X(t) = [p(t), r(t), g(t), s(t), d(t)]
```

Where:

* **p(t)** = Stock Price vs Time
* **r(t)** = Revenue vs Quarter
* **g(t)** = Profit vs Quarter
* **s(t)** = Market Index (Sensex) vs Time
* **d(t)** = USD–INR Exchange Rate vs Time

The goal is to predict the future stock price:

```
p̂(t + Δt) = f_θ(S_t)
```

Where `f_θ` is the CNN model and `S_t` is the spectrogram at time `t`.

---

## Algorithm Components Implemented

### 1. Short-Time Fourier Transform (STFT)

Converts the non-stationary time signal into a time–frequency representation using a sliding window:

```
STFT(t, f) = ∫ X(τ) · w(τ − t) · e^(−j2πfτ) dτ
```

Key parameters:

* **Window Length (L)** – Number of samples per segment
* **Hop Size (H)** – Step size between windows
* **Overlap** – L − H

---

### 2. Spectrogram

The spectrogram is computed as the magnitude squared of the STFT:

```
S(t, f) = |STFT(t, f)|²
```

This produces a 2D matrix `S ∈ R^(T × F)` representing signal energy at each time-frequency point. It is treated as an image and used as CNN input.

---

### 3. CNN Regression Model

A Convolutional Neural Network is trained on spectrogram patches to predict future stock prices.

Architecture:

```
Input (Spectrogram)
→ Conv2D (32 filters, 3×3) + BatchNorm + MaxPool
→ Conv2D (64 filters, 3×3) + BatchNorm + MaxPool
→ Conv2D (128 filters, 3×3) + BatchNorm
→ Global Average Pooling
→ Dense (128 units, ReLU) + Dropout
→ Output (1 unit — predicted price)
```

---

### 4. Evaluation Metrics

```
MSE  = mean((y_actual − y_predicted)²)
MAE  = mean(|y_actual − y_predicted|)
R²   = 1 − (SS_res / SS_tot)
MAPE = mean(|y_actual − y_predicted| / |y_actual|) × 100
```

---

## Inputs

The application allows configuration of:

* Company selection (NSE/BSE/Global tickers via Yahoo Finance)
* Date range for historical data
* STFT window length (L) and hop size (H)
* Forecast lookahead (Δt in days)
* Number of training epochs
* Dropout rate

Data is fetched automatically — no manual file upload is required.

---

## Outputs

After running the full pipeline, the application displays:

* Normalized time series plot for all selected companies
* FFT frequency spectrum with dominant frequency
* STFT spectrogram S(t, f) heatmap
* CNN architecture diagram
* Live training loss curve (Train vs Validation)
* Actual vs Predicted price overlay on test set
* Residual plot and error distribution
* MSE, MAE, R², MAPE evaluation metrics
* Feature analysis and spectral entropy comparison
* Simulated future price forecast with confidence band

---

## Visualization

The application provides:

* Time domain signal plot (raw and normalised)
* Frequency spectrum (FFT amplitude vs frequency)
* Spectrogram heatmap (time–frequency energy map)
* CNN architecture block diagram
* Training history (loss vs epoch)
* Prediction vs actual scatter plot
* Residual bar chart
* Correlation matrix across companies
* Feature entropy and dominant frequency comparison

---

## Technologies Used

* Python 3.11
* NumPy
* Pandas
* SciPy (STFT, FFT)
* Plotly (interactive charts)
* TensorFlow / Keras (CNN model)
* yfinance (stock data)
* Streamlit (web interface)

---

## Project Structure

```
files/
│
├── app.py
├── requirements.txt
├── README.md
└── _pages/
    ├── __init__.py
    ├── overview.py
    ├── data.py
    ├── signal_proc.py
    ├── cnn_model.py
    ├── predictions.py
    └── analysis.py
```

---

## How to Run the Project

### 1. Install Python 3.11

Download from:

[https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)

Ensure Python is added to system PATH during installation.

### 2. Install Dependencies

```
pip install streamlit yfinance pandas numpy scipy plotly tensorflow
```

> Note: TensorFlow is optional. If not installed, the app runs a simulated training loop automatically.

### 3. Run Application

```
cd files
python -m streamlit run app.py
```

The application opens at: `http://localhost:8501`

---

## Sample Input

To reproduce the results:

| Setting | Value |
|---------|-------|
| Companies | Reliance Industries, TCS, Infosys |
| Start Date | 2020-01-01 |
| End Date | 2024-12-31 |
| Feature | Close Price |
| Window Length (L) | 32 |
| Hop Size (H) | 8 |
| Lookahead (Δt) | 5 days |
| Epochs | 20 |
| Dropout | 0.3 |

---

## Important Notes

* All signal processing is implemented using SciPy — no black-box STFT libraries.
* CNN is implemented from scratch using TensorFlow/Keras.
* Data is fetched live from Yahoo Finance — internet connection is required.
* If TensorFlow is unavailable, a simulation mode demonstrates all outputs.
* Spectrogram shape and resolution depend on window length and hop size selection.

---

## References

1. Y. Zhang and C. Aggarwal, "Stock Market Prediction Using Deep Learning," IEEE Access.
2. A. Tsantekidis et al., "Deep Learning for Financial Time Series Forecasting."
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
4. A. Borovykh et al., "Conditional Time Series Forecasting with CNNs."

---

## Conclusion

This project demonstrates that financial time series can be treated as non-stationary signals, transformed into meaningful time–frequency representations via STFT, and used to train a CNN model for stock price forecasting. The spectrogram reveals hidden periodic patterns in price data that standard time-domain analysis cannot capture, validating the signal processing approach to financial prediction.

---
