# Revisiting-the-Efficacy-of-Signal-Decomposition-in-AI-based-Time-Series-Forecasting

- Code for the paper Revisiting the Efficacy of Signal Decomposition in AI-based Time Series Forecasting.

# Code Files

- The experimental code we provide is implemented using the significant wave height(Hs) data set as an example. 
- The following is the description of the code we provide:
- MLP_EMD.py: Use the MLP model and the subsequence obtained by EMD to predict Hs, including data preprocessing, model construction, training and testing.
- LSTM_EMD.py: Use the LSTM model and the subsequence obtained by EMD to predict Hs, including data preprocessing, model construction, training and testing.
- LSTM_DWT.py: Use the LSTM model and the subsequence obtained by DWT to predict Hs, including data preprocessing, model construction, training and testing.
- LSTM_SSA.py: Use the LSTM model and the subsequence obtained by SSA to predict Hs, including data preprocessing, model construction, training and testing.
- Transformer_EMD.py: Use the Transformer model and the subsequence obtained by EMD to predict Hs, including data preprocessing, model construction, training and testing.
- In the same py file, by modifying the emd, dwt, and ssa files read, the experiments of Leak and No Leak methods can be realized.

# Running Instructions

- We provide demo data in the path "../demo_data". Each of these code files can be run independently. The output will be obtained after each run of the file: MSE, MAE, RMSE, MAPE, R2 and the best checkpoint file of the model (all saved in the path "../Code/LSTMcheckpoints"). 

- For an average computer (with an Intel CPU), downloading this repository typically takes about 2 minutes, and individual code files take:

  MLP_EMD.py is within 40 minutes, LSTM_EMD.py and LSTM_DWT.py are about two hours, and LSTM_SSA.py and Transformer_EMD.py are within 90 minutes.