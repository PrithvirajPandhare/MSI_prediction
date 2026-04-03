# Motor Stress Index (MSI) Prediction

This repository contains an end-to-end machine learning pipeline for predicting the **Motor Stress Index (MSI)** using vehicle telemetry and IoT data. The project leverages **Apache Spark (PySpark)** for distributed data processing and implements an ensemble modeling approach, prominently featuring **Long Short-Term Memory (LSTM)** neural networks for sequential data forecasting.

## Project Structure

The project is structured as a series of sequential Jupyter Notebooks located in the `end to end/` directory:

*   **`01_base_data_preprocessing.ipynb`**: Loads raw sensor data from Databricks Delta tables. Performs data cleaning, handles missing values (mean imputation), scales numeric features (`StandardScaler`), and converts vectors utilizing `VectorAssembler`. It concludes with a train-test split and exports the cleaned datasets as Parquet files to DBFS.
*   **`02_lstm_seq_generation.ipynb`**: Transforms the preprocessed tabular data into time-series sequences with optimal look-back windows suitable for recurrent neural networks (LSTM). 
*   **`03_models_and_prediction.ipynb`**: Trains baseline machine learning models to establish a performance benchmark and generates initial predictions.
*   **`04_lstm_training_and_ensemble_integration.ipynb`**: Trains the core LSTM network on the sequential data and integrates its predictions with other baseline models using an ensemble method to improve overall accuracy and robustness.
*   **`05_metrics_calculations_and_ranking.ipynb`**: Evaluates model performance across standard regression metrics (e.g., RMSE, MAE, R-squared) and scores/ranks different models and ensemble configurations.
*   **`06_figures.ipynb`**: Generates visualizations and plots to analyze model forecasting quality, error distributions, and overall behavior.
*   **`07_stress_parameter_ranking.ipynb`**: Conducts feature importance and interpretability analysis to identify which initial base parameters contribute most heavily to the overall Motor Stress Index (Thermal Load).

## Tech Stack
*   **Environment**: Jupyter Notebook / Databricks
*   **Data Processing**: Apache Spark (PySpark.sql, PySpark.ml)
*   **Deep Learning**: TensorFlow / Keras (LSTM)
*   **Data Storage**: DBFS, Parquet

## Prerequisites
*   Python 3.8 or higher
*   Merged data of 4 months.   

## How to Run

1.  Install the dependencies using the requirements.txt file.
    ``` 
    python -m pip install -r requirements.txt 
    ```
2.  Execute the notebooks sequentially from `01` to `07`.
3.  Processed data and pipeline artifacts are cached in your DBFS path.
4.  Change the DBFS path in the notebooks to your desired path.
