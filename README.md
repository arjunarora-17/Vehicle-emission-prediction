# Federated Learning for Vehicle's Emission Prediction  
**Model Aggregation Using Knowledge Distillation**

## Overview  
This project explores the application of federated learning to predict vehicle emissions, focusing on distributed training and model aggregation. The study implements federated learning techniques, such as FedAvg, FedProx, and FedDistill, to combine client models effectively, leveraging knowledge distillation for improved global model accuracy.

---

## Table of Contents  
- [Objective](#objective)  
- [Dataset and Libraries](#dataset-and-libraries)  
- [Implementation Details](#implementation-details)  
- [Results and Comparisons](#results-and-comparisons)  
- [Contributors](#contributors)  

---

## Objective 
- Implementation of three federated aggregation methods: **FedAvg**, **FedProx**, and **FedDistill**.  
- Simulated training of models across four clients: Bulgaria, Estonia, Latvia, Lithuania, and Slovenia.  
- Evaluation of three machine learning models: Linear Regression, Random Forest, and Artificial Neural Networks (MLP Regressor).  
- Data analysis and visualization for model comparison.  

---


## Dataset and Libraries
### Libraries  
The project makes use of the following Python libraries: 

```python
import os  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.neural_network import MLPRegressor 
```

### Dataset
The datasets represent vehicle emission data collected from Estonia, Latvia, Lithuania, and Slovenia. Each dataset contains unique characteristics, enabling testing of federated learning on non-IID (non-independent and identically distributed) data.  

**Reference:**  
European Commission. (Year). *CO2 emissions from new passenger cars registered in EU27, Bulgaria (from 2023), Estonia (from 2023), Latvia (from 2023), Lithuania (from 2023), Slovenia (from 2023) – Regulation (EU) 2019/631 [Data set].* European Environment Agency.  
[Limk to the dataset](https://co2cars.apps.eea.europa.eu/source=%7B%22track_total_hits%22%3Atrue%2C%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22constant_score%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22year%22%3A2023%7D%7D%5D%7D%7D%2C%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22scStatus%22%3A%22Provisional%22%7D%7D%5D%7D%7D%5D%7D%7D%7D%7D%5D%7D%7D%2C%22display_type%22%3A%22tabular%22%7D).


## Implementation Details
### Import Libraries and Dataset
All necessary libraries are imported, and the dataset is preprocessed for analysis.

### Data Analysis
Exploratory Data Analysis (EDA) is conducted to understand the dataset's distribution and identify trends influencing emission levels.

### Model Selection
Three machine learning models are evaluated:

-  Linear Regression: A baseline model for regression tasks.
-  Random Forest Regressor: A robust ensemble learning method.
-  Artificial Neural Networks (ANN): Implemented using MLPRegressor.

### Federated Learning Architecture
- **Step 1:** Create the Base Neural Network Model: The selected ANN (MLP Regressor) is used as the base model, which will be distributed to clients for training.
- **Step 2:** Distribute the Model Across Clients: The base model is distributed to client datasets from Estonia, Latvia, Lithuania, and Slovenia. Each client trains the model locally, and parameters (weights) are extracted post-training.
- **Step 3:** Aggregate Parameters Using FedAvg, FedProx, and FedDistill
    -  FedAvg: Averages model weights across all clients to form a global model.
    -  FedProx: Enhances FedAvg by adding a proximal term to manage divergence in client models due to non-IID data.
    -  FedDistill: Utilizes knowledge distillation to create a global model by mimicking client outputs rather than averaging weights.

## Results-and-Comparisons
### Performance Metrics Comparison  

The table below compares the performance of different models and federated learning techniques on Bulgaria test dataset:  

| Model        | MAE       | MSE        | R²         |
|--------------|-----------|------------|------------|
| Direct NN    | 2.116229  | 29.570969  | 0.981572   |
| FedAvg       | 16.433031 | 472.638781 | 0.705464   |
| FedProx      | 19.941837 | 607.884084 | 0.621183   |
| FedDistill   | 7.076949  | 138.806412 | 0.913500   |

### Observations:  
- **Direct NN** achieves the best performance with the lowest MAE and MSE and the highest R² score.  
- Among federated methods, **FedDistill** outperforms FedAvg and FedProx, indicating its superior ability to generalize on non-IID data.  
- **FedProx** shows the highest error metrics, suggesting limitations in handling data heterogeneity compared to the other methods.  

## Contributors
- **Arjun Arora (M24CSA003)**

- **Pooja Naveen Poonia (M24CSA020)**

- **Prateek (M24CSA022)**

- **Shivani Tiwari (M24CSA029)**

- **Suvigya Sharma (M24CSA033)**
