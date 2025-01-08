# Fraud Detection using Self-Organizing Map (SOM)

This project demonstrates the use of a Self-Organizing Map (SOM) for fraud detection in a dataset. The dataset contains transaction records, and the goal is to identify potential fraudulent transactions using unsupervised learning techniques.

## Overview

In this project, we use the `MiniSom` library, a Python implementation of a Self-Organizing Map, to cluster and visualize transaction data. The dataset contains various features such as transaction details, user information, and merchant information. After training the SOM, fraudulent transactions are identified based on their location in the SOM grid.

![Fraud Detection using Self-Organizing Map](https://github.com/user-attachments/assets/43b8adf7-1344-4e93-a5e9-445307007401)

## Dataset

The dataset used in this project is the [Fraud Detection Dataset](https://www.kaggle.com/kartik2112/fraud-detection) from Kaggle. The dataset contains transaction records labeled as fraudulent or legitimate.

## Features

- **Data Preprocessing**: 
    - Conversion of categorical columns to numeric timestamps (where applicable).
    - Scaling of features using `MinMaxScaler` for normalization.
    
- **SOM Training**: 
    - Self-Organizing Map (SOM) is trained using the transaction data.
    - The SOM is visualized using a heatmap, which helps to observe clusters and potential outliers.

- **Fraud Detection**: 
    - Fraudulent transactions are identified by locating them on specific clusters in the SOM grid.
    - The resulting fraudulent transactions are outputted.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/BaseMax/fraud-detection-som.git
   ```

2. Navigate to the project directory:

```bash
cd fraud-detection-som
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

If kagglehub, minisom, scikit-learn, and matplotlib are not installed yet, install them using:

```bash
pip install kagglehub minisom scikit-learn matplotlib
```

## Usage

Download the dataset from Kaggle and place it in the appropriate directory.

Update the path to your dataset in the script (main.py).

Run the script:

```bash
python main.py
```

The script will output a heatmap showing the SOM and print the fraudulent transactions detected.

### Output

- A heatmap visualization of the SOM distance map.
- A list of fraudulent transactions identified based on their position in the SOM grid.

### Code Explanation

- Data Preprocessing: We start by loading the dataset and checking the data types. Any categorical columns are converted into numeric values (timestamps) if they represent date-related information.
- SOM Training: We initialize the SOM grid with a size of 10x10 and train it on the preprocessed and scaled transaction data.
- Fraud Detection: After training, we map the transactions to their corresponding positions in the SOM grid and identify fraudulent transactions based on their position (for example, located in specific clusters).

### Contributing

Feel free to fork the repository, contribute by submitting pull requests, or open issues if you encounter any bugs or have suggestions for improvements.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgements

- Kaggle Fraud Detection Dataset
- MiniSom Library
- scikit-learn
- Matplotlib
- python

Copyright 2025, Max Base
