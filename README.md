# Heart Disease Prediction using Logistic Regression

This project implements a logistic regression model from scratch to predict heart disease based on various health indicators. The model is trained using gradient descent to optimize weights and bias. It also includes data preprocessing, visualization, and performance evaluation on test data.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Dependencies](#dependencies)
- [Results](#results)
- [License](#license)

### Project Overview

This project uses logistic regression to classify whether a patient has heart disease based on a set of features, including age, sex, cholesterol levels, blood pressure, and more. It includes:
- Data preprocessing: normalization and splitting of data.
- Data visualization with histograms and a correlation heatmap.
- Training of a logistic regression model using gradient descent.
- Evaluation of the model on test data, including accuracy metrics.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ArpitKaushik7/Heart-Disease-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Heart-Disease-Prediction
   ```
3. Install the required Python packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

### Usage

1. **Load and preprocess data:**
   - Ensure your dataset `heart.csv` is located in the specified path or update the path in the code.
   - The code loads data, checks for missing values, and normalizes the features.

2. **Run the logistic regression model:**
   ```python
   logistic_regression(features, labels, test_features, test_labels, learning_rate=1.5, num_iterations=300)
   ```
3. **View the output:**
   - The model will output test accuracy and a cost iteration plot, showing the cost function convergence during training.

### Code Explanation

#### Key Steps

1. **Data Loading and Exploration:**
   - `data.info()` provides a summary of the data.
   - `data.describe()` shows basic statistical details.
   - Histograms and heatmaps provide a visual summary of feature distributions and correlations.

2. **Data Preprocessing:**
   - Normalization using Min-Max scaling to ensure faster convergence.
   - Data split into training and test sets using `train_test_split`.

3. **Logistic Regression Implementation:**
   - **initialize_weights_and_bias(dimension):** Initializes weights and bias.
   - **sigmoid(z):** Applies the sigmoid activation function.
   - **foward_and_backward_propagation(w, b, x_train, y_train):** Computes the cost and gradients.
   - **update(w, b, x_train, y_train, learning_rate, number_of_iterations):** Updates weights and bias over iterations.
   - **predict(w, b, x_test):** Predicts outcomes on test data.
   - **logistic_regression(...)**: The main function that ties all components together, performs training, and evaluates accuracy.

#### Visualization

- **Histograms**: Display distributions of individual features using `seaborn`.
- **Heatmap**: Visualizes feature correlations to assess relationships between features.

### Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Results

- After training, the model outputs test accuracy, which indicates its performance on the test data.
- The cost iteration plot shows how the cost function decreases over time, indicating learning progress.

### License

This project is licensed under the MIT License. 
