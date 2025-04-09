# Boston Housing Regression Project

This project provides hands-on exercises using the Boston Housing dataset. Participants will learn how to preprocess data, build regression models using Scikit-learn and TensorFlow, and evaluate model performance.

## Project Structure

- **notebooks/**: Contains Jupyter notebooks for interactive exercises.
  - `boston_housing_regression.ipynb`: A notebook for hands-on exercises with sections for data preprocessing, building a Scikit-learn Linear Regression model, and constructing a TensorFlow neural network for regression.

- **data/**: Directory containing information about the dataset.
  - `README.md`: Provides details about the dataset, including its source and structure.

- **models/**: Contains scripts for model building and evaluation.
  - `sklearn_models.py`: Functions or classes for building and evaluating Scikit-learn models for the Boston Housing dataset.
  - `tensorflow_models.py`: Functions or classes for building and evaluating TensorFlow models for the Boston Housing dataset.

- **utils/**: Utility scripts for common tasks.
  - `preprocessing.py`: Functions for data preprocessing tasks such as loading the dataset, checking for missing values, scaling features, and splitting the data into training and testing sets.
  - `evaluation.py`: Functions for evaluating model performance, including calculating mean squared error and other relevant metrics.

- **requirements.txt**: Lists the required Python packages and their versions needed to run the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd boston-housing-regression
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook:
   ```
   jupyter notebook notebooks/boston_housing_regression.ipynb
   ```

## Usage Guidelines

- Follow the instructions in the Jupyter notebook to complete the exercises.
- Refer to the `data/README.md` for information about the dataset.
- Use the `models/` scripts to explore different modeling approaches.
- Utilize the `utils/` scripts for preprocessing and evaluation tasks.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.