# ML Tools

The "ML_tools" repository is a collection of utility classes designed to streamline and enhance the workflow of machine learning engineers. These tools offer robust support for common machine learning tasks such as regression, classification, clustering, and recommendation systems. The classes encapsulate complex processes into simplified, reusable components, making it easier to develop, evaluate, and deploy machine learning models.

## Repository Structure

### 1. `models_ml.py`
This file contains four specialized classes tailored for different machine learning tasks:

- **Regressor**: A class dedicated to regression tasks, providing a comprehensive interface for training, evaluating, and reporting on regression models. It integrates seamlessly with scikit-learn’s regression models and adds functionality for generating detailed performance reports.

- **Classifier**: Designed for classification tasks, this class simplifies the process of training and evaluating classifiers. It includes methods for generating ROC curves, confusion matrices, and detailed performance reports, making it a versatile tool for various classification problems.

- **Cluster**: This class is built for clustering tasks and supports a wide range of clustering algorithms. It includes methods for determining the optimal number of clusters, plotting dendrograms, and reporting on clustering metrics, helping to visualize and interpret clustering results effectively.

- **RecommendSystem**: A class designed for building and managing recommendation systems using the k-Nearest Neighbors (k-NN) algorithm. It offers flexibility in choosing distance metrics, including custom metrics like Levenshtein distance, and provides tools for generating and sorting recommendations based on different criteria.

### 2. `tools_pandas.py`
The `tools_pandas.py` file introduces a custom `DataFrame` class that extends the capabilities of the standard `pandas.DataFrame` (from [pandas](https://pandas.pydata.org/) library), specifically tailored for machine learning and data analysis tasks. This enhanced DataFrame retains all the familiar functionality of a typical Pandas DataFrame while incorporating additional methods to simplify and accelerate common data manipulation and analysis workflows.

#### Key Features:
- **Outlier Detection**: Advanced methods for detecting outliers using statistical and machine learning techniques, including the 3-sigma rule, Tukey's fences, and clustering-based approaches.
- **Feature Selection and Engineering**: Direct integration with [`scikit-learn`](https://scikit-learn.org/stable/index.html) for feature selection using methods like `SelectKBest` and `Lasso`, enabling easy selection of the most relevant features for model training.
- **Clustering and Dimensionality Reduction**: Built-in support for clustering algorithms such as DBSCAN and dimensionality reduction techniques like PCA and Linear Discriminant Analysis.
- **Handling Imbalanced Data**: Utilities for balancing datasets through oversampling and undersampling methods like SMOTE, improving the performance of models on imbalanced datasets.
- **Visualization**: Simplified plotting functions using [`matplotlib`](https://matplotlib.org/) and [`seaborn`](https://seaborn.pydata.org/) for quick visualization of data distributions, correlations, and analysis results.

---

This repository is an essential toolkit for machine learning engineers, offering powerful, ready-to-use classes that simplify model development and data handling. Whether you're building models for regression, classification, clustering, or recommendation systems, "ML_tools" provides the functionality needed to streamline your workflow and improve productivity.
