import joblib
from typing import Any, Dict, List, Tuple, Type, Union, Optional
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.utils import all_estimators
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.neighbors import NearestNeighbors
from Levenshtein import distance as lev_dist
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram
from sklearn.tree import plot_tree, BaseDecisionTree
from pandas import DataFrame, Series
import pandas as pd
# Импорт метрик для оценки модели (scores)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, r2_score,
    d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
    adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
    jaccard_score, v_measure_score, brier_score_loss, d2_tweedie_score,
    cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
    average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
    top_k_accuracy_score, calinski_harabasz_score, roc_auc_score, davies_bouldin_score,
    normalized_mutual_info_score, fowlkes_mallows_score
)
# Импорт метрик ошибок (errors)
from sklearn.metrics import (
    max_error, mean_absolute_percentage_error, median_absolute_error,
    mean_squared_log_error, mean_squared_error, mean_absolute_error
)
import sklearn

scores = (
    accuracy_score, precision_score, recall_score, f1_score, r2_score,
    d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
    adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
    jaccard_score, v_measure_score, brier_score_loss, d2_tweedie_score,
    cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
    average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
    top_k_accuracy_score, calinski_harabasz_score, roc_auc_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score, v_measure_score, adjusted_rand_score,
    normalized_mutual_info_score, fowlkes_mallows_score
)

errors = (
    max_error, mean_absolute_percentage_error, median_absolute_error,
    mean_squared_log_error, mean_squared_error, mean_absolute_error
)


class Model:
    def __init__(self, model: Type[BaseEstimator] = None):
        """
        Initializes an instance of the Model class.

        Parameters:
        model (Type[BaseEstimator], optional): The model to use. Defaults to None.

        Example:
        --------
        >>> from sklearn.linear_model import LogisticRegression
        >>> model_instance = Model(model=LogisticRegression())
        """
        all_models = self.__check_model_type()

        if model is not None:
            assert isinstance(model, tuple(all_models)), ('Incorrect input model type. '
                                                          f'Should be one of {type(self)} models from sklearn')
        self.__model: BaseEstimator = model  # Приватизируем атрибут model

    @property
    def model(self):
        return self.__model

    def __check_model_type(self) -> List[Type[BaseEstimator]]:
        """
        Checks and returns the types of available models.

        Returns:
        --------
        List[Type[BaseEstimator]]: A list of model types.

        Example:
        --------
        >>> model_instance = Model()
        >>> model_instance.__check_model_type()
        [<class 'sklearn.linear_model._logistic.LogisticRegression'>, ...]
        """

        self.__model_types_with_names: List[Tuple[str, Type[BaseEstimator]]] = all_estimators(
            type_filter=type(self).__name__.lower())
        all_models = [t[1] for t in self.__model_types_with_names]
        return all_models

    def fit(self, X: Any, y: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Fits the model to the data.

        Parameters:
        -----------
        X : Any
            Training data.
        y : Any, optional
            Target values.
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.

        Raises:
        -------
        AssertionError: If the model is not defined.

        Example:
        --------
        >>> model_instance = Model(model=LogisticRegression())
        >>> model_instance.fit(X_train, y_train)
        """
        assert self.__model is not None, "Model is not defined."
        self.__model.fit(X, y, *args, **kwargs)

    def predict(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Predicts using the model.

        Parameters:
        -----------
        X : Any
            Data to predict.
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.

        Returns:
        --------
        Any: Predicted values.

        Raises:
        -------
        AssertionError: If the model is not defined.

        Example:
        --------
        >>> model_instance = Model(model=LogisticRegression())
        >>> predictions = model_instance.predict(X_test)
        """
        assert self.__model is not None, "Model is not defined."
        return self.__model.predict(X, *args, **kwargs)

    def predict_proba(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Predicts class probabilities using the model.

        Parameters:
        -----------
        X : Any
            Data to predict.
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.

        Returns:
        --------
        Any: Predicted class probabilities.

        Raises:
        -------
        AssertionError: If the model is not defined or is not a classifier model.

        Example:
        --------
        >>> model_instance = Model(model=LogisticRegression())
        >>> probabilities = model_instance.predict_proba(X_test)
        """
        classifier_models = [t[1] for t in all_estimators(type_filter='classifier')]
        assert isinstance(self.__model, tuple(classifier_models)), ('Incorrect model type for predict_proba. '
                                                                    f'Should be one of {classifier_models}')
        return self.__model.predict_proba(X, *args, **kwargs)

    def save_model(self, path: str, *args: Any, **kwargs: Any) -> None:
        """
        Saves the model to a file.

        Parameters:
        -----------
        path : str
            The path to save the model.
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.

        Raises:
        -------
        AssertionError: If the model is not defined.

        Example:
        --------
        >>> model_instance = Model(model=LogisticRegression())
        >>> model_instance.save_model('model.pkl')
        """
        assert self.__model is not None, "Model is not defined."
        joblib.dump(self, path, *args, **kwargs)  # Сохраняем текущий объект Model

    @classmethod
    def load_model(cls, path: str, *args: Any, **kwargs: Any) -> 'Model':
        """
        Loads a model from a file.

        Parameters:
        -----------
        path : str
            The path to load the model from.
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.

        Returns:
        --------
        Model: An instance of the Model class with the loaded model.

        Raises:
        -------
        ValueError: If the loaded object is not an instance of the Model class.

        Example:
        --------
        >>> loaded_model_instance = Model.load_model('model.pkl')
        """
        try:
            model_instance = joblib.load(path, *args, **kwargs)
            assert isinstance(model_instance, cls), "Loaded object is not an instance of the expected class."
            return model_instance
        except:
            raise ValueError("You're tying to load incorrect model")

    def fit_all(self, X: Any, y: Any = None) -> Tuple[
        Dict[str, 'Model'], Dict[str, Exception]]:
        """
        Fits all available models to the data.

        Parameters:
        -----------
        X : Any
            Training data.
        y : Any, optional
            Target values.

        Returns:
        --------
        Tuple[Dict[str, Model], Dict[str, Exception]]: A tuple containing a dictionary of fitted models and a dictionary of errors.

        Example:
        --------
        >>> model_instance = Model()
        >>> fitted_models, errors = model_instance.fit_all(X_train, y_train)
        """
        fitted_models: Dict[str, 'Model'] = {}
        error_fitting: Dict[str, Exception] = {}

        for model_name, model_type in self.__model_types_with_names:
            try:
                model_instance = model_type()
                model_instance.fit(X, y)
                wrapped_model = self.__class__(model_instance)
                fitted_models[model_name] = wrapped_model
            except Exception as e:
                error_fitting[model_name] = e

        return fitted_models, error_fitting

    def get_params(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Gets parameters of the model.

        Parameters:
        -----------
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.

        Returns:
        --------
        Dict[str, Any]: Model parameters.

        Raises:
        -------
        AssertionError: If the model is not defined.

        Example:
        --------
        >>> model_instance = Model(model=LogisticRegression())
        >>> params = model_instance.get_params()
        """
        assert self.__model is not None, "Model is not defined."
        return self.__model.get_params(*args, **kwargs)


class Regressor(Model):
    def __init__(self, model: Type[BaseEstimator] = None) -> None:
        """
        Initializes an instance of the Regressor class.

        Parameters:
        -----------
        model : Type[BaseEstimator], optional
            The regression model to use. Defaults to None.

        Example:
        --------
        >>> from sklearn.linear_model import LinearRegression
        >>> regressor_instance = Regressor(model=LinearRegression())
        """
        super().__init__(model)

    def report(self, y_true, y_pred):
        """
        Generates a report of regression metrics based on the true and predicted values.

        Parameters:
        -----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.

        Returns:
        --------
        dict : A dictionary containing various regression metrics:
            - 'r2_score': R^2 (coefficient of determination) regression score.
            - 'mean_absolute_error': Mean absolute error regression loss.
            - 'mean_squared_error': Mean squared error regression loss.
            - 'max_error': Maximum residual error.
            - 'mean_absolute_percentage_error': Mean absolute percentage error regression loss.
            - 'median_absolute_error': Median absolute error regression loss.
            - 'mean_squared_log_error': Mean squared logarithmic error regression loss.
            - 'd2_absolute_error_score': D^2 (coefficient of determination) regression score based on absolute error.
            - 'root_mean_squared_error': Root mean squared error regression loss.
            - 'root_mean_squared_log_error': Root mean squared logarithmic error regression loss.

        Example:
        --------
        >>> y_true = [3.0, -0.5, 2.0, 7.0]
        >>> y_pred = [2.5, 0.0, 2.0, 8.0]
        >>> regressor_instance = Regressor(model=LinearRegression())
        >>> metrics_report = regressor_instance.report(y_true, y_pred)
        >>> print(metrics_report)
        {'r2_score': 0.9486081370449679, ...}
        """

        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'mean_absolute_error': mean_absolute_error(y_true, y_pred),
            'mean_squared_error': mean_squared_error(y_true, y_pred),
            'max_error': max_error(y_true, y_pred),
            'mean_absolute_percentage_error': mean_absolute_percentage_error(y_true, y_pred),
            'median_absolute_error': median_absolute_error(y_true, y_pred),
            'mean_squared_log_error': mean_squared_log_error(y_true, y_pred),
            'd2_absolute_error_score': d2_absolute_error_score(y_true, y_pred),
            'root_mean_squared_error': np.sqrt(mean_squared_error(y_true, y_pred)),
            'root_mean_squared_log_error': np.sqrt(mean_squared_log_error(y_true, y_pred)),
        }

        return metrics


class Classifier(Model):
    def __init__(self, model: Type[BaseEstimator] = None) -> None:
        """
        Initializes an instance of the Classifier class.

        Parameters:
        -----------
        model : Type[BaseEstimator], optional
            The classification model to use. Defaults to None.

        Example:
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> classifier_instance = Classifier(model=RandomForestClassifier())
        """
        super().__init__(model)

    def roc_auc_plot(self, y_true: Any, y_score: Any, *args: Any,
                     **kwargs: Any) -> None:
        """
        Plots the ROC (Receiver Operating Characteristic) curve.

        Parameters:
        -----------
        y_true : array-like
            True target values.
        y_score : array-like
            Predicted scores.
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.

        Returns:
        --------
        None

        Example:
        --------
        >>> y_true = [0, 1, 1, 0]
        >>> y_score = [0.1, 0.4, 0.35, 0.8]
        >>> classifier_instance.roc_auc_plot(y_true, y_score)
        """
        fpr, tpr, _ = roc_curve(y_true, y_score, *args, **kwargs)
        roc_auc = roc_auc_score(y_true, y_score)

        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()

    def confusion_matrix_display(self, x_test: Any, y_test: Any, *args: Any,
                                 **kwargs: Any) -> ConfusionMatrixDisplay:
        """
        Displays the Confusion Matrix.

        Parameters:
        -----------
        x_test : Any
            Test data.
        y_test : Any
            True target values.
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.

        Returns:
        --------
        None

        Example:
        --------
        >>> x_test = [[0, 1], [1, 0]]
        >>> y_test = [0, 1]
        >>> classifier_instance.confusion_matrix_display(x_test, y_test)
        """
        ConfusionMatrixDisplay.from_estimator(self.model, x_test, y_test, *args, **kwargs)
        plt.show()

    def __gini(self, y_true, y_pred):
        """
        Computes the Gini coefficient.

        Parameters:
        -----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted values.

        Returns:
        --------
        float : Gini coefficient.
        """

        return 2 * roc_auc_score(y_true, y_pred) - 1

    def __get_grid(self, data):
        """
        Generate a grid for plotting decision boundaries.

        Parameters:
        -----------
        data : array-like
            Input data for which to generate the grid.

        Returns:
        --------
        tuple: Meshgrid arrays for plotting.
        """

        x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
        y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
        return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    def test_tree_classification(self, X, y, cy=0, fit_clf=True):
        """
        Test and plot decision boundaries for a tree-based classifier.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        y : pd.DataFrame or pd.Series
            True target values.
        cy : array-like, optional
            Colors for plotting the scatter points. Defaults to 0.
        fit_clf : bool, optional
            If True, fit the classifier before plotting. Defaults to True.

        Returns:
        --------
        model: The fitted model, if fit_clf is True.

        Example:
        --------
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> import pandas as pd
        >>> X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
        >>> y = pd.Series([0, 1, 0, 1])
        >>> model = DecisionTreeClassifier()
        >>> cluster_instance = Classifier(model)
        >>> cluster_instance.test_tree_classification(X, y, cy=y['target'])
        """

        valid_models = (
            'DecisionTreeClassifier', 'ExtraTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier')
        assert self.model.__class__.__name__ in valid_models, f'Model must be an instance of the tree class. Valid models: {valid_models}'
        assert isinstance(X, pd.DataFrame), f'Incorrect X paramnetr type. {type(X)} instead of {pd.DataFrame}'
        assert isinstance(y, (
            pd.DataFrame, pd.Series)), f'Incorrect y paramnetr type. {type(y)} instead of {pd.DataFrame | pd.Series}'

        xx, yy = self.__get_grid(X.values)

        if fit_clf:
            self.model.fit(X, y)

        predicted = self.model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.figure(figsize=(8, 8))
        plt.pcolormesh(xx, yy, predicted, cmap='Pastel1')
        plt.scatter(X.values[:, 0], X.values[:, 1], s=50, cmap='tab10', c=cy)
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.show()

        if fit_clf:
            return self.model

    def tree_plot(self, **kwargs):
        """
        Plot the tree structure of a decision tree model.

        Parameters:
        -----------
        kwargs : Additional keyword arguments for the tree plotting function. Possible parameters include:
            - feature_names : list of str
                Names of each of the features.
            - class_names : list of str or bool
                Names of each of the target classes. If True, shows the string representation of the class.
            - filled : bool
                When set to True, paint nodes to indicate the majority class for classification, extremity of values for regression, or purity of node for multi-output.
            - rounded : bool
                When set to True, draw node boxes with rounded corners and use Helvetica fonts instead of Times-Roman.
            - proportion : bool
                When set to True, change the display of 'values' and/or 'samples' to be proportions and percentages instead of absolute numbers.
            - precision : int
                Number of decimal places to display.

        Returns:
        --------
        None

        Example:
        --------
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
        >>> y = pd.Series([0, 1, 0, 1])
        >>> model = DecisionTreeClassifier().fit(X, y)
        >>> cluster_instance = Classifier(model)
        >>> cluster_instance.tree_plot(feature_names=X.columns, class_names=['Class 0', 'Class 1'], filled=True)
        """
        assert isinstance(self.model, BaseDecisionTree), 'Model must be an instance of the tree class'
        plot_tree(self.model, **kwargs)
        plt.show()

    def report(self, y_true, y_pred):
        """
        Generates a report of classification metrics based on the true and predicted values.

        Parameters:
        -----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.

        Returns:
        --------
        dict : A dictionary containing various classification metrics:
            - 'accuracy_score': Accuracy classification score.
            - 'precision_score': Precision classification score.
            - 'recall_score': Recall classification score.
            - 'f1_score': F1 score.
            - 'd2_absolute_error_score': D^2 (coefficient of determination) regression score based on absolute error.
            - 'ndcg_score': Normalized Discounted Cumulative Gain score.
            - 'dcg_score': Discounted Cumulative Gain score.
            - 'fbeta_score': F-beta score.
            - 'completeness_score': Completeness score.
            - 'homogeneity_score': Homogeneity score.
            - 'jaccard_score': Jaccard similarity coefficient score.
            - 'brier_score_loss': Brier score loss.
            - 'd2_tweedie_score': D^2 (coefficient of determination) regression score for Tweedie distribution.
            - 'cohen_kappa_score': Cohen's kappa score.
            - 'd2_pinball_score': D^2 (coefficient of determination) regression score based on pinball loss.
            - 'mutual_info_score': Mutual information score.
            - 'adjusted_mutual_info_score': Adjusted mutual information score.
            - 'average_precision_score': Average precision score.
            - 'label_ranking_average_precision_score': Label ranking average precision score.
            - 'balanced_accuracy_score': Balanced accuracy classification score.
            - 'top_k_accuracy_score': Top-k accuracy classification score.
            - 'roc_auc_score': ROC AUC score.
            - '__gini': Gini coefficient.

        Example:
        --------
        >>> y_true = [0, 1, 1, 0]
        >>> y_pred = [0.1, 0.4, 0.35, 0.8]
        >>> metrics_report = classifier_instance.report(y_true, y_pred)
        >>> print(metrics_report)
        {'accuracy_score': 0.75, ...}
        """

        metrics = [
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            d2_absolute_error_score,
            ndcg_score,
            dcg_score,
            fbeta_score,
            completeness_score,
            homogeneity_score,
            jaccard_score,
            brier_score_loss,
            d2_tweedie_score,
            cohen_kappa_score,
            d2_pinball_score,
            mutual_info_score,
            adjusted_mutual_info_score,
            average_precision_score,
            label_ranking_average_precision_score,
            balanced_accuracy_score,
            top_k_accuracy_score,
            roc_auc_score,
            self.__gini
        ]

        result = {}

        for metric in metrics:
            try:
                result[metric.__name__] = metric(y_true, y_pred)
            except Exception as e:
                result[metric.__name__] = e
        result
        return result


class Cluster(Model):
    def __init__(self, model: Type[BaseEstimator] = None) -> None:
        """
        Initializes an instance of the Cluster class.

        Parameters:
        -----------
        model : Type[BaseEstimator], optional
            The clustering model to use. Defaults to None.

        Example:
        --------
        >>> from sklearn.cluster import DBSCAN
        >>> classifier_instance = Classifier(model=DBSCAN())
        """
        super().__init__(model)

    @property
    def labels_(self):
        return self.model.labels_

    @property
    def n_clusters(self):
        return self.model.n_clusters

    def elbow_method(self, x_train: Any, max_k: int, change_n_clusters: bool = True) -> List[float]:
        """
        Apply the elbow method to determine the optimal number of clusters and optionally update the model.

        Parameters:
        -----------
        x_train : array-like
            Training data.
        max_k : int
            Maximum number of clusters to consider.
        change_n_clusters : bool, optional
            If True, update the model's 'n_clusters' parameter to the optimal number and fit the model. Defaults to True.

        Returns:
        --------
        list : WCSS (within-cluster sum of squares) for each number of clusters.

        Example:
        --------
        >>> from sklearn.cluster import KMeans
        >>> cluster_instance = Cluster(model=KMeans())
        >>> wcss = cluster_instance.elbow_method(x_train, max_k=10)
        >>> print(wcss)
        [1234.56, 789.01, 456.78, ...]
        """
        assert isinstance(max_k, int), f'Incorrect max_k param type. {type(max_k)} instead of {int}'
        assert self.model.__class__.__name__ in ('BisectingKMeans', 'KMeans', 'MiniBatchKMeans'), \
            f"This model doesn't support the elbow method. Valid models: {('BisectingKMeans', 'KMeans', 'MiniBatchKMeans')}"

        default_num_clusters = self.model.n_clusters

        wcss = []
        for k in range(1, max_k + 1):
            self.model.n_clusters = k
            model = self.model.fit(x_train)
            wcss.append(model.inertia_)

        n_clust = self.__elbow_method_best_k(wcss)
        if change_n_clusters:
            self.model.n_clusters = n_clust
            self.model.fit(x_train)
            print(f"Your model's parameter 'n_clusters' was changed to optimal: {n_clust} and model was fitted on it.")
        else:
            self.model.n_clusters = default_num_clusters

        return wcss

    def elbow_method_plot(self, wcss: Union[List[float], Tuple[float, ...]]) -> None:
        """
        Plot the results of the elbow method.

        Parameters:
        -----------
        wcss : list or tuple
            WCSS values for different numbers of clusters.

        Returns:
        --------
        None

        Example:
        --------
        >>> cluster_instance.elbow_method_plot(wcss)
        """
        assert isinstance(wcss, (list, tuple)), f'Incorrect wcss param type. {type(wcss)} instead of {list | tuple}'

        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.plot(range(1, len(wcss) + 1), wcss, marker='o', mfc='red')
        plt.title('Selecting the number of clusters using the elbow method')
        plt.xlabel('num clusters')
        plt.ylabel('WCSS (error)')
        plt.xticks(range(1, len(wcss) + 1))
        plt.show()

    def __elbow_method_best_k(self, wcss: Union[List[float], Tuple[float, ...]]) -> Union[int, str]:
        """
        Determine the best number of clusters using the elbow method with a given threshold.

        Parameters:
        -----------
        wcss : list or tuple
            WCSS values for different numbers of clusters.

        Returns:
        --------
        int : Optimal number of clusters.
        """
        assert isinstance(wcss, (list, tuple)), f'Incorrect wcss parameter type. {type(wcss)} instead of {list | tuple}'
        assert len(wcss) >= 3, 'max_k len must be >= 3'

        # подробное описание работы алгоритма в файле про кластеризацию и метрики качества
        diff = np.diff(wcss)
        diff_r = diff[1:] / diff[:-1]
        k_opt = range(1, len(wcss))[np.argmin(diff_r) + 1]

        return k_opt

    def __dunn_index(self, x_train: Any, labels: Any) -> float:
        """
        Calculate the Dunn Index for the given data and labels.

        Parameters:
        -----------
        x_train : array-like
            Training data.
        labels : array-like
            Cluster labels.

        Returns:
        --------
        float : Dunn Index.
        """
        clusters = np.unique(labels)
        if len(clusters) < 2:
            return 0

        distances = cdist(x_train, x_train)
        intra_cluster_dists = [np.max(distances[labels == cluster]) for cluster in clusters]
        inter_cluster_dists = [np.min(distances[labels == c1][:, labels == c2])
                               for i, c1 in enumerate(clusters) for c2 in clusters[i + 1:]]

        return np.min(inter_cluster_dists) / np.max(intra_cluster_dists)

    def __smape(self, y_true, y_pred):
        """
        Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between true and predicted values.

        Parameters:
        -----------
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.

        Returns:
        --------
        float : SMAPE value.
        """

        return np.mean(2 * np.abs(y_pred - y_true) / (y_true + y_pred))

    def __plot_dendrogram(self, model, **kwargs):
        """
        Generate the linkage matrix and plot the dendrogram for hierarchical clustering.

        Parameters:
        -----------
        model : object
            Fitted clustering model.
        kwargs : additional keyword arguments
            Additional keyword arguments for the dendrogram plotting function. Possible parameters include:
            - truncate_mode : str, optional
                The truncation mode: 'level' or 'lastp'.
            - p : int, optional
                The number of levels to plot or the number of last clusters to show.
            - show_contracted : bool, optional
                Whether to show the contracted branches (default is False).
            - annotate_above : float, optional
                Annotate only the above threshold.
            - leaf_rotation : float, optional
                The rotation angle for leaf labels (default is 90).
            - leaf_font_size : float, optional
                The font size for leaf labels (default is 10).

        Returns:
        --------
        None
        """

        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)
        dendrogram(linkage_matrix, **kwargs)

    def dendrogram_plot(self, **kwargs):
        """
        Plot a dendrogram for the agglomerative clustering model.

        Parameters:
        -----------
        kwargs : additional keyword arguments
            Additional keyword arguments for the dendrogram plotting function. Possible parameters include:
            - truncate_mode : str, optional
                The truncation mode: 'level' or 'lastp'.
            - p : int, optional
                The number of levels to plot or the number of last clusters to show.
            - show_contracted : bool, optional
                Whether to show the contracted branches (default is False).
            - annotate_above : float, optional
                Annotate only the above threshold.
            - leaf_rotation : float, optional
                The rotation angle for leaf labels (default is 90).
            - leaf_font_size : float, optional
                The font size for leaf labels (default is 10).

        Returns:
        --------
        None

        Example:
        --------
        >>> from sklearn.cluster import AgglomerativeClustering
        >>> import numpy as np
        >>> model = AgglomerativeClustering().fit(np.random.rand(10, 2))
        >>> cluster_instance = Cluster(model)
        >>> cluster_instance.dendrogram_plot(truncate_mode='level', p=3)
        """

        assert self.model.__class__.__name__ in ('AgglomerativeClustering'), f'Only support AgglomerativeClustering'
        assert hasattr(self.model, 'children_'), f'The model must be fitted'

        plt.figure(figsize=(10, 8))
        plt.title('Hierarchical Clustering Dendrogram')
        self.__plot_dendrogram(self.model, **kwargs)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

    def report(self, x_train: Any, y_true: Any) -> Dict[str, float]:
        """
        Calculate various clustering metrics for the given training data and true labels.

        Parameters:
        -----------
        x_train : array-like
            Training data.
        y_true : array-like
            True labels for the training data.

        Returns:
        --------
        dict : A dictionary containing various clustering metrics:
            - 'Silhouette Score': Silhouette coefficient for the clusters.
            - 'Calinski-Harabasz Index': Calinski-Harabasz score.
            - 'Davies-Bouldin Index': Davies-Bouldin score.
            - 'Dunn Index': Dunn index.
            - 'V-Measure': V-measure score.
            - 'Adjusted Rand Index': Adjusted Rand index.
            - 'Rand Index': Rand index.
            - 'Symmetric Mean Absolute Percentage Error (SMAPE)': Symmetric Mean Absolute Percentage Error.
            - 'Mean Absolute Percentage Error (MAPE)': Mean Absolute Percentage Error.
            - 'Normalized Mutual Information (NMI)': Normalized Mutual Information score.
            - 'Fowlkes-Mallows Index (FMI)': Fowlkes-Mallows score.
            - 'Calinski-Harabasz Index (CHI)': Calinski-Harabasz score.
            - 'Davies-Bouldin Index (DBI)': Davies-Bouldin score.

        Example:
        --------
        >>> from sklearn.datasets import make_blobs
        >>> from sklearn.cluster import KMeans
        >>> import numpy as np
        >>> X, y = make_blobs(n_samples=100, centers=3, random_state=42)
        >>> model = KMeans(n_clusters=3).fit(X)
        >>> cluster_instance = Cluster(model)
        >>> metrics_report = cluster_instance.report(X, y)
        >>> print(metrics_report)
        {'Silhouette Score': 0.68, 'Calinski-Harabasz Index': 345.31, ...}
        """

        labels = self.labels_
        metrics = {
            'Silhouette Score': silhouette_score(x_train, labels),
            'Calinski-Harabasz Index': calinski_harabasz_score(x_train, labels),
            'Davies-Bouldin Index': davies_bouldin_score(x_train, labels),
            'Dunn Index': self.__dunn_index(x_train, labels),
            'V-Measure': v_measure_score(y_true, labels),
            'Adjusted Rand Index': adjusted_rand_score(y_true, labels),
            'Rand Index': rand_score(y_true, labels),
            'Symmetric Mean Absolute Percentage Error (SMAPE)': self.__smape(y_true, labels),
            'Mean_Absolute_Percentage_Error': mean_absolute_percentage_error(y_true, labels),
            'Normalized Mutual Information (NMI)': normalized_mutual_info_score(y_true, labels),
            'Fowlkes-Mallows Index (FMI)': fowlkes_mallows_score(y_true, labels),
            'Calinski-Harabasz Index (CHI)': calinski_harabasz_score(x_train, labels),
            'Davies-Bouldin Index (DBI)': davies_bouldin_score(x_train, labels)
        }
        return metrics


class RecommendSystem:
    def __init__(self, based_on: Optional[Any] = None) -> None:
        """
        Constructor to initialize the RecommendSystem class.

        Parameters:
        based_on (optional): An optional parameter that can be used to customize the initialization.
        """
        self.__based_on = based_on

    def fit(self, X: DataFrame, y: Series, **kwargs: Any) -> None:
        """
        Fit the recommendation model with provided data.

        Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target vector.
        **kwargs: Additional keyword arguments for configuring the NearestNeighbors model, such as:
            - n_neighbors (int): Number of neighbors to use. Default is the number of rows in the weighted average DataFrame.
            - radius (float): Range of parameter space to use by default for neighbors search. Default is 1.0.
            - metric (str): Metric to use for distance computation. Default is "minkowski".

        Returns:
        None: This method does not return any value. It fits the model with the provided data.
        """
        df = pd.concat([X, y], axis=1)
        weighted_avg = df.groupby(df.columns[-1]).apply(
            lambda g: g.iloc[:, :-1].multiply(len(g), axis=0).sum() / len(g))
        self.df = weighted_avg

        self.__model = NearestNeighbors(
            n_neighbors=kwargs.get('n_neighbors', self.df.shape[0]),
            radius=kwargs.get('radius', 1.0),
            algorithm="auto",
            leaf_size=30,
            metric=kwargs.get('metric', "minkowski"),
            p=2,
            metric_params=None,
            n_jobs=None
        )
        self.__model.fit(self.df)

    def predict(self, x: Union[DataFrame, List[Any]], **kwargs: Any) -> List[DataFrame]:
        """
        Predict recommendations for the given input.

        Parameters:
        x (DataFrame or list): Input data for which recommendations are to be made. If a list is provided, it will be converted to a DataFrame.
        **kwargs: Additional keyword arguments for configuring the prediction, such as:
            - ascending (bool): Whether to sort the distances in ascending order. Default is False.

        Returns:
        list of DataFrames: Each DataFrame contains the recommendations and distances for the corresponding input.
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame([x])

        result = self.__model.kneighbors(x, return_distance=True)
        res_recomends = []
        for example, dist in zip(result[1], result[0]):
            temp_df = self.df.copy()
            temp_df['recommendation'] = example
            temp_df['distance'] = dist
            temp_df.sort_values('distance', inplace=True, ascending=kwargs.get('ascending', False))
            temp_df.reset_index(inplace=True, drop=True)
            res_recomends.append(temp_df)
        return res_recomends

    def levenshtein_distance_handmade(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.

        Parameters:
        s1 (str): First string.
        s2 (str): Second string.

        Returns:
        int: The Levenshtein distance between the two strings.
        """
        len_s1 = len(s1)
        len_s2 = len(s2)

        dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

        for i in range(len_s1 + 1):
            dp[i][0] = i

        for j in range(len_s2 + 1):
            dp[0][j] = j

        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1,
                               dp[i][j - 1] + 1,
                               dp[i - 1][j - 1] + cost)

        return dp[len_s1][len_s2]

    def report(self, x: Union[DataFrame, List[Any]], **kwargs: Any) -> DataFrame:
        """
        Generate a report of recommendations using various distance metrics.

        Parameters:
        x (DataFrame or list): Input data for which the report is to be generated. If a list is provided, it will be converted to a DataFrame.
        **kwargs: Additional keyword arguments for configuring the report, such as:
            - n_neighbors (int): Number of neighbors to use. Default is the number of rows in the weighted average DataFrame.
            - radius (float): Range of parameter space to use by default for neighbors search. Default is 1.0.
            - sort_by (str): The distance metric to sort the recommendations by. Default is 'minkowski'.
            - ascending (bool): Whether to sort the distances in ascending order. Default is False.

        Returns:
        DataFrame: A DataFrame containing the recommendations and distances for the given input, sorted by the specified metric.
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame([x])

        recommendation_metrics = ['minkowski', 'cityblock', 'cosine', 'euclidean', 'haversine',
                                  'l1', 'l2', 'manhattan', 'nan_euclidean', lev_dist]
        temp_df = self.df.copy()
        for metric in recommendation_metrics:
            model = NearestNeighbors(
                n_neighbors=kwargs.get('n_neighbors', self.df.shape[0]),
                radius=kwargs.get('radius', 1.0),
                algorithm="auto",
                leaf_size=30,
                metric=metric,
                p=2,
                metric_params=None,
                n_jobs=None
            )
            model.fit(self.df)
            result = model.kneighbors(x, return_distance=True)
            name_metric = f'distance_{metric}' if metric != recommendation_metrics[-1] else f'distance_levenshtein'
            temp_df[name_metric] = result[0][0]
        temp_df.sort_values(f"distance_{kwargs.get('sort_by', 'minkowski')}", inplace=True,
                            ascending=kwargs.get('ascending', False))
        temp_df.reset_index(inplace=True, drop=True)
        return temp_df
