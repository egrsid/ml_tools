import scipy
from tqdm import tqdm
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.feature_selection import SelectPercentile
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import imblearn.under_sampling as usampling
import imblearn.over_sampling as osampling


class DataFrame(pd.DataFrame):
    def __init__(self, data):
        """
        Initialize a custom DataFrame object, ensuring the provided data is a pandas DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The data to initialize the custom DataFrame. Must be of type pd.DataFrame.

        Raises:
        -------
        AssertionError:
            If the provided data is not a pandas DataFrame.
        """

        assert isinstance(data, pd.DataFrame), f'data parameter has to be {pd.DataFrame} type. Got {type(data)} instead'
        super().__init__(data)

    def detect_outliers(self, method: str = '3sigma', **kwargs) -> pd.DataFrame:
        """
        Detect outliers in the DataFrame using various methods.

        Parameters:
        -----------
        method : str, optional, default='3sigma'
            The method to use for outlier detection. Valid methods are:
            - '3sigma': Detects outliers using the 3-sigma rule.
            - 'Tukey': Detects outliers using Tukey's fences (IQR method).
            - 'Shovene': Detects outliers based on a specific statistical measure.
            - 'Grabbs': Uses Grubbs' test for detecting outliers.
            - 'Clusters': Uses clustering methods to detect outliers.

        kwargs : additional keyword arguments, optional
            Additional parameters specific to the chosen method:
            - For 'Grabbs':
                - threshold : float, default=0.5
                    The significance level for Grubbs' test.
            - For 'Clusters':
                - scale : bool, default=True
                    Whether to scale the data before applying clustering methods.
                - eps : float, default=0.5
                    The maximum distance between two samples for them to be considered as in the same neighborhood (used in DBSCAN).
                - min_samples : int, default=10
                    The number of samples in a neighborhood for a point to be considered a core point (used in DBSCAN).
                - kernel : str, default='rbf'
                    The kernel type to be used in the algorithm (used in OCSVM).
                - nu : float, default=0.1
                    An upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors (used in OCSVM).
                - contamination : float, default=0.1
                    The proportion of outliers in the data set (used in EllipticEnvelope).

        Returns:
        --------
        pd.DataFrame :
            A DataFrame containing the detected outliers. The structure of the output depends on the chosen method.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, 3, 100, 5], "B": [10, 20, 30, 40, 50]}))
        >>> outliers = df.detect_outliers(method='3sigma')
        >>> print(outliers)
        """

        valid_methods = ('3sigma', 'Tukey', 'Shovene', 'Grabbs', 'Clusters')
        method = method.capitalize()
        assert method in valid_methods, f"This method doesn't support. Valid methods: {valid_methods}"

        if method == 'Grabbs':
            threshold = kwargs.get('threshold', 0.5)
            mean_val = self.mean()
            std_val = self.std()
            n = self.count()
            t_value = (n - 1) * (abs(self.max() - mean_val) / std_val)
            critical_value = scipy.stats.t.sf(1 - threshold / (2 * n), n - 2)
            return t_value > critical_value

        if method == 'Clusters':
            models = {
                "DBSCAN": DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 10)),
                "OCSVM": OneClassSVM(kernel=kwargs.get('kernel', 'rbf'), nu=kwargs.get('nu', 0.1)),
                "IsoForest": IsolationForest(),
                "ELL": EllipticEnvelope(contamination=kwargs.get('contamination', 0.1)),
                "LOF": LocalOutlierFactor(novelty=True)
            }

            res_df = self.copy()  # нужно, чтобы не изменять добавлением новых колонок изначальный датасет
            temp_df = res_df  # нужно, чтобы учить на отскейленных данных, в выводить изначальные
            if kwargs.get('scale', True):
                scaler = StandardScaler()
                temp_df = scaler.fit_transform(res_df)

            for model_name, model in models.items():
                model.fit(temp_df)
                if model_name == 'DBSCAN':
                    labels = model.labels_
                    pred = np.where(labels == 0, True, False)
                else:
                    pred = model.predict(temp_df)
                    pred = np.where(pred == 1, True, False)
                res_df[model_name] = pred
            return res_df

        res_df = pd.DataFrame()
        for feature in self:
            if not isinstance(self[feature][0], (float, int)): continue
            match method:
                case '3sigma':
                    mu, sigma = self[feature].mean(), self[feature].std()
                    d = self[abs(self[feature]) > mu + 3 * sigma]
                    res_df = pd.concat((res_df, d)).drop_duplicates(keep=False)
                case 'Tukey':
                    q1 = self[feature].quantile(0.25)
                    q3 = self[feature].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)
                    d = self[(self[feature] < lower_bound) | (self[feature] > upper_bound)]
                    res_df = pd.concat((res_df, d)).drop_duplicates(keep=False)
                case 'Shovene':
                    d = self[scipy.special.erfc(abs(self[feature] - self[feature].mean()) / self[feature].std()) < 1 / (
                            2 * len(self[feature]))]
                    res_df = pd.concat((res_df, d)).drop_duplicates(keep=False)
        return res_df.sort_index()

    def fillna(self, method: str = 'mean', inplace: bool = False) -> pd.DataFrame | None:
        """
        Fill missing values in the DataFrame using various strategies.

        Parameters:
        -----------
        method : str, optional, default='mean'
            The method to use for filling missing values. Valid methods are:
            - 'mean': Fill with the mean of the column.
            - 'median': Fill with the median of the column.
            - 'mode': Fill with the mode of the column.
            - 'hmean': Fill with the harmonic mean of the column.
            - 'indicator': Add an indicator column to flag missing values.
            - 'ffill': Forward fill missing values.
            - 'interpolation': Interpolate missing values.

        inplace : bool, optional, default=False
            If True, modify the DataFrame in place. Otherwise, return a new DataFrame with filled values.

        Returns:
        --------
        pd.DataFrame or None:
            Returns a DataFrame with missing values filled according to the specified method.
            If `inplace` is True, returns None.

        Raises:
        -------
        AssertionError:
            If an unsupported method is provided.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4]}))
        >>> df.fillna(method='median')
        """

        valid_methods = ('mean', 'median', 'mode', 'hmean', 'indicator', 'ffill', 'interpolation')
        assert method.lower() in valid_methods, f"This method doesn't support. Valid methods: {valid_methods}"

        res_df = self.copy()
        match method:
            case 'mean':
                res_df = res_df.fillna(res_df.mean()).round(2)
            case 'median':
                res_df = res_df.fillna(res_df.median()).round(2)
            case 'mode':
                res_df = res_df.fillna(res_df.mode().mean()).round(2)
            case 'hmean':
                res_df = res_df.fillna(
                    pd.Series([np.round(scipy.stats.hmean(col[~np.isnan(col)]), 2) for col in res_df.values.T],
                              index=res_df.columns).round(2))
            case 'indicator':
                temple_df = pd.DataFrame(res_df.isna().astype(int).to_numpy(),
                                         columns=res_df.isna().columns + '_indicator')
                res_df = pd.concat((res_df, temple_df), axis=1)
                res_df = res_df.loc[:, (res_df != 0).any(axis=0)]
            case 'ffill':
                res_df = res_df.fillna(res_df.ffill()).fillna(res_df.bfill()).round(2)
            case 'interpolation':
                res_df = res_df.interpolate(method='linear', limit_direction='both').round(2)
        if inplace:
            super().__init__(res_df)
            return

        return res_df

    def corr_features(self, threshold: int | float = 0.85, **kwargs) -> dict:
        """
        Identify and return highly correlated feature pairs in the DataFrame.

        Parameters:
        -----------
        threshold : int or float, optional, default=0.85
            The correlation coefficient threshold above which features are considered highly correlated.

        kwargs : additional keyword arguments, optional
            Additional arguments for correlation calculation:
            - method : str, optional, default='pearson'
                Method of correlation: {'pearson', 'kendall', 'spearman'}.

        Returns:
        --------
        dict:
            A dictionary where keys are tuples of highly correlated feature pairs, and values are their correlation coefficients.

        Raises:
        -------
        AssertionError:
            If the threshold is not a number or is not in the range (0, 1].

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, 3], "B": [2, 4, 6], "C": [3, 6, 9]}))
        >>> df.corr_features(threshold=0.9)
        """

        assert isinstance(threshold,
                          (int, float)), f'Incorrect dtype. threshold: {type(threshold)} instead of any {(int, float)}'
        assert 0 < threshold <= 1, 'Incorrect threshold value. It must be (0, 1]'

        method = kwargs.get('method', 'pearson').lower()
        corr_matrix = self.corr(method=method).abs()
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        corr_features = corr_matrix.where(upper_tri).unstack()

        return corr_features[corr_features > threshold].to_dict()

    def balance_df(self, target: str | int | float, threshold: int | float) -> None:
        """
        Check the balance of the DataFrame based on the target column.

        Parameters:
        -----------
        target : str, int, or float
            The column name or index to be checked for balance.

        threshold : int or float
            The threshold above which the DataFrame is considered imbalanced.

        Raises:
        -------
        AssertionError:
            If the target is not in the DataFrame's columns or the threshold is not valid.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, 3, 3, 2, 2, 1], "B": [1, 1, 1, 2, 2, 2, 2]}))
        >>> df.balance_df(target='B', threshold=1.5)
        """

        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'
        assert isinstance(threshold,
                          (int, float)), f'Incorrect dtype. threshold: {type(threshold)} instead of any {(int, float)}'
        assert threshold >= 0, 'Incorrect threshold value. It must be >= 0'

        _, count_unique = np.unique(self[target], return_counts=True)
        balance = max(count_unique) / min(count_unique)
        print(f'Disbalanced: {balance}') if balance > threshold else print(f'Balanced: {balance}')

    def check_norm_distribution(self, target: str) -> dict:
        """
        Check if the distribution of the target column is approximately normal by evaluating skewness and kurtosis.

        Parameters:
        -----------
        target : str
            The column name to check for normal distribution.

        Returns:
        --------
        dict:
            A dictionary with the skewness ('asymmetry'), kurtosis ('kurt'),
            and a boolean ('normal') indicating whether the distribution is normal
            based on the criteria |skewness| <= 2 and |kurtosis| <= 7.

        Raises:
        -------
        AssertionError:
            If the target column is not found in the DataFrame.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, 3, 4, 5]}))
        >>> df.check_norm_distribution(target='A')
        """

        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'

        asymmetry = pd.DataFrame(self[target].to_numpy()).skew()[0].round(3)
        kurt = pd.DataFrame(self[target].to_numpy()).kurtosis()[0].round(3)
        print(
            f'ass: {asymmetry}, kurt: {kurt} -> '
            f'Distribution is {"normal" if abs(asymmetry) <= 2 and abs(kurt) <= 7 else "not normal"}')
        return {'asymmetry': asymmetry, 'kurt': kurt, 'normal': abs(asymmetry) <= 2 and abs(kurt) <= 7}

    def under_sampling(self, target: str | int | float, method: str = 'RandomUnderSampler', **kwargs) -> pd.DataFrame:
        """
        Perform under-sampling on the DataFrame to balance the target class distribution.

        Parameters:
        -----------
        target : str, int, or float
            The column name or index to be used as the target for under-sampling.

        method : str, optional, default='RandomUnderSampler'
            The under-sampling method to be used. Valid methods include:
            - 'randomundersampler'
            - 'editednearestneighbours'
            - 'repeatededitednearestneighbours'
            - 'allknn'
            - 'condensednearestneighbour'
            - 'onesidedselection'
            - 'neighbourhoodcleaningrule'
            - 'clustercentroids'
            - 'tomeklinks'
            - 'nearmiss'
            - 'instancehardnessthreshold'

        kwargs : additional keyword arguments
            Additional parameters for the selected under-sampling method.

        Returns:
        --------
        pd.DataFrame:
            A DataFrame with the under-sampled data.

        Raises:
        -------
        AssertionError:
            If the method is not supported, the target is not in the DataFrame's columns,
            or the target type is incorrect.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 1, 1, 2, 2], "B": [0, 0, 0, 1, 1]}))
        >>> df.under_sampling(target='A', method='randomundersampler')
        """
        valid_methods = {
            'randomundersampler': usampling.RandomUnderSampler,
            'editednearestneighbours': usampling.EditedNearestNeighbours,
            'repeatededitednearestneighbours': usampling.RepeatedEditedNearestNeighbours,
            'allknn': usampling.AllKNN,
            'condensednearestneighbour': usampling.CondensedNearestNeighbour,
            'onesidedselection': usampling.OneSidedSelection,
            'neighbourhoodcleaningrule': usampling.NeighbourhoodCleaningRule,
            'clustercentroids': usampling.ClusterCentroids,
            'tomeklinks': usampling.TomekLinks,
            'nearmiss': usampling.NearMiss,
            'instancehardnessthreshold': usampling.InstanceHardnessThreshold
        }

        method = method.lower()
        assert isinstance(target, (
            str, int, float)), f'Incorrect dtype for target: {type(target)} instead of any of (str, int, float)'
        assert method in valid_methods.keys(), f"This method doesn't support. Valid methods: {list(valid_methods.keys())}"
        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'

        sampler_class = valid_methods[method]
        sampler = sampler_class(**kwargs)

        X_resampled, y_resampled = sampler.fit_resample(self.drop(columns=[target]), self[target])
        resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=self.drop(columns=[target]).columns),
                                  pd.DataFrame(y_resampled, columns=[target])], axis=1)
        return resampled_df

    def over_sampling(self, target: str | int | float, method: str = 'RandomOverSampler', **kwargs) -> pd.DataFrame:
        """
        Perform over-sampling on the DataFrame to balance the target class distribution.

        Parameters:
        -----------
        target : str, int, or float
            The column name or index to be used as the target for over-sampling.

        method : str, optional, default='RandomOverSampler'
            The over-sampling method to be used. Valid methods include:
            - 'randomoversampler'
            - 'smote'
            - 'adasyn'
            - 'borderlinesmote'
            - 'svmsmote'
            - 'kmeanssmote'
            - 'smotenc'
            - 'smoten'

        kwargs : additional keyword arguments
            Additional parameters for the selected over-sampling method.

        Returns:
        --------
        pd.DataFrame:
            A DataFrame with the over-sampled data.

        Raises:
        -------
        AssertionError:
            If the method is not supported, the target is not in the DataFrame's columns,
            or the target type is incorrect.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, 2], "B": [0, 1, 1]}))
        >>> df.over_sampling(target='A', method='smote')
        """
        valid_methods = {
            'randomoversampler': osampling.RandomOverSampler,
            'smote': osampling.SMOTE,
            'adasyn': osampling.ADASYN,
            'borderlinesmote': osampling.BorderlineSMOTE,
            'svmsmote': osampling.SVMSMOTE,
            'kmeanssmote': osampling.KMeansSMOTE,
            'smotenc': osampling.SMOTENC,
            'smoten': osampling.SMOTEN
        }

        method = method.lower()
        assert isinstance(target, (
            str, int, float)), f'Incorrect dtype for target: {type(target)} instead of any of (str, int, float)'
        assert method in valid_methods.keys(), f"This method doesn't support. Valid methods: {list(valid_methods.keys())}"
        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'

        sampler_class = valid_methods[method]
        sampler = sampler_class(**kwargs)

        X_resampled, y_resampled = sampler.fit_resample(self.drop(columns=[target]), self[target])
        resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=self.drop(columns=[target]).columns),
                                  pd.DataFrame(y_resampled, columns=[target])], axis=1)
        return resampled_df

    def __l1_models(self, X: pd.DataFrame, y: pd.Series, l1: tuple = tuple(2 ** np.linspace(-10, 10, 100)),
                    scale: bool = False, early_stopping: bool = False) -> list:
        """
        Fit L1-regularized models (Lasso regression) over a range of regularization strengths (alpha values).

        Parameters:
        -----------
        X : pd.DataFrame
            The feature matrix.

        y : pd.Series
            The target vector.

        l1 : tuple, optional, default=tuple(2 ** np.linspace(-10, 10, 100))
            The tuple of L1 regularization strengths to test.

        scale : bool, optional, default=False
            If True, scales the feature matrix X using StandardScaler.

        early_stopping : bool, optional, default=False
            If True, stops fitting models early if all coefficients in the model are zero.

        Returns:
        --------
        list:
            A list of fitted Lasso models.
        """
        X = StandardScaler().fit_transform(X) if scale else X

        result = list()
        for alpha in tqdm(l1, desc='Fitting L1-models'):
            model = Lasso(alpha=alpha).fit(X, y)
            result.append(model)
            if early_stopping and all(map(lambda c: c == 0, model.coef_)): break  # if all weights are zero, stop
        return result

    def __l1_importance(self, X: pd.DataFrame, y: pd.Series, l1: tuple = tuple(2 ** np.linspace(-10, 10, 100)),
                        scale: bool = False, early_stopping: bool = False) -> pd.DataFrame:
        """
        Calculate the feature importance using L1-regularized models across different regularization strengths.

        Parameters:
        -----------
        X : pd.DataFrame
            The feature matrix.

        y : pd.Series
            The target vector.

        l1 : tuple, optional, default=tuple(2 ** np.linspace(-10, 10, 100))
            The tuple of L1 regularization strengths to test.

        scale : bool, optional, default=False
            If True, scales the feature matrix X using StandardScaler.

        early_stopping : bool, optional, default=False
            If True, stops fitting models early if all coefficients in the model are zero.

        Returns:
        --------
        pd.DataFrame:
            A DataFrame containing the L1 regularization strengths and corresponding feature coefficients.
        """
        l1_models_ = self.__l1_models(X, y, l1=l1, scale=scale, early_stopping=early_stopping)

        df = pd.DataFrame([l1_model.coef_ for l1_model in l1_models_], columns=X.columns)
        return pd.concat([pd.DataFrame({'L1': l1}), df], axis=1)

    def l1_importance_plot(self, target: str, l1: tuple = tuple(2 ** np.linspace(-10, 10, 100)), scale: bool = False,
                           early_stopping: bool = False, **kwargs) -> None:
        """
        Plot the L1-regularized coefficients for features across different regularization strengths.

        Parameters:
        -----------
        target : str
            The target column name.

        l1 : tuple, optional, default=tuple(2 ** np.linspace(-10, 10, 100))
            The tuple of L1 regularization strengths to test.

        scale : bool, optional, default=False
            If True, scales the feature matrix X using StandardScaler.

        early_stopping : bool, optional, default=False
            If True, stops fitting models early if all coefficients in the model are zero.

        kwargs : additional keyword arguments
            Additional parameters for plot customization (e.g., figsize, grid).

        Returns:
        --------
        None
        """
        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'
        assert isinstance(l1, (list, tuple, type(np.array)))
        assert isinstance(scale, bool), f'Incorrect dtype. scale: {type(scale)} instead of {bool}'
        assert isinstance(early_stopping,
                          bool), f'Incorrect dtype. early_stopping: {type(early_stopping)} instead of {bool}'

        x = self.drop(target, axis=1)
        y = self[target]

        df = self.__l1_importance(x, y, l1=l1, scale=scale, early_stopping=early_stopping)
        df.dropna(axis=0, inplace=True)
        x = df.pop('L1')

        plt.figure(figsize=kwargs.get('figsize', (12, 9)))
        plt.grid(kwargs.get('grid', True))
        for column in df.columns:
            plt.plot(x, df[column])
        plt.legend(df.columns, fontsize=12)
        plt.xlabel('L1', fontsize=14)
        plt.ylabel('coef', fontsize=14)
        plt.xlim([0, l1[x.shape[0]]])
        plt.show()

    def l1_best_features(self, target: str, l1_threshold: float, min_coef: float,
                         l1: tuple = tuple(2 ** np.linspace(-10, 10, 100)), scale: bool = False,
                         early_stopping: bool = False) -> pd.Series:
        """
        Identify the best features based on L1-regularized model coefficients at a specific L1 threshold.

        Parameters:
        -----------
        target : str
            The target column name.

        l1_threshold : float
            The L1 regularization strength at which to select the best features.

        min_coef : float
            The minimum absolute coefficient value to consider a feature important.

        l1 : tuple, optional, default=tuple(2 ** np.linspace(-10, 10, 100))
            The tuple of L1 regularization strengths to test.

        scale : bool, optional, default=False
            If True, scales the feature matrix X using StandardScaler.

        early_stopping : bool, optional, default=False
            If True, stops fitting models early if all coefficients in the model are zero.

        Returns:
        --------
        pd.Series:
            A Series containing the important features and their coefficients.

        Raises:
        -------
        AssertionError:
            If the input types are incorrect or if the target column is not in the DataFrame.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 3, 4, 5], "C": [5, 6, 7, 8]}))
        >>> df.l1_best_features(target="C", l1_threshold=0.01, min_coef=0.1)
        """
        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'
        assert isinstance(l1, (list, tuple, type(np.array)))
        assert isinstance(scale, bool), f'Incorrect dtype. scale: {type(scale)} instead of {bool}'
        assert isinstance(early_stopping,
                          bool), f'Incorrect dtype. early_stopping: {type(early_stopping)} instead of {bool}'

        x = self.drop(target, axis=1)
        y = self[target]

        df = self.__l1_importance(x, y, l1=l1, scale=scale, early_stopping=early_stopping)
        l1_place = df['L1'].to_list().index(
            df['L1'][min(range(len(df['L1'])), key=lambda i: abs(df['L1'][i] - l1_threshold))])
        res = df.iloc[l1_place, 1:]
        res = res[abs(res) > min_coef]
        return res

    def rfc_best_features(self, target: str, **kwargs) -> pd.Series:
        """
        Identify the best features using a RandomForestClassifier.

        Parameters:
        -----------
        target : str
            The target column name.

        kwargs : additional keyword arguments
            Additional parameters for RandomForestClassifier.

        Returns:
        --------
        pd.Series:
            A Series containing the feature importances, sorted in descending order.

        Raises:
        -------
        AssertionError:
            If the target column is not in the DataFrame.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 3, 4, 5], "C": [0, 1, 0, 1]}))
        >>> df.rfc_best_features(target="C")
        """
        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'

        X = self.drop(columns=[target])
        Y = self[target]

        model = RandomForestClassifier(random_state=1, **kwargs)
        model.fit(X, Y)
        imps = pd.Series(model.feature_importances_, index=X.columns)
        return imps.sort_values(ascending=False)

    def rfc_permutation_importance(self, target: str, **kwargs) -> pd.Series:
        """
        Calculate feature importance using permutation importance with a RandomForestClassifier.

        Parameters:
        -----------
        target : str
            The target column name.

        kwargs : additional keyword arguments
            Additional parameters for RandomForestClassifier.

        Returns:
        --------
        pd.Series:
            A Series containing the mean permutation importances, sorted in descending order.

        Raises:
        -------
        AssertionError:
            If the target column is not in the DataFrame.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 3, 4, 5], "C": [0, 1, 0, 1]}))
        >>> df.rfc_permutation_importance(target="C")
        """
        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'

        X = self.drop(columns=[target])
        Y = self[target]

        model = RandomForestClassifier(random_state=1, **kwargs)

        model.fit(X, Y)
        result = permutation_importance(model, X, Y)
        imps = pd.Series(result['importances_mean'], index=X.columns)
        return imps.sort_values(ascending=False)

    def compare_PCA_LDA_NCA(self, target: str, test_size: float = 0.2) -> None:
        """
        Compare dimensionality reduction techniques: PCA, LDA, and NCA, and visualize their results.

        Parameters:
        -----------
        target : str
            The target column name.

        test_size : float, optional, default=0.2
            The proportion of the data to include in the test split.

        Returns:
        --------
        None

        Raises:
        -------
        AssertionError:
            If the target column is not in the DataFrame.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({
        >>>     "A": np.random.rand(100), "B": np.random.rand(100), "C": np.random.randint(0, 2, 100)
        >>> }))
        >>> df.compare_PCA_LDA_NCA(target="C")
        """

        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'

        test_size = int(test_size * len(self))
        train_set, test_set = self.sample(frac=1, random_state=42)[test_size:], self.sample(frac=1, random_state=42)[
                                                                                :test_size]

        X_train, X_test = train_set.drop(target, axis=1), test_set.drop(target, axis=1)
        y_train, y_test = train_set[target], test_set[target]

        # Создание объектов для методов снижения размерности
        dim_reduction_methods = [
            ('PCA', PCA(n_components=1), PCA(n_components=2)),
            ('LDA', LinearDiscriminantAnalysis(n_components=1), LinearDiscriminantAnalysis(n_components=2)),
            ('NCA', NeighborhoodComponentsAnalysis(n_components=1), NeighborhoodComponentsAnalysis(n_components=2))
        ]

        # Создание фигуры и сетки подграфиков
        fig, axa = plt.subplots(3, 2, figsize=(12, 12))

        # Цикл по каждому методу снижения размерности
        for i, (name, model1, model2) in enumerate(dim_reduction_methods):
            # Обучение моделей на тренировочных данных
            model1.fit(X_train, y_train)
            model2.fit(X_train, y_train)

            # Преобразование данных
            X_tr1 = model1.transform(X_train)
            X_tr2 = model2.transform(X_train)

            # Построение scatter plot для данных, уменьшенных до 2 компонент
            axa[i, 0].scatter(X_tr2[:, 0], X_tr2[:, 1], c=y_train, s=30, cmap='Set1')
            axa[i, 0].set_title(f"{name}")
            axa[i, 0].grid()

            # Построение гистограммы для данных, уменьшенных до 1 компоненты
            sns.histplot(x=X_tr1.ravel(), hue=y_train, ax=axa[i, 1], element="poly")
            axa[i, 1].grid()

            # Создание и обучение классификатора KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_tr2, y_train)

            # Оценка точности классификатора на тестовых данных
            test_score = knn.score(model2.transform(X_test), y_test)
            axa[i, 0].set_title(f"{name}\nTest Score: {test_score:.2f}")

        # Отображение всех графиков
        plt.tight_layout()
        plt.show()

    def select_best_features(self, target: str, selector: str, test_size: float = 0.2, **kwargs) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Select the best features from the dataset using feature selection techniques.

        Parameters:
        -----------
        target : str
            The target column name.

        selector : str
            The feature selection method. Options: 'SelectPercentile', 'SelectKBest'.

        test_size : float, optional, default=0.2
            The proportion of the data to include in the test split.

        kwargs : additional keyword arguments
            Additional parameters for the feature selection methods.

        Returns:
        --------
        tuple[pd.DataFrame, pd.DataFrame]:
            Transformed training and test DataFrames with the selected features.

        Raises:
        -------
        AssertionError:
            If the target column is not in the DataFrame or if an incorrect selector is provided.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 3, 4, 5], "C": [0, 1, 0, 1]}))
        >>> x_train_new, x_test_new = df.select_best_features(target="C", selector="SelectKBest")
        """

        assert target in self.columns, f'target must be in columns of df: {list(self.columns)}'

        test_size = int(test_size * len(self))
        train_set, test_set = self.sample(frac=1, random_state=42)[test_size:], self.sample(frac=1, random_state=42)[
                                                                                :test_size]

        x_train, x_test = train_set.drop(target, axis=1), test_set.drop(target, axis=1)
        y_train = train_set[target]

        available_selectors = ('SelectPercentile'.lower(), 'SelectKBest'.lower())
        assert isinstance(selector, str), f'Incorrect parameter selector type. {type(selector)} instead of {str}'
        assert selector.lower() in available_selectors, f'Incorrect parameter selector. Available selectors: {available_selectors}'

        if selector.lower() == 'selectpercentile':
            selector = SelectPercentile(score_func=kwargs.get('score_func', f_classif),
                                        percentile=kwargs.get('percentile', 80))
        elif selector.lower() == 'selectkbest':
            selector = SelectKBest(score_func=kwargs.get('score_func', chi2),
                                   k=kwargs.get('k', round(len(x_train.columns) * 0.5)))

        x_train_new = selector.fit_transform(x_train, y_train)
        cols = selector.get_feature_names_out()
        x_test_new = selector.transform(x_test)

        return pd.DataFrame(x_train_new, columns=cols), pd.DataFrame(x_test_new, columns=cols)

    def cluster_outliers_search_plot_2d(self) -> None:
        """
        Detect and plot outliers in a 2D dataset using four different clustering-based methods.

        This method identifies outliers in a dataset with exactly two features by applying various outlier detection algorithms. The algorithms used are:

        1. **One-Class SVM (OCSVM)**: A support vector machine-based method that attempts to separate the majority of the data from the outliers by learning a decision boundary around the data.
        2. **Isolation Forest (IsoForest)**: An ensemble method that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
        3. **Elliptic Envelope (ELL)**: Assumes the data is Gaussian distributed and fits an ellipse to the central data points, labeling points outside the ellipse as outliers.
        4. **Local Outlier Factor (LOF)**: Measures the local deviation of density of a data point with respect to its neighbors, identifying points that have a substantially lower density than their neighbors as outliers.

        Each algorithm creates a contour plot representing the decision boundary for outlier detection. The data points are then plotted, and contours are visualized to show the regions identified as containing outliers by each algorithm.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Raises:
        -------
        AssertionError:
            If the DataFrame does not have exactly 2 features.

        Example:
        --------
        >>> df = DataFrame(pd.DataFrame({"A": np.random.rand(100), "B": np.random.rand(100)}))
        >>> df.cluster_outliers_search_plot_2d()
        """

        assert len(self.columns) == 2, f'DataFrame must have 2 features. Got {len(self.columns)} instead'

        X = self

        # Создаем словарь классификаторов для обнаружения выбросов
        classifiers = {
            "OCSVM": OneClassSVM(nu=0.1),  # One-Class SVM, где nu - это процент выбросов
            "IsoForest": IsolationForest(),  # Лес изоляции
            "ELL": EllipticEnvelope(contamination=0.2),
            # Эллиптическая оболочка, где contamination - это процент выбросов
            "LOF": LocalOutlierFactor(novelty=True)
            # Локальный фактор выбросов, с опцией novelty для работы с новыми данными
        }

        # Определяем цвета для контуров каждого классификатора
        colors = ['m', 'g', 'b', 'y']
        # Словарь для хранения контурных объектов для каждого классификатора
        legend1 = {}

        # Создание сетки координат для построения контуров
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx1, yy1 = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

        plt.figure(1, figsize=(12, 9))

        # Проходим по каждому классификатору и строим контуры
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            # Обучение классификатора на данных X
            clf.fit(X.to_numpy())
            # Вычисление функции решения для каждой точки на сетке координат
            Z1 = clf.decision_function(np.c_[np.ravel(xx1), np.ravel(yy1)])
            # Преобразование одномерного массива Z1 обратно в двумерный массив
            Z1 = Z1.reshape(xx1.shape)

            # Построение контура для уровня 0 и сохранение его в словарь legend1
            legend1[clf_name] = plt.contour(xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])

        # Преобразование значений и ключей словаря legend1 в списки
        legend1_values_list = list(legend1.values())
        legend1_keys_list = list(legend1.keys())

        # Создание новой фигуры для отображения результатов
        plt.figure(1, figsize=(12, 9))  # Фигура для отображения двух кластеров
        plt.title("Обнаружение выбросов в наборе данных")

        # Отрисовка исходных данных
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], color='black')

        plt.xlim((xx1.min(), xx1.max()))
        plt.ylim((yy1.min(), yy1.max()))

        # Создание легенды, используя элементы контуров из legend1
        plt.legend(
            [legend1_values_list[0].legend_elements()[0][0],
             legend1_values_list[1].legend_elements()[0][0],
             legend1_values_list[2].legend_elements()[0][0],
             legend1_values_list[3].legend_elements()[0][0]],
            legend1_keys_list,
            loc="upper center",
            prop=matplotlib.font_manager.FontProperties(size=11)
        )

        plt.ylabel(X.columns[0])
        plt.xlabel(X.columns[1])
        plt.show()
