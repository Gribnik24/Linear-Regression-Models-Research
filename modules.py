import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from typing import Union, List

class Custom_OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom OneHotEncoder. Takes `columns_to_encode` and `column_to_compare` in transform method.
    Param `columns_to_encode` contains the name (str) or names (list) to find in `column_to_compare` column
    containing list of features in each cell
    """
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    @staticmethod
    def _feature_onehotencoder(df: pd.DataFrame, df_length: int,
                               feature_name: str, column_to_compare: str) -> pd.Series:
        encoded_feature = pd.Series([0] * df_length)
        for index, row in df.iterrows():
            if feature_name in row[column_to_compare]:
                encoded_feature.loc[index] = 1
        return encoded_feature

    def transform(self, X, columns_to_encode: Union[str, List[str]] = None, column_to_compare: str = None):
        if column_to_compare is None:
            raise ValueError("'columns_to_encode' param shouldn't be None")
        if columns_to_encode is None:
            raise ValueError("'column_to_compare' param shouldn't be None")
        
        columns_vartype = type(columns_to_encode).__name__ 

        df = X.copy().reset_index(drop=True)
        df_length = df.shape[0]

        if columns_vartype == 'str':
            df[columns_to_encode] = self._feature_onehotencoder(df=df, df_length=df_length,
                                                                feature_name=columns_to_encode,
                                                                column_to_compare=column_to_compare)
        elif columns_vartype == 'list':
            for col in columns_to_encode:
                df[col] = self._feature_onehotencoder(df=df, df_length=df_length,
                                                      feature_name=col,
                                                      column_to_compare=column_to_compare)   
        else:
            raise ValueError("'columns_to_encode' param should be list or str type")
                
        return df
    

class Custom_MinMaxScaler(BaseEstimator, TransformerMixin):
    """Custom MinMaxScaler. Scales features to a given range, typically [0, 1]."""
    def __init__(self, feature_range=(0,1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.feature_names_in_ = None
        
    def fit(self, X, y=None):
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X = X.values
            
        X = np.array(X)
        
        # Store min and max for each feature
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        # Calculate scale factor
        feature_range_min, feature_range_max = self.feature_range
        self.scale_ = (feature_range_max - feature_range_min) / (self.data_max_ - self.data_min_)
        
        # Handle constant features (avoid division by zero)
        self.scale_[self.data_max_ == self.data_min_] = 0
        
        self.min_ = feature_range_min - self.data_min_ * self.scale_
        
        return self
    
    def transform(self, X):
        if self.scale_ is None:
            raise ValueError("Scaler must be fitted before transform")
        
        # Store original type and columns
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            columns = X.columns
            index = X.index
            X = X.values
            
        X = np.array(X)
        
        # Apply scaling
        X_scaled = X * self.scale_ + self.min_
        
        # Return in original format
        if is_dataframe:
            return pd.DataFrame(X_scaled, columns=columns, index=index)
        return X_scaled
        
class Custom_StandardScaler(BaseEstimator, TransformerMixin):
    """Custom StandardScaler. Standardizes features by removing mean and scaling to unit variance."""
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X = X.values
        
        X = np.array(X)
        self.n_features_in_ = X.shape[1]
        
        # Calculate mean
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        
        # Calculate standard deviation
        if self.with_std:
            self.var_ = np.var(X, axis=0)
            self.scale_ = np.sqrt(self.var_)
            # Handle constant features (zero variance)
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_ = np.ones(X.shape[1])
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if self.scale_ is None:
            raise ValueError("Scaler must be fitted before transform")
        
        # Store original type and columns
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            columns = X.columns
            index = X.index
            X = X.values
        
        X = np.array(X)
        
        # Apply standardization: (X - mean) / scale
        X_scaled = (X - self.mean_) / self.scale_
        
        # Return in original format
        if is_dataframe:
            return pd.DataFrame(X_scaled, columns=columns, index=index)
        return X_scaled

class Custom_AnalyticalLinearRegression(BaseEstimator, RegressorMixin):
    """
    Linear Regression algorithm with analytical solution (also called the Normal Equation method)
    """
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept=fit_intercept
        self.weights=None
        
    def fit(self, X, y):
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
            
        # Reshape y to column vector if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Add intercept column if needed
        if self.fit_intercept:
            front_addition = np.ones(X.shape[0])
            X = np.column_stack([front_addition, X])

        # Finding weights with analytical solution
        X_t = X.T
        try:
            self.weights = np.linalg.inv((X_t @ X)) @ X_t @ y
        except np.linalg.LinAlgError:
            # Hadling singluar matrices
            self.weights = np.linalg.pinv((X_t @ X)) @ X_t @ y

        if self.fit_intercept:
            self.intercept_ = self.weights[0]
            self.coef_ = self.weights[1:]
        else:
            self.intercept_ = 0
            self.coef_ = self.weights

        return self

    def predict(self, X) -> np.array:
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Add intercept column for prediction
        if self.fit_intercept:
            front_addition = np.ones(X.shape[0])
            X = np.column_stack([front_addition, X])

        predictions = X.dot(self.weights).flatten() # Return as 1D array

        return predictions
    
class Custom_SGDLinearRegression(BaseEstimator, RegressorMixin):
    """
    Linear Regression algorithm with SGD (stochastic gradient descent) implementation
    Possible 'regularization' param options:
    - none: classic SGDLinearRegression
    - l1: L1 regularization (with 'alpha' param)
    - l2: L2 regularization (with 'alpha' param)
    - ElasticNet: ElasticNet (with 'alpha' and 'l1_ratio' (0 = L2, 1 = L1) params)
    """
    def __init__(self, fit_intercept: bool = True,
                 random_state: int = 42,
                 learning_rate=0.001,
                 epochs=1000,
                 batch_size=32,
                 regularization='none',  # 'none', 'l1', 'l2', 'elasticnet'
                 alpha=0.01,  # Regularization strength
                 l1_ratio=0.5):  # Only for elasticnet (0 = L2, 1 = L1)
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    
    def fit(self, X, y):
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
            
        # Reshape y to column vector if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_length, n_features = X.shape

        # Add intercept column if needed
        if self.fit_intercept:
            front_addition = np.ones(X_length)
            X = np.column_stack([front_addition, X])
            n_features += 1
        
        # Initialize weights
        np.random.seed(self.random_state)
        self.weights = np.random.randn(n_features, 1) # Random values from Gaussian distribution

        # SGD training
        for epoch in range(self.epochs):
            # Shuffle data
            np.random.seed(self.random_state + epoch)  # Different but deterministic per epoch
            indices = np.random.permutation(X_length)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch SGD
            for i in range(0, X_length, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Calculate predictions and gradients
                batch_size_actual = len(X_batch)
                predictions = X_batch.dot(self.weights)
                error = predictions - y_batch
                
                # Gradient: (2/m) * X^T * (Xw - y)
                gradients = (2 / batch_size_actual) * X_batch.T.dot(error) 
                if self.regularization == 'none':
                    pass
                elif self.regularization == 'l1':
                    # Apply L1 regularization to all weights except intercept
                    reg_term = self.alpha * np.sign(self.weights)
                    if self.fit_intercept:
                        reg_term[0] = 0
                    gradients += reg_term
                elif self.regularization == 'l2':
                    # Apply L2 regularization to all weights except intercept
                    reg_term = 2 * self.alpha * self.weights
                    if self.fit_intercept:
                        reg_term[0] = 0
                    gradients += reg_term
                elif self.regularization == 'ElasticNet':
                    # Apply ElasticNet regularization to all weights except intercept
                    l1_grad = self.alpha * self.l1_ratio * np.sign(self.weights)
                    l2_grad = 2 * self.alpha * (1 - self.l1_ratio) * self.weights
                    if self.fit_intercept:
                        l1_grad[0] = 0
                        l2_grad[0] = 0 
                    gradients = gradients + l1_grad + l2_grad 
                else:
                    raise ValueError('Wrong regularization param.')
                
                # Gradient clipping
                grad_norm = np.linalg.norm(gradients)
                if grad_norm > 1.0:
                    gradients = gradients / grad_norm
                    
                # Update weights
                self.weights -= self.learning_rate * gradients
         
        if self.fit_intercept:
            self.intercept_ = self.weights[0]
            self.coef_ = self.weights[1:]
        else:
            self.intercept_ = 0
            self.coef_ = self.weights
         
    def predict(self, X) -> np.array:
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Input NaN check 
        if np.isnan(X).any():
            print("Warning: NaN in input features")
            return np.zeros(X.shape[0])
            
        # Add intercept column for prediction
        if self.fit_intercept:
            front_addition = np.ones(X.shape[0])
            X = np.column_stack([front_addition, X])
            
        # Weights NaN check 
        if np.isnan(self.weights).any():
            print("Warning: NaN in weights")
            return np.zeros(X.shape[0])

        predictions = X.dot(self.weights).flatten() # Return as 1D array

        return predictions

def custom_R2_coef(y_true: Union[pd.Series, np.ndarray[float]], y_pred: Union[pd.Series, np.ndarray[float]]) -> float:
    """Calculate the coefficient of determination (R^2) for a regression model."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    numerator = ((y_true - y_pred)**2).sum()
    denominator = ((y_true - y_true.mean())**2).sum()
    r_squared = 1.0 - numerator / denominator
    
    return r_squared