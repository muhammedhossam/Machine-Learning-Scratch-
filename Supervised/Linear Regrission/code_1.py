import numpy as np


class SimpleLinearRegression:
    
    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = None
        self.residuals_ = None
        self.RSS = None
        self.TSS = None
        self.r2score_ = None
        self.MAE = None
        self.MSE = None
        self.RMSE = None
        
    def fit(self,X,y):
        x_ = X.mean()
        y_ = y.mean()
        
        up = np.dot((X - x_).T,(y - y_))
        down = np.sum((X - x_)**2)
        
        self.coefficients_ = up / down
        
        x_ = np.mean(X)
        y_ = np.mean(y)
        self.intercept_ = y_ - self.coefficients_ * x_
        y_pred = self.coefficients_ * X + self.intercept_ 
        
        self.residuals_ = y - y_pred
        
        # Calculate the residual sum of squares (RSS) and total sum of squares (TSS) 
        self.RSS = np.sum(self.residuals_ ** 2)
        self.TSS = np.sum((y - np.mean(y)) ** 2)   
        
        # Calculate the coefficient of determination (R-squared)
        self.r2score_ = 1 - (self.RSS / self.TSS)
        
        # Calculate the mean absolute error (MAE)
        self.MAE = np.mean(np.abs(self.residuals_))

        # Calculate the mean squared error (MSE)
        self.MSE = np.mean(self.residuals_ ** 2)

        # Calculate the root mean squared error (RMSE)
        self.RMSE = np.sqrt(self.MSE)   
        
    def predict(self, X):
        return X * self.coefficients_  + self.intercept_
    
    
    
class MultipleLinearRegression:
    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = None
        self.residuals_ = None
        self.RSS = None
        self.TSS = None
        self.r2score_ = None
        self.MAE = None
        self.MSE = None
        self.RMSE = None
        
    def fit(self, X, y):
        n, p = X.shape[0], X.shape[1]
        
        # Add a column of ones to the X matrix for the intercept_ 
        ones = np.ones((n, 1))
        X = np.hstack((ones, X))
        
        # Calculate the coefficients (b1, b2, ... bp) and intercept (bo)
        # b = (X.T * X)^-1 * X.T . y
        X_transpose = np.transpose(X)
        X_transpose_X = np.dot(X_transpose, X)
        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
        X_transpose_X_inv_X_transpose = np.dot(X_transpose_X_inv, X_transpose)
        
        self.coefficients_ = np.dot(X_transpose_X_inv_X_transpose, y)
        self.intercept_ = self.coefficients_[0]
        
        # Calculate the predicted values and residuals
        y_pred = np.dot(X, self.coefficients_)
        self.residuals_ = y - y_pred
        
        # Calculate the residual sum of squares (RSS) anf total sum of squares (TSS) 
        self.RSS = np.sum(self.residuals_ ** 2)
        self.TSS = np.sum((y - np.mean(y)) ** 2)   
        
        # Calculate the coefficient of determination (R-squared)
        self.r2score_ = 1 - (self.RSS / self.TSS)
        
        # Calculate the mean absolute error (MAE)
        self.MAE = np.mean(np.abs(self.residuals_))

        # Calculate the mean squared error (MSE)
        self.MSE = np.mean(self.residuals_ ** 2)

        # Calculate the root mean squared error (RMSE)
        self.RMSE = np.sqrt(self.MSE)
        
    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        return np.dot(X, self.coefficients_)
    
    
    
class Metrics:
    @staticmethod
    def r2_score(y_true, y_pred):
        SST = np.sum((y_true - np.mean(y_true)) ** 2)
        RSS = np.sum((y_true - y_pred) ** 2)
        return 1 - (RSS / SST)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse