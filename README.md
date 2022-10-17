# assignmnet
# ML_labs_Assignments

      import matplotlib.pyplot as plt 
      from sklearn import datasets
      import seaborn as sns
      import numpy as np
      diabetes = datasets.load_diabetes()

      print(diabetes.DESCR)

       print(diabetes.keys()) 
       print(diabetes.data)

       X = diabetes.data
       Y = diabetes.target

      X.shape, Y.shape

       from sklearn.model_selection import train_test_split
       X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

       X_train.shape, Y_train.shape
       X_test.shape, Y_test.shape

      from sklearn import linear_model
      from sklearn.metrics import mean_squared_error, r2_score

      model.fit(X_train, Y_train)

      Y_pred = model.predict(X_test)

      print('Coefficients:', model.coef_)
       print('Intercept:', model.intercept_)

      print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))

        print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))
      
      
      Y_pred = model.predict(X_train)

      print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred))

      print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred))
      
      
      plt.scatter(X_test[:,1], Y_test) 
       plt.plot(X_test, Y_pred)
      plt.show()
