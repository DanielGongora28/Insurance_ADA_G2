import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor


def getXY(df):
    y=df['Precio']
    x=df.drop(columns='Precio')
    return x,y



db=pd.read_csv('Preprocesamiento_Finanzas.csv')

#Not Dummis 'CANCER', 'EPOC','DIABETES', 'HIPERTENSION', 'ENF_CARDIOVASCULAR'

#Drop useless columns
db.drop(columns=['Unnamed: 0'],inplace=True)

#Get Dummies
df_pred=pd.get_dummies(db,columns=['Reclamacion_codigo', 'Diagnostico_Codigo',
                            'Sexo_codigo', 'Regional_codigo'],drop_first=True)

x,y=getXY(df_pred)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.66, random_state=42)


SGDreg = make_pipeline(StandardScaler(),
                 SGDRegressor(max_iter=1000, tol=1e-3,learning_rate='optimal'))
SGDreg.fit(X_train, y_train)
pred=SGDreg.predict(X_test)
mean_squared_error(y_test, pred)

#GBR = GradientBoostingRegressor()
#parameters = {'learning_rate': [0.01,0.02,0.03,0.04],
                  #'subsample'    : [0.9, 0.5, 0.2, 0.1],
                  #'n_estimators' : [100,500,1000, 1500],
                  #'max_depth'    : [4,6,8,10]
                 #}
#grid_GBR = GridSearchCV(estimator=GBR, param_grid = parameters, cv = 2, n_jobs=-1)
#grid_GBR.fit(X_train, y_train)


#print(" Results from Grid Search " )
#print("\n The best estimator across ALL searched params:\n",grid_GBR.best_estimator_)
#print("\n The best score across ALL searched params:\n",grid_GBR.best_score_)
#print("\n The best parameters across ALL searched params:\n",grid_GBR.best_params_)
###Using SGD Regressor for Scikit model sleection#

 ##Using Lasso Regression for try another model##
##

#LassoReg = Lasso(alpha=2,max_iter=3000)
#LassoReg.fit(X_train,y_train)
#pred2=LassoReg.predict(X_test)

###Hacer agrupaciones de los codigos de enfermedad###
#Escalar los valores numericos#
#Feature selection#


