import pandas as pd
import numpy as np
from unicodedata import normalize
#%pip install scikit-learn
from sklearn.model_selection import RandomizedSearchCV
#import plotly.graph_objs as go #Graficos
#import plotly.express as px
#from matplotlib.pyplot import figure
#import matplotlib.pyplot as plt ### gráficos
from sklearn import linear_model ## para regresión lineal
from sklearn import tree ###para ajustar arboles de decisión
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor ##Ensamble con bagging
from sklearn.ensemble import GradientBoostingRegressor ###Ensamble boosting
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sqlite3 as sql
import datetime as datetime
#import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def RMSE(y_actual,y_predicted):
    import math
    MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
    RMSE = math.sqrt(MSE)
    return RMSE

df=pd.read_csv(r"D:\Users\USUARIO\Desktop\Analitica3\Preprocesamiento_Finanzas1.csv")
df
df.drop(columns='Unnamed: 0',inplace=True)
scaler=MinMaxScaler()
df["Edad"]=scaler.fit_transform(df[["Edad"]])
df["Precio"]=scaler.fit_transform(df[["Precio"]])
df
x=df[['Reclamacion_codigo', 'Diagnostico_Codigo', 'Cantidad',
       'Sexo_codigo', 'Edad', 'Regional_codigo']]
y=df['Precio']
X=pd.get_dummies(x,columns=['Reclamacion_codigo', 'Diagnostico_Codigo',
       'Sexo_codigo', 'Regional_codigo'])


#crear un modelo de selección
est_prueba = SelectKBest(score_func=f_regression, k=3)
est_ajustado = est_prueba.fit(X, y)

#Se muestran el desempeño de los features basado en el valor F
np.set_printoptions(precision=3)
print(est_ajustado.scores_)
features = est_ajustado.transform(X)
print(features)
#Filers
d1={'features':X.columns,'score':est_ajustado.scores_}
fs=pd.DataFrame(d1)
fs.sort_values(by=['score'],ascending=False,inplace=True)

selected_features=fs[fs['score']>fs['score'].mean()]['features'].values

X_pred=X[selected_features]

with pd.ExcelWriter('grouping.xlsx') as writer:  
    X_pred.to_excel(writer, sheet_name='Hoja1')
#---------Renamed columns--------#
df1=pd.read_csv(r"D:\Users\USUARIO\Desktop\Analitica3\grouping2.csv")
df1
df1.drop(columns='Unnamed: 0',inplace=True)

sorted_cols=["31","0","A09X","A400","A418","B211","B582","C021",
             "C099","C189","C20X","C257","C312","C349","C482",
             "C509","C56X","C577","C61X","C66X","C710","C712",
             "C720","C762","C833","C845","C851","C859","C900",
             "C919","C921","D039","D057","D059","D075","D166","D250",
             "D259","D371","D391","D397","D430","D441","D443","D469",
             "D477","D489","D649","E46X","E725","F412","G040","G360",
             "G448","G459","G618","G619","G638","G737","G822","H023",
             "H024","H251","H258","H259","H269","H46X","I200","I209",
             "I214","I219","I251","I340","I350","I441","I470","I471",
             "I48X","I495","I498","I500","I620","I698","I702","I714",
             "I743","I770","I839","J159","J15X","J189","J209","J219",
             "J329","J342","J343","J451","J90X","K108","K359","K409",
             "K449","K509","K564","K578","K660","K801","K802","K808",
             "K922","L03X","L984","M150","M161","M169","M170","M171",
             "M179","M199","M211","M220","M233","M238","M332","M410",
             "M464","M484","M508","M511","M518","M541","M542","M544",
             "M545","M678","M705","M751","M869","M872","N179","N200",
             "N201","N209","N23X","N370","N390","N40X","N411","N801",
             "N939","O109","O367","O471","P038","P059","P073","P229",
             "Q211","Q443","Q614","Q670","R065","R074","R101","R102",
             "R103","R104","R17X","R509","R51X","R55","R568","R571",
             "R935","S525","S720","S721","S730","S819","S826","S830",
             "S832","S833","S835","S860","S921","T814","T889","Y225",
             "Z321","Z966","Z988","Edad","1","13","18","21","25","27",
             "29","30","32","4","8","9"]

dfinal=df1[sorted_cols]

with pd.ExcelWriter('sortedngrouped.xlsx') as writer:  
    dfinal.to_excel(writer, sheet_name='Hoja1')

grouped=pd.read_csv(r'D:\Users\USUARIO\Desktop\Analitica3\sortedngrouped.csv')
grouped

drop_cols=["A09X","A400","A418","B211","B582","C021",
             "C099","C189","C20X","C257","C312","C349","C482",
             "C509","C56X","C577","C61X","C66X","C710","C712",
             "C720","C762","C833","C845","C851","C859","C900",
             "C919","C921","D039","D057","D059","D075","D166","D250",
             "D259","D371","D391","D397","D430","D441","D443","D469",
             "D477","D489","D649","E46X","E725","F412","G040","G360",
             "G448","G459","G618","G619","G638","G737","G822","H023",
             "H024","H251","H258","H259","H269","H46X","I200","I209",
             "I214","I219","I251","I340","I350","I441","I470","I471",
             "I48X","I495","I498","I500","I620","I698","I702","I714",
             "I743","I770","I839","J159","J15X","J189","J209","J219",
             "J329","J342","J343","J451","J90X","K108","K359","K409",
             "K449","K509","K564","K578","K660","K801","K802","K808",
             "K922","L03X","L984","M150","M161","M169","M170","M171",
             "M179","M199","M211","M220","M233","M238","M332","M410",
             "M464","M484","M508","M511","M518","M541","M542","M544",
             "M545","M678","M705","M751","M869","M872","N179","N200",
             "N201","N209","N23X","N370","N390","N40X","N411","N801",
             "N939","O109","O367","O471","P038","P059","P073","P229",
             "Q211","Q443","Q614","Q670","R065","R074","R101","R102",
             "R103","R104","R17X","R509","R51X","R55","R568","R571",
             "R935","S525","S720","S721","S730","S819","S826","S830",
             "S832","S833","S835","S860","S921","T814","T889","Y225",
             "Z321","Z966","Z988",'Unnamed: 0']
grouped.drop(columns=drop_cols,inplace=True)


minmax_cols=['Infec_Paras', 'tumors', 'neu_sys_issues', 'eye_iss',
       'circ_sys_iss', 'res_sys_iss', 'diges_sys_iss', 'bone_musc_sys_iss',
       'gen_ur_sys_iss', 'pregnancy', 'perinatal_iss', 'abnormalities',
       'traumas_poison_ext_causes', 'health_feat',"1","13","18","21","25","27",
             "29","30","32","4","8","9"]

scaler=MinMaxScaler()
for i in minmax_cols:
    grouped[i]=scaler.fit_transform(grouped[[i]])
grouped.shape
grouped

X_train, X_test, y_train, y_test = train_test_split(grouped, y, test_size=0.2, random_state=42)



#______________MODELOS_____________#

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
# Always scale the input. The most convenient way is to use a pipeline.
reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))
reg.fit(X_train, y_train)

SGDpred=reg.predict(X_test)

RMSE(y_test,SGDpred)

from sklearn.metrics import r2_score
r2_score(y_test, SGDpred)
#-----------------------------------------------------#

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)

RF_pred=regr.predict(X_test)

print('RMSE',RMSE(y_test,RF_pred))

print('R2',r2_score(y_test, RF_pred))

print("MSE",mean_squared_error(y_test, RF_pred))

print("MAE",mean_absolute_error(y_test, RF_pred))
#------------------------------------------------------#
from sklearn.linear_model import LinearRegression

regl = linear_model.LinearRegression()
regl.fit(X_train, y_train)

RF_pred=regr.predict(X_test)

print('RMSE',RMSE(y_test,RF_pred))

print('R2',r2_score(y_test, RF_pred))

print("MSE",mean_squared_error(y_test, RF_pred))

print("MAE",mean_absolute_error(y_test, RF_pred))

#------------------------------------------------------#

from sklearn import tree

regr = tree.DecisionTreeRegressor()
regr.fit(X_train, y_train)

RF_pred=regr.predict(X_test)

print('RMSE',RMSE(y_test,RF_pred))

print('R2',r2_score(y_test, RF_pred))

print("MSE",mean_squared_error(y_test, RF_pred))

print("MAE",mean_absolute_error(y_test, RF_pred))

#------------------------------------------------------#

from sklearn.linear_model import Ridge

regr= Ridge(alpha=1)

regr.fit(X_train, y_train)

RF_pred=regr.predict(X_test)

print('RMSE',RMSE(y_test,RF_pred))

print('R2',r2_score(y_test, RF_pred))

print("MSE",mean_squared_error(y_test, RF_pred))

print("MAE",mean_absolute_error(y_test, RF_pred))

#------------------------------------------------------#
from sklearn.model_selection import GridSearchCV

parameters = {'fit_intercept': [True, False], 'normalize': [True, False]}

# Búsqueda en cuadrícula para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(regl, parameters, cv=5)
grid_search.fit(X_train, y_train)

# Mejores hiperparámetros encontrados
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)

#------------------------------------------------------#

regl = linear_model.LinearRegression(fit_intercept=True, normalize=True)

regl.fit(X_train, y_train)

RF_predML=regl.predict(X_test)

print('RMSE',RMSE(y_test,RF_pred))

print('R2',r2_score(y_test, RF_pred))

print("MSE",mean_squared_error(y_test, RF_pred))

print("MAE",mean_absolute_error(y_test, RF_pred))

#------------------------------------------------------#

#DESPLIEGUE DEL MODELO 

Dfprediccion=pd.DataFrame({"Prediccion": RF_pred})
Dfprediccion=scaler.inverse_transform(Dfprediccion[["Prediccion"]])
Dfprediccionf=pd.DataFrame(Dfprediccion)
Dfprediccionf.rename(columns={"0":"Prediccion"}, inplace=True)
Dfprediccionf

