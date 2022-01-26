import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


#######################################################################

# 1 - VERI SETININ OKUNMASI

data_set = pd.read_csv("insurance.csv")

#print(data_set.to_string())

#print(data_set.describe())

##############################################################################################################################################

# 2 - VERI ON ISLEME ASAMSI

# 2.1 - Eksik Veri Analizi

#print(data_set.isnull().sum())
# bu kisimda veri setimizde eksik veri olmadigini ogreniyoruz

##############################################################################################################################################

# 2.2 - Verilerin, Sutunlarda Dengeli Dagilip Dagilmadiklarini Inceliyoruz

age_statistic = data_set["age"].value_counts()
sex_statistic = data_set["sex"].value_counts()
bmi_statistic = data_set["bmi"].value_counts()
children_statistic = data_set["children"].value_counts()
smoker_statistic = data_set["smoker"].value_counts()
region_statistic = data_set["region"].value_counts()

"""print("yaş kolonundaki değerkerin kaç defa tekrarladıkları : \n",age_statistic)
print("\n\n")
print("medeni durum degerlerinin kac defa tekrarladiklari : \n",sex_statistic)
print("\n\n")
print("vucut kitle indeksi durumlarinin kac defa tekrarladiklari : \n",bmi_statistic)
print("\n\n")
print("cocuk sayisi verilerinin kac defa tekrarladiklari : \n",children_statistic)
print("\n\n")
print("sigara icip icmediklerine iliskin degerlerinin kac defa tekrarladiklari : \n",smoker_statistic)
print("\n\n")
print("bolge ozniteligine iliskin verilerin kac defa tekrarladiklari : |n",region_statistic)
"""
# burada verilerin kolonlarda dengeli dagildigini goruyoruz.

##############################################################################################################################################

# 2.3 - ONE HOT ENCODING ILE KATEGORIK VERILERIN NOMINAL VERILERE DONUSTURULMESI

sex_types = pd.get_dummies(data_set.sex, prefix='sex')
smoker_types = pd.get_dummies(data_set.smoker, prefix='smoker')
region_types = pd.get_dummies(data_set.region, prefix='region')

data_set = pd.concat([data_set, sex_types, smoker_types, region_types], axis=1)

#print(data_set.to_string())

data_set.drop(['sex', 'smoker', 'region', 'sex_female', 'smoker_no'], axis=1, inplace = True)

#print(data_set.to_string())

##############################################################################################################################################

# 2.4 - VERI SETININ X VE Y DEGISKENLERINE AYIRMA

y = data_set["charges"]

data_set.drop(["charges"], axis=1, inplace=True)

x = data_set


#print(x.to_string())

#print(y.to_string())

##############################################################################################################################################

# 3 - MAKINE OGRENMESI ASAMASINA HAZIRLIK

# 3.1 - verileri egitim ve test verisi olmak uzerek 2 ye ayiriyoruz.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=46)

################################################################################################################################

# 3.2 - veri setinde normalizasyon islemi yapiyoruz.

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

################################################################################################################################

# 4 - MAKINE OGRENMESI VE MODEL PERFORMANSI

# 4.1 - KARAR AGACI REGRESYON ALGORITMASI

tree_regression = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_regression = tree_regression.fit(x_train, y_train)
tahmin_tree_regression = tree_regression.predict(x_test)


# 4.2 - RANDOM FORREST REGRESYON ALGORITMASI

random_regression = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
random_regression.fit(x_train, y_train)
tahmin_random_regression = random_regression.predict(x_test)


# 4.3 - LASSO REGRESYON ALGORITMASI

lassoReg = Lasso(alpha=2)
lassoReg.fit(x_train,y_train)
tahmin_lasso = lassoReg.predict(x_test)


# 4.4 - Elastic REGRESYON ALGORITMASI

elastic_reg = ElasticNet(random_state=0)
elastic_reg.fit(x_train, y_train)
tahmin_elastic = elastic_reg.predict(x_test)


# 4.5 RIDGE REGRESYON ALGORITMASI

ridge_reg = Ridge()
ridge_reg.fit(x_train, y_train)
tahmin_ridge = ridge_reg.predict(x_test)

################################################################################################################################

# 5 - SONUCLAR

predicts = [tahmin_tree_regression,tahmin_random_regression,tahmin_lasso, tahmin_elastic, tahmin_ridge]

algoritma_names = ["Karar Ağacı Regresyon", "Random Forrest regresyon", "Lasso Regresyon", "Elastic Regresyon", "Ridge Regresyon"]


# SONUCLARI HESAPLAMA FONKSIYONU
def performance_calculate(predict):
    mae = mean_absolute_error(y_test, predict)
    mse = mean_squared_error(y_test, predict)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = r2_score(y_test, predict)

    data = [mae, mse, rmse, r2]

    return data



# EKRANA YAZDIRMAK
seriler = []
metrics = ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)" , "R2"]

for i in predicts:
    data = performance_calculate(i)
    seriler.append(data)

from IPython.display import HTML

df = pd.DataFrame(data = seriler, index = algoritma_names, columns = metrics)
pd.set_option('display.colheader_justify', 'center') # kolon isimlerini ortaliyoruz

print(df.to_string())


###############################################################



