import pandas as pd    #używamy 'as' i dzieki temu w dalszej czesci programu wystarczy napisac "pd" zamiast pandas
from sklearn.linear_model import LinearRegression

#Doinstalować należy ("TERMMINAL"):
#python -m pip install -U scikit-learn
#pip install sklearn
#pip install pandas

df = pd.read_csv("weight-height.csv")   #df - data frame
print(type(df))
print(df)
print(df.head(10))   #10 wierszy