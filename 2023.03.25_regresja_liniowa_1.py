import pandas as pd    #używamy 'as' i dzieki temu w dalszej czesci programu wystarczy napisac "pd" zamiast pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Doinstalować należy ("TERMMINAL"):
#python -m pip install -U scikit-learn
#pip install sklearn
#pip install pandas
#pip install seaborn

df = pd.read_csv("weight-height.csv")   #df - data frame
print(type(df))
print(df)
print(df.head(10))   #10 wierszy
#print(df.Height) # dana kolumna
print(df.Gender.value_counts())
df.Height *= 2.54 #mnożymy całe kolumny
df.Weight /= 2.2 #dzielimy całe kolumny
print(df.head(5))

#2 zmienne niezależne - gender i height
#1 zmienna zależna - weight

#sns.displot(df.Weight) #wybieramy dane do wygenerowania wykresu, kobiety i mezczyzni razem

sns.displot(df.query("Gender=='Male'").Weight)
sns.displot(df.query("Gender=='Female'").Weight)
plt.show() #funkcja wyświetlania wykresu

df = pd.get_dummies(df)  # zamienia dane niemeryczne na numeryczne
del(df["Gender_Male"]) #usuń kolumnę
df.rename(columns={'Gender_Female': 'Gender'}, inplace=True)
print(df.head(5))
#dane na stole

model = LinearRegression()   #wybieram algorytm
model.fit(df[ ['Height', 'Gender'] ], df['Weight'] )   #policz
print(model.coef_, model.intercept_)
print('wzór: Height * ', model.coef_[0], '+ Gender *', model.coef_[1],' = Weight')

#własna formuła
gender = 0  #male
height = 192
weight = model.intercept_ + model.coef_[0] * height + model.coef_[1] * gender
print(weight)