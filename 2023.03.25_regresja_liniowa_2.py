import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('otodom.csv')
print(df.head(5).to_string())   #wyświetlanie wszystkich kolumn

print(df.iloc[2, 4])        #pokaż wartość z wiersza 2 i kolumny 4
print(df.iloc[:4, 1:4])     #pokaż rzędy o początku do 4 oraz kolumny 1-4

print(df.describe())    #dane statystyczne naszych danych
#print(df.corr())        #korelacje między danymi

sns.heatmap(df.corr(), annot=True)      #korelacje w postaci mapy
plt.show()

sns.displot(df.cena)
plt.show()
plt.scatter(df.powierzchnia, df.cena)
plt.show()
print(df.describe())

#najtańsze mieszkania chcemy odrzucić
#aby nasze statystyki były najbardziej rzetelne (bez krajnych wartości)
_min = df.describe().loc['min','cena']
q1 = df.describe().loc['25%','cena']
q3 = df.describe().loc['75%','cena']

print(_min, q1, q3)

df1 = df[ (df.cena >= _min) & (df.cena <= q3) ]
sns.displot(df1.cena)
plt.show()

#podział na dane treningowe i testowe
X = df1.iloc[:, 2: ]        #liczba pięter liczba pokoi pietro powierzchnia rok budowy
y = df.cena
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))