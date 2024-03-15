# %% [markdown]
# # Predictia numerelor castigatoare la lotto

# %%
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

# %% [markdown]
# 1. Reading the data from our database
# 

# %%
df_extr = pd.read_excel('extrageri.xlsx', sheet_name ='EXTRAGERI')
df_extr.head()

# %% [markdown]
# Verificarea tipurilor de date.

# %%
df_extr.dtypes

# %% [markdown]
# Functii pentru prelucrarea datei

# %%
luni = {
    'ianuarie' :1,
    'februarie' :2,
    'martie':3,
    'aprilie':4,
    'mai':5,
    'iunie':6,
    'iulie':7,
    'august':8,
    'septembrie':9,
    'octombrie':10,
    'noiembrie':11,
    'decembrie':12
}

# Function to remove day abbreviation from a date
def remove_day_abbreviation(date):
    day, rest_of_date = date.split(",", 1)
    return rest_of_date.split('\n')[0].strip()

# Function to extract and convert the month to its corresponding number
def extract_and_convert_month(date):
    parts = date.split()
    month_name = parts[1]  
    month_number = luni.get(month_name.lower(), None)
    parts[1] = str(month_number)
    modified_date = '.'.join(parts)
    return modified_date


def process_dates(dataframe):
    dates = []
    for date in dataframe['DATA']:
        dates.append(remove_day_abbreviation(date))

    for i in range(0,len(dates)):
        dates[i] = extract_and_convert_month(dates[i])
    return dates



# %%
df_extr['DATA'] = process_dates(df_extr)
df_extr['DATA'] = pd.to_datetime(df_extr['DATA'], format ='%d.%m.%Y')
df_extr.head()

# %% [markdown]
# Studying the numbers extracted

# %%
n1_values = df_extr['N1'].tolist()

n2_values = df_extr['N2'].tolist()

n3_values = df_extr['N3'].tolist()

n4_values = df_extr['N4'].tolist()

n5_values = df_extr['N5'].tolist()

n6_values = df_extr['N6'].tolist()

# Combine all ball values into a single list
all_values = (
    n1_values +
    n2_values +
    n3_values +
    n4_values +
    n5_values +
    n6_values
)


my_list = np.zeros(49)

for no in all_values:
    for i in range(1,50):
        # print(i)
        if(no == i):
            my_list[i-1] = my_list[i-1] + 1

print(my_list)

# Checking result
msum = 0
for no in my_list:
    msum = msum + no
print(msum)
print(len(all_values))


# %%
balls = list(range(1, 50, 1))
all_balls = pd.Series(balls)

values = all_balls.values

colors = plt.cm.YlOrRd(np.linspace(0, 1, len(values)))

plt.bar(balls, my_list, color=colors)

plt.xlabel('Balls')
plt.ylabel('Values')
plt.title('Distribution of all balls')


plt.show()


# %% [markdown]
# Most common numbers

# %%
ball_counts = pd.Series(all_values).value_counts().sort_index()

most_frequent_numbers = ball_counts.head(6).index.tolist()
print("The 6 most frequent numbers:", most_frequent_numbers)

# %%
ball_counts_df = pd.DataFrame({'Ball': ball_counts.index, 'Count': ball_counts.values})

X = ball_counts_df[['Ball']]
y = ball_counts_df['Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(X_test)

plt.scatter(X_test, y_test, color='skyblue', label='Actual Data')
plt.plot(X_test, y_pred, color='orange', linewidth=2, label='Linear Regression')
plt.xlabel('Ball')
plt.ylabel('Count')
plt.title('Linear Regression on Ball Counts')
plt.legend()
plt.show()

# %%
from collections import Counter

ball_counts = Counter(all_values)

most_frequent_numbers = [num for num, count in ball_counts.most_common(6)]
print("The 6 most frequent numbers:", most_frequent_numbers)



# %% [markdown]
# Din seturile de numere voiam sa extrag si sa etichez cu 1- castigator ce date aveam de la arhiva loteriei si sa creez eu siruri necastigatoare, urmand sa folosesc RandomForestClassifier din libraria sklearn pentru a face o predictie. Acest lucru mi-a depasit capacitatile caci selectarea a 10 siruri castigatoare combina atat de multe numere incat functia mea nu gasea destule siruri pe care sa le etichetez necastigatoare. Esecul de a construii un set de date echilibrat pentru a putea face o predictie cat de cat aproape de a fi relevanta m-a facut sa renunt la partea aceasta atasand proiectului doar niste statistici asupra setului de numere extrase.

# %% [markdown]
# ## Predictia numarului de castigatori
# Verificarea datelor din dataframes si verificarea tipurilor de date

# %%
df_cat1 = pd.read_excel('extrageri.xlsx', sheet_name ='CATEGORIA I')
df_cat1.head()

# %%
df_cat2 = pd.read_excel('extrageri.xlsx', sheet_name ='CATEGORIA II')
df_cat2.head()

# %%
df_cat3 = pd.read_excel('extrageri.xlsx', sheet_name ='CATEGORIA III')
df_cat3.head()

# %%

df_cat4 = pd.read_excel('extrageri.xlsx', sheet_name ='reduced_data')
df_cat4.head()


# %%
df_test = pd.read_excel('extrageri.xlsx', sheet_name ='CATEGORIA IV')
df_test.head()

# %%
df_cat1.dtypes

# %%
df_cat2.dtypes

# %%
df_cat3.dtypes

# %%
df_cat4.dtypes

# %% [markdown]
# ### Procesare de date
# Transformarea datelor in string pentru a putea lucra cu ele
# Transformarea datei('DATA') in datetime si popularea dataframe-ului cu noile valori

# %%
df_extr['DATA'] = df_extr['DATA'].astype(str)

df_cat4['DATA'] = df_cat4['DATA'].astype(str)


# %%
df_cat4['DATA'] = process_dates(df_cat4)
df_cat4['DATA'] = pd.to_datetime(df_cat4['DATA'], format ='%d.%m.%Y')
df_cat4.head()


df_test['DATA'] = process_dates(df_test)
df_test['DATA'] = pd.to_datetime(df_test['DATA'], format ='%d.%m.%Y')
df_test.head()


# %% [markdown]
# Check datatypes

# %%
df_extr.dtypes

# %%
df_cat4.dtypes

# %%
df_cat2['NumarCastiguri'] = df_cat2['NumarCastiguri'].replace('REPORT', '0')
df_cat1['NumarCastiguri'] = df_cat1['NumarCastiguri'].replace('REPORT', '0')
df_cat4['NumarCastiguri'] = df_cat4['NumarCastiguri'].astype(str).str.replace('.', '').astype(int)
df_test['NumarCastiguri'] = df_cat4['NumarCastiguri'].astype(str).str.replace('.', '').astype(int)

# %%
df_cat1['NumarCastiguri'] = pd.to_numeric(df_cat1['NumarCastiguri'], errors='coerce')
df_cat2['NumarCastiguri'] = pd.to_numeric(df_cat2['NumarCastiguri'], errors='coerce')
df_cat3['NumarCastiguri'] = pd.to_numeric(df_cat3['NumarCastiguri'], errors='coerce')
df_cat4['NumarCastiguri'] = pd.to_numeric(df_cat4['NumarCastiguri'], errors='coerce')
df_test['NumarCastiguri'] = pd.to_numeric(df_test['NumarCastiguri'], errors='coerce')

# %%
#Verificarea lungimii coloanelor pentru a nu intampina probleme
print(len(df_cat4['DATA']))
print(len(df_cat4['NumarCastiguri']))

# %% [markdown]
# #### Evaluarea numarului de castigatori pe axa timpului

# %%
plt.plot(df_extr['DATA'], df_cat1['NumarCastiguri'])
plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time')
plt.show()


# %%
plt.plot(df_extr['DATA'],df_cat2['NumarCastiguri'])
plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time')
plt.show()

# %%
plt.plot(df_extr['DATA'],df_cat3['NumarCastiguri'])
plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time')
plt.show()

# %%
plt.plot(df_cat4['DATA'],df_cat4['NumarCastiguri'])
plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time')
plt.show()

# %% [markdown]
# Dupa analizarea datelor, am luat decizia de a alege categoria IV pentru ca este cea mai consistenta

# %% [markdown]
# Precizare:  Acesta reprezentare a df_cat4 este dupa impartirea datelor in antrenare si validare 
#             Acest interval de timp a fost ales pentru ca nu are date lipsa.
#             Restul garficelor reprezentate au toate datele pe axa de timp

# %% [markdown]
# Interpolarea datelor din categoria 4
# (sau o incercare)

# %%
plt.plot(df_cat4['DATA'], df_cat4['NumarCastiguri'], label='Original Data', color='blue')

# Interpolare
df_cat4['NumarCastiguri_interpolated'] = df_cat4['NumarCastiguri'].interpolate(method='linear')

plt.plot(df_cat4['DATA'], df_cat4['NumarCastiguri_interpolated'], label='Interpolated Data',linestyle='dashed', color='orange')

plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time')
plt.legend()
plt.show()

# %% [markdown]
# Netezire cu Transformata Fourier

# %%
x = np.array(df_cat4['DATA'])
y = np.array(df_cat4['NumarCastiguri'])

# Aplicam transformata Fourier
fft_result = np.fft.fft(y)
# Frecvența semnalului
frequencies = np.fft.fftfreq(len(fft_result))

cutoff_frequency = 0.001 
fft_result[frequencies > cutoff_frequency] = 0
smoothed_data = np.fft.ifft(fft_result).real

plt.plot(x, y, label='Original Data')

plt.plot(x, smoothed_data, label='Smoothed Data', linewidth=1, color='red')

plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time (Smoothed)')
plt.legend()
plt.show()


# %% [markdown]
# Netezirea cu media alunecatoare

# %%
x = df_cat4['DATA']
y = df_cat4['NumarCastiguri']

window_size = 5

smoothed_data = 0
smoothed_data = y.rolling(window=window_size).mean()

plt.plot(x, y, label='Original Data')

plt.plot(x, smoothed_data, label=f'Smoothed Data (Window Size = {window_size})', linewidth=2, color='red')

plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time (Smoothed with Moving Average)')
plt.legend()
plt.show()


# %%
df_cat4['NumarCastiguri_netezite'] = smoothed_data
for i in range(0,10):
    print(df_cat4['NumarCastiguri_netezite'][i])

# %% [markdown]
# Observam ca netezirea a adaugat niste NaN de care trebuie sa scapam.

# %%
df_cat4['NumarCastiguri_netezite'] = smoothed_data

df_cat4 = df_cat4.iloc[4:]

df_cat4.reset_index(drop=True, inplace=True)

df_cat4[['DATA', 'NumarCastiguri_netezite']].head()

# %% [markdown]
# Aplicarea algoritmului de regresie liniara pe numarul castigatorilor.

# %%
df_cat4.dtypes

# %%
df_cat4['NumarCastiguri_netezite'].astype(int)

# %% [markdown]
# ### Antrenarea modelului si reprezentarea grafica a predictiei

# %%
x = np.array(pd.to_datetime(df_cat4['DATA']).apply(lambda x: x.toordinal())).reshape(-1, 1)
y = np.array(df_cat4['NumarCastiguri_netezite'])

if len(x) != len(y):
    raise ValueError("Lengths of x and y are not the same.")

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

plt.scatter(df_cat4['DATA'], df_cat4['NumarCastiguri_netezite'], label='Original data')

plt.plot(df_cat4['DATA'], y_pred, color='red', linewidth=2, label='Linear Regression')

plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time with Linear Regression')
plt.legend()
plt.show()


# %%
model = LinearRegression()

future_dates = pd.date_range(start='2019-1-18', end='2022-5-15', freq='2D')
future_dates = future_dates[:-4]

scaler = MinMaxScaler()
future_X =  scaler.fit_transform(x).reshape(-1, 1)

model.fit(future_X, y) 

future_pred = model.predict(future_X)

print(len(x),len(future_dates),len(future_pred))

plt.scatter(future_dates, future_pred, color='blue', label='Future Predictions')

plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time with Linear Regression and Future Predictions')
plt.legend()
plt.show()

# %%
X = np.array(pd.to_datetime(df_cat4['DATA']).apply(lambda x: x.toordinal())).reshape(-1, 1)
y = np.array(df_cat4['NumarCastiguri_netezite'])


scaler = MinMaxScaler()

X_normalized = scaler.fit_transform(X)

if len(X_normalized) != len(y):
    raise ValueError("Lengths of X and y are not the same.")

model =  GradientBoostingRegressor(n_estimators=150, random_state=42)

model.fit(X_normalized, y)

future_dates = pd.date_range(start='2018-1-18', end='2023-1-17', freq='D')

future_X = np.arange(len(X_normalized), len(X_normalized) + len(future_dates)).reshape(-1, 1)

future_pred = model.predict(future_X)

plt.scatter(df_cat4['DATA'], df_cat4['NumarCastiguri_netezite'], label='Original data')

plt.plot(future_dates, future_pred, color='blue', linewidth=2, label='Future Predictions')

plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time with Linear Regression and Future Predictions')
plt.legend()
plt.show()


# %%
x = np.array(pd.to_datetime(df_test['DATA']).apply(lambda x: x.toordinal())).reshape(-1, 1)
y = np.array(df_test['NumarCastiguri'])

if len(x) != len(y):
    raise ValueError("Lengths of x and y are not the same.")

model = LinearRegression()

model.fit(x, y)


y_pred = model.predict(x)

plt.scatter(df_test['DATA'], df_test['NumarCastiguri'], label='Original data')

plt.plot(df_test['DATA'], y_pred, color='red', linewidth=2, label='Linear Regression')

plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time with Linear Regression')
plt.legend()
plt.show()


# %%
plt.plot(df_test['DATA'],df_test['NumarCastiguri'])
plt.xlabel('Date')
plt.ylabel('NumarCastiguri')
plt.title('NumarCastiguri over Time')
plt.show()

# %% [markdown]
# ## Concluzie:
# Regresia Liniara nu a fost chiar eficienta in a da un rezultat asupra acestui set de date, estimand o crestere a numarului de catigatori. Acest rezultat nu poate fi etichetat drept incorect caci valorile trec prin perioade de crestere si de descrestere periodic, insa analizand vizual valorile se poate constata ca tendinta numarului de castigatori este de a se micsora. Poate acesta concluzie sa fie afectata de numarul de oameni care participa la lotto?(cu siguranta).


