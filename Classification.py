import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


pd.set_option('display.max_columns', 1008)

df = pd.read_csv('KNN_Project_Data')

#print(df.head())
#print(df.info())
print(df.describe())

#sns.pairplot(df,hue='TARGET CLASS')
#plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())

from sklearn.model_selection import train_test_split

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print(pred)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


error_rate = []

for i in range(1,85):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(24,12))
print(error_rate)
plt.plot(range(1,85),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# Based on graph, it appears a better K value would be


