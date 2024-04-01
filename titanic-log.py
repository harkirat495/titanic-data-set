import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('titanic.csv')

sex = pd.get_dummies(df['Sex'])
embark = pd.get_dummies(df['Embarked'])

sex = sex[['male', 'female']].replace({True : 1, False : 0})
embark = embark[['C', 'Q', 'S']].replace({True : 1, False : 0})

df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'])

df = pd.concat([df, sex, embark], axis = 1)


each_class = df[df['Pclass'] == 1]
print(each_class['Age'].mean())


def fill(age, pclass):
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 30
        else:
            return 25
    else:
        return age
    
for i in range(len(df)):
    df.loc[i, 'Age'] = fill(df.loc[i, 'Age'], df.loc[i, 'Pclass'])



#                                                                             Data Splitting    

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = df.drop(columns=['Survived'])
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

confu = confusion_matrix(y_test, y_pred)
print(confu)

accu = accuracy_score(y_test, y_pred)
print(accu)

f1 = f1_score(y_pred, y_test)
print(f1)