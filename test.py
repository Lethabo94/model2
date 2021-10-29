import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
df = pd.read_csv('heart.csv')
# empty=[]
# for iterm in df['sex']:
#     if iterm == 'male':
#         empty.append(1)
#     if iterm == 'female':
#         empty.append(2)
# df['sex']=empty 
df = df[['age','sex','trestbps','thalach','chol','fbs','target']]
df = df.reset_index().drop(['index'],axis=1)

data = df
X = data.drop(['target'],axis=1).values
y = data['target'].values
y = y.astype('int')
X = X.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

inputt=[x for x in "45 32 60 20 10 29".split(' ')]
final=pd.DataFrame(inputt)
final.to_csv('test_data.csv',index=False)
