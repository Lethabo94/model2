import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
df = pd.read_csv('heart.csv')
#shuffling data 

data = df
X = data.drop(['target'],axis=1).values
y = data['target'].values
y = y.astype('int')
X = X.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

inputt=[x for x in "45 32 60 20 10 29".split(' ')]
for i in range(len(int_features)):
    int_features[i] = int_features[i].lower()
empty=[]
for iterm in inputt:
    if iterm == 'male':
        empty.append(1)
    elif iterm == 'female':
        empty.append(0)
    elif iterm == 'yes':
        empty.append(1)
    elif iterm == 'no':
        empty.append(0)
    else:
        empty.append(iterm)

final=pd.DataFrame(empty)
# final.to_csv('bobo.csv',index=False) i was using this to test 
final = final.T.values
b = log_reg.predict_proba(final)
pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))