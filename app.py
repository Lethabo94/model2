from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("forest_fire2.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[x for x in request.form.values()]
    #converting input from the website to a dataframe
    # final=pd.DataFrame(int_features)
    for i in range(len(int_features)):
        int_features[i] = int_features[i].lower()

    empty = []
    for iterm in int_features:
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
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.6):
        return render_template('forest_fire2.html',pred='You May have a heart Disease Please visit a doctor.\nProbability is {}'.format(output),bhai="")
    else:
        return render_template('forest_fire2.html',pred='All is well.\n Probability is {}'.format(output),bhai="")


if __name__ == '__main__':
    app.run(debug=True)
