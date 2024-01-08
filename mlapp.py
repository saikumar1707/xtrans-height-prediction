from flask import Flask, redirect, url_for, render_template, request,jsonify,flash
from pymongo.errors import AutoReconnect
from pymongo import MongoClient
import pandas as pd
import traceback
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
import os

MONGO_USERNAME = 'saikumarmanchala2003'
MONGO_PASSWORD = 'dsNxl8PJ1gKSmOu2'
MONGO_CLUSTER_URL = 'cluster0.lclg6oj.mongodb.net'
MONGO_DATABASE = 'cluster0'

MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_CLUSTER_URL}/{MONGO_DATABASE}?retryWrites=true&w=majority"

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client[MONGO_DATABASE]

df=pd.read_csv(r"bmi.csv")
zero_not_accepted = ['Height', 'Weight',"Age", "Bmi"]
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

X = df[[ 'Age','Weight']]
y = df['Height']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def welcome():
    return render_template('ml.html')

@app.route('/<string:model>/<int:score>/<int:age>')
def success(model, score,age):
    if model == "linear":
        from sklearn.linear_model import LinearRegression
        regression=LinearRegression(n_jobs=-1)
        regression.fit(X_train,y_train)
        linear_y_pred=regression.predict(X_test)
        p=plt.scatter(df['Height'],df['Weight'])
        plt.xlabel("Height")
        plt.ylabel("Weight")
        mse =round(mean_squared_error(y_test, linear_y_pred),4)
        linear_score=round(r2_score(y_test,linear_y_pred),2)
        line=float(regression.predict(scaler.transform([[int(age),int(score)]])))
        line=round(line,2)
        fo = {'Model':model,'Age':age,'Given Weight (Kg)': score, 'Height(m)': line,'r2 score': linear_score,'Mean square error':mse,'graph':p}
        return render_template('result.html', result=fo)

    if model == "SVM":
        from sklearn.svm import SVR
        Sreg=SVR(kernel="rbf")
        Sreg.fit(X_train,y_train)
        svm_y_pred = Sreg.predict(X_test)
        mse =round(mean_squared_error(y_test, svm_y_pred),4)
        svm_score=round(r2_score(y_test,svm_y_pred),2)
        svm=float(Sreg.predict(np.array([int(age),int(score)]).reshape(1,-1)))
        svm=round(svm,2)
        fo = {'Model':model,'Age':age,'Given Weight (Kg)': score, 'Height(m)': svm,'r2 score': svm_score,'Mean square error':mse}
        return render_template('result.html', result=fo)
    
    if model == "KNN":
        from sklearn.neighbors import KNeighborsRegressor
        knn_regression = KNeighborsRegressor(n_neighbors=13, metric='euclidean')
        knn_regression.fit(X_train, y_train)
        # Convert entered_value to 2D array with one feature
        entered_value_2D = np.array([[int(age),int(score)]])
         # Scale the entered_value using the same scaler used for training data
        entered_value_scaled = scaler.transform(entered_value_2D)
        knn_y_pred = knn_regression.predict(entered_value_scaled)
        mse =round(mean_squared_error(y_test, knn_regression.predict(X_test)),4)
        knn_score = round(r2_score(y_test, knn_regression.predict(X_test)),2)
        knn = round(float(knn_y_pred[0]), 2)
        fo = {'Model':model,'Age':age,'Given Weight (Kg)': score, 'Height(m)': knn,'r2 score': knn_score,'Mean square error':mse}
        return render_template('result.html', result=fo)
    
    if model == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        reg= RandomForestRegressor(n_estimators= 13, criterion="poisson")
        reg.fit(X_train, y_train)
        rf_y_pred = reg.predict(X_test)
        svm_score=r2_score(y_test,rf_y_pred)
        rf_score=round(r2_score(y_test,rf_y_pred),2)
        mse =round(mean_squared_error(y_test, rf_y_pred),4)
        rf=float(reg.predict(np.array([int(age),int(score)]).reshape(1,-1)))
        rf=round(rf,2)
        rf_score=round(rf_score,2)
        fo = {'Model':model,'Age':age,'Given Weight (Kg)': score, 'Height(m)': rf,'r2 score': rf_score,'Mean square error':mse}
        return render_template('result.html', result=fo)
    
    if model == "ANN":
        
        y_pred = model1.predict(X_test)
        ann_score = round(r2_score(y_test, y_pred),2)
        mse=round(mean_squared_error(y_test, y_pred),4)
        new_weight = np.array([[int(age),int(score)]])
        new_weight_scaled = scaler.transform(new_weight)
        ann_prediction = round(model1.predict(new_weight_scaled)[0][0],2)
        fo = {'Model':model,'Age':age,'Given Weight (Kg)': score, 'Height(m)': ann_prediction,'r2 score': ann_score,'Mean square error':mse}
        return render_template('result.html', result=fo)



@app.route('/submit', methods=['POST', 'GET'])
def submit():
    try:
        if request.method == 'POST':
            age = float(request.form['Age'])
            model = request.form['model']
            weight = float(request.form['weight'])
            name=request.form['name']
            data = {
                'Age': age,
                'Model': model,
                'Weight': weight,
                'name': name  # Add the prediction field if needed
            }
            if not name:
                flash('Name is required', 'error')
                return render_template('ml.html')
       

            db[name].insert_one(data)
            return redirect(url_for('success', model=model, score=weight, age=age))
        
    except ValueError:
        flash('Invalid input.Please enter given parameters', 'error')
        return render_template('ml.html')
    except AutoReconnect :
        flash('Please check your internet connection or your IP address', 'error')
        return render_template('ml.html')

    except Exception as e:
        traceback.print_exc()  # Print the traceback to the console
        raise e 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

