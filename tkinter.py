

from tkinter import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

df_csv=pd.read_csv(r"C:\Users\hp\Desktop\bmi.csv")
df=df_csv.drop(columns=["Age","Bmi","BmiClass"],axis=1)
zero_not_accepted = ['Height','Weight']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)
X=df[['Weight']]
y=df['Height']
X_series=df['Weight']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)



def click(event):

    entered_value = input.get()
    try:
        entered_value = int(entered_value)
        # Call your prediction or processing function with the entered value here
        result_label.config(text=f"Given Weight: {entered_value :}KG")  # Just a sample calculation
    except ValueError:
        result_label.config(text="Invalid input. Please enter a numeric value.")

    text=event.widget.cget("text")
    if text == "linear regression":
        from sklearn.linear_model import LinearRegression
        regression=LinearRegression(n_jobs=-1)
        regression.fit(X_train,y_train)
        linear_y_pred=regression.predict(X_test)
        linear_score=r2_score(y_test,linear_y_pred)
        print(linear_score)
        line=float(regression.predict(scaler.transform([[int(entered_value)]])))
        print(line)
        line=round(line,2)
        result_label2.config(text=f"Predicted height : {round(line,2):}m")
        result_label1.config(text=f"Accuracy: {round(linear_score,2)*100:}%")

    if text == "polynomial regression":
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        poly_features = PolynomialFeatures(degree=6, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)

        # Linear regression with polynomial features
        poly_reg = LinearRegression()
        poly_reg.fit(X_train_poly, y_train)

        # Make predictions on the training set
        y_train_pred = poly_reg.predict(X_train_poly)

        # Make predictions on the test set
        plt.scatter(X_train,y_train)
        plt.scatter(X_train,poly_reg.predict(X_train_poly))
        y_pred = poly_reg.predict(X_test_poly)
        score=r2_score(y_test,y_pred)
        print(score)
        q = int(entered_value)
        p = poly_features.fit_transform(np.array([q]).reshape(-1, 1))
        print(p)
        final_pred = poly_reg.predict(p)
        print(final_pred)
        result_label2.config(text=f"Predicted height : {round(final_pred[0],2):}m")
        result_label1.config(text=f"Accuracy: {round(score,2)*100:}%")

            
    


    if text=="KNN regression":
        from sklearn.neighbors import KNeighborsRegressor
        knn_regression = KNeighborsRegressor(n_neighbors=13, metric='euclidean')
        knn_regression.fit(X_train, y_train)

        # Convert entered_value to 2D array with one feature
        entered_value_2D = np.array([[int(entered_value)]])
        
        # Scale the entered_value using the same scaler used for training data
        entered_value_scaled = scaler.transform(entered_value_2D)

        knn_y_pred = knn_regression.predict(entered_value_scaled)
        knn_score = r2_score(y_test, knn_regression.predict(X_test))
        print(knn_score)

        knn = round(float(knn_y_pred[0]), 2)
        result_label2.config(text=f"Predicted height : {round(knn, 2):}m")
        result_label1.config(text=f"Accuracy: {round(knn_score, 2) * 100:}%")

    if text=="SVM regression":
        from sklearn.svm import SVR
        Sreg=SVR(kernel="rbf")
        Sreg.fit(X_train,y_train)
        svm_y_pred = Sreg.predict(X_test)
        svm_score=r2_score(y_test,svm_y_pred)
        print(svm_score)
        svm=float(Sreg.predict(np.array([int(entered_value)]).reshape(1,-1)))
        print(svm)
        svm=round(svm,2)
        result_label2.config(text=f"Predicted height : {round(svm,2):}m")
        result_label1.config(text=f"Accuracy: {round(svm_score*100,2):}%")

    if text=="Random forest regression":
        from sklearn.ensemble import RandomForestRegressor
        reg= RandomForestRegressor(n_estimators= 13, criterion="poisson")
        reg.fit(X_train, y_train)
        rf_y_pred = reg.predict(X_test)
        rf_score=r2_score(y_test,rf_y_pred)
        print(rf_score)
        rf=float(reg.predict(np.array([int(entered_value)]).reshape(1,-1)))
        print(rf)
        rf=round(rf,2)
        rf_score=round(rf_score,2)
        result_label2.config(text=f"Predicted height : {round(rf,2):}m")
        result_label1.config(text=f"Accuracy: {round(rf_score,2)*100:}%")


    
root =Tk()
root.geometry("900x600")
root.configure(bg="#002D62")
f=Frame(root,bg="#B9D9EB")
input= StringVar()
input.set("")
result_label1 = Label(root, text="Enter Your weight in KG", font="lucida 14")
result_label1.pack(pady=10)
entry =Entry(root,textvar=input,font='lucida 20 bold')
entry.pack(padx=10,pady=10)


b=Label(f,text="Select the required model to predict the height",padx=10,pady=6,font="lucida 16 bold")
b.pack()
b.bind("<Button-1>",click)  
f.pack(padx=10,pady=10)
result_label = Label(root, text="Given Weight : None", font="lucida 14")
result_label.pack(pady=10)




root.title("Height Predictor using Weight")
result_label2 = Label(root, text="Predicted Height : None", font="lucida 20 bold")
result_label2.pack(pady=10)
f=Frame(root,bg="#B9D9EB")
b=Button(f,text="linear regression",padx=10,pady=6,font="lucida 20 bold",bg="#041E42",fg="#E0FFFF")
b.pack(side=LEFT,padx=10,pady=10)
b.bind("<Button-1>",click)  
#b=Button(f,text="polynomial regression",padx=10,pady=6,font="lucida 20 bold",bg="#041E42",fg="#E0FFFF")
#b.pack(side=LEFT,padx=10,pady=10)
#b.bind("<Button-1>",click)
b=Button(f,text="KNN regression",padx=10,pady=6,font="lucida 20 bold",bg="#041E42",fg="#E0FFFF")
b.pack(side=LEFT,padx=10,pady=10)
b.bind("<Button-1>",click)
f.pack( )
f=Frame(root,bg="#B9D9EB")
b=Button(f,text="Random forest regression",padx=10,pady=6,font="lucida 20 bold",bg="#041E42",fg="#E0FFFF")
b.pack(side=LEFT,padx=10,pady=10)
b.bind("<Button-1>",click)
b=Button(f,text="SVM regression",padx=10,pady=6,font="lucida 20 bold",bg="#041E42",fg="#E0FFFF")
b.pack(side=LEFT,padx=10,pady=10)
b.bind("<Button-1>",click)
f.pack( padx=10,pady=10)

result_label1 = Label(root, text="Accuracy : None", font="lucida 14")
result_label1.pack(pady=10)





root.mainloop()