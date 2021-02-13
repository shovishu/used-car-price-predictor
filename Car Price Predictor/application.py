from flask import Flask, render_template,request
import pandas as pd
import pickle
import numpy as np


app=Flask(__name__)
# price=pd.read_csv("file.csv")

model=pickle.load(open("LinearRegressionModel_new.pkl",'rb'))
price=pd.read_csv("abcdef.csv")

@app.route('/')
def index():
    companies=sorted(price['company'].unique())
    car_models=sorted(price['model'].unique())
    year=sorted(price['year'].unique(),reverse=True)
    km_driven=sorted(price['km_driven'].unique())
    fuel=sorted(price['fuel'].unique())
    transmission = sorted(price['transmission'].unique())
    owner = sorted(price['owner'].unique())
    mileage=sorted(price['mileage'].unique())
    engine = sorted(price['engine'].unique())
    seats = sorted(price['seats'].unique())

    companies.insert(0,"Select Company ")
    year.insert(0, "Select Year")
    fuel.insert(0, "Select Fuel Type ")
    km_driven.insert(0, "Km driven")
    transmission.insert(0, "Select Transmission ")
    owner.insert(0, "Select Owner Type ")
    # engine.insert(0, "Select Engine(CC)")
    seats.insert(0, "Select no. of Seats")

    return render_template("index.html",companies=companies,car_models=car_models,year=year,km_driven=km_driven,fuel=fuel,transmission=transmission,owner=owner,mileage=mileage,engine=engine,seats=seats)




@app.route('/predict', methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model=request.form.get('car_model')
    year=int(request.form.get('year'))
    km_driven=int(request.form.get('km_driven'))
    fuel=request.form.get('fuel')
    transmission=request.form.get('transmission')
    owner=request.form.get('owner')
    mileage=int(request.form.get('mileage'))
    engine=int(request.form.get('engine'))
    seats=int(request.form.get('seats'))

    # print(company, car_model, year,km_driven, fuel, transmission, owner, mileage, engine,seats)


    prediction = model.predict(pd.DataFrame([[company, car_model, year,km_driven, fuel, transmission, owner, mileage, engine,seats]],columns=['company', 'model', 'year','km_driven', 'fuel', 'transmission', 'owner', 'mileage','engine', 'seats']))

    return str(np.round(prediction[0],2))







if __name__=="__main__":
    app.run(debug=True)