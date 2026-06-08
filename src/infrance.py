import joblib

model = joblib.load("./../output/model.pkl")
pred = model.predict([[15.1, 3.5, 1.4, 0.2]])
print(pred)
