import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_json('data/times.json')['session1'].values

times = []
i = 1
for l in data:
    nl = list()
    nl.append(i)
    nl.append(l[0][1]/1000)
    times.append(nl)
    i += 1
times = np.array(times)

lower_threshold = np.percentile(times[:,1], 10)
upper_threshold = np.percentile(times[:,1], 90)

filtered_times = times[(times[:,1] > lower_threshold) & (times[:,1] < upper_threshold)]

y = times[:, 1]
X = times[:, 0]

from sklearn import linear_model
lin1 = linear_model.LinearRegression()
Xsample = np.c_[X]
ysample = np.c_[y]
lin1.fit(Xsample, ysample)
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]

st.title("Time vs Index")
st.scatter_chart(X, y, label="Time")
x = np.linspace(min(X), max(X), 100)
st.line_chart(x, t0 + t1*x, label="Regression Line")
st.legend()

y = filtered_times[:, 1]
X = filtered_times[:, 0]
lin1 = linear_model.LinearRegression()
Xsample = np.c_[X]
ysample = np.c_[y]
lin1.fit(Xsample, ysample)
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]

st.title("Filtered Time vs Index")
st.scatter_chart(X, y, label="Filtered Time")
x = np.linspace(min(X), max(X), 100)
st.line_chart(x, t0 + t1*x, label="Regression Line")
st.legend()

model = sklearn.linear_model.LinearRegression()
model.fit(Xsample, ysample)
X_new = [[200]]
prediction = model.predict(X_new)
st.write("Prediction for 200th solve:", prediction)
