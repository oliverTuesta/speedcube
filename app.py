import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

@st.cache
def load_data():
    data = pd.read_json('data/times.json')['session1'].values
    times = []
    i = 1
    for l in data:
        nl = list()
        nl.append(i)
        nl.append(l[0][1]/1000)
        times.append(nl)
        i += 1
    return np.array(times)

times = load_data()

# Calculate the 10th and 90th percentile thresholds to filter the times
lower_threshold = np.percentile(times[:,1], 10)
upper_threshold = np.percentile(times[:,1], 90)
filtered_times = times[(times[:,1] > lower_threshold) & (times[:,1] < upper_threshold)]

# Linear Regression Model
def linear_regression(times, filtered=False):
    if filtered:
        y = filtered_times[:, 1]
        X = filtered_times[:, 0]
    else:
        y = times[:, 1]
        X = times[:, 0]

    model = sklearn.linear_model.LinearRegression()
    Xsample = np.c_[X]
    ysample = np.c_[y]
    model.fit(Xsample, ysample)
    return model

# Plot Time vs Index
def plot_time_vs_index(times, filtered=False):
    if filtered:
        y = filtered_times[:, 1]
        X = filtered_times[:, 0]
        plt.scatter(X, y, s=5)
        plt.xlabel('Index')
        plt.ylabel('Filtered Time')
        plt.title('Filtered Time vs Index')
    else:
        y = times[:, 1]
        X = times[:, 0]
        plt.scatter(X, y, s=5)
        plt.xlabel('Index')
        plt.ylabel('Time')
        plt.title('Time vs Index')

    t0, t1 = model.intercept_[0], model.coef_[0][0]
    plt.plot(X, t0 + t1*X, "b")
    plt.show()

# Predict average time in certain number of solves
def predict_average_time(model, num_solves):
    X_new = [[num_solves]]
    prediction = model.predict(X_new)
    return prediction

# Streamlit App
st.title("Rubik's Cube Solving Time Predictor")
st.write("This app predicts the time it will take you to solve the Rubik's cube with a certain number of practices using a linear regression model.")

if st.checkbox("Show raw times"):
    st.write(times)

if st.checkbox("Show filtered times"):
    st.write(filtered_times)

model = linear_regression(times)

if st.checkbox("Show linear regression model"):
    st.write(model)

if st.checkbox("Show linear regression model coefficients"):
    st.write(model.intercept_, model.coef_)

if st.checkbox("Show time vs index plot"):
    plot_time_vs_index(times)
    st.pyplot()

if st.checkbox("Show filtered time vs index plot"):
    plot_time_vs_index(times, filtered=True)
    st.pyplot()

num_solves = st.number_input("Number of solves", min_value=1, max_value=1000, value=1)
prediction = predict_average_time(model, num_solves)
st.write("The average time it will take you to solve the Rubik's cube with {} solves is {} seconds.".format(num_solves, prediction[0][0]))

st.write("This app was created by Oliver Tuesta")
