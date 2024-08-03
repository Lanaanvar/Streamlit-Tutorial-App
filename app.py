import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache_data
def get_data():
    hotel_data = pd.read_csv('data/hotel_bookings.csv/hotel_bookings.csv')
    return hotel_data

with header:
    st.title("Welcome to the awsome datascience project...")
    st.text("In this project I look into the transactions of taxis in NYC ...")

with dataset:
    st.header("NYC taxi dataset")
    st.text("I found this dataset on blablabla.com..")

    hotel_data = get_data()
    st.write(hotel_data.head())
    st.subheader("Reservation status in the dataset")
    reservation_status = pd.DataFrame(hotel_data['reservation_status'].value_counts()).head(50)
    st.bar_chart(reservation_status)

with features:
    st.header("I like to train the model!")
    st.text("Here you get to choose the hyperparameters of the model and see the performance change..")

    st.markdown("* **First feature:** I created this feature because of this... I calculatedd it using..")
    st.markdown("* **Second feature:** I created this feature because of this... I calculatedd it using..")

with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparamters of the model and see how the performance changes..')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value = 10, max_value= 100, value=20, step =10)
    n_estimators = sel_col.selectbox("How many trees should be there", options=[100, 200, 300, "no limit"])

    sel_col.text("The following is a list of input features to try")
    sel_col.write(hotel_data.columns)

    input_text = sel_col.text_input("Enter the feature to search", "lead_time")

    X = hotel_data[[input_text]]
    y = hotel_data[['is_canceled']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if n_estimators=="no limit":
        regression_model = RandomForestClassifier(max_depth=max_depth)
    else:
        regression_model = RandomForestClassifier(n_estimators= n_estimators, max_depth=max_depth)
    
    regression_model.fit(X_train,y_train)

    prediction = regression_model.predict(X_test)

    disp_col.subheader("Mean Absolute Error : ")
    disp_col.write(mean_absolute_error(y_test,prediction))

    disp_col.subheader("Mean Squared Error : ")
    disp_col.write(mean_squared_error(y_test, prediction))

    disp_col.subheader("R2 score : ")
    disp_col.write(r2_score(y_test, prediction))