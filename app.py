import streamlit as st
import pickle as pkl
import json

with open("accuracy.json","r") as file:
    metrics = json.load(file)



model = pkl.load(open("model.pkl","rb"))


st.set_page_config(page_title="Exam Score Predictor",page_icon="🎓")

st.title("🎓 Exam Score Predictor")

st.write("")
st.subheader("About the Model")
st.write("This model predicts student exam scores using by study hours, attendance percentage, sleep hours, age, and internet access.")
st.write("It helps to estimate academic performance of a student based on the above key factors.")

st.write("")
st.subheader("Model Performance")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("R² Score", f"{metrics['r2_score']:.2f}")

with col2:
    st.metric("MAE", f"{metrics['mae']:.2f}")

with col3:
    st.metric("RMAE", f"{metrics['rmae']:.2f}")

with col4:
    st.metric("MSE", f"{metrics['mse']:.2f}")

with col5:
    st.metric("RMSE", f"{metrics['rmse']:.2f}")


st.write("")
st.write("")

age = st.number_input("Age",min_value=16,max_value=24,value=17)
hours = st.number_input("Enter Study Hours",min_value=2.0,max_value=9.0,value=5.0,step=0.1)
attendance = st.number_input("Attendance Percentange",min_value=0.0,max_value=100.0,value=75.0,step=0.5)
sleep_hours = st.number_input("Sleep Hours",min_value=4.0,max_value=10.0,value=8.0,step=0.1)
internet_access = st.selectbox("Internet Access",["Yes","No"])




if st.button("Predict Score"):
    internet = 1 if internet_access == "Yes" else 0
    input_data = [[age, hours,attendance,sleep_hours, internet]]
    prediction = model.predict(input_data)
    st.success(f"Estimated Score of the Student is {prediction[0]:.2f} %")