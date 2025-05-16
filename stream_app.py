import joblib
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Hàm chuyển đổi object thành int, có thể tái sử dụng label
def object_to_int(dataframe_series, label_encoders=None):
    if dataframe_series.dtype == 'object':
        # Nếu chưa có label_encoder thì khởi tạo mới
        if label_encoders is None:
            label_encoders = {}
        
        # Kiểm tra nếu cột chưa có encoder, tạo mới
        if dataframe_series.name not in label_encoders:
            label_encoders[dataframe_series.name] = LabelEncoder()
            dataframe_series = label_encoders[dataframe_series.name].fit_transform(dataframe_series)
        else:
            dataframe_series = label_encoders[dataframe_series.name].transform(dataframe_series)
    
    return dataframe_series, label_encoders
# with open(model_file, 'rb') as f_in:
lr_model = joblib.load('best_lr_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')
num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']

def main():

    # image = Image.open('images/icone.png')
    # image2 = Image.open('images/image.png')
    # st.image(image,use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online"))
    st.sidebar.info('This app is created to predict Customer Churn')
    # st.sidebar.image(image2)
    st.title("Predicting Customer Churn")
    if add_selectbox == 'Online':
        gender = st.selectbox('Gender:', ['Male', 'Female'])
        seniorcitizen= st.selectbox(' Customer is a senior citizen:', [0, 1])
        partner= st.selectbox(' Customer has a partner:', ['Yes', 'No'])
        dependents = st.selectbox(' Customer has  dependents:', ['Yes', 'No'])
        tenure = st.number_input('Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
        phoneservice = st.selectbox(' Customer has phoneservice:', ['Yes', 'No'])
        multiplelines = st.selectbox(' Customer has multiplelines:', ['Yes', 'No', 'No phone service'])
        internetservice= st.selectbox(' Customer has internetservice:', ['DSL', 'no', 'Fiber optic'])
        onlinesecurity= st.selectbox(' Customer has onlinesecurity:', ['Yes', 'No', 'No internet service'])
        onlinebackup = st.selectbox(' Customer has onlinebackup:', ['Yes', 'No', 'No internet service'])
        deviceprotection = st.selectbox(' Customer has deviceprotection:', ['Yes', 'No', 'No internet service'])
        techsupport = st.selectbox(' Customer has techsupport:', ['Yes', 'No', 'No internet service'])
        streamingtv = st.selectbox(' Customer has streamingtv:', ['Yes', 'No', 'No internet service'])
        streamingmovies = st.selectbox(' Customer has streamingmovies:', ['Yes', 'No', 'No internet service'])
        contract= st.selectbox(' Customer has a contract:', ['Month-to-month', 'One year', 'Two year'])
        paperlessbilling = st.selectbox(' Customer has a paperlessbilling:', ['Yes', 'No'])
        paymentmethod= st.selectbox('Payment Option:', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check' ,'Mailed check'])
        monthlycharges = st.number_input('Monthly charges:', min_value=0.0, max_value=240.0, value=0.0, format="%.2f")
        totalcharges = tenure*monthlycharges
        output= ""
        output_prob = ""
        input_dict={
                "gender":gender ,
                "SeniorCitizen": seniorcitizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phoneservice,
                "MultipleLines": multiplelines,
                "InternetService": internetservice,
                "OnlineSecurity": onlinesecurity,
                "OnlineBackup": onlinebackup,
                "DeviceProtection": deviceprotection,
                "TechSupport": techsupport,
                "StreamingTV": streamingtv,
                "StreamingMovies": streamingmovies,
                "Contract": contract,
                "PaperlessBilling": paperlessbilling,
                "PaymentMethod": paymentmethod,
                "MonthlyCharges": monthlycharges,
                "TotalCharges": totalcharges
            }

        if st.button("Predict"):
            X = pd.DataFrame([input_dict])
            X = X.apply(lambda x: object_to_int(x, label_encoders)[0])
            X[num_cols] = scaler.transform(X[num_cols])
            y_pred = lr_model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
        st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))
    # if add_selectbox == 'Batch':
    #     file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
    #     if file_upload is not None:
    #         data = pd.read_csv(file_upload)
    #         X = dv.transform([data])
    #         y_pred = model.predict_proba(X)[0, 1]
    #         churn = y_pred >= 0.5
    #         churn = bool(churn)
    #         st.write(churn)

if __name__ == '__main__':
    main()