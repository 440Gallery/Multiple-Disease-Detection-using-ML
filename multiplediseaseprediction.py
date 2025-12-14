import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu


# loading the saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Detection using ML',
                          ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)

st.title("Multiple Disease Detection using ML")

# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    st.subheader('Diabetes Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0.0, max_value=300.0, value=120.0, step=0.1)
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0.0, max_value=200.0, value=70.0, step=0.1)
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0.0, max_value=1000.0, value=79.0, step=0.1)
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=5.0, value=0.5, step=0.01)
    with col2:
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, value=30, step=1)

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            X = [[float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                  float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]]
            diab_prediction = diabetes_model.predict(X)
            diab_diagnosis = 'The person is diabetic' if int(diab_prediction[0]) == 1 else 'The person is not diabetic'
            st.success(diab_diagnosis)
        except Exception as e:
            st.error(f'Prediction error: {e}')

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    st.subheader('Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=50, step=1)
    with col2:
        sex = st.selectbox('Sex', options=[0,1], format_func=lambda x: 'Female' if x==0 else 'Male')
    with col3:
        cp = st.selectbox('Chest Pain type', options=[0,1,2,3])
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0.0, max_value=300.0, value=130.0, step=0.1)
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0.0, max_value=1000.0, value=250.0, step=0.1)
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0,1])
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results', options=[0,1,2])
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0.0, max_value=300.0, value=150.0, step=0.1)
    with col3:
        exang = st.selectbox('Exercise Induced Angina', options=[0,1])
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.01)
    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', options=[0,1,2])
    with col3:
        ca = st.selectbox('Major vessels colored by flourosopy (0-4)', options=[0,1,2,3,4])
    with col1:
        thal = st.selectbox('Thalassemia (0=normal,1=fixed,2=reversible)', options=[0,1,2])

    if st.button('Heart Disease Test Result'):
        try:
            Xh = [[float(age), int(sex), int(cp), float(trestbps), float(chol), int(fbs),
                   int(restecg), float(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]]
            heart_prediction = heart_disease_model.predict(Xh)
            heart_diagnosis = 'The person is having heart disease' if int(heart_prediction[0]) == 1 else 'The person does not have any heart disease'
            st.success(heart_diagnosis)
        except Exception as e:
            st.error(f'Prediction error: {e}')

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    st.subheader("Parkinson's Disease Prediction")
    # Use multiple rows of columns; ensure numeric inputs with defaults
    c1, c2, c3 = st.columns(3)
    with c1:
        fo = st.number_input('MDVP:Fo(Hz)', value=119.0, format="%.6f")
    with c2:
        fhi = st.number_input('MDVP:Fhi(Hz)', value=157.0, format="%.6f")
    with c3:
        flo = st.number_input('MDVP:Flo(Hz)', value=74.0, format="%.6f")

    c1, c2, c3 = st.columns(3)
    with c1:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.007, format="%.6f")
    with c2:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', value=0.00007, format="%.6f")
    with c3:
        RAP = st.number_input('MDVP:RAP', value=0.003, format="%.6f")

    c1, c2, c3 = st.columns(3)
    with c1:
        PPQ = st.number_input('MDVP:PPQ', value=0.005, format="%.6f")
    with c2:
        DDP = st.number_input('Jitter:DDP', value=0.02, format="%.6f")
    with c3:
        Shimmer = st.number_input('MDVP:Shimmer', value=0.043, format="%.6f")

    c1, c2, c3 = st.columns(3)
    with c1:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', value=1.0, format="%.6f")
    with c2:
        APQ3 = st.number_input('Shimmer:APQ3', value=0.0, format="%.6f")
    with c3:
        APQ5 = st.number_input('Shimmer:APQ5', value=2.1, format="%.6f")

    c1, c2, c3 = st.columns(3)
    with c1:
        APQ = st.number_input('MDVP:APQ', value=130.0, format="%.6f")
    with c2:
        DDA = st.number_input('Shimmer:DDA', value=41.0, format="%.6f")
    with c3:
        NHR = st.number_input('NHR', value=3.0, format="%.6f")

    c1, c2, c3 = st.columns(3)
    with c1:
        HNR = st.number_input('HNR', value=0.0, format="%.6f")
    with c2:
        RPDE = st.number_input('RPDE', value=32.0, format="%.6f")
    with c3:
        DFA = st.number_input('DFA', value=30.0, format="%.6f")

    c1, c2, c3 = st.columns(3)
    with c1:
        spread1 = st.number_input('spread1', value=145.0, format="%.6f")
    with c2:
        spread2 = st.number_input('spread2', value=0.0, format="%.6f")
    with c3:
        D2 = st.number_input('D2', value=3.0, format="%.6f")

    c1, c2 = st.columns(2)
    with c1:
        PPE = st.number_input('PPE', value=612.0, format="%.6f")

    if st.button("Parkinson's Test Result"):
        try:
            Xp = [[float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                   float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                   float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR),
                   float(HNR), float(RPDE), float(DFA), float(spread1), float(spread2),
                   float(D2), float(PPE)]]
            parkinsons_prediction = parkinsons_model.predict(Xp)
            parkinsons_diagnosis = "The person has Parkinson's disease" if int(parkinsons_prediction[0]) == 1 else "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)
        except Exception as e:
            st.error(f'Prediction error: {e}')

def set_bg_from_url(url, opacity=1):
    footer = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">
    <footer>
        <div style='visibility: visible;margin-top:7rem;justify-content:center;display:flex;'>
            <p style="font-size:1.1rem;">
                Made by Anusha B T
                &nbsp;
                <a href="https://www.linkedin.com/in/anusha-b-t-b8592b286">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="Black" class="bi bi-linkedin" viewBox="0 0 16 16">
                        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                    </svg>          
                </a>
                &nbsp;
                <a href="https://github.com/440Gallery">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="Black" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                </a>
            </p>
        </div>
    </footer>
"""
    st.markdown(footer, unsafe_allow_html=True)
    
    
    # Set background image using HTML and CSS
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image from URL
set_bg_from_url("https://images.everydayhealth.com/homepage/health-topics-2.jpg?w=768", opacity=0.875)
