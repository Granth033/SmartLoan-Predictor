import streamlit as st
import pickle
import numpy as np

# Page setup & Styling
st.set_page_config(page_title="Pro Loan Predictor", page_icon="🏦", layout="wide")
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0078D7; color: white; width: 100%; border-radius: 10px; font-weight: bold; font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Load models safely
try:
    model = pickle.load(open('loan_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("System Error: Model files not found. Please ensure 'loan_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# Header
st.title("💸 Advance Loan Predictor")
st.write("Enter your details below to check your loan eligibility and approval chances.")
st.divider()

# --- INPUT SECTION (16 Features) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Details")
    income = st.number_input("Applicant Income ($)", value=0)
    co_income = st.number_input("Co-Applicant Income ($)", value=0)
    loan_amount = st.number_input("Loan Amount ($)", value=0)
    loan_term = st.number_input("Loan Amount Term (e.g., 360 days)", value=360.0)
    credit_history = st.selectbox("Credit History", options=[1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)")
    
    st.subheader("Personal Details")
    gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    married = st.selectbox("Married", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    education = st.selectbox("Education", options=[1, 0], format_func=lambda x: "Graduate" if x==1 else "Not Graduate")

with col2:
    st.subheader("Other Details")
    self_employed = st.selectbox("Self Employed", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    
    st.write("**Dependents**")
    dep_1 = st.selectbox("Dependents = 1?", [0, 1])
    dep_2 = st.selectbox("Dependents = 2?", [0, 1])
    dep_3 = st.selectbox("Dependents = 3+?", [0, 1])
    
    st.write("**Property Area**")
    prop_semiurban = st.selectbox("Area: Semiurban?", [0, 1])
    prop_urban = st.selectbox("Area: Urban?", [0, 1])
    prop_rural = st.selectbox("Area: Rural?", [0, 1])
    
    extra_feature = st.number_input("Extra Feature (If any in your dataset)", value=0)

st.divider()

# --- PREDICTION SECTION ---
if st.button("🚀 Check Loan Status"):
    # Create array with all 16 inputs
    input_data = np.array([[income, co_income, loan_amount, loan_term, credit_history, 
                            gender, married, education, self_employed, 
                            dep_1, dep_2, dep_3, 
                            prop_semiurban, prop_urban, prop_rural, extra_feature]])
    
    try:
        # Scale Data
        scaled_features = scaler.transform(input_data)
        
        # Predict Output
        prediction = model.predict(scaled_features)
        
        # Predict Probability (Confidence Score)
        try:
            probability = model.predict_proba(scaled_features)[0]
            approval_chance = round(probability[1] * 100, 2)
            rejection_chance = round(probability[0] * 100, 2)
        except AttributeError:
            approval_chance, rejection_chance = None, None

        # --- DISPLAY RESULTS ---
        st.markdown("### 📊 Prediction Results")
        
        if prediction[0] == 1:
            st.balloons()
            st.success("🎉 Congratulations! Your Loan is APPROVED.")
            if approval_chance:
                st.info(f"**Approval Probability:** The AI model is **{approval_chance}%** confident in approving your loan.")
        else:
            st.error("❌ Sorry, Your Loan application is REJECTED.")
            if rejection_chance:
                st.warning(f"**Rejection Probability:** The AI model is **{rejection_chance}%** confident in rejecting this loan.")
        
        # --- DISPLAY SUMMARY DASHBOARD ---
        st.markdown("#### 📝 Application Summary")
        
        # Creating a nice 4-column metric layout
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(label="Total Income", value=f"${income + co_income}")
        m2.metric(label="Loan Amount", value=f"${loan_amount}")
        m3.metric(label="Loan Term", value=f"{loan_term} Days")
        m4.metric(label="Credit Score", value="Good" if credit_history == 1.0 else "Bad")
            
    except Exception as e:
        st.error(f"Error while predicting: {e}")