import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# 8. Light Custom CSS for spacing, rounded cards, and clean typography
st.markdown("""
<style>
    /* Main container padding and max width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1050px;
    }
    
    /* Section title styling */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Rounded corners and subtle border for section cards */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
        background-color: #fafafa;
        border: 1px solid #e2e8f0;
        padding: 0.5rem 1rem 1rem 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 7. Sidebar with project information and disclaimer
with st.sidebar:
    st.header("About this project")
    st.write(
        "This is a beginner machine learning project created for a 1st-year B.Tech CSE (AI/ML) course. "
        "It uses a Random Forest Classifier trained on 45,000 applicant records to predict loan approvals."
    )
    st.divider()
    st.caption(
        "⚠️ **Disclaimer**: This is an educational student project for learning AI/ML concepts "
        "and is not a real financial decision tool."
    )

# 2. Header Section
st.title("🏦 Loan Approval Predictor")
st.write("Fill in the applicant information below to predict whether a loan application is likely to be approved or rejected.")
st.divider()

# Load the saved model (logic unchanged)
@st.cache_resource
def load_saved_model():
    if os.path.exists("model.joblib"):
        return joblib.load("model.joblib")
    return None

data = load_saved_model()

if data is None:
    st.error("Error: 'model.joblib' file not found. Please train and save the model first.")
else:
    model = data['model']
    feature_names = data['feature_names']

    # 3 & 4. Grouped Input Form Sections with 2-3 Columns Layout
    
    # Section 1: Personal Details
    with st.container(border=True):
        st.markdown('<div class="section-title">👤 Personal Details</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=28)
            income = st.number_input("Annual Income ($)", min_value=1000, max_value=5000000, value=65000, step=1000)
        with col2:
            gender = st.selectbox("Gender", ["female", "male"])
            emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=50, value=4)
        with col3:
            education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    # Section 2: Loan Details
    with st.container(border=True):
        st.markdown('<div class="section-title">💰 Loan Details</div>', unsafe_allow_html=True)
        col4, col5 = st.columns(2)
        with col4:
            loan_amount = st.number_input("Loan Amount Requested ($)", min_value=500, max_value=100000, value=10000, step=500)
            loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        with col5:
            int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=11.0, step=0.1)
            # Default loan percent calculation based on inputs, clipped between 0 and 1
            calculated_pct = min(1.0, round(loan_amount / max(income, 1), 2))
            loan_percent = st.number_input(
                "Loan as Fraction of Income", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(calculated_pct), 
                step=0.01, 
                help="For example, 0.15 means the loan is 15% of annual income."
            )

    # Section 3: Credit Details
    with st.container(border=True):
        st.markdown('<div class="section-title">📊 Credit Details</div>', unsafe_allow_html=True)
        col6, col7, col8 = st.columns(3)
        with col6:
            cred_hist = st.number_input("Credit History Length (years)", min_value=0.0, max_value=35.0, value=5.0, step=0.5)
        with col7:
            credit_score = st.number_input("Credit Score (300 to 850)", min_value=300, max_value=850, value=650, step=5)
        with col8:
            defaults = st.selectbox("Previous Loan Default on File", ["No", "Yes"])

    # 5. Prominent Predict Button
    predict_clicked = st.button("🔍 Predict Loan Status", type="primary", use_container_width=True)

    if predict_clicked:
        # Put inputs into a dataframe (logic unchanged)
        applicant_dict = {
            'person_age': float(age),
            'person_gender': gender,
            'person_education': education,
            'person_income': float(income),
            'person_emp_exp': int(emp_exp),
            'person_home_ownership': home_ownership,
            'loan_amnt': float(loan_amount),
            'loan_intent': loan_intent,
            'loan_int_rate': float(int_rate),
            'loan_percent_income': float(loan_percent),
            'cb_person_cred_hist_length': float(cred_hist),
            'credit_score': int(credit_score),
            'previous_loan_defaults_on_file': defaults
        }
        
        app_df = pd.DataFrame([applicant_dict])

        # Label encode binary columns (logic unchanged)
        app_df['person_gender'] = app_df['person_gender'].map({'female': 0, 'male': 1})
        app_df['previous_loan_defaults_on_file'] = app_df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})

        # One-hot encode multi-category columns (logic unchanged)
        categorical_cols = ['person_education', 'person_home_ownership', 'loan_intent']
        app_encoded = pd.get_dummies(app_df, columns=categorical_cols, dtype=int)

        # Align columns with the model's training features (logic unchanged)
        app_aligned = app_encoded.reindex(columns=feature_names, fill_value=0)

        # Make prediction (logic unchanged)
        prediction = model.predict(app_aligned)[0]
        probabilities = model.predict_proba(app_aligned)[0]

        # 6. Improved Result Display
        st.write("")
        st.subheader("Prediction Result")
        
        res_col1, res_col2 = st.columns([1, 1])

        if prediction == 1:
            confidence = probabilities[1] * 100
            with res_col1:
                st.success("### 🎉 Loan Approved")
                st.metric(label="Model Confidence", value=f"{confidence:.1f}%")
                st.progress(float(probabilities[1]))
            with res_col2:
                st.markdown("**Key factors influencing this prediction:**")
                factors = []
                if defaults == "No":
                    factors.append("Clean credit history (no previous defaults on file)")
                if loan_percent >= 0.15:
                    factors.append(f"Requested loan proportion ({loan_percent * 100:.0f}% of income) aligns with approved loan patterns")
                if int_rate >= 10.0:
                    factors.append(f"Interest rate ({int_rate:.1f}%) matches the risk category")
                if credit_score >= 600:
                    factors.append(f"Satisfactory credit score ({credit_score})")
                if home_ownership in ["OWN", "MORTGAGE"]:
                    factors.append(f"Home ownership status ({home_ownership}) provides asset backing")
                if not factors:
                    factors.append("Applicant profile meets the criteria learned by the Random Forest model")
                for f in factors[:3]:
                    st.write(f"- {f}")
        else:
            confidence = probabilities[0] * 100
            with res_col1:
                st.error("### ❌ Loan Rejected")
                st.metric(label="Model Confidence", value=f"{confidence:.1f}%")
                st.progress(float(probabilities[0]))
            with res_col2:
                st.markdown("**Key factors influencing this prediction:**")
                factors = []
                if defaults == "Yes":
                    factors.append("Prior loan default on file significantly reduces approval chances")
                if credit_score < 620:
                    factors.append(f"Credit score ({credit_score}) is below standard approval levels")
                if home_ownership == "RENT":
                    factors.append("Rental residence without home equity")
                if loan_percent < 0.15:
                    factors.append(f"Loan-to-income ratio ({loan_percent * 100:.0f}%) does not match approved tier patterns")
                if int_rate < 10.0:
                    factors.append(f"Offered interest rate ({int_rate:.1f}%) is outside typical approval margins")
                if not factors:
                    factors.append("Overall combination of income, loan amount, and credit indicators did not meet the model's approval threshold")
                for f in factors[:3]:
                    st.write(f"- {f}")
