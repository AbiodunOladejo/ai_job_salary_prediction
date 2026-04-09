import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(page_title="AI Job Salary Predictor", page_icon="💼",
                   layout="wide", initial_sidebar_state="expanded")

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "salary_model_final.pkl"))

@st.cache_data
def load_ref():
    with open(os.path.join(BASE, "encoding_ref.json")) as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "ai_job_dataset.csv"))

model = load_model()
ref   = load_ref()
df    = load_data()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=72)
    st.title("AI Salary Predictor")
    st.markdown("*XGBoost Pipeline · R² = 0.884 · MAE ≈ $12,945*")
    st.divider()

    st.subheader("📋 Job Details")
    job_title  = st.selectbox("Job Title",  ref['all_titles'])
    industry   = st.selectbox("Industry",   ref['all_industries'])
    experience = st.selectbox("Experience Level", list(ref['exp_options'].values()))
    exp_code   = {v:k for k,v in ref['exp_options'].items()}[experience]
    education_disp = st.selectbox("Education Required", ref['edu_options'])
    education  = ref['edu_map'][education_disp]

    st.subheader("🏢 Company Details")
    location   = st.selectbox("Company Location", ref['all_locations'])
    size_disp  = st.selectbox("Company Size", list(ref['size_options'].values()))
    size_code  = {v:k for k,v in ref['size_options'].items()}[size_disp]
    emp_disp   = st.selectbox("Employment Type", list(ref['emp_options'].values()))
    emp_code   = {v:k for k,v in ref['emp_options'].items()}[emp_disp]
    remote_disp = st.selectbox("Work Mode", list(ref['remote_options'].keys()))
    remote_val  = ref['remote_options'][remote_disp]
    same_country = st.checkbox("Employee lives in same country as company", value=True)

    st.subheader("📊 Role Specifics")
    years_exp   = st.slider("Years of Experience Required", 0, 20, 3)
    benefits    = st.slider("Benefits Score (1–10)", 5.0, 10.0, 7.5, step=0.1)
    desc_length = st.slider("Job Description Length (chars)", 500, 2500, 1500, step=50)

    predict_btn = st.button("🔮 Predict Salary", use_container_width=True, type="primary")

st.title("💼 Global AI Job Salary Predictor")
st.markdown("Estimate annual salary (USD) for AI/tech roles worldwide using **XGBoost + sklearn Pipeline** trained on 14,517 real job postings.")
st.divider()

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Training Records", f"{len(df):,}")
c2.metric("Avg Salary", f"${df['salary_usd'].mean():,.0f}")
c3.metric("Model R²", "0.884")
c4.metric("Model MAE", "~$12,945")
c5.metric("5-fold CV R²", f"{ref['cv_mean']} ± {ref['cv_std']}")
st.divider()

if predict_btn:
    input_df = pd.DataFrame([{
        'job_title': job_title, 'experience_level': exp_code,
        'employment_type': emp_code, 'company_location': location,
        'company_size': size_code, 'education_required': education,
        'industry': industry, 'remote_ratio': remote_val,
        'years_experience': years_exp, 'job_description_length': desc_length,
        'benefits_score': benefits, 'same_country': int(same_country),
    }])

    prediction = model.predict(input_df)[0]
    monthly    = prediction / 12

    st.success("✅ Prediction complete!")
    r1,r2,r3 = st.columns(3)
    r1.metric("💰 Predicted Annual Salary", f"${prediction:,.0f}")
    r2.metric("📅 Estimated Monthly", f"${monthly:,.0f}")
    avg_title = df[df['job_title']==job_title]['salary_usd'].mean()
    if not np.isnan(avg_title):
        r3.metric("vs Dataset Avg for This Role", f"${avg_title:,.0f}",
                  delta=f"${prediction-avg_title:+,.0f}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📝 Input Summary")
        st.dataframe(pd.DataFrame({
            "Feature": ["Job Title","Industry","Experience","Education","Location",
                        "Employment Type","Company Size","Work Mode","Years Exp",
                        "Benefits","Same Country"],
            "Value": [job_title, industry, experience, education_disp, location,
                      emp_disp, size_disp, remote_disp, years_exp, benefits,
                      "Yes" if same_country else "No"]
        }), use_container_width=True, hide_index=True)

    with col_b:
        with st.expander("🔍 How was this predicted?", expanded=True):
            st.markdown(f"""
**Model Architecture: make_pipeline**

The pipeline bundles two steps into one object:

1. **ColumnTransformer + OneHotEncoder**  
   Converts all categorical features (`job_title`, `experience_level`, `company_location`, etc.) into binary columns. Each unique value gets its own column — this preserves full information.

2. **XGBoost Regressor** (tuned via GridSearchCV)  
   - `n_estimators`: {ref['best_params']['n_estimators']}  
   - `learning_rate`: {ref['best_params']['learning_rate']}  
   - `max_depth`: {ref['best_params']['max_depth']}

**Why this works better than manual encoding:**  
OneHotEncoder gives each job title its own feature — XGBoost learns salary patterns for each title independently, rather than treating titles as an ordered scale.

**Validation:**  
- Test R² = 0.884  
- 5-fold CV R² = {ref['cv_mean']} ± {ref['cv_std']}
            """)
else:
    st.info("👈 Fill in the job details in the sidebar, then click **Predict Salary**.")

st.divider()
st.subheader("📊 Dataset Insights")
tab1,tab2,tab3,tab4,tab5 = st.tabs(["💵 Distribution","🎓 Experience","🌍 Country","🏭 Industry","🏠 Work Mode"])

try:
    import plotly.express as px

    with tab1:
        fig = px.histogram(df, x="salary_usd", nbins=60, title="Salary Distribution",
                           color_discrete_sequence=["#4F8BF9"], labels={"salary_usd":"Salary (USD)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Most salaries cluster between $50K–$160K. Right-skewed distribution — executive salaries pull the mean above the median.")

    with tab2:
        exp_map = {"EN":"Entry","MI":"Mid","SE":"Senior","EX":"Executive"}
        df2 = df.copy(); df2["Experience"] = df2["experience_level"].map(exp_map)
        fig2 = px.box(df2, x="Experience", y="salary_usd", color="Experience",
                      category_orders={"Experience":["Entry","Mid","Senior","Executive"]},
                      title="Salary by Experience Level", labels={"salary_usd":"Salary (USD)"})
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Experience level is the #1 salary predictor. Executives earn ~3× more than Entry-level workers.")

    with tab3:
        c_avg = df.groupby("company_location")["salary_usd"].mean().sort_values(ascending=False).reset_index()
        fig3 = px.bar(c_avg, x="salary_usd", y="company_location", orientation="h",
                      title="Average Salary by Country", color="salary_usd",
                      color_continuous_scale="Blues",
                      labels={"salary_usd":"Avg Salary (USD)","company_location":""})
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Switzerland, Denmark, Norway lead. Geographic location has a major salary impact — up to 2× difference between highest and lowest countries.")

    with tab4:
        i_avg = df.groupby("industry")["salary_usd"].mean().sort_values(ascending=False).reset_index()
        fig4 = px.bar(i_avg, x="salary_usd", y="industry", orientation="h",
                      title="Average Salary by Industry", color="salary_usd",
                      color_continuous_scale="Greens",
                      labels={"salary_usd":"Avg Salary (USD)","industry":""})
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("AI talent commands premium pay across all industries. The spread is narrow — unlike traditional sectors, industry choice matters less than experience and location.")

    with tab5:
        remote_map = {0:"On-site",50:"Hybrid",100:"Fully Remote"}
        df3 = df.copy(); df3["Work Mode"] = df3["remote_ratio"].map(remote_map)
        fig5 = px.box(df3, x="Work Mode", y="salary_usd", color="Work Mode",
                      category_orders={"Work Mode":["On-site","Hybrid","Fully Remote"]},
                      title="Salary by Work Mode", labels={"salary_usd":"Salary (USD)"})
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Fully remote roles earn slightly more on average. The AI market is nearly evenly split across all three work modes.")

except ImportError:
    st.warning("Install plotly: pip install plotly")

st.divider()
st.markdown("<p style='text-align:center;color:gray'>💼 AI Job Salary Predictor · XGBoost Pipeline · R²=0.884 · MAE≈$12,945 · Portfolio Project</p>",
            unsafe_allow_html=True)
