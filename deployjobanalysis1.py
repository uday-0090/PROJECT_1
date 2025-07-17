# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 23:17:23 2025

@author: sucha
"""

# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# 1. Streamlit Setup
st.set_page_config(page_title="Salary Range Prediction", layout="centered")

# 🔧 Inject CSS to enlarge input fields and button
st.markdown("""
<style>
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        font-size: 18px !important;
        padding: 10px 12px !important;
    }
    button[kind="secondary"] {
        font-size: 18px !important;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        background-color: #3366ff;
        color: white;
    }
    .stTextInput input::placeholder {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Model Loader
@st.cache_resource
def load_model():
    df = pd.read_csv("C://Users//sucha//project_deployments_nja//final_data.csv")
    df.drop(["Unnamed: 0", "URL"], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    # Experience split
    df['Min_Experience'] = df['Experience'].str.extract(r'(\d+)', expand=False).astype(int)
    df['Max_Experience'] = df['Experience'].str.extract(r'(\d+)\D+(\d+)', expand=True)[1]
    df['Max_Experience'] = df['Max_Experience'].fillna(df['Min_Experience']).astype(int)

    # Reviews
    df['Reviews'] = df['Reviews'].str.extract(r'(\d+)').astype(int)

    # Job Post Days Ago
    def convert_to_days(post_str):
        if isinstance(post_str, str):
            if 'day' in post_str.lower():
                return int(post_str.split()[0])
            elif 'month' in post_str.lower():
                return int(post_str.split()[0]) * 30
            elif 'hour' in post_str.lower() or 'few' in post_str.lower():
                return 0
        return np.nan
    df['Job_Post_Days_Ago'] = df['Job_Post_History'].apply(convert_to_days)

    # Salary processing
    df['Salary_Disclosed'] = df['Salary'].apply(lambda x: 'Not disclosed' not in x)
    df['Min_Salary'] = df['Salary'].str.replace(',', '').str.extract(r'(\d+\.?\d*)').astype(float)
    df['Max_Salary'] = df['Salary'].str.replace(',', '').str.extract(r'to[ ]*(\d+\.?\d*)').astype(float)
    df['Max_Salary'] = df['Max_Salary'].fillna(df['Min_Salary'])

    # Company Industry Mapping
    company_industry_map = {
        'Accenture': 'IT', 'Oracle': 'IT', 'Rave Technologies': 'IT', 'Snaphunt': 'IT', 'Aspire Systems': 'IT', 'CompuCom': 'IT',
        'Citibank, N.A': 'Banking', 'Credit Suisse': 'Banking', 'BNY Mellon': 'Banking', 'NatWest Group': 'Banking',
        'Duff & Phelps': 'Finance', 'Thinksynq Solutions': 'Finance/Consulting',
        'CoinDCX': 'Fintech', 'Siemens': 'Industrial', 'Prodair Air Products': 'Industrial',
        'Air Products': 'Industrial', 'Sona Comstar': 'Industrial', 'Shell': 'Energy',
        'Ubisoft': 'Gaming', 'Method Studios': 'Media', 'Company3 Method India Private Limited ': 'Media',
        'HealthSpring': 'Healthcare', 'Icon Clinical Research': 'Healthcare', 'Icon Pharmaceutical s': 'Healthcare',
        'RRD': 'Media/Printing', 'Kraftmaid Services India': 'Manufacturing',
        'Associated Auto Solutions International Pvt. Ltd.': 'Auto', 'Eversendai': 'Construction'
    }
    df['Industry'] = df['Company'].map(company_industry_map).fillna('Other/Unknown')

    # Salary class simulation
    def simulate_salary(row):
        title = row['Title'].lower()
        if any(k in title for k in ['senior', 'lead', 'manager', 'consultant']):
            return 'High'
        elif any(k in title for k in ['analyst', 'associate', 'specialist']):
            return 'Medium'
        return 'Low'
    df['Salary_Class'] = df.apply(simulate_salary, axis=1)

    y = df['Salary_Class']
    df = df.drop(columns=['Salary', 'Experience', 'Job_Post_History'])

    X = df[['Company', 'Location', 'Ratings', 'Reviews', 'Title', 'Skills',
            'Min_Experience', 'Max_Experience', 'Job_Post_Days_Ago',
            'Salary_Disclosed', 'Min_Salary', 'Max_Salary', 'Industry']]

    categorical_features = ['Company', 'Location', 'Title', 'Skills', 'Industry']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough')

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=42))
    ])

    model.fit(X, y)
    return model, company_industry_map, df

model, company_industry_map, df_data = load_model()

# 3. UI Layout
st.markdown("<h1 style='text-align: center;'>🔍 Salary Range Prediction</h1>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([3, 3, 3, 3])
with col1:
    company = st.text_input("🏢 Company Name")
with col2:
    title = st.text_input("🧑‍💼 Job Title")
with col3:
    skills = st.text_input("🛠️ Required Skills")
with col4:
    experience = st.selectbox("🎓 Experience", [
    "0-1 Years", "0-2 Years", "0-3 Years", "0-4 Years", "0-5 Years",
    "1-3 Years", "1-4 Years", "2-3 Years", "2-4 Years", "2-6 Years", "2-7 Years",
    "3-5 Years", "3-6 Years", "3-7 Years", "3-8 Years",
    "4-6 Years", "4-8 Years", "4-9 Years", "5-8 Years", "5-10 Years", "12-15 Years"
])

location = st.text_input("📍 Job Location")
search = st.button("🔍 Search")

# 4. Search Logic
if search:
    if not company or not title or not skills or not experience or not location:
        st.error("Please fill in all fields.")
    else:
        with st.spinner('Predicting Salary Range...'):
            match = re.findall(r'\d+', experience)
            min_exp, max_exp = (int(match[0]), int(match[-1])) if match else (0, 0)
            industry = company_industry_map.get(company, 'Other/Unknown')

            company_row = df_data[df_data['Company'].str.lower() == company.lower()]
            if not company_row.empty:
                dynamic_ratings = company_row['Ratings'].values[0]
                dynamic_reviews = company_row['Reviews'].values[0]
            else:
                dynamic_ratings = 4.0
                dynamic_reviews = 100

            title_lower = title.lower()
            if any(k in title_lower for k in ['senior', 'lead', 'manager', 'consultant']):
                min_salary = 12.0
                max_salary = 20.0
            elif any(k in title_lower for k in ['analyst', 'associate', 'specialist']):
                min_salary = 6.0
                max_salary = 12.0
            else:
                min_salary = 3.0
                max_salary = 6.0

            job_row = df_data[
                (df_data['Company'].str.lower() == company.lower()) &
                (df_data['Title'].str.lower() == title.lower())
            ]
            if not job_row.empty:
                job_post_days = job_row['Job_Post_Days_Ago'].values[0]
            else:
                job_post_days = 30

            input_data = {
                'Company': company,
                'Location': location,
                'Ratings': dynamic_ratings,
                'Reviews': dynamic_reviews,
                'Title': title,
                'Skills': skills,
                'Min_Experience': min_exp,
                'Max_Experience': max_exp,
                'Job_Post_Days_Ago': job_post_days,
                'Salary_Disclosed': True,
                'Min_Salary': min_salary,
                'Max_Salary': max_salary,
                'Industry': industry
            }

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]

        # 5. Nice Results Display
        left_col, right_col = st.columns(2)
        with left_col:
            st.info(f"🏢 **Company**: {company.title()}")
            st.info(f"📍 **Location**: {location}")
            st.info(f"🧑‍💼 **Title**: {title.title()}")
            st.info(f"🛠️ **Skills**: {skills}")
            st.info(f"🎓 **Experience**: {min_exp}-{max_exp} Years")

        with right_col:
            st.info(f"🏭 **Industry**: {industry}")
            st.info(f"⭐ **Ratings**: {dynamic_ratings}")
            st.info(f"🗣️ **Reviews**: {dynamic_reviews}")
            st.info(f"🗓️ **Posted**: {job_post_days} days ago")
            st.info(f"💰 **Min Salary (est)**: {min_salary} LPA")
            st.info(f"💰 **Max Salary (est)**: {max_salary} LPA")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #D0F0C0; border-radius: 10px;'>
            <h2 style='color: #1a237e;'>💰 Predicted Salary Class: <b>{prediction}</b></h2>
        </div>
        """, unsafe_allow_html=True)
