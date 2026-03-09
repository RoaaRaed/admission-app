import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="University Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Predicting University Admission Chances")
st.markdown("Using Machine Learning to predict your chances of getting admitted!")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Admission.csv"
    df = pd.read_csv(url)
    df = df.drop("Serial No.", axis=1)
    return df

data = load_data()

@st.cache_resource
def train_model(data):
    X = data.drop("Chance of Admit ", axis=1)
    y = data["Chance of Admit "]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return model, scaler, mse, r2

model, scaler, mse, r2 = train_model(data)

# ==================== Sidebar ====================
st.sidebar.header("📋 Enter Student Information")

st.sidebar.markdown("---")
gre = st.sidebar.slider("🎓 GRE Score (Graduate Record Examination)", 260, 340, 310)
st.sidebar.caption("Standardized test for graduate school admissions — max 340")

st.sidebar.markdown("---")
toefl = st.sidebar.slider("🌍 TOEFL Score (English Proficiency Test)", 90, 120, 105)
st.sidebar.caption("Test of English as a Foreign Language — max 120")

st.sidebar.markdown("---")
uni_rating = st.sidebar.selectbox("🏫 University Rating (1 = Low, 5 = High)", [1, 2, 3, 4, 5], index=2)
st.sidebar.caption("The prestige/ranking of the university you're applying to")

st.sidebar.markdown("---")
sop = st.sidebar.slider("📝 SOP — Statement of Purpose (1.0 - 5.0)", 1.0, 5.0, 3.5, step=0.5)
st.sidebar.caption("A written essay explaining your goals and motivation")

st.sidebar.markdown("---")
lor = st.sidebar.slider("💌 LOR — Letter of Recommendation (1.0 - 5.0)", 1.0, 5.0, 3.5, step=0.5)
st.sidebar.caption("A letter written by a professor or employer supporting your application")

st.sidebar.markdown("---")
cgpa = st.sidebar.slider("📊 CGPA — Cumulative Grade Point Average (0 - 10)", 6.0, 10.0, 8.5, step=0.1)
st.sidebar.caption("Your overall academic grade average out of 10")

st.sidebar.markdown("---")
research = st.sidebar.selectbox("🔬 Research Experience", [0, 1], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
st.sidebar.caption("Have you published or participated in any research?")

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict My Chances!", use_container_width=True)

# ==================== Tabs ====================
tab1, tab2, tab3 = st.tabs(["📊 Data & Visualization", "📈 Model Performance", "🔮 Prediction"])

# ---- Tab 1 ----
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CGPA vs Chance of Admission")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=data["CGPA"], y=data["Chance of Admit "], ax=ax1)
        X_cgpa = data[["CGPA"]]
        y_target = data["Chance of Admit "]
        m = LinearRegression().fit(X_cgpa, y_target)
        ax1.plot(data["CGPA"], m.predict(X_cgpa), color="red", linewidth=2)
        ax1.set_xlabel("CGPA")
        ax1.set_ylabel("Chance of Admit")
        st.pyplot(fig1)

    with col2:
        st.subheader("GRE Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(data["GRE Score"], bins=20, kde=True, ax=ax2)
        ax2.set_xlabel("GRE Score")
        ax2.set_ylabel("Number of Students")
        st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# ---- Tab 2 ----
with tab2:
    st.subheader("Model Evaluation Metrics")
    col1, col2 = st.columns(2)
    col1.metric("📉 Mean Squared Error", f"{mse:.4f}")
    col2.metric("📈 R² Score", f"{r2:.4f}")

    st.info("R² Score close to 1.0 means the model predicts very accurately!")

    st.subheader("Feature Importance (Coefficients)")
    features = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    coef_df = coef_df.sort_values("Coefficient", ascending=False)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="viridis", ax=ax4)
    st.pyplot(fig4)

# ---- Tab 3 ----
with tab3:
    st.subheader("🔮 Admission Prediction")

    if predict_btn:
        student = pd.DataFrame([[gre, toefl, uni_rating, sop, lor, cgpa, research]],
                                columns=["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR ", "CGPA", "Research"])
        student_scaled = scaler.transform(student)
        chance = model.predict(student_scaled)[0]
        chance = max(0, min(1, chance))

        st.markdown(f"### Predicted Admission Chance: **{chance*100:.1f}%**")

        if chance >= 0.8:
            st.success("🎉 Excellent! Very high chance of admission!")
        elif chance >= 0.6:
            st.warning("👍 Good chance! Keep improving your profile.")
        else:
            st.error("📚 Work on improving your scores before applying.")

        st.progress(chance)

        st.subheader("Your Profile Summary")
        summary = pd.DataFrame({
            "Factor": [
                "GRE Score (Graduate Record Examination)",
                "TOEFL Score (English Proficiency Test)",
                "University Rating (1=Low, 5=High)",
                "SOP (Statement of Purpose)",
                "LOR (Letter of Recommendation)",
                "CGPA (Cumulative Grade Point Average)",
                "Research Experience"
            ],
            "Your Value": [gre, toefl, uni_rating, sop, lor, cgpa, "✅ Yes" if research else "❌ No"]
        })
        st.table(summary)
    else:
        st.info("👈 Fill in your information in the sidebar and click **Predict**!")