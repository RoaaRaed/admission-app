import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== Language ====================
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

st.set_page_config(
    page_title="University Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

col_lang1, col_lang2 = st.columns([8, 1])
with col_lang2:
    if st.button("🌐 العربية" if st.session_state.lang == "EN" else "🌐 English"):
        st.session_state.lang = "AR" if st.session_state.lang == "EN" else "EN"
        st.rerun()

lang = st.session_state.lang

# ==================== Translations ====================
T = {
    "title": {
        "EN": "🎓 Predicting University Admission Chances",
        "AR": "🎓 التنبؤ بفرص القبول الجامعي"
    },
    "subtitle": {
        "EN": "Using Machine Learning to predict your chances of getting admitted!",
        "AR": "استخدام تعلم الآلة للتنبؤ بفرصتك في القبول!"
    },
    "sidebar_header": {
        "EN": "📋 Enter Student Information",
        "AR": "📋 أدخل بيانات الطالب"
    },
    "university": {
        "EN": "🏫 Target University",
        "AR": "🏫 الجامعة المستهدفة"
    },
    "university_caption": {
        "EN": "Select the university you're applying to",
        "AR": "اختر الجامعة التي تتقدم إليها"
    },
    "major": {
        "EN": "📚 Field of Study",
        "AR": "📚 التخصص الدراسي"
    },
    "major_caption": {
        "EN": "Select your intended field of study",
        "AR": "اختر تخصصك الدراسي"
    },
    "gre": {
        "EN": "🎓 GRE Score (Graduate Record Examination)",
        "AR": "🎓 درجة GRE (اختبار السجل الجامعي)"
    },
    "gre_caption": {
        "EN": "Standardized test for graduate school admissions — max 340",
        "AR": "اختبار معياري للقبول في الدراسات العليا — الحد الأقصى 340"
    },
    "toefl": {
        "EN": "🌍 TOEFL Score (English Proficiency Test)",
        "AR": "🌍 درجة TOEFL (اختبار اللغة الإنجليزية)"
    },
    "toefl_caption": {
        "EN": "Test of English as a Foreign Language — max 120",
        "AR": "اختبار اللغة الإنجليزية كلغة أجنبية — الحد الأقصى 120"
    },
    "uni_rating": {
        "EN": "⭐ University Rating (1 = Low, 5 = High)",
        "AR": "⭐ تصنيف الجامعة (1 = منخفض، 5 = مرتفع)"
    },
    "uni_rating_caption": {
        "EN": "The prestige/ranking of the university you're applying to",
        "AR": "مستوى وتصنيف الجامعة التي تتقدم إليها"
    },
    "sop": {
        "EN": "📝 SOP — Statement of Purpose (1.0 - 5.0)",
        "AR": "📝 SOP — رسالة الغرض الدراسي (1.0 - 5.0)"
    },
    "sop_caption": {
        "EN": "A written essay explaining your goals and motivation",
        "AR": "مقال مكتوب يشرح أهدافك ودوافعك الدراسية"
    },
    "lor": {
        "EN": "💌 LOR — Letter of Recommendation (1.0 - 5.0)",
        "AR": "💌 LOR — خطاب التوصية (1.0 - 5.0)"
    },
    "lor_caption": {
        "EN": "A letter written by a professor or employer supporting your application",
        "AR": "خطاب من أستاذ أو صاحب عمل يدعم طلبك"
    },
    "cgpa": {
        "EN": "📊 CGPA — Cumulative Grade Point Average (0 - 10)",
        "AR": "📊 CGPA — المعدل التراكمي (0 - 10)"
    },
    "cgpa_caption": {
        "EN": "Your overall academic grade average out of 10",
        "AR": "معدلك الأكاديمي الإجمالي من 10"
    },
    "research": {
        "EN": "🔬 Research Experience",
        "AR": "🔬 الخبرة البحثية"
    },
    "research_caption": {
        "EN": "Have you published or participated in any research?",
        "AR": "هل نشرت أو شاركت في أي بحث علمي؟"
    },
    "predict_btn": {
        "EN": "🔮 Predict My Chances!",
        "AR": "🔮 توقع فرصتي!"
    },
    "tab1": {"EN": "📊 Data & Visualization", "AR": "📊 البيانات والتحليل"},
    "tab2": {"EN": "📈 Model Performance", "AR": "📈 أداء النموذج"},
    "tab3": {"EN": "🔮 Prediction", "AR": "🔮 التنبؤ"},
    "dataset_preview": {"EN": "Dataset Preview", "AR": "عرض البيانات"},
    "cgpa_chart": {"EN": "CGPA vs Chance of Admission", "AR": "المعدل التراكمي مقابل فرصة القبول"},
    "gre_chart": {"EN": "GRE Score Distribution", "AR": "توزيع درجات GRE"},
    "heatmap": {"EN": "Correlation Heatmap", "AR": "خريطة الارتباط"},
    "model_metrics": {"EN": "Model Evaluation Metrics", "AR": "مقاييس تقييم النموذج"},
    "r2_info": {
        "EN": "R² Score close to 1.0 means the model predicts very accurately!",
        "AR": "كلما اقترب R² من 1.0، كان النموذج أكثر دقة في التنبؤ!"
    },
    "feature_importance": {"EN": "Feature Importance (Coefficients)", "AR": "أهمية المتغيرات (المعاملات)"},
    "prediction_title": {"EN": "🔮 Admission Prediction", "AR": "🔮 نتيجة التنبؤ"},
    "predicted_chance": {"EN": "Predicted Admission Chance", "AR": "فرصة القبول المتوقعة"},
    "excellent": {"EN": "🎉 Excellent! Very high chance of admission!", "AR": "🎉 ممتاز! فرصة قبول عالية جداً!"},
    "good": {"EN": "👍 Good chance! Keep improving your profile.", "AR": "👍 فرصة جيدة! استمر في تحسين ملفك."},
    "low": {"EN": "📚 Work on improving your scores before applying.", "AR": "📚 اعمل على تحسين درجاتك قبل التقديم."},
    "profile_summary": {"EN": "Your Profile Summary", "AR": "ملخص ملفك الدراسي"},
    "factor": {"EN": "Factor", "AR": "العامل"},
    "value": {"EN": "Your Value", "AR": "قيمتك"},
    "no_predict": {
        "EN": "👈 Fill in your information in the sidebar and click **Predict**!",
        "AR": "👈 أدخل بياناتك في الشريط الجانبي واضغط **توقع**!"
    },
    "yes": {"EN": "✅ Yes", "AR": "✅ نعم"},
    "no": {"EN": "❌ No", "AR": "❌ لا"},
    "factors": {
        "EN": [
            "Target University",
            "Field of Study",
            "GRE Score (Graduate Record Examination)",
            "TOEFL Score (English Proficiency Test)",
            "University Rating (1=Low, 5=High)",
            "SOP (Statement of Purpose)",
            "LOR (Letter of Recommendation)",
            "CGPA (Cumulative Grade Point Average)",
            "Research Experience"
        ],
        "AR": [
            "الجامعة المستهدفة",
            "التخصص الدراسي",
            "درجة GRE (اختبار السجل الجامعي)",
            "درجة TOEFL (اختبار اللغة الإنجليزية)",
            "تصنيف الجامعة (1=منخفض، 5=مرتفع)",
            "SOP (رسالة الغرض الدراسي)",
            "LOR (خطاب التوصية)",
            "CGPA (المعدل التراكمي)",
            "الخبرة البحثية"
        ]
    }
}

def t(key):
    return T[key][lang]

# ==================== Header ====================
st.title(t("title"))
st.markdown(t("subtitle"))

# ==================== Load Data ====================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Admission.csv"
    df = pd.read_csv(url)
    df = df.drop("Serial No.", axis=1)
    return df

data = load_data()

# ==================== Train Model ====================
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
st.sidebar.header(t("sidebar_header"))

st.sidebar.markdown("---")
university = st.sidebar.selectbox(t("university"), [
    "MIT - Massachusetts Institute of Technology",
    "Stanford University",
    "Harvard University",
    "University of California, Berkeley",
    "Carnegie Mellon University",
    "University of Toronto",
    "University of Melbourne",
    "National University of Singapore",
    "Other"
])
st.sidebar.caption(t("university_caption"))

st.sidebar.markdown("---")
major = st.sidebar.selectbox(t("major"), [
    "Computer Science & AI",
    "Data Science & Statistics",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Business Administration (MBA)",
    "Biomedical Sciences",
    "Mathematics & Physics",
    "Other"
])
st.sidebar.caption(t("major_caption"))

st.sidebar.markdown("---")
gre = st.sidebar.slider(t("gre"), 260, 340, 310)
st.sidebar.caption(t("gre_caption"))

st.sidebar.markdown("---")
toefl = st.sidebar.slider(t("toefl"), 90, 120, 105)
st.sidebar.caption(t("toefl_caption"))

st.sidebar.markdown("---")
uni_rating = st.sidebar.selectbox(t("uni_rating"), [1, 2, 3, 4, 5], index=2)
st.sidebar.caption(t("uni_rating_caption"))

st.sidebar.markdown("---")
sop = st.sidebar.slider(t("sop"), 1.0, 5.0, 3.5, step=0.5)
st.sidebar.caption(t("sop_caption"))

st.sidebar.markdown("---")
lor = st.sidebar.slider(t("lor"), 1.0, 5.0, 3.5, step=0.5)
st.sidebar.caption(t("lor_caption"))

st.sidebar.markdown("---")
cgpa = st.sidebar.slider(t("cgpa"), 6.0, 10.0, 8.5, step=0.1)
st.sidebar.caption(t("cgpa_caption"))

st.sidebar.markdown("---")
research = st.sidebar.selectbox(t("research"), [0, 1],
    format_func=lambda x: t("yes") if x == 1 else t("no"))
st.sidebar.caption(t("research_caption"))

st.sidebar.markdown("---")
predict_btn = st.sidebar.button(t("predict_btn"), use_container_width=True)

# ==================== Tabs ====================
tab1, tab2, tab3 = st.tabs([t("tab1"), t("tab2"), t("tab3")])

# ---- Tab 1 ----
with tab1:
    st.subheader(t("dataset_preview"))
    st.dataframe(data.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t("cgpa_chart"))
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
        st.subheader(t("gre_chart"))
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(data["GRE Score"], bins=20, kde=True, ax=ax2)
        ax2.set_xlabel("GRE Score")
        ax2.set_ylabel("Number of Students")
        st.pyplot(fig2)

    st.subheader(t("heatmap"))
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# ---- Tab 2 ----
with tab2:
    st.subheader(t("model_metrics"))
    col1, col2 = st.columns(2)
    col1.metric("📉 Mean Squared Error", f"{mse:.4f}")
    col2.metric("📈 R² Score", f"{r2:.4f}")
    st.info(t("r2_info"))

    st.subheader(t("feature_importance"))
    features = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    coef_df = coef_df.sort_values("Coefficient", ascending=False)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="viridis", ax=ax4)
    st.pyplot(fig4)

# ---- Tab 3 ----
with tab3:
    st.subheader(t("prediction_title"))

    if predict_btn:
        student = pd.DataFrame([[gre, toefl, uni_rating, sop, lor, cgpa, research]],
                                columns=["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR ", "CGPA", "Research"])
        student_scaled = scaler.transform(student)
        chance = model.predict(student_scaled)[0]
        chance = max(0, min(1, chance))

        st.markdown(f"### {t('predicted_chance')}: **{chance*100:.1f}%**")

        if chance >= 0.8:
            st.success(t("excellent"))
        elif chance >= 0.6:
            st.warning(t("good"))
        else:
            st.error(t("low"))

        st.progress(chance)

        st.subheader(t("profile_summary"))
        summary = pd.DataFrame({
            t("factor"): t("factors"),
            t("value"): [
                university, major, gre, toefl, uni_rating, sop, lor, cgpa,
                t("yes") if research else t("no")
            ]
        })
        st.table(summary)
    else:
        st.info(t("no_predict"))
