
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Healthcare AI | Personalized Health",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM PROFESSIONAL BLUE AND TEAL THEME
# =========================================================

st.markdown("""
<style>

/* Main application background */

.stApp {
    background-color: #F3F7FB;
    color: #263B50;
    font-family: Arial, sans-serif;
}

/* Main content spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102A43 0%, #163D5B 100%);
    border-right: 1px solid #244B68;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background-color: transparent;
    border-radius: 9px;
    padding: 9px 10px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #24516D;
}

/* Headings */

h1, h2, h3 {
    color: #153B59 !important;
    font-weight: 700 !important;
}

p {
    color: #526B80;
}

/* Main title */

.main-heading {
    font-size: 34px;
    font-weight: 800;
    color: #153B59;
    margin-bottom: 4px;
}

.main-subheading {
    font-size: 15px;
    color: #71869A;
    margin-bottom: 24px;
}

/* Hero banner */

.hero-card {
    background: linear-gradient(110deg, #123B5A, #087F8C);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 7px 20px rgba(15, 55, 85, 0.12);
}

.hero-card h2 {
    color: white !important;
    font-size: 27px;
    margin-bottom: 8px;
}

.hero-card p {
    color: #E2F4F6;
    font-size: 15px;
    margin-bottom: 0;
}

/* Custom metric cards */

.metric-card {
    background: #FFFFFF;
    border: 1px solid #DCE8F1;
    border-radius: 15px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 4px 14px rgba(23, 61, 89, 0.05);
}

.metric-label {
    font-size: 13px;
    color: #72869A;
    font-weight: 600;
}

.metric-value {
    font-size: 29px;
    font-weight: 800;
    color: #153B59;
    margin-top: 8px;
}

.metric-icon {
    font-size: 22px;
}

/* Section heading */

.section-heading {
    color: #153B59;
    font-size: 22px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Feature cards */

.feature-card {
    background: #FFFFFF;
    border: 1px solid #DCE8F1;
    border-radius: 15px;
    padding: 22px;
    min-height: 150px;
    box-shadow: 0 4px 14px rgba(23, 61, 89, 0.04);
}

.feature-card h3 {
    color: #087F8C !important;
    font-size: 18px;
    margin-bottom: 10px;
}

.feature-card p {
    font-size: 14px;
    line-height: 1.6;
}

/* Recommendation cards */

.recommendation-card {
    background: #FFFFFF;
    border: 1px solid #DCE8F1;
    border-left: 5px solid #0A8992;
    border-radius: 12px;
    padding: 19px;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(23, 61, 89, 0.04);
}

.recommendation-card h3 {
    color: #153B59 !important;
    font-size: 18px;
    margin-bottom: 10px;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background: #087F8C !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 650 !important;
    padding: 10px 18px !important;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #056773 !important;
    color: white !important;
}

/* Inputs */

.stTextInput input,
.stNumberInput input {
    border-radius: 9px;
    border: 1px solid #CBDCE8;
}

div[data-baseweb="select"] > div {
    border-radius: 9px;
    border-color: #CBDCE8;
}

/* Form and containers */

[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #DCE8F1;
    border-radius: 15px;
    padding: 22px;
}

div[data-testid="stExpander"] {
    background: white;
    border-radius: 12px;
    border: 1px solid #DCE8F1;
}

/* Divider */

hr {
    border-color: #DFE9F1;
}

/* Footer */

.footer {
    text-align: center;
    color: #8294A5;
    font-size: 13px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. PROJECT FOLDER PATHS
# =========================================================

# streamlit_app.py is inside the app.py folder.
# parent.parent takes us to the main project folder.

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "dataset"


# =========================================================
# 4. CHECK PROJECT FILES
# =========================================================

required_files = [
    OUTPUT_DIR / "healthcare_model.pkl",
    OUTPUT_DIR / "health_vectorizer.pkl",
    OUTPUT_DIR / "health_information.csv",
    DATASET_DIR / "Cleaned_Dataset.csv"
]

missing_files = [
    file for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("Some required project files are missing.")

    for file in missing_files:
        st.write(str(file))

    st.stop()


# =========================================================
# 5. LOAD PROJECT FILES
# =========================================================

try:

    model = joblib.load(
        OUTPUT_DIR / "healthcare_model.pkl"
    )

    vectorizer = joblib.load(
        OUTPUT_DIR / "health_vectorizer.pkl"
    )

    health_df = pd.read_csv(
        OUTPUT_DIR / "health_information.csv"
    )

    original_df = pd.read_csv(
        DATASET_DIR / "Cleaned_Dataset.csv"
    )

except Exception as error:

    st.error(f"Error loading project files: {error}")
    st.stop()


# =========================================================
# 6. HELPER FUNCTIONS
# =========================================================

def metric_card(icon, label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title):

    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


def normalize_outcome_counts(data):

    outcomes = (
        data["outcome_variable"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    return outcomes.value_counts().reindex(
        ["Positive", "Negative"],
        fill_value=0
    )


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 5px 10px;">
            <div style="font-size:48px;">🏥</div>
            <h2 style="color:white !important;
                       margin-bottom:2px;
                       font-size:23px;">
                HealthCare AI
            </h2>
            <p style="color:#BBD2E2 !important;
                      font-size:12px;">
                PERSONALIZED HEALTH PLATFORM
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "<p style='color:#9EBBCD !important;"
        "font-size:11px;font-weight:700;'>MAIN MENU</p>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🧑‍⚕️ Health Prediction",
            "💡 Recommendations",
            "📊 Analytics",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "<p style='color:#9EBBCD !important;"
        "font-size:11px;font-weight:700;'>TECHNOLOGIES</p>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="line-height:2.1; font-size:14px;">
        🐍 Python<br>
        🤖 Random Forest<br>
        🔎 TF-IDF<br>
        📐 Cosine Similarity<br>
        📊 Streamlit
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        """
        <p style="text-align:center;
                  color:#BBD2E2 !important;
                  font-size:11px;">
            Academic Project Prototype<br>
            Version 1.0 | 2026
        </p>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 8. COMMON PAGE HEADER
# =========================================================

st.markdown(
    """
    <div class="main-heading">
        Personalized Healthcare System
    </div>
    <div class="main-subheading">
        Intelligent healthcare analytics and
        personalized health-information recommendations
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "Educational prototype only. Model predictions are "
    "experimental dataset classifications, not medical "
    "diagnoses. Recommendations are general educational "
    "information, not treatment or medicine advice."
)


# =========================================================
# 9. DASHBOARD HOME PAGE
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero-card">
            <h2>Welcome to Healthcare AI 👋</h2>
            <p>
                Explore machine learning-based experimental
                classification, health-information recommendations
                and interactive dataset analytics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_title("Platform Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "📁",
            "Dataset Records",
            f"{len(original_df):,}"
        )

    with col2:
        metric_card(
            "📊",
            "Dataset Features",
            str(len(original_df.columns))
        )

    with col3:
        metric_card(
            "💡",
            "Health Information",
            str(len(health_df))
        )

    with col4:
        metric_card(
            "🎯",
            "Test Accuracy",
            "78.57%"
        )

    st.caption(
        "The accuracy is the previously measured test-set result. "
        "Recalculate it if the model or test split changes."
    )

    st.divider()

    section_title("Explore Platform Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.markdown(
            """
            <div class="feature-card">
                <h3>🧑‍⚕️ Health Prediction</h3>
                <p>
                    Enter available dataset features and view
                    the experimental Random Forest classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with feature_col2:

        st.markdown(
            """
            <div class="feature-card">
                <h3>💡 Health Recommendations</h3>
                <p>
                    Discover related educational health information
                    using TF-IDF and Cosine Similarity.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    feature_col3, feature_col4 = st.columns(2)

    with feature_col3:

        st.markdown(
            """
            <div class="feature-card">
                <h3>📊 Dataset Analytics</h3>
                <p>
                    Explore outcome distribution, dataset records,
                    features and summary statistics.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with feature_col4:

        st.markdown(
            """
            <div class="feature-card">
                <h3>ℹ️ Project Information</h3>
                <p>
                    Understand the technologies, algorithms
                    and modules used in this academic project.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    section_title("Dataset Outcome Distribution")

    outcome_counts = normalize_outcome_counts(original_df)

    chart_col, count_col = st.columns([2, 1])

    with chart_col:

        st.bar_chart(
            outcome_counts,
            color="#0A8992"
        )

    with count_col:

        st.markdown(
            """
            <div class="feature-card">
                <h3>Outcome Summary</h3>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Positive Records",
            int(outcome_counts["Positive"])
        )

        st.metric(
            "Negative Records",
            int(outcome_counts["Negative"])
        )

        st.metric(
            "Total Records",
            int(outcome_counts.sum())
        )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 10. HEALTH PREDICTION PAGE
# =========================================================

elif page == "🧑‍⚕️ Health Prediction":

    section_title("🧑‍⚕️ Experimental Health Classification")

    st.write(
        "Enter the available dataset features below to view "
        "the Random Forest model's experimental output."
    )

    with st.form("prediction_form"):

        st.markdown("### Personal Information")

        col1, col2 = st.columns(2)

        with col1:

            age_min = int(original_df["age"].min())
            age_max = int(original_df["age"].max())

            age = st.number_input(
                "Age",
                min_value=age_min,
                max_value=age_max,
                value=min(max(25, age_min), age_max)
            )

            gender_options = sorted(
                original_df["gender"]
                .dropna()
                .unique()
                .tolist()
            )

            gender = st.selectbox(
                "Gender",
                gender_options
            )

        with col2:

            fever = st.selectbox(
                "Fever",
                sorted(
                    original_df["fever"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            cough = st.selectbox(
                "Cough",
                sorted(
                    original_df["cough"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        st.markdown("### Additional Dataset Features")

        col3, col4 = st.columns(2)

        with col3:

            fatigue = st.selectbox(
                "Fatigue",
                sorted(
                    original_df["fatigue"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            difficulty_breathing = st.selectbox(
                "Difficulty Breathing",
                sorted(
                    original_df["difficulty_breathing"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        with col4:

            blood_pressure = st.selectbox(
                "Blood Pressure Dataset Code",
                sorted(
                    original_df["blood_pressure"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            cholesterol_level = st.selectbox(
                "Cholesterol Dataset Code",
                sorted(
                    original_df["cholesterol_level"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        st.caption(
            "Blood pressure and cholesterol selections are "
            "dataset codes, not actual clinical measurements."
        )

        submitted = st.form_submit_button(
            "🔍 Run Experimental Prediction",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame({
            "fever": [fever],
            "cough": [cough],
            "fatigue": [fatigue],
            "difficulty_breathing": [difficulty_breathing],
            "age": [age],
            "gender": [gender],
            "blood_pressure": [blood_pressure],
            "cholesterol_level": [cholesterol_level]
        })

        try:

            prediction = model.predict(input_data)

            st.divider()

            section_title("Model Output")

            st.success(
                f"Experimental classification: {prediction[0]}"
            )

            st.info(
                "This is a classification result generated from "
                "the project dataset. It cannot diagnose a disease "
                "or confirm a person's health status."
            )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# =========================================================
# 11. HEALTH INFORMATION RECOMMENDATION PAGE
# =========================================================

elif page == "💡 Recommendations":

    section_title("💡 Personalized Health Information")

    st.write(
        "Enter a health topic or keywords to find related "
        "educational information from the project database."
    )

    with st.container(border=True):

        user_input = st.text_input(
            "Search Health Topics",
            placeholder="Example: fever, cough, fatigue, hydration"
        )

        find_button = st.button(
            "✨ Find Health Information",
            use_container_width=True
        )

    if find_button:

        if user_input.strip():

            try:

                user_vector = vectorizer.transform(
                    [user_input]
                )

                health_vectors = vectorizer.transform(
                    health_df["combined_text"].fillna("")
                )

                scores = cosine_similarity(
                    user_vector,
                    health_vectors
                ).flatten()

                indices = scores.argsort()[::-1]

                matching_indices = [
                    index
                    for index in indices
                    if scores[index] > 0
                ][:3]

                if matching_indices:

                    st.success(
                        "Related educational information found!"
                    )

                    for rank, index in enumerate(
                        matching_indices,
                        start=1
                    ):

                        title = health_df.iloc[index]["title"]

                        description = (
                            health_df.iloc[index]["description"]
                        )

                        score = scores[index]

                        st.markdown(
                            f"""
                            <div class="recommendation-card">
                                <h3>
                                    {rank}. {title}
                                </h3>
                                <p>{description}</p>
                                <p style="color:#087F8C;
                                          font-size:12px;
                                          font-weight:600;">
                                    Similarity Score: {score:.3f}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    st.info(
                        "No matching educational information found. "
                        "Try another keyword such as fever, cough "
                        "or hydration."
                    )

            except Exception as error:

                st.error(
                    f"Recommendation error: {error}"
                )

        else:

            st.warning(
                "Please enter a topic before searching."
            )

    st.divider()

    st.caption(
        "Recommendations are based on text similarity. "
        "They are not medical advice or treatment instructions."
    )


# =========================================================
# 12. DATASET ANALYTICS PAGE
# =========================================================

elif page == "📊 Analytics":

    section_title("📊 Dataset Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "📁",
            "Total Records",
            str(len(original_df))
        )

    with col2:
        metric_card(
            "📋",
            "Total Features",
            str(len(original_df.columns))
        )

    with col3:
        metric_card(
            "🔎",
            "Missing Values",
            str(int(original_df.isnull().sum().sum()))
        )

    st.divider()

    section_title("Outcome Distribution")

    outcome_counts = normalize_outcome_counts(original_df)

    left, right = st.columns([2, 1])

    with left:

        st.bar_chart(
            outcome_counts,
            color="#0A8992"
        )

    with right:

        st.metric(
            "Positive",
            int(outcome_counts["Positive"])
        )

        st.metric(
            "Negative",
            int(outcome_counts["Negative"])
        )

        st.metric(
            "Total",
            int(outcome_counts.sum())
        )

    st.divider()

    section_title("Dataset Preview")

    st.dataframe(
        original_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    section_title("Statistical Summary")

    st.dataframe(
        original_df.describe(include="all").transpose(),
        use_container_width=True
    )


# =========================================================
# 13. ABOUT PROJECT PAGE
# =========================================================

elif page == "ℹ️ About Project":

    section_title("ℹ️ About the Project")

    st.markdown(
        """
        <div class="hero-card">
            <h2>Personalized Healthcare Recommendation System</h2>
            <p>
                An academic prototype combining machine learning
                classification and content-based educational
                health-information retrieval.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_title("Project Objective")

    st.write(
        "The objective of this project is to demonstrate how "
        "machine learning and text similarity techniques can "
        "be used for experimental healthcare data analysis "
        "and educational information retrieval."
    )

    section_title("Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas and NumPy",
            "Scikit-learn",
            "Random Forest",
            "TF-IDF",
            "Cosine Similarity",
            "Streamlit",
            "Jupyter Notebook"
        ],
        "Purpose": [
            "Programming language",
            "Data processing",
            "Machine learning library",
            "Experimental classification",
            "Text feature extraction",
            "Text similarity measurement",
            "Interactive dashboard",
            "Development environment"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    section_title("Project Modules")

    modules = [
        "Dataset Loading and Preprocessing",
        "Exploratory Data Analysis",
        "Random Forest Classification",
        "TF-IDF Feature Extraction",
        "Cosine Similarity Recommendation",
        "Streamlit Dashboard"
    ]

    for number, module in enumerate(modules, start=1):

        st.markdown(
            f"**{number}.** {module}"
        )

    st.divider()

    section_title("Important Disclaimer")

    st.warning(
        "This is an academic demonstration prototype. "
        "It is not a clinically validated decision-support "
        "system. It does not provide diagnoses, medicine "
        "recommendations or dosage instructions. Please consult "
        "a qualified healthcare professional for medical care."
    )


# =========================================================
# 14. COMMON FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🏥 <b>Personalized Healthcare Recommendation System</b>
        <br>
        Academic Project Prototype | 2026
        <br><br>
        Designed with Python, Machine Learning and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
