import streamlit as st
import pandas as pd
import plotly.express as px
from src.engine import (
    TECH_SKILLS, SOFT_SKILLS, ALL_SKILLS, CAREERS, load_dataset,
    train_models, weighted_match, build_roadmap
)

st.set_page_config(page_title="AI Career & Skill Development", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp {background: #f7f9fc;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: white; padding: 14px; border-radius: 14px; border: 1px solid #e7ebf2;}
div.stButton > button, div.stDownloadButton > button {border-radius: 10px; font-weight: 600;}
h1, h2, h3 {color: #172554;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_dataset()

@st.cache_resource
def get_models():
    return train_models(get_data())

df = get_data()
models, metrics, X_test, y_test, predictions = get_models()

if "profile" not in st.session_state:
    st.session_state.profile = {s: 2 for s in ALL_SKILLS}
if "student_name" not in st.session_state:
    st.session_state.student_name = ""

pages = [
    "🏠 Dashboard", "🧑‍🎓 Student Assessment", "🎯 Career Recommendation",
    "📊 Skill Gap Analysis", "🤖 ML Model Comparison",
    "📚 Learning Hub", "🗺️ Personalized Roadmap", "📄 Download Report"
]
page = st.sidebar.radio("NAVIGATION", pages)
st.sidebar.markdown("---")
st.sidebar.caption("Technical skills + soft skills + ML-based career guidance")
st.sidebar.info("Educational prototype. The included training dataset is synthetic/demo data, not real employer or student records.")

def profile_match():
    return weighted_match(st.session_state.profile)

def section_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)

if page == "🏠 Dashboard":
    section_header("🎓 AI Career Intelligence", "Understand your strengths, career fit and the skills you can develop next.")
    matches = profile_match()
    c1, c2, c3 = st.columns(3)
    c1.metric("Skills assessed", len(ALL_SKILLS))
    c2.metric("Technical skills", len(TECH_SKILLS))
    c3.metric("Soft skills", len(SOFT_SKILLS))
    st.subheader("Your skill profile")
    chart_data = pd.DataFrame({"Skill": ALL_SKILLS, "Level": [st.session_state.profile[s] for s in ALL_SKILLS],
                               "Category": ["Technical"]*len(TECH_SKILLS)+["Soft"]*len(SOFT_SKILLS)})
    st.plotly_chart(px.bar(chart_data, x="Skill", y="Level", color="Category", range_y=[0,5],
                           title="Self-assessed skill levels (1 = Beginner, 5 = Strong)"), use_container_width=True)
    st.subheader("Top career matches")
    st.dataframe(pd.DataFrame([{"Career": m["Career"], "Profile match": f'{m["Match score"]}%', "Description": m["Description"]} for m in matches[:4]]),
                 use_container_width=True, hide_index=True)
    st.caption("Profile match is a weighted comparison against illustrative role requirements. It is not a probability of employment.")

elif page == "🧑‍🎓 Student Assessment":
    section_header("🧑‍🎓 Student Skill Assessment", "Rate both technical and professional skills honestly. These ratings are self-assessments, not certificates.")
    with st.form("student_profile"):
        name = st.text_input("Student name", value=st.session_state.student_name)
        st.markdown("### Technical skills")
        tech_cols = st.columns(2)
        for i, skill in enumerate(TECH_SKILLS):
            with tech_cols[i % 2]:
                st.session_state.profile[skill] = st.slider(skill, 1, 5, int(st.session_state.profile[skill]), key=f"tech_{skill}")
        st.markdown("### Soft skills / professional skills")
        st.caption("Soft skills matter in interviews, teamwork, presentations and workplace communication.")
        soft_cols = st.columns(2)
        for i, skill in enumerate(SOFT_SKILLS):
            with soft_cols[i % 2]:
                st.session_state.profile[skill] = st.slider(skill, 1, 5, int(st.session_state.profile[skill]), key=f"soft_{skill}")
        submitted = st.form_submit_button("Save my assessment", type="primary")
        if submitted:
            st.session_state.student_name = name
            st.success("Your assessment has been saved for this browser session.")

elif page == "🎯 Career Recommendation":
    section_header("🎯 Career Recommendation", "Four trained classification models estimate a career category from your skill profile.")
    vector = pd.DataFrame([[st.session_state.profile[s] for s in ALL_SKILLS]], columns=ALL_SKILLS)
    rows = []
    for name, model in models.items():
        pred = model.predict(vector)[0]
        rows.append({"Algorithm": name, "Predicted career": pred})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Recommended career paths (weighted skill fit)")
    for item in profile_match()[:5]:
        with st.container(border=True):
            st.subheader(item["Career"])
            st.progress(item["Match score"]/100, text=f'Profile match: {item["Match score"]}%')
            st.write(item["Description"])
            gaps = item["Gaps"][:5]
            st.write("**Priority skills to improve:** " + (", ".join(g["skill"] for g in gaps) if gaps else "No major gaps in this profile."))
    st.caption("Model predictions are trained on the included synthetic demonstration dataset. Validate with real, representative data before making research or hiring claims.")

elif page == "📊 Skill Gap Analysis":
    section_header("📊 Technical + Soft Skill Gap Analysis", "Compare your current ratings with the selected career's illustrative skill requirements.")
    career = st.selectbox("Choose target career", list(CAREERS.keys()))
    req = CAREERS[career]["requirements"]
    rows = []
    for s in ALL_SKILLS:
        current, required = st.session_state.profile[s], req[s]
        rows.append({"Skill": s, "Category": "Technical" if s in TECH_SKILLS else "Soft",
                     "Current": current, "Required": required, "Gap": max(0, required-current)})
    gap_df = pd.DataFrame(rows)
    st.dataframe(gap_df, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(gap_df, x="Skill", y=["Current","Required"], barmode="group", color_discrete_sequence=px.colors.qualitative.Set2,
                           title=f"Current vs required skills — {career}"), use_container_width=True)
    st.metric("Total skill-level gap", int(gap_df["Gap"].sum()))

elif page == "🤖 ML Model Comparison":
    section_header("🤖 Machine Learning Model Comparison", "Compare four classification algorithms using a held-out test split.")
    st.warning("The included CSV is synthetic demonstration data generated from illustrative role profiles. Report these metrics as demo evaluation only, not real-world accuracy.")
    st.dataframe(metrics.rename(columns={"Accuracy":"Accuracy","Precision (macro)":"Precision","Recall (macro)":"Recall","F1-score (macro)":"F1-score"}),
                 use_container_width=True, hide_index=True)
    metric_choice = st.selectbox("Metric to visualize", ["Accuracy","Precision (macro)","Recall (macro)","F1-score (macro)"])
    st.plotly_chart(px.bar(metrics, x="Algorithm", y=metric_choice, range_y=[0,1], title=f"Algorithm comparison — {metric_choice}"),
                    use_container_width=True)
    chosen = st.selectbox("Confusion matrix for", list(models.keys()))
    labels = list(CAREERS.keys())
    cm = pd.DataFrame(
        __import__("sklearn.metrics").metrics.confusion_matrix(y_test, predictions[chosen], labels=labels),
        index=labels, columns=labels
    )
    st.write("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(cm, use_container_width=True)
    st.caption("Models: Random Forest, Decision Tree, K-Nearest Neighbors (KNN), Logistic Regression.")

elif page == "📚 Learning Hub":
    section_header("📚 Soft Skills & Technical Learning Hub", "Choose a skill, understand why it matters and complete a practical activity.")
    category = st.radio("Skill category", ["Soft skills", "Technical skills"], horizontal=True)
    selected_skills = SOFT_SKILLS if category == "Soft skills" else TECH_SKILLS
    selected = st.selectbox("Choose a skill to learn", selected_skills)
    from src.engine import learning_resources
    suggestion, url = learning_resources(selected)
    st.subheader(selected)
    st.write(suggestion)
    if selected in SOFT_SKILLS:
        activities = {
            "Communication": ["Explain a technical topic in 2 minutes using simple language.", "Write a clear project update in 4 sentences.", "Practise listening and summarising another person's point."],
            "Teamwork": ["Plan a mini-project with roles and deadlines.", "Practise giving respectful feedback.", "Write how you handled a disagreement constructively."],
            "Problem Solving": ["Define one problem clearly.", "List at least three possible solutions.", "Test one solution and record what happened."],
            "Critical Thinking": ["Separate facts from assumptions in a short article.", "Find two sources that support or challenge a claim.", "Explain why you selected one solution."],
            "Adaptability": ["Learn one unfamiliar tool feature.", "Write what changed and how you adjusted.", "Identify one lesson you can reuse."],
            "Time Management": ["List today's tasks by priority.", "Break one large task into smaller steps.", "Review what was completed and what needs rescheduling."],
            "Leadership": ["Set a small team goal.", "Assign responsibilities fairly.", "Review progress and ask teammates what support they need."],
            "Creativity": ["Generate three ideas for one problem.", "Combine two ideas into a new approach.", "Create a small sketch or prototype."]
        }
        st.markdown("### Practice activities")
        for activity in activities.get(selected, []):
            st.checkbox(activity, key=f"activity_{selected}_{activity}")
    else:
        st.info("Create a small practical project to demonstrate this technical skill.")
        st.write("Suggested practice: " + suggestion)
    st.markdown(f"**Free learning resource:** [{url}]({url})")
    st.caption("Learning links are suggestions; check the provider's current course content and availability.")

elif page == "🗺️ Personalized Roadmap":
    section_header("🗺️ Personalized Skill Development Roadmap", "Your learning priorities include both technical and soft skills.")
    matches = profile_match()
    default_career = matches[0]["Career"]
    career = st.selectbox("Target career", list(CAREERS.keys()), index=list(CAREERS.keys()).index(default_career))
    roadmap = build_roadmap(st.session_state.profile, career)
    if roadmap.empty:
        st.success("Your current ratings meet all illustrative requirements for this career.")
    else:
        st.dataframe(roadmap, use_container_width=True, hide_index=True)
        for _, item in roadmap.iterrows():
            with st.container(border=True):
                st.markdown(f"**{item['Type']} — {item['Skill']}**")
                st.write(f"Current: {item['Current']}/5 | Target: {item['Target']}/5 | Gap: {item['Gap']}")
                st.write(item["Practice activity"])
                st.markdown(f"[Open learning resource]({item['Learning resource']})")

elif page == "📄 Download Report":
    section_header("📄 Student Career & Skill Development Report", "Download a report of the current self-assessment and model recommendation.")
    matches = profile_match()
    career = st.selectbox("Report target career", list(CAREERS.keys()), index=list(CAREERS.keys()).index(matches[0]["Career"]))
    gaps = build_roadmap(st.session_state.profile, career)
    vector = pd.DataFrame([[st.session_state.profile[s] for s in ALL_SKILLS]], columns=ALL_SKILLS)
    predicted = {name: model.predict(vector)[0] for name, model in models.items()}
    report = [
        "AI CAREER INTELLIGENCE & SKILL DEVELOPMENT PLATFORM",
        f"Student: {st.session_state.student_name or 'Not provided'}",
        f"Target career: {career}",
        f"Weighted profile match: {next(x['Match score'] for x in matches if x['Career']==career)}%",
        "", "MODEL PREDICTIONS:"
    ] + [f"{k}: {v}" for k,v in predicted.items()] + ["", "SKILL RATINGS (1-5):"] + [
        f"{s}: {st.session_state.profile[s]}/5 ({'Technical' if s in TECH_SKILLS else 'Soft'})" for s in ALL_SKILLS
    ] + ["", "LEARNING PRIORITIES:"] + [
        f"{r['Type']} | {r['Skill']} | current {r['Current']} | target {r['Target']} | gap {r['Gap']}"
        for _, r in gaps.iterrows()
    ] + ["", "DATA NOTE: ML models were trained on synthetic demonstration data. This is an educational prototype, not a hiring decision or employment guarantee."]
    st.download_button("Download report (TXT)", "\n".join(report), file_name="student_career_skill_report.txt", mime="text/plain", type="primary")
    st.caption("The report is generated from the current browser session. Save or download it before closing the app.")

st.sidebar.markdown("---")
st.sidebar.caption("Python • Streamlit • scikit-learn • Plotly")
