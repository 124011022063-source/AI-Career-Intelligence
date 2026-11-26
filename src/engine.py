from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TECH_SKILLS = [
    "Python", "SQL", "Machine Learning", "Statistics", "Data Visualization",
    "Java", "DSA", "Web Development", "Cybersecurity", "Cloud & Networking"
]
SOFT_SKILLS = [
    "Communication", "Teamwork", "Problem Solving", "Critical Thinking",
    "Adaptability", "Time Management", "Leadership", "Creativity"
]
ALL_SKILLS = TECH_SKILLS + SOFT_SKILLS

CAREERS = {
    "Data Analyst": {
        "description": "Turns data into reports, dashboards and useful business insights.",
        "requirements": {"Python":3,"SQL":5,"Machine Learning":2,"Statistics":4,"Data Visualization":5,"Java":1,"DSA":1,"Web Development":1,"Cybersecurity":1,"Cloud & Networking":2,"Communication":5,"Teamwork":4,"Problem Solving":4,"Critical Thinking":5,"Adaptability":3,"Time Management":4,"Leadership":2,"Creativity":3}
    },
    "Data Scientist": {
        "description": "Uses programming, statistics and machine learning to solve data problems.",
        "requirements": {"Python":5,"SQL":4,"Machine Learning":5,"Statistics":5,"Data Visualization":3,"Java":1,"DSA":2,"Web Development":1,"Cybersecurity":1,"Cloud & Networking":2,"Communication":4,"Teamwork":4,"Problem Solving":5,"Critical Thinking":5,"Adaptability":4,"Time Management":3,"Leadership":2,"Creativity":4}
    },
    "Machine Learning Engineer": {
        "description": "Builds, tests and deploys machine-learning systems.",
        "requirements": {"Python":5,"SQL":3,"Machine Learning":5,"Statistics":4,"Data Visualization":2,"Java":2,"DSA":4,"Web Development":2,"Cybersecurity":2,"Cloud & Networking":4,"Communication":3,"Teamwork":4,"Problem Solving":5,"Critical Thinking":5,"Adaptability":4,"Time Management":4,"Leadership":2,"Creativity":3}
    },
    "Software Developer": {
        "description": "Designs, develops, tests and maintains software applications.",
        "requirements": {"Python":3,"SQL":3,"Machine Learning":1,"Statistics":1,"Data Visualization":1,"Java":5,"DSA":5,"Web Development":4,"Cybersecurity":2,"Cloud & Networking":3,"Communication":3,"Teamwork":5,"Problem Solving":5,"Critical Thinking":4,"Adaptability":4,"Time Management":4,"Leadership":2,"Creativity":4}
    },
    "Cybersecurity Analyst": {
        "description": "Helps protect systems by monitoring risks and supporting defensive security.",
        "requirements": {"Python":3,"SQL":2,"Machine Learning":1,"Statistics":2,"Data Visualization":2,"Java":1,"DSA":2,"Web Development":2,"Cybersecurity":5,"Cloud & Networking":4,"Communication":4,"Teamwork":4,"Problem Solving":5,"Critical Thinking":5,"Adaptability":4,"Time Management":4,"Leadership":2,"Creativity":3}
    },
    "Cloud / Network Engineer": {
        "description": "Builds and maintains cloud services, infrastructure and networks.",
        "requirements": {"Python":3,"SQL":1,"Machine Learning":1,"Statistics":1,"Data Visualization":1,"Java":2,"DSA":2,"Web Development":2,"Cybersecurity":3,"Cloud & Networking":5,"Communication":3,"Teamwork":4,"Problem Solving":5,"Critical Thinking":4,"Adaptability":5,"Time Management":4,"Leadership":3,"Creativity":3}
    },
    "Frontend Developer": {
        "description": "Creates accessible, interactive user interfaces for websites and apps.",
        "requirements": {"Python":1,"SQL":1,"Machine Learning":1,"Statistics":1,"Data Visualization":2,"Java":3,"DSA":3,"Web Development":5,"Cybersecurity":2,"Cloud & Networking":2,"Communication":4,"Teamwork":4,"Problem Solving":4,"Critical Thinking":3,"Adaptability":4,"Time Management":4,"Leadership":2,"Creativity":5}
    }
}

def load_dataset(path=None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "demo_career_dataset.csv"
    df = pd.read_csv(path)
    missing = set(ALL_SKILLS + ["Career"]) - set(df.columns)
    if missing:
        raise ValueError("Dataset missing columns: " + ", ".join(sorted(missing)))
    return df

def train_models(df):
    X = df[ALL_SKILLS].astype(float)
    y = df["Career"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced")
    }
    trained, rows, predictions = {}, [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        trained[name] = model
        predictions[name] = pred
        rows.append({
            "Algorithm": name,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision (macro)": round(precision_score(y_test, pred, average="macro", zero_division=0), 4),
            "Recall (macro)": round(recall_score(y_test, pred, average="macro", zero_division=0), 4),
            "F1-score (macro)": round(f1_score(y_test, pred, average="macro", zero_division=0), 4)
        })
    metrics = pd.DataFrame(rows).sort_values("F1-score (macro)", ascending=False)
    return trained, metrics, X_test, y_test, predictions

def weighted_match(profile):
    results = []
    for career, info in CAREERS.items():
        req = info["requirements"]
        total_weight = sum(req.values())
        score = sum(min(float(profile.get(s, 1)), req[s]) / 5 * req[s] for s in ALL_SKILLS) / total_weight * 100
        gaps = sorted(
            [{"skill": s, "current": int(profile.get(s, 1)), "required": req[s],
              "gap": max(0, req[s] - int(profile.get(s, 1))), "type": "Technical" if s in TECH_SKILLS else "Soft"}
             for s in ALL_SKILLS if int(profile.get(s, 1)) < req[s]],
            key=lambda x: x["gap"], reverse=True
        )
        results.append({"Career": career, "Match score": round(score, 1), "Description": info["description"], "Gaps": gaps})
    return sorted(results, key=lambda x: x["Match score"], reverse=True)

def learning_resources(skill):
    resources = {
        "Python": ("Python basics, functions, collections and mini automation project", "https://docs.python.org/3/tutorial/"),
        "SQL": ("SELECT, JOIN, GROUP BY and query practice", "https://sqlbolt.com/"),
        "Machine Learning": ("Supervised learning, train/test split and model evaluation", "https://scikit-learn.org/stable/tutorial/index.html"),
        "Statistics": ("Mean, median, probability and distributions", "https://www.khanacademy.org/math/statistics-probability"),
        "Data Visualization": ("Create charts and explain insights using dashboards", "https://plotly.com/python/"),
        "Java": ("Java syntax, OOP and collections", "https://dev.java/learn/"),
        "DSA": ("Arrays, strings, stacks, queues and trees", "https://www.geeksforgeeks.org/data-structures/"),
        "Web Development": ("HTML, CSS, JavaScript and responsive pages", "https://developer.mozilla.org/en-US/docs/Learn"),
        "Cybersecurity": ("Networking basics and defensive security principles", "https://www.cisa.gov/secure-our-world"),
        "Cloud & Networking": ("Networking and cloud fundamentals", "https://learn.microsoft.com/en-us/training/azure/"),
        "Communication": ("Practise a 2-minute explanation of a topic; structure it as point, reason, example", "https://www.toastmasters.org/resources/public-speaking-tips"),
        "Teamwork": ("Complete a small group task; agree on roles, deadlines and a shared outcome", "https://www.atlassian.com/team-playbook"),
        "Problem Solving": ("Use: define the problem, list options, test one solution, review the result", "https://www.mindtools.com/"),
        "Critical Thinking": ("Check evidence, identify assumptions and compare alternative explanations", "https://www.skillsyouneed.com/learn/critical-thinking.html"),
        "Adaptability": ("Reflect on a new tool or change; write what you learned and how you adjusted", "https://www.skillsyouneed.com/ps/adaptability.html"),
        "Time Management": ("Break a task into 25-minute focus blocks and review priorities", "https://todoist.com/productivity-methods/pomodoro-technique"),
        "Leadership": ("Practise setting a clear goal, sharing responsibilities and checking progress", "https://www.atlassian.com/team-playbook"),
        "Creativity": ("Generate three different approaches before choosing one and prototype a small idea", "https://www.ideou.com/blogs/inspiration/creative-problem-solving")
    }
    return resources.get(skill, ("Practise this skill with a small project.", "https://www.coursera.org/"))

def build_roadmap(profile, career):
    req = CAREERS[career]["requirements"]
    items = []
    for skill in ALL_SKILLS:
        current, required = int(profile.get(skill, 1)), req[skill]
        if current < required:
            suggestion, url = learning_resources(skill)
            items.append({"Skill": skill, "Type": "Technical" if skill in TECH_SKILLS else "Soft",
                          "Current": current, "Target": required, "Gap": required-current,
                          "Practice activity": suggestion, "Learning resource": url})
    return pd.DataFrame(items).sort_values(["Gap", "Type"], ascending=[False, True]) if items else pd.DataFrame(
        columns=["Skill","Type","Current","Target","Gap","Practice activity","Learning resource"]
    )
