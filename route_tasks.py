import re
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Pre-compiled regex pattern for speed optimization
TOKEN_PATTERN = re.compile(r'\b[a-z0-9]{3,}\b')
SENTENCE_SPLIT_PATTERN = re.compile(r'[.\n!?;,]')

@dataclass
class AdvancedSubTask:
    id: int
    title: str
    detailed_scope: str
    category: str
    priority: str
    estimated_hours: float
    required_skills: List[str]
    dependencies: List[int]

@dataclass
class ProjectInstanceSchema:
    project_name: str
    overall_priority: str
    extracted_tasks: List[AdvancedSubTask]

@dataclass
class EnhancedEmployee:
    name: str
    department: str
    role: str
    reports_to: str
    raw_responsibilities: str
    experience_years: float
    technical_skills: List[str]
    kpis: List[str]
    kpi_tokens: Set[str]
    estimated_capacity_hours: float = 40.0
    current_workload_hours: float = 0.0

    @property
    def semantic_profile(self) -> str:
        skills_chunk = " ".join(self.technical_skills)
        kpi_chunk = " ".join(self.kpis)
        return f"{self.department} {self.role} {self.raw_responsibilities} {skills_chunk} {kpi_chunk}"

def clean_and_tokenize(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()
    return set(TOKEN_PATTERN.findall(text.lower()))

class CognitiveUnderstandingEngine:
    """Parses raw text input briefs into classified contextual sub-tasks using mapped rule matrixing."""
    
    # Declarative rule matrix to easily add/modify categories without shifting core code logic
    RULE_MATRIX = {
        "Content": (["write", "copy", "script", "content", "create", "journey"], ["Copywriting", "Content Strategy"], 12.0),
        "Design": (["design", "graphic", "logo", "visual", "branding", "ui", "figma"], ["Figma", "UI Design", "Branding"], 14.0),
        "Ads": (["meta", "ad", "budget", "campaign", "traffic", "pixel", "marketing", "buyer", "media"], ["Paid Traffic", "Meta Ads", "Optimization"], 10.0),
        "Coordination": (["department", "maintain", "coordinate", "reporting", "execution", "regular", "communication", "oversee"], ["Management", "Operations", "Communications"], 8.0),
        "Backend": (["backend", "sql", "analytics", "database", "hooks", "program", "app", "integrate"], ["Python", "SQL", "Analytics", "Backend"], 16.0)
    }

    def decompose_brief(self, brief: str) -> ProjectInstanceSchema:
        raw_statements = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(brief) if len(s.strip()) > 12]
        if not raw_statements:
            raw_statements = [brief.strip()]

        extracted_tasks: List[AdvancedSubTask] = []
        
        for idx, statement in enumerate(raw_statements, start=1):
            stmt_low = statement.lower()
            category, req_skills, base_hours = "General", ["Operations", "Coordination"], 8.0
            
            # Linear lookup optimization over declared rule configurations
            for cat_name, (keywords, skills, hours) in self.RULE_MATRIX.items():
                if any(kw in stmt_low for kw in keywords):
                    category, req_skills, base_hours = cat_name, skills, hours
                    break

            priority = "High" if any(w in stmt_low for w in ["critical", "urgent", "asap", "immediately", "improve", "maintain"]) else "Medium"
            words = statement.split()
            title_text = " ".join(words[:4]) + "..." if len(words) > 4 else statement

            dependencies = [idx - 1] if (idx > 1 and category not in ["General", "Coordination"]) else []

            if priority == "High": 
                base_hours = round(base_hours * 1.15, 1)
            if dependencies: 
                base_hours = round(base_hours * 1.10, 1)

            extracted_tasks.append(
                AdvancedSubTask(
                    id=idx, title=title_text, detailed_scope=statement,
                    category=category, priority=priority, estimated_hours=base_hours,
                    required_skills=req_skills, dependencies=dependencies
                )
            )

        return ProjectInstanceSchema(
            project_name="AIVORA Global Core Operational Pipeline",
            overall_priority="High" if any(t.priority == "High" for t in extracted_tasks) else "Medium",
            extracted_tasks=extracted_tasks
        )


class IntelligenceDrivenRouter:
    """Manages machine learning model training, routing, and gap analytics."""
    def __init__(self):
        self.employees: List[EnhancedEmployee] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=1, sublinear_tf=True)
        self.model = LogisticRegression(max_iter=5000, class_weight='balanced', C=4.0)
        self.raw_gaps: List[Dict] = []

    def ingest_local_file(self, file_path: str) -> None:
        with pd.ExcelFile(file_path) as xls:
            df_emp = pd.read_excel(xls, sheet_name="Employees")
            df_kpi = pd.read_excel(xls, sheet_name="KPIs")
            df_tasks = pd.read_excel(xls, sheet_name="Task_Routing")
            
            if "Workflow_Gaps" in xls.sheet_names:
                self.raw_gaps = pd.read_excel(xls, sheet_name="Workflow_Gaps").to_dict(orient="records")

        kpi_dict = df_kpi.groupby("Role")["KPI"].apply(list).to_dict()

        for _, row in df_emp.iterrows():
            role_title = str(row["Role"]).strip()
            raw_resp = str(row["Responsibilities"])
            emp_kpis = kpi_dict.get(role_title, ["General Execution Tracking"])
            
            self.employees.append(EnhancedEmployee(
                name=str(row["Employee"]).strip(), 
                department=str(row["Department"]).strip(), 
                role=role_title,
                reports_to=str(row["Reports To"]).strip(), 
                raw_responsibilities=raw_resp, 
                kpis=emp_kpis,
                technical_skills=list(clean_and_tokenize(raw_resp))[:5], 
                kpi_tokens=clean_and_tokenize(" ".join(emp_kpis)),
                experience_years=5.0 if any(w in role_title.lower() for w in ["senior", "specialist", "manager", "coordinator", "ceo"]) else 3.0
            ))

        # Train data augmentation logic
        X_raw = (df_tasks["Task"].astype(str) + " " + df_tasks["Category"].astype(str)).tolist()
        y_raw = df_tasks["Assigned Role"].str.strip().tolist()

        X_augmented, y_augmented = list(X_raw), list(y_raw)
        synonyms = {"write": ["generate", "draft"], "design": ["render", "mockup"], "coordinate": ["align", "sync"]}
        
        for txt, lbl in zip(X_raw, y_raw):
            txt_low = txt.lower()
            for key, reps in synonyms.items():
                if key in txt_low:
                    for rep in reps:
                        augmented_text = re.sub(r'\b' + key + r'\b', rep, txt, flags=re.IGNORECASE)
                        X_augmented.append(augmented_text)
                        y_augmented.append(lbl)

        X_vec = self.vectorizer.fit_transform(X_augmented)
        self.model.fit(X_vec, y_augmented)

    def evaluate_and_route(self, project: ProjectInstanceSchema) -> List[Dict]:
        classes_list = list(self.model.classes_)
        emp_vectors = self.vectorizer.transform([emp.semantic_profile for emp in self.employees])
        routed_manifest = []

        for task in project.extracted_tasks:
            query_text = f"{task.title} {task.detailed_scope} {task.category}"
            task_vec = self.vectorizer.transform([query_text])
            ml_probs = self.model.predict_proba(task_vec)[0]
            cos_sims = cosine_similarity(task_vec, emp_vectors)[0]

            candidate_rankings = []
            query_tokens = clean_and_tokenize(query_text)

            for e_idx, emp in enumerate(self.employees):
                skill_fit = float(cos_sims[e_idx])
                
                # Defensively avoid ValueError index lookups for untrained roles
                ml_fit = float(ml_probs[classes_list.index(emp.role)]) if emp.role in classes_list else 0.0
                kpi_fit = 1.0 if query_tokens.intersection(emp.kpi_tokens) else 0.0

                role_low, cat_low = emp.role.lower(), task.category.lower()
                structural_boost = 0.0
                if (cat_low == "coordination" and "coordinator" in role_low) or \
                   (cat_low == "ads" and "buyer" in role_low) or \
                   (cat_low == "content" and "creator" in role_low) or \
                   (cat_low == "design" and "designer" in role_low):
                    structural_boost = 0.85

                if max(ml_probs) < 0.40 or emp.role not in classes_list:
                    suitability = (0.50 * structural_boost) + (0.20 * skill_fit) + (0.10 * kpi_fit) + (0.20 * (min(1.0, emp.experience_years / 10.0)))
                else:
                    suitability = (0.35 * ml_fit) + (0.35 * structural_boost) + (0.15 * skill_fit) + (0.15 * kpi_fit)

                # Penalize structural overload conditions
                if (emp.current_workload_hours + task.estimated_hours) > emp.estimated_capacity_hours:
                    suitability *= 0.20

                candidate_rankings.append((emp, suitability))

            candidate_rankings.sort(key=lambda x: x[1], reverse=True)
            primary_choice, primary_score = candidate_rankings[0]
            primary_choice.current_workload_hours += task.estimated_hours

            target_gate = 0.15 if (any(w in primary_choice.role.lower() for w in ["coordinator", "creator", "buyer", "specialist"]) or max(ml_probs) < 0.40) else 0.50
            gating_status = "AUTOMATED_ROUTING" if primary_score >= target_gate else "HUMAN_REVIEW_REQUIRED"

            routed_manifest.append({
                "task_id": task.id,
                "task_title": task.title,
                "category": task.category,
                "estimated_hours": task.estimated_hours,
                "assigned_employee": primary_choice.name,
                "assigned_role": primary_choice.role,
                "match_confidence": round(primary_score, 2),
                "routing_status": gating_status,
                "backups": [f"{e.name} ({e.role})" for e, _ in candidate_rankings[1:3]]
            })

        return routed_manifest

    def analyze_structural_gaps(self, brief: str) -> List[Dict]:
        detected_gaps = []
        brief_low = brief.lower()

        for gap in self.raw_gaps:
            missing_role = gap.get("Missing Role", "Unknown Role")
            resp_str = str(gap.get("Responsibilities", ""))
            tokens_to_check = [t.strip().lower() for t in resp_str.split(";") if len(t.strip()) > 2]
            
            matches = [token for token in tokens_to_check if token in brief_low]
            if matches:
                detected_gaps.append({
                    "missing_role": missing_role,
                    "risk_reason": gap.get("Reason", "Process bottlenecks"),
                    "matched_indicators": ", ".join(matches),
                    "suggested_mandate": resp_str
                })
        return detected_gaps