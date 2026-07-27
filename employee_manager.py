# ==============================================================================
# MODULE: services/employee_manager.py
# DESCRIPTION: Production-ready Enterprise Employee Management System for AIVORA.
#              Orchestrates Workforce Data, Attendance, Leave Workflows, Roles,
#              AI Burnout & Risk Predictive Models, Performance & Analytics,
#              and Cross-Service Modular Integrations.
# ==============================================================================

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

# Setup Enterprise Module Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EmployeeManagerService")


# ---------------------------------------------------------------------------
# DOMAIN ENUMS & ROLE DEFINITIONS
# ---------------------------------------------------------------------------

class RoleEnum(str, Enum):
    ADMIN = "Admin"
    CEO = "CEO"
    MANAGER = "Manager"
    SUPERVISOR = "Supervisor"
    EMPLOYEE = "Employee"
    CUSTOM = "Custom"


class LeaveType(str, Enum):
    ANNUAL = "Annual Leave"
    SICK = "Sick Leave"
    EMERGENCY = "Emergency Leave"
    MATERNITY = "Maternity Leave"


class LeaveStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class WarningType(str, Enum):
    LATE = "Late Arrival Warning"
    PERFORMANCE = "Performance Warning"
    SECURITY = "Security Violation"


class WorkLocation(str, Enum):
    OFFICE = "Office"
    REMOTE = "Remote"


# Default Permission Matrix Hierarchy
DEFAULT_PERMISSION_MATRIX: Dict[str, List[str]] = {
    RoleEnum.ADMIN.value: ["read", "write", "delete", "manage_roles", "approve_leave", "view_analytics", "manage_finance"],
    RoleEnum.CEO.value: ["read", "write", "manage_roles", "approve_leave", "view_analytics", "view_finance"],
    RoleEnum.MANAGER.value: ["read", "write", "approve_leave", "view_analytics", "assign_tasks"],
    RoleEnum.SUPERVISOR.value: ["read", "write", "approve_leave", "assign_tasks"],
    RoleEnum.EMPLOYEE.value: ["read", "clock_in_out", "request_leave"],
}


# ---------------------------------------------------------------------------
# DATACLASS DOMAIN MODELS
# ---------------------------------------------------------------------------

@dataclass
class Role:
    role_name: str
    permissions: List[str] = field(default_factory=list)


@dataclass
class Attendance:
    attendance_id: str
    emp_id: str
    date_str: str
    clock_in: str
    clock_out: Optional[str] = None
    working_hours: float = 0.0
    location: str = WorkLocation.OFFICE.value
    gps_coords: Optional[Tuple[float, float]] = None
    is_late: bool = False
    is_early_leave: bool = False


@dataclass
class LeaveRequest:
    request_id: str
    emp_id: str
    leave_type: str
    start_date: str
    end_date: str
    total_days: int
    status: str = LeaveStatus.PENDING.value
    approved_by: Optional[str] = None
    reason: str = ""


@dataclass
class Performance:
    record_id: str
    emp_id: str
    period: str
    kpis: Dict[str, float] = field(default_factory=dict)
    completed_tasks: int = 0
    quality_score: float = 100.0  # 0 to 100
    projects_completed: int = 0
    meeting_attendance_pct: float = 100.0
    ai_performance_score: float = 85.0
    productivity_trend: str = "STABLE"  # "UPWARD", "STABLE", "DECLINING"


@dataclass
class Warning:
    warning_id: str
    emp_id: str
    warning_type: str
    reason: str
    issued_date: str
    auto_generated: bool = False


@dataclass
class Reward:
    reward_id: str
    emp_id: str
    title: str
    bonus_points: int
    achievement_name: str
    issued_date: str
    certificate_url: Optional[str] = None


@dataclass
class Department:
    dept_id: str
    name: str
    manager_id: Optional[str] = None
    annual_budget: float = 0.0
    employee_ids: List[str] = field(default_factory=list)


@dataclass
class Employee:
    emp_id: str
    full_name: str
    email: str
    role: str
    department_id: str
    is_active: bool = True
    skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    emergency_contacts: List[Dict[str, str]] = field(default_factory=list)
    employment_history: List[Dict[str, str]] = field(default_factory=list)
    leave_balances: Dict[str, int] = field(default_factory=lambda: {
        LeaveType.ANNUAL.value: 21,
        LeaveType.SICK.value: 10,
        LeaveType.EMERGENCY.value: 5,
        LeaveType.MATERNITY.value: 90
    })
    bonus_points: int = 0


# ---------------------------------------------------------------------------
# CORE EMPLOYEE MANAGER SERVICE
# ---------------------------------------------------------------------------

class EmployeeManagerService:
    """
    Production Enterprise Service handling all workforce management, attendance tracking,
    leave approval workflows, performance evaluations, AI analytics, and cross-module integrations.
    """

    def __init__(self, storage_file: str = "outputs/employee_db.json"):
        self.storage_file = storage_file
        self.employees: Dict[str, Employee] = {}
        self.departments: Dict[str, Department] = {}
        self.attendance_records: List[Attendance] = []
        self.leave_requests: Dict[str, LeaveRequest] = {}
        self.performance_records: Dict[str, List[Performance]] = {}
        self.warnings: List[Warning] = []
        self.rewards: List[Reward] = []
        self.permissions: Dict[str, List[str]] = DEFAULT_PERMISSION_MATRIX.copy()

        self._ensure_storage_directory()
        self._load_database()

    # ---------------------------------------------------------------------------
    # DATABASE PERSISTENCE & INITIALIZATION
    # ---------------------------------------------------------------------------

    def _ensure_storage_directory(self) -> None:
        """Creates storage directory if absent."""
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)

    def _save_database(self) -> bool:
        """Serializes dataclass models to JSON disk file."""
        try:
            payload = {
                "employees": {eid: asdict(emp) for eid, emp in self.employees.items()},
                "departments": {did: asdict(dept) for did, dept in self.departments.items()},
                "attendance_records": [asdict(att) for att in self.attendance_records],
                "leave_requests": {lid: asdict(req) for lid, req in self.leave_requests.items()},
                "performance_records": {
                    eid: [asdict(p) for p in plist] for eid, plist in self.performance_records.items()
                },
                "warnings": [asdict(w) for w in self.warnings],
                "rewards": [asdict(r) for r in self.rewards],
            }
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info("Successfully persisted employee database to %s", self.storage_file)
            return True
        except Exception as e:
            logger.error("Database write error: %s", str(e))
            return False

    def _load_database(self) -> None:
        """Loads and inflates dataclass models from persistent JSON file or initializes defaults."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.employees = {eid: Employee(**e) for eid, e in data.get("employees", {}).items()}
                self.departments = {did: Department(**d) for did, d in data.get("departments", {}).items()}
                self.attendance_records = [Attendance(**a) for a in data.get("attendance_records", [])]
                self.leave_requests = {lid: LeaveRequest(**l) for lid, l in data.get("leave_requests", {}).items()}
                self.performance_records = {
                    eid: [Performance(**p) for p in plist]
                    for eid, plist in data.get("performance_records", {}).items()
                }
                self.warnings = [Warning(**w) for w in data.get("warnings", [])]
                self.rewards = [Reward(**r) for r in data.get("rewards", [])]
                logger.info("Employee Manager loaded %d active employees.", len(self.employees))
                return
            except Exception as e:
                logger.warning("Could not parse storage JSON (%s). Reverting to default setup.", str(e))

        self._seed_default_enterprise_data()

    def _seed_default_enterprise_data(self) -> None:
        """Populates baseline enterprise layout."""
        d1 = Department(dept_id="DEPT-ENG", name="Engineering", annual_budget=500000.0)
        d2 = Department(dept_id="DEPT-FIN", name="Finance & Analytics", annual_budget=350000.0)
        self.departments = {d1.dept_id: d1, d2.dept_id: d2}

        e1 = Employee(
            emp_id="EMP-101",
            full_name="Sarah Chen",
            email="sarah.chen@aivora.io",
            role=RoleEnum.MANAGER.value,
            department_id="DEPT-ENG",
            skills=["Python", "Streamlit", "AI System Architecture"],
            certifications=["AWS Certified Solutions Architect"]
        )
        e2 = Employee(
            emp_id="EMP-102",
            full_name="Marcus Vance",
            email="marcus.vance@aivora.io",
            role=RoleEnum.EMPLOYEE.value,
            department_id="DEPT-ENG",
            skills=["Database Security", "PostgreSQL", "Docker"],
            certifications=["Certified Information Systems Security Professional"]
        )
        self.employees = {e1.emp_id: e1, e2.emp_id: e2}
        d1.manager_id = e1.emp_id
        d1.employee_ids = [e1.emp_id, e2.emp_id]

        self._save_database()

    # ---------------------------------------------------------------------------
    # FEATURE 1: EMPLOYEE MANAGEMENT
    # ---------------------------------------------------------------------------

    def create_employee(self, full_name: str, email: str, role: str, department_id: str, skills: Optional[List[str]] = None) -> str:
        emp_id = f"EMP-{len(self.employees) + 101}"
        new_emp = Employee(
            emp_id=emp_id,
            full_name=full_name,
            email=email,
            role=role if role in RoleEnum._value2member_map_ else RoleEnum.EMPLOYEE.value,
            department_id=department_id,
            skills=skills or []
        )
        self.employees[emp_id] = new_emp

        if department_id in self.departments:
            self.departments[department_id].employee_ids.append(emp_id)

        self._save_database()
        logger.info("Created new employee record: %s (%s)", full_name, emp_id)
        return emp_id

    def update_employee(self, emp_id: str, **kwargs) -> bool:
        if emp_id not in self.employees:
            return False
        emp = self.employees[emp_id]
        for key, value in kwargs.items():
            if hasattr(emp, key):
                setattr(emp, key, value)
        self._save_database()
        return True

    def deactivate_employee(self, emp_id: str) -> bool:
        return self.update_employee(emp_id, is_active=False)

    def delete_employee(self, emp_id: str) -> bool:
        if emp_id in self.employees:
            dept_id = self.employees[emp_id].department_id
            if dept_id in self.departments and emp_id in self.departments[dept_id].employee_ids:
                self.departments[dept_id].employee_ids.remove(emp_id)
            del self.employees[emp_id]
            self._save_database()
            return True
        return False

    def search_employees(self, query: str) -> List[Employee]:
        q = query.lower()
        return [
            emp for emp in self.employees.values()
            if q in emp.full_name.lower() or q in emp.email.lower() or any(q in s.lower() for s in emp.skills)
        ]

    def get_employee_profile(self, emp_id: str) -> Optional[Dict[str, Any]]:
        if emp_id not in self.employees:
            return None
        emp = self.employees[emp_id]
        dept = self.departments.get(emp.department_id)
        return {
            "profile": asdict(emp),
            "department_name": dept.name if dept else "Unassigned",
            "active_warnings": len([w for w in self.warnings if w.emp_id == emp_id]),
            "total_rewards": len([r for r in self.rewards if r.emp_id == emp_id])
        }

    # ---------------------------------------------------------------------------
    # FEATURE 2: DEPARTMENTS & ROLES
    # ---------------------------------------------------------------------------

    def create_department(self, name: str, budget: float = 0.0) -> str:
        dept_id = f"DEPT-{name[:3].upper()}"
        dept = Department(dept_id=dept_id, name=name, annual_budget=budget)
        self.departments[dept_id] = dept
        self._save_database()
        return dept_id

    def assign_department_manager(self, dept_id: str, manager_emp_id: str) -> bool:
        if dept_id in self.departments and manager_emp_id in self.employees:
            self.departments[dept_id].manager_id = manager_emp_id
            self.employees[manager_emp_id].role = RoleEnum.MANAGER.value
            self._save_database()
            return True
        return False

    def check_permission(self, role_name: str, permission: str) -> bool:
        return permission in self.permissions.get(role_name, [])

    # ---------------------------------------------------------------------------
    # FEATURE 3: ATTENDANCE & LEAVE WORKFLOWS
    # ---------------------------------------------------------------------------

    def clock_in(self, emp_id: str, location: str = WorkLocation.OFFICE.value, gps: Optional[Tuple[float, float]] = None) -> str:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        clock_in_str = now.strftime("%H:%M:%S")

        # Late threshold: After 09:15 AM
        is_late = now.time() > time(9, 15)

        att_id = f"ATT-{len(self.attendance_records) + 1:04d}"
        att = Attendance(
            attendance_id=att_id,
            emp_id=emp_id,
            date_str=date_str,
            clock_in=clock_in_str,
            location=location,
            gps_coords=gps,
            is_late=is_late
        )
        self.attendance_records.append(att)

        if is_late:
            self.check_and_auto_generate_warnings(emp_id, "LATE_ARRIVAL")

        self._save_database()
        return att_id

    def clock_out(self, emp_id: str) -> bool:
        today_str = date.today().isoformat()
        for att in reversed(self.attendance_records):
            if att.emp_id == emp_id and att.date_str == today_str and not att.clock_out:
                now = datetime.now()
                att.clock_out = now.strftime("%H:%M:%S")

                # Parse working hours
                fmt = "%H:%M:%S"
                t_in = datetime.strptime(att.clock_in, fmt)
                t_out = datetime.strptime(att.clock_out, fmt)
                att.working_hours = round((t_out - t_in).seconds / 3600.0, 2)
                att.is_early_leave = now.time() < time(17, 0)

                self._save_database()
                return True
        return False

    def request_leave(self, emp_id: str, leave_type: str, start_date: str, end_date: str, reason: str = "") -> Optional[str]:
        if emp_id not in self.employees:
            return None

        emp = self.employees[emp_id]
        d_start = datetime.strptime(start_date, "%Y-%m-%d")
        d_end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = max(1, (d_end - d_start).days + 1)

        if emp.leave_balances.get(leave_type, 0) < total_days:
            logger.warning("Leave Request Rejected: Insufficient balance for %s", emp_id)
            return None

        req_id = f"LEV-{len(self.leave_requests) + 1:03d}"
        req = LeaveRequest(
            request_id=req_id,
            emp_id=emp_id,
            leave_type=leave_type,
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            reason=reason
        )
        self.leave_requests[req_id] = req
        self._save_database()
        return req_id

    def approve_leave(self, request_id: str, approver_emp_id: str, approve: bool = True) -> bool:
        if request_id not in self.leave_requests:
            return False

        req = self.leave_requests[request_id]
        if approve:
            req.status = LeaveStatus.APPROVED.value
            req.approved_by = approver_emp_id
            emp = self.employees.get(req.emp_id)
            if emp and req.leave_type in emp.leave_balances:
                emp.leave_balances[req.leave_type] -= req.total_days
        else:
            req.status = LeaveStatus.REJECTED.value

        self._save_database()
        return True

    # ---------------------------------------------------------------------------
    # FEATURE 4: PERFORMANCE, REWARDS & AUTOMATED WARNINGS
    # ---------------------------------------------------------------------------

    def check_and_auto_generate_warnings(self, emp_id: str, trigger_event: str) -> None:
        """Evaluates operational bounds and issues automated system warnings."""
        if trigger_event == "LATE_ARRIVAL":
            recent_lates = sum(
                1 for a in self.attendance_records[-10:] if a.emp_id == emp_id and a.is_late
            )
            if recent_lates >= 3:
                w_id = f"WRN-{len(self.warnings) + 1:03d}"
                w = Warning(
                    warning_id=w_id,
                    emp_id=emp_id,
                    warning_type=WarningType.LATE.value,
                    reason="Automated System Trigger: Exceeded 3 late arrivals within 10 shifts.",
                    issued_date=date.today().isoformat(),
                    auto_generated=True
                )
                self.warnings.append(w)
                logger.warning("Auto Warning issued to %s for repeated late arrivals.", emp_id)

    def issue_reward(self, emp_id: str, title: str, bonus_points: int, achievement_name: str) -> str:
        r_id = f"RWD-{len(self.rewards) + 1:03d}"
        reward = Reward(
            reward_id=r_id,
            emp_id=emp_id,
            title=title,
            bonus_points=bonus_points,
            achievement_name=achievement_name,
            issued_date=date.today().isoformat()
        )
        self.rewards.append(reward)
        if emp_id in self.employees:
            self.employees[emp_id].bonus_points += bonus_points
        self._save_database()
        return r_id

    # ---------------------------------------------------------------------------
    # FEATURE 5: AI PREDICTIVE ENGINE & HEURISTICS
    # ---------------------------------------------------------------------------

    def predict_burnout(self, emp_id: str) -> Dict[str, Any]:
        """Calculates employee burnout index based on hours, leaves, and activity."""
        emp_atts = [a for a in self.attendance_records if a.emp_id == emp_id]
        if not emp_atts:
            return {
                "emp_id": emp_id,
                "burnout_index": 15.0,
                "risk_level": "LOW",
                "average_daily_hours": 0.0,
                "recommendation": "Optimal operational balance (No shift data logged yet).",
                "reasons": ["Insufficient telemetry context."]
            }

        avg_hours = sum(a.working_hours for a in emp_atts) / len(emp_atts)
        overtime_penalty = max(0.0, (avg_hours - 8.0) * 15.0)

        emp = self.employees.get(emp_id)
        leave_penalty = 0.0
        if emp and emp.leave_balances.get(LeaveType.ANNUAL.value, 0) > 18:
            leave_penalty = 20.0  # Employee is not taking adequate rest time

        burnout_score = min(100.0, round(20.0 + overtime_penalty + leave_penalty, 1))
        risk_level = "HIGH" if burnout_score > 70 else ("MEDIUM" if burnout_score > 40 else "LOW")

        return {
            "emp_id": emp_id,
            "burnout_index": burnout_score,
            "risk_level": risk_level,
            "average_daily_hours": round(avg_hours, 2),
            "recommendation": "Enforce mandatory 3-day leave reset." if risk_level == "HIGH" else "Optimal operational balance."
        }

    def predict_promotion(self, emp_id: str) -> Dict[str, Any]:
        """Evaluates readiness score for corporate advancement."""
        emp = self.employees.get(emp_id)
        if not emp:
            return {"promotion_readiness_pct": 0.0, "status": "UNKNOWN"}

        perf_list = self.performance_records.get(emp_id, [])
        avg_quality = (sum(p.quality_score for p in perf_list) / len(perf_list)) if perf_list else 85.0
        bonus_factor = min(30.0, emp.bonus_points * 0.5)

        readiness = min(100.0, round(avg_quality * 0.6 + bonus_factor + len(emp.certifications) * 5.0, 1))
        return {
            "emp_id": emp_id,
            "promotion_readiness_pct": readiness,
            "recommendation": "Eligible for Senior Title Elevation" if readiness >= 80.0 else "Requires broader project ownership."
        }

    def analyze_skill_gaps(self, department_id: str, target_skills: List[str]) -> Dict[str, Any]:
        """Isolates capability gaps across department staff."""
        dept_emps = [e for e in self.employees.values() if e.department_id == department_id]
        existing_skills = set(skill for e in dept_emps for skill in e.skills)
        missing_skills = [s for s in target_skills if s not in existing_skills]

        return {
            "department_id": department_id,
            "missing_skills": missing_skills,
            "coverage_pct": round(((len(target_skills) - len(missing_skills)) / len(target_skills)) * 100.0, 1) if target_skills else 100.0
        }

    # ---------------------------------------------------------------------------
    # FEATURE 6: ANALYTICS & REPORTING
    # ---------------------------------------------------------------------------

    def get_employee_rankings_dataframe(self) -> pd.DataFrame:
        """Returns clean DataFrame ranking employees by bonus points and performance."""
        data = []
        for emp_id, emp in self.employees.items():
            dept = self.departments.get(emp.department_id)
            data.append({
                "Employee ID": emp_id,
                "Name": emp.full_name,
                "Department": dept.name if dept else "N/A",
                "Role": emp.role,
                "Bonus Points": emp.bonus_points,
                "Active Status": "Active" if emp.is_active else "Inactive"
            })
        df = pd.DataFrame(data)
        return df.sort_values(by="Bonus Points", ascending=False) if not df.empty else df

    # ---------------------------------------------------------------------------
    # FEATURE 7: CROSS-SERVICE MODULAR INTEGRATIONS
    # ---------------------------------------------------------------------------

    def integrate_with_project_manager(self, pm_service: Any) -> Dict[str, Any]:
        """Syncs employee assignments directly with ProjectManagerService task states."""
        try:
            summary = pm_service.get_portfolio_summary()
            logger.info("Integrated with ProjectManager: Synced %d tasks across portfolio.", summary.get("total_tasks", 0))
            return {"status": "SUCCESS", "synced_tasks": summary.get("total_tasks", 0)}
        except Exception as e:
            logger.error("ProjectManager Integration failed: %s", str(e))
            return {"status": "FAILED", "reason": str(e)}

    def integrate_with_finance(self, finance_service: Any) -> Dict[str, Any]:
        """Calculates total payroll expenditure footprint for FinanceService."""
        total_payroll_estimate = len(self.employees) * 7500.0  # Est. baseline average
        return {"total_active_headcount": len(self.employees), "monthly_payroll_estimate": total_payroll_estimate}


# ---------------------------------------------------------------------------
# LOCAL MODULE TEST HARNESS
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("==========================================================")
    print("⚡ TESTING PRODUCTION EMPLOYEE MANAGER SERVICE (AIVORA)")
    print("==========================================================")

    ems = EmployeeManagerService()

    # 1. Employee Operations
    new_id = ems.create_employee("Elena Rostova", "elena@aivora.io", RoleEnum.EMPLOYEE.value, "DEPT-ENG", ["Python", "Rust"])
    print(f"✅ Created Employee: {new_id}")

    # 2. Attendance & Warning Workflow
    att_id = ems.clock_in(new_id, location=WorkLocation.REMOTE.value)
    print(f"✅ Clocked In ({att_id}). Clocking Out...")
    ems.clock_out(new_id)

    # 3. Leave Request Workflow
    l_id = ems.request_leave(new_id, LeaveType.ANNUAL.value, "2026-08-01", "2026-08-05", "Summer vacation")
    if l_id:
        ems.approve_leave(l_id, "EMP-101", approve=True)
        print(f"✅ Leave Request {l_id} Approved.")

    # 4. AI Engine Predictive Checks
    burnout = ems.predict_burnout(new_id)
    print("\n🔥 AI Burnout Telemetry:", burnout)

    promo = ems.predict_promotion(new_id)
    print("🚀 AI Promotion Prediction:", promo)

    # 5. Rankings & DataFrames
    print("\n📊 Employee Rankings Matrix:")
    print(ems.get_employee_rankings_dataframe())