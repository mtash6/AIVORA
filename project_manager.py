# ==============================================================================
# MODULE: services/project_manager.py
# DESCRIPTION: AIVORA Unified Enterprise Project Lifecycle & AI Orchestration Service.
#              Handles Projects, Milestones, Sprints, Kanban Boards, Nested Subtasks,
#              Dependency Validation, Time Tracking, Analytics Reports, and AI Insights.
# ==============================================================================

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd

# Configure logger for enterprise monitoring
logger = logging.getLogger("AIVORA.ProjectManager")
logger.setLevel(logging.INFO)


# ==============================================================================
# ENUMS & DOMAIN CONSTANTS
# ==============================================================================

class ProjectStatus(Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    ON_HOLD = "ON_HOLD"
    ARCHIVED = "ARCHIVED"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


class Priority(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class KanbanColumn(Enum):
    BACKLOG = "BACKLOG"
    TO_DO = "TO_DO"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEW = "REVIEW"
    TESTING = "TESTING"
    COMPLETED = "COMPLETED"


class DependencyType(Enum):
    FINISH_TO_START = "FINISH_TO_START"  # Predecessor must finish before Successor starts
    START_TO_START = "START_TO_START"    # Predecessor must start before Successor starts
    FINISH_TO_FINISH = "FINISH_TO_FINISH"# Predecessor must finish before Successor finishes


class SprintStatus(Enum):
    PLANNING = "PLANNING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class ProjectManagementException(Exception):
    """Base exception for all Project Manager operations."""
    pass


class TaskBlockedException(ProjectManagementException):
    """Raised when an operation is attempted on a blocked task."""
    pass


class InvalidDependencyException(ProjectManagementException):
    """Raised when dependency constraints are violated or form a cyclic graph."""
    pass


class EntityNotFoundException(ProjectManagementException):
    """Raised when a project, task, milestone, or sprint is not found."""
    pass


# ==============================================================================
# CORE DOMAIN ENTITIES
# ==============================================================================

@dataclass
class Dependency:
    """Represents a directional relationship constraint between two tasks."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    predecessor_task_id: str = ""
    successor_task_id: str = ""
    dependency_type: DependencyType = DependencyType.FINISH_TO_START

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "predecessor_task_id": self.predecessor_task_id,
            "successor_task_id": self.successor_task_id,
            "dependency_type": self.dependency_type.value
        }


@dataclass
class TimeEntry:
    """Tracks manual and live operational time logs for tasks."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    user_id: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    total_paused_seconds: float = 0.0
    manual_hours: float = 0.0
    is_running: bool = False
    is_paused: bool = False

    def start_timer(self) -> None:
        """Starts live timer tracking."""
        if self.is_running:
            logger.warning(f"Timer already running for TimeEntry {self.id}")
            return
        self.start_time = datetime.now(timezone.utc)
        self.is_running = True
        self.is_paused = False

    def pause_timer(self) -> None:
        """Pauses active live timer."""
        if not self.is_running or self.is_paused:
            return
        self.paused_at = datetime.now(timezone.utc)
        self.is_paused = True

    def resume_timer(self) -> None:
        """Resumes a paused live timer."""
        if not self.is_running or not self.is_paused or not self.paused_at:
            return
        pause_duration = (datetime.now(timezone.utc) - self.paused_at).total_seconds()
        self.total_paused_seconds += pause_duration
        self.paused_at = None
        self.is_paused = False

    def stop_timer(self) -> float:
        """Stops live timer and calculates elapsed tracked hours."""
        if not self.is_running or not self.start_time:
            return self.get_total_hours()
        
        if self.is_paused and self.paused_at:
            self.resume_timer()
            
        self.end_time = datetime.now(timezone.utc)
        self.is_running = False
        self.is_paused = False
        return self.get_total_hours()

    def get_total_hours(self) -> float:
        """Returns total calculated hours including manual and live timer entries."""
        live_hours = 0.0
        if self.start_time:
            end = self.end_time or datetime.now(timezone.utc)
            elapsed = (end - self.start_time).total_seconds() - self.total_paused_seconds
            live_hours = max(0.0, elapsed / 3600.0)
        return round(self.manual_hours + live_hours, 2)


@dataclass
class Task:
    """Represents a granular work deliverable supporting recursive subtask structures."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    project_id: str = ""
    assignee_id: Optional[str] = None
    status: KanbanColumn = KanbanColumn.BACKLOG
    priority: Priority = Priority.MEDIUM
    due_date: Optional[datetime] = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    parent_task_id: Optional[str] = None
    subtasks: List["Task"] = field(default_factory=list)
    attachments: List[Dict[str, str]] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    time_entries: List[TimeEntry] = field(default_factory=list)
    is_ai_generated: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __getitem__(self, key: str) -> Any:
        """Enables dictionary subscripting access for main.py UI rendering."""
        mapping = {
            "task_id": self.id,
            "title": self.title,
            "assigned_to": self.assignee_id or "Unassigned",
            "estimated_hours": self.estimated_hours,
            "status": self.status.value if isinstance(self.status, KanbanColumn) else str(self.status),
            "due_date": self.due_date.strftime("%Y-%m-%d") if self.due_date else "N/A"
        }
        if key in mapping:
            return mapping[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def add_subtask(self, subtask: "Task") -> None:
        """Nests a subtask under the current task."""
        subtask.parent_task_id = self.id
        subtask.project_id = self.project_id
        self.subtasks.append(subtask)
        self._recalculate_auto_completion()

    def calculate_progress(self) -> float:
        """Recursively calculates task completion percentage across nested subtasks."""
        if not self.subtasks:
            return 100.0 if self.status == KanbanColumn.COMPLETED else 0.0
        
        total_subtask_progress = sum(sub.calculate_progress() for sub in self.subtasks)
        return round(total_subtask_progress / len(self.subtasks), 2)

    def _recalculate_auto_completion(self) -> None:
        """Automatically sets task to COMPLETED if all nested subtasks are finished."""
        if self.subtasks and all(sub.status == KanbanColumn.COMPLETED for sub in self.subtasks):
            self.status = KanbanColumn.COMPLETED
            self.updated_at = datetime.now(timezone.utc)

    def calculate_actual_hours(self) -> float:
        """Aggregates all time tracking entries for this task and nested subtasks."""
        direct_hours = sum(entry.get_total_hours() for entry in self.time_entries)
        subtask_hours = sum(sub.calculate_actual_hours() for sub in self.subtasks)
        self.actual_hours = round(direct_hours + subtask_hours, 2)
        return self.actual_hours


@dataclass
class Milestone:
    """Represents a critical event or gateway date in the project scope."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    title: str = ""
    deadline: Optional[datetime] = None
    task_ids: List[str] = field(default_factory=list)
    is_completed: bool = False

    def calculate_completion_percentage(self, project_tasks: Dict[str, Task]) -> float:
        """Calculates milestone completion rate based on linked tasks."""
        if not self.task_ids:
            return 100.0 if self.is_completed else 0.0
        
        linked_tasks = [project_tasks[tid] for tid in self.task_ids if tid in project_tasks]
        if not linked_tasks:
            return 0.0
        
        completed_count = sum(1 for t in linked_tasks if t.status == KanbanColumn.COMPLETED)
        progress = round((completed_count / len(linked_tasks)) * 100.0, 2)
        if progress >= 100.0:
            self.is_completed = True
        return progress


@dataclass
class Sprint:
    """Represents an active or planned agile iterative execution cycle."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    name: str = ""
    goal: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: SprintStatus = SprintStatus.PLANNING
    task_ids: List[str] = field(default_factory=list)
    planned_velocity: float = 0.0
    completed_velocity: float = 0.0

    def calculate_burndown(self, project_tasks: Dict[str, Task]) -> Dict[str, Any]:
        """Calculates burndown metrics for active sprint management."""
        total_estimated = sum(project_tasks[tid].estimated_hours for tid in self.task_ids if tid in project_tasks)
        remaining_estimated = sum(
            project_tasks[tid].estimated_hours 
            for tid in self.task_ids 
            if tid in project_tasks and project_tasks[tid].status != KanbanColumn.COMPLETED
        )
        completed_hours = total_estimated - remaining_estimated
        self.completed_velocity = sum(
            project_tasks[tid].estimated_hours 
            for tid in self.task_ids 
            if tid in project_tasks and project_tasks[tid].status == KanbanColumn.COMPLETED
        )
        return {
            "sprint_id": self.id,
            "total_estimated_hours": total_estimated,
            "remaining_estimated_hours": remaining_estimated,
            "completed_hours": completed_hours,
            "completion_percentage": round((completed_hours / total_estimated * 100.0), 2) if total_estimated > 0 else 0.0
        }


@dataclass
class Project:
    """Core domain container orchestrating tasks, sprints, members, and budgets."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    owner_id: str = ""
    member_ids: List[str] = field(default_factory=list)
    status: ProjectStatus = ProjectStatus.DRAFT
    priority: Priority = Priority.MEDIUM
    budget: float = 0.0
    due_date: Optional[datetime] = None
    tasks: Dict[str, Task] = field(default_factory=dict)
    milestones: Dict[str, Milestone] = field(default_factory=dict)
    sprints: Dict[str, Sprint] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    client: str = "Enterprise Client"

    def __getitem__(self, key: str) -> Any:
        """Enables dictionary subscripting access for main.py UI rendering."""
        mapping = {
            "project_id": self.id,
            "title": self.name,
            "client": self.client,
            "owner": self.owner_id,
            "target_completion": self.due_date.strftime("%Y-%m-%d") if self.due_date else "N/A",
            "budget": self.budget,
            "tasks": [t.to_dict() if hasattr(t, "to_dict") else {
                "task_id": t.id,
                "title": t.title,
                "assigned_to": t.assignee_id or "Unassigned",
                "estimated_hours": t.estimated_hours,
                "status": t.status.value if isinstance(t.status, KanbanColumn) else str(t.status),
                "due_date": t.due_date.strftime("%Y-%m-%d") if t.due_date else "N/A"
            } for t in self.tasks.values()]
        }
        if key in mapping:
            return mapping[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def get_progress_percentage(self) -> float:
        """Calculates overall project progress across all root tasks."""
        root_tasks = [t for t in self.tasks.values() if t.parent_task_id is None]
        if not root_tasks:
            return 100.0 if self.status == ProjectStatus.COMPLETED else 0.0
        
        total_progress = sum(t.calculate_progress() for t in root_tasks)
        return round(total_progress / len(root_tasks), 2)


# ==============================================================================
# MAIN SERVICE: PROJECT MANAGER ENGINE
# ==============================================================================

class ProjectManagerService:
    """
    Enterprise Application Service layer orchestrating projects, time tracking,
    dependency validation, agile sprints, and predictive AI insights.
    """

    def __init__(self):
        self._projects: Dict[str, Project] = {}
        self._seed_default_projects()
        logger.info("AIVORA Project Manager Service initialized successfully.")

    @property
    def projects(self) -> Dict[str, Project]:
        """Exposes projects dictionary property for UI integration."""
        return self._projects

    def _seed_default_projects(self) -> None:
        """Populates initial sample projects if database is empty."""
        p1 = self.create_project(
            name="Cloud Infrastructure Migration",
            owner_id="Sarah Chen",
            description="Migrate enterprise services to serverless cloud infrastructure.",
            budget=45000.0,
            priority=Priority.HIGH,
            due_date=datetime.now(timezone.utc) + timedelta(days=90)
        )
        p1.client = "Alpha Corp"

        t1 = self.create_task(p1.id, "Database Schema Export", assignee_id="Marcus Vance", estimated_hours=16.0)
        t1.status = KanbanColumn.COMPLETED
        t2 = self.create_task(p1.id, "Serverless Pipeline Setup", assignee_id="Elena Rostova", estimated_hours=32.0)
        t2.status = KanbanColumn.IN_PROGRESS
        t3 = self.create_task(p1.id, "Automated QA Verification", assignee_id="David Kim", estimated_hours=24.0)

        p2 = self.create_project(
            name="AI Analytics Engine V2",
            owner_id="Alex Mercer",
            description="Enterprise BI & Predictive Financial Analytics Pipeline.",
            budget=85000.0,
            priority=Priority.URGENT,
            due_date=datetime.now(timezone.utc) + timedelta(days=120)
        )
        p2.client = "AIVORA R&D"

        t4 = self.create_task(p2.id, "Consensus Engine Refactor", assignee_id="Alex Mercer", estimated_hours=40.0)
        t4.status = KanbanColumn.COMPLETED
        t5 = self.create_task(p2.id, "Executive Dashboard Telemetry", assignee_id="Sarah Chen", estimated_hours=20.0)
        t5.status = KanbanColumn.IN_PROGRESS

    # --------------------------------------------------------------------------
    # STREAMLIT COMPATIBILITY & TELEMETRY INTERFACES
    # --------------------------------------------------------------------------

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Calculates global portfolio metrics for Streamlit telemetry panels."""
        total_projects = len(self._projects)
        total_budget = sum(p.budget for p in self._projects.values())
        
        all_tasks = [t for p in self._projects.values() for t in p.tasks.values()]
        total_tasks = len(all_tasks)
        completed_tasks = sum(1 for t in all_tasks if t.status == KanbanColumn.COMPLETED)
        blocked_tasks = sum(1 for t in all_tasks if self.is_task_blocked(t.project_id, t.id)[0])

        global_completion = (completed_tasks / total_tasks * 100.0) if total_tasks > 0 else 0.0

        return {
            "total_projects": total_projects,
            "total_budget": round(total_budget, 2),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "blocked_tasks": blocked_tasks,
            "global_completion_pct": round(global_completion, 1)
        }

    def calculate_project_health(self, project_id: str) -> Dict[str, Any]:
        """Computes health index, completion percentage, and risk factors for UI."""
        project = self.get_project(project_id)
        tasks = list(project.tasks.values())
        total_tasks = len(tasks)

        if total_tasks == 0:
            return {
                "completion_pct": 0.0,
                "health_score": 100.0,
                "status_flag": "GREEN",
                "blocked_tasks": 0,
                "overdue_tasks": 0,
                "total_tasks": 0,
                "completed_tasks": 0
            }

        completed = sum(1 for t in tasks if t.status == KanbanColumn.COMPLETED)
        blocked = sum(1 for t in tasks if self.is_task_blocked(project_id, t.id)[0])

        now = datetime.now(timezone.utc)
        overdue = sum(1 for t in tasks if t.status != KanbanColumn.COMPLETED and t.due_date and t.due_date < now)

        completion_pct = (completed / total_tasks) * 100.0
        penalty = (blocked * 15) + (overdue * 20)
        health_score = max(0.0, min(100.0, 100.0 - penalty))

        if health_score >= 80:
            status_flag = "GREEN"
        elif health_score >= 50:
            status_flag = "AMBER"
        else:
            status_flag = "RED"

        return {
            "completion_pct": round(completion_pct, 1),
            "health_score": round(health_score, 1),
            "status_flag": status_flag,
            "blocked_tasks": blocked,
            "overdue_tasks": overdue,
            "total_tasks": total_tasks,
            "completed_tasks": completed
        }

    def get_project_tasks_dataframe(self, project_id: str) -> pd.DataFrame:
        """Returns Pandas DataFrame formatted for direct Streamlit table display."""
        project = self.get_project(project_id)
        if not project.tasks:
            return pd.DataFrame()

        rows = []
        for task in project.tasks.values():
            rows.append({
                "Task ID": task.id,
                "Task Title": task.title,
                "Assigned Personnel": task.assignee_id or "Unassigned",
                "Est. Hours": task.estimated_hours,
                "Status": task.status.value if isinstance(task.status, KanbanColumn) else str(task.status),
                "Due Date": task.due_date.strftime("%Y-%m-%d") if task.due_date else "N/A"
            })
        return pd.DataFrame(rows)

    def add_task_to_project(
        self,
        project_id: str,
        task_title: str,
        assigned_to: str,
        estimated_hours: float,
        due_date: str
    ) -> str:
        """Compatibility wrapper to register a task from string UI form values."""
        parsed_due = None
        if due_date:
            try:
                parsed_due = datetime.strptime(due_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        task = self.create_task(
            project_id=project_id,
            title=task_title,
            assignee_id=assigned_to,
            estimated_hours=float(estimated_hours),
            due_date=parsed_due
        )
        return task.id

    def update_task_status(self, project_id: str, task_id: str, new_status: str) -> bool:
        """Compatibility wrapper to update task status from UI inputs."""
        mapping = {
            "BACKLOG": KanbanColumn.BACKLOG,
            "TO_DO": KanbanColumn.TO_DO,
            "IN_PROGRESS": KanbanColumn.IN_PROGRESS,
            "REVIEW": KanbanColumn.REVIEW,
            "TESTING": KanbanColumn.TESTING,
            "COMPLETED": KanbanColumn.COMPLETED
        }
        target_column = mapping.get(new_status.upper(), KanbanColumn.BACKLOG)

        try:
            self.move_kanban_task(project_id, task_id, target_column)
            return True
        except ProjectManagementException as e:
            logger.warning(f"Could not update task status: {e}")
            return False

    # --------------------------------------------------------------------------
    # PROJECT OPERATIONS
    # --------------------------------------------------------------------------

    def create_project(
        self,
        name: str,
        owner_id: str,
        description: str = "",
        member_ids: Optional[List[str]] = None,
        budget: float = 0.0,
        priority: Priority = Priority.MEDIUM,
        due_date: Optional[datetime] = None
    ) -> Project:
        """Creates and registers a new Project entity."""
        project = Project(
            name=name,
            owner_id=owner_id,
            description=description,
            member_ids=member_ids or [owner_id],
            budget=budget,
            priority=priority,
            due_date=due_date,
            status=ProjectStatus.ACTIVE
        )
        self._projects[project.id] = project
        logger.info(f"Project '{name}' (ID: {project.id}) created successfully.")
        return project

    def update_project(self, project_id: str, **updates: Any) -> Project:
        """Updates project specifications dynamically."""
        project = self.get_project(project_id)
        for key, value in updates.items():
            if hasattr(project, key):
                setattr(project, key, value)
        project.updated_at = datetime.now(timezone.utc)
        logger.info(f"Project '{project_id}' updated.")
        return project

    def archive_project(self, project_id: str) -> Project:
        """Archives a project and hides it from active pipeline execution."""
        return self.update_project(project_id, status=ProjectStatus.ARCHIVED)

    def delete_project(self, project_id: str) -> bool:
        """Permanently deletes a project from registry."""
        if project_id in self._projects:
            del self._projects[project_id]
            logger.info(f"Project '{project_id}' deleted.")
            return True
        raise EntityNotFoundException(f"Project '{project_id}' not found.")

    def get_project(self, project_id: str) -> Project:
        """Retrieves a project by ID."""
        if project_id not in self._projects:
            raise EntityNotFoundException(f"Project ID '{project_id}' not found.")
        return self._projects[project_id]

    # --------------------------------------------------------------------------
    # TASK MANAGEMENT & DEPENDENCY CHECKS
    # --------------------------------------------------------------------------

    def create_task(
        self,
        project_id: str,
        title: str,
        description: str = "",
        assignee_id: Optional[str] = None,
        priority: Priority = Priority.MEDIUM,
        estimated_hours: float = 0.0,
        due_date: Optional[datetime] = None,
        parent_task_id: Optional[str] = None,
        is_ai_generated: bool = False
    ) -> Task:
        """Creates a task or nested subtask within a target project."""
        project = self.get_project(project_id)
        
        task = Task(
            project_id=project_id,
            title=title,
            description=description,
            assignee_id=assignee_id,
            priority=priority,
            estimated_hours=estimated_hours,
            due_date=due_date,
            parent_task_id=parent_task_id,
            is_ai_generated=is_ai_generated
        )

        project.tasks[task.id] = task

        if parent_task_id and parent_task_id in project.tasks:
            project.tasks[parent_task_id].add_subtask(task)

        logger.info(f"Task '{title}' (ID: {task.id}) added to Project '{project_id}'.")
        return task

    def add_dependency(
        self,
        project_id: str,
        predecessor_id: str,
        successor_id: str,
        dep_type: DependencyType = DependencyType.FINISH_TO_START
    ) -> Dependency:
        """Establishes a dependency constraint between two tasks and validates for cycles."""
        project = self.get_project(project_id)
        
        if predecessor_id not in project.tasks or successor_id not in project.tasks:
            raise EntityNotFoundException("Predecessor or Successor task not found in project.")

        if predecessor_id == successor_id:
            raise InvalidDependencyException("A task cannot depend on itself.")

        # Cycle Detection Check
        if self._detect_dependency_cycle(project, predecessor_id, successor_id):
            raise InvalidDependencyException("Dependency creates a cyclic loop in project tasks.")

        dependency = Dependency(
            predecessor_task_id=predecessor_id,
            successor_task_id=successor_id,
            dependency_type=dep_type
        )
        
        project.tasks[successor_id].dependencies.append(dependency)
        logger.info(f"Dependency created: Task {predecessor_id} -> Task {successor_id} ({dep_type.value})")
        return dependency

    def is_task_blocked(self, project_id: str, task_id: str) -> Tuple[bool, List[str]]:
        """Verifies if a task is currently blocked by unresolved predecessors."""
        project = self.get_project(project_id)
        task = project.tasks.get(task_id)
        if not task:
            raise EntityNotFoundException(f"Task '{task_id}' not found.")

        blocking_reasons: List[str] = []

        for dep in task.dependencies:
            pred_task = project.tasks.get(dep.predecessor_task_id)
            if not pred_task:
                continue

            if dep.dependency_type == DependencyType.FINISH_TO_START:
                if pred_task.status != KanbanColumn.COMPLETED:
                    blocking_reasons.append(f"Predecessor '{pred_task.title}' is not finished.")

            elif dep.dependency_type == DependencyType.START_TO_START:
                if pred_task.status in [KanbanColumn.BACKLOG, KanbanColumn.TO_DO]:
                    blocking_reasons.append(f"Predecessor '{pred_task.title}' has not started.")

            elif dep.dependency_type == DependencyType.FINISH_TO_FINISH:
                if pred_task.status != KanbanColumn.COMPLETED and task.status == KanbanColumn.COMPLETED:
                    blocking_reasons.append(f"Predecessor '{pred_task.title}' must complete before finishing.")

        return (len(blocking_reasons) > 0, blocking_reasons)

    def move_kanban_task(self, project_id: str, task_id: str, target_column: KanbanColumn) -> Task:
        """Moves a task across Kanban board columns with strict dependency enforcement."""
        project = self.get_project(project_id)
        task = project.tasks.get(task_id)
        if not task:
            raise EntityNotFoundException(f"Task '{task_id}' not found.")

        # Guard: Prevent completion or execution of blocked tasks
        if target_column in [KanbanColumn.IN_PROGRESS, KanbanColumn.COMPLETED]:
            is_blocked, reasons = self.is_task_blocked(project_id, task_id)
            if is_blocked:
                error_msg = f"Cannot move task '{task.title}' to {target_column.value}. Reasons: {'; '.join(reasons)}"
                logger.error(error_msg)
                raise TaskBlockedException(error_msg)

        task.status = target_column
        task.updated_at = datetime.now(timezone.utc)

        # Trigger parent auto-completion checks if applicable
        if task.parent_task_id and task.parent_task_id in project.tasks:
            project.tasks[task.parent_task_id]._recalculate_auto_completion()

        logger.info(f"Task '{task_id}' moved to '{target_column.value}'")
        return task

    # --------------------------------------------------------------------------
    # SPRINT & MILESTONE MANAGEMENT
    # --------------------------------------------------------------------------

    def create_sprint(self, project_id: str, name: str, goal: str, planned_velocity: float = 0.0) -> Sprint:
        """Creates a new agile sprint iteration."""
        project = self.get_project(project_id)
        sprint = Sprint(project_id=project_id, name=name, goal=goal, planned_velocity=planned_velocity)
        project.sprints[sprint.id] = sprint
        return sprint

    def start_sprint(self, project_id: str, sprint_id: str, duration_days: int = 14) -> Sprint:
        """Activates a sprint and sets its start/end dates."""
        project = self.get_project(project_id)
        sprint = project.sprints.get(sprint_id)
        if not sprint:
            raise EntityNotFoundException(f"Sprint '{sprint_id}' not found.")
        
        sprint.start_date = datetime.now(timezone.utc)
        sprint.end_date = sprint.start_date + timedelta(days=duration_days)
        sprint.status = SprintStatus.ACTIVE
        logger.info(f"Sprint '{sprint.name}' activated for project '{project_id}'.")
        return sprint

    def create_milestone(self, project_id: str, title: str, deadline: datetime, task_ids: List[str]) -> Milestone:
        """Establishes a milestone boundary linked to key project tasks."""
        project = self.get_project(project_id)
        milestone = Milestone(project_id=project_id, title=title, deadline=deadline, task_ids=task_ids)
        project.milestones[milestone.id] = milestone
        return milestone

    # --------------------------------------------------------------------------
    # TIME TRACKING
    # --------------------------------------------------------------------------

    def start_task_timer(self, project_id: str, task_id: str, user_id: str) -> TimeEntry:
        """Starts real-time session tracking for a user on a task."""
        project = self.get_project(project_id)
        task = project.tasks.get(task_id)
        if not task:
            raise EntityNotFoundException(f"Task '{task_id}' not found.")

        time_entry = TimeEntry(task_id=task_id, user_id=user_id)
        time_entry.start_timer()
        task.time_entries.append(time_entry)
        return time_entry

    def stop_task_timer(self, project_id: str, task_id: str, entry_id: str) -> float:
        """Stops live session tracking and updates task actual hours."""
        project = self.get_project(project_id)
        task = project.tasks.get(task_id)
        if not task:
            raise EntityNotFoundException(f"Task '{task_id}' not found.")

        entry = next((e for e in task.time_entries if e.id == entry_id), None)
        if not entry:
            raise EntityNotFoundException(f"TimeEntry '{entry_id}' not found.")

        logged_hours = entry.stop_timer()
        task.calculate_actual_hours()
        return logged_hours

    def log_manual_time(self, project_id: str, task_id: str, user_id: str, hours: float) -> TimeEntry:
        """Registers manual time entry against a task."""
        project = self.get_project(project_id)
        task = project.tasks.get(task_id)
        if not task:
            raise EntityNotFoundException(f"Task '{task_id}' not found.")

        entry = TimeEntry(task_id=task_id, user_id=user_id, manual_hours=hours)
        task.time_entries.append(entry)
        task.calculate_actual_hours()
        return entry

    # --------------------------------------------------------------------------
    # REPORTING ENGINE
    # --------------------------------------------------------------------------

    def generate_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Generates a high-density status summary report for executive decision-making."""
        project = self.get_project(project_id)
        total_tasks = len(project.tasks)
        completed_tasks = sum(1 for t in project.tasks.values() if t.status == KanbanColumn.COMPLETED)
        
        total_est = sum(t.estimated_hours for t in project.tasks.values())
        total_act = sum(t.calculate_actual_hours() for t in project.tasks.values())

        return {
            "project_id": project.id,
            "name": project.name,
            "status": project.status.value,
            "owner_id": project.owner_id,
            "progress_percentage": project.get_progress_percentage(),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "estimated_total_hours": total_est,
            "actual_logged_hours": total_act,
            "overbudget_hours": max(0.0, total_act - total_est),
            "sprints_count": len(project.sprints),
            "milestones_count": len(project.milestones)
        }

    def generate_employee_workload_report(self) -> Dict[str, Any]:
        """Calculates workload distribution and total task allocations per team member."""
        workload: Dict[str, Dict[str, Any]] = {}

        for project in self._projects.values():
            for task in project.tasks.values():
                if task.assignee_id:
                    if task.assignee_id not in workload:
                        workload[task.assignee_id] = {
                            "assigned_tasks": 0,
                            "total_estimated_hours": 0.0,
                            "total_actual_hours": 0.0,
                            "urgent_tasks": 0
                        }
                    workload[task.assignee_id]["assigned_tasks"] += 1
                    workload[task.assignee_id]["total_estimated_hours"] += task.estimated_hours
                    workload[task.assignee_id]["total_actual_hours"] += task.calculate_actual_hours()
                    if task.priority == Priority.URGENT:
                        workload[task.assignee_id]["urgent_tasks"] += 1

        return workload

    # --------------------------------------------------------------------------
    # ADVANCED AI INTELLIGENCE & ANALYTICS
    # --------------------------------------------------------------------------

    def predict_project_delays(self, project_id: str) -> Dict[str, Any]:
        """AI Feature: Predicts project schedule slippage based on burn rates and task velocity."""
        project = self.get_project(project_id)
        if not project.due_date:
            return {"delay_predicted": False, "reason": "No due date set on project."}

        total_est = sum(t.estimated_hours for t in project.tasks.values())
        total_act = sum(t.calculate_actual_hours() for t in project.tasks.values())
        progress = project.get_progress_percentage()

        if progress == 0 and total_est > 0:
            return {"delay_predicted": True, "risk_level": "HIGH", "confidence": 0.88, "message": "Zero progress logged on active project tasks."}

        expected_total_hours = (total_act / (progress / 100.0)) if progress > 0 else total_est
        hours_overrun = expected_total_hours - total_est

        is_delayed = hours_overrun > (total_est * 0.15)  # Threshold > 15% overrun
        
        return {
            "project_id": project_id,
            "delay_predicted": is_delayed,
            "risk_level": "HIGH" if hours_overrun > (total_est * 0.3) else "MEDIUM" if is_delayed else "LOW",
            "projected_overrun_hours": round(max(0.0, hours_overrun), 2),
            "estimated_completion_date": (datetime.now(timezone.utc) + timedelta(hours=expected_total_hours)).isoformat()
        }

    def detect_inactive_projects(self, threshold_days: int = 14) -> List[Dict[str, Any]]:
        """AI Feature: Isolates stagnant projects with zero updates in the threshold window."""
        inactive = []
        now = datetime.now(timezone.utc)
        
        for project in self._projects.values():
            if project.status == ProjectStatus.ACTIVE:
                days_since_update = (now - project.updated_at).days
                if days_since_update >= threshold_days:
                    inactive.append({
                        "project_id": project.id,
                        "name": project.name,
                        "days_inactive": days_since_update,
                        "recommendation": "Archive or re-assign project leadership."
                    })
        return inactive

    def recommend_task_priorities(self, project_id: str) -> List[Dict[str, Any]]:
        """AI Feature: Recommends priority adjustments for critical path tasks."""
        project = self.get_project(project_id)
        recommendations = []

        for task in project.tasks.values():
            if task.status != KanbanColumn.COMPLETED:
                dependent_count = sum(
                    1 for t in project.tasks.values() 
                    if any(d.predecessor_task_id == task.id for d in t.dependencies)
                )
                if dependent_count >= 3 and task.priority != Priority.URGENT:
                    recommendations.append({
                        "task_id": task.id,
                        "title": task.title,
                        "current_priority": task.priority.value,
                        "recommended_priority": Priority.URGENT.value,
                        "reason": f"Task blocks {dependent_count} downstream tasks on the critical path."
                    })

        return recommendations

    # --------------------------------------------------------------------------
    # INTERNAL UTILITIES & INTEGRATION HOOKS
    # --------------------------------------------------------------------------

    def _detect_dependency_cycle(self, project: Project, predecessor_id: str, successor_id: str) -> bool:
        """Depth-First Search (DFS) graph traversal for cycle detection."""
        visited: Set[str] = set()

        def dfs(current_id: str) -> bool:
            if current_id == predecessor_id:
                return True
            visited.add(current_id)
            for t in project.tasks.values():
                for d in t.dependencies:
                    if d.predecessor_task_id == current_id and d.successor_task_id not in visited:
                        if dfs(d.successor_task_id):
                            return True
            return False

        return dfs(successor_id)

    def hook_recovery_engine_trigger(self, project_id: str, task_id: str) -> Dict[str, Any]:
        """Integration Hook: Bridges into services/recovery_engine.py."""
        is_blocked, reasons = self.is_task_blocked(project_id, task_id)
        return {
            "recovery_engine_notified": True,
            "task_id": task_id,
            "escalation_required": is_blocked,
            "reasons": reasons
        }

    def hook_finance_budget_sync(self, project_id: str) -> Dict[str, Any]:
        """Integration Hook: Bridges into financial services for labor cost sync."""
        summary = self.generate_project_summary(project_id)
        calculated_spend = summary["actual_logged_hours"] * 85.0
        project = self.get_project(project_id)
        
        return {
            "project_id": project_id,
            "allocated_budget": project.budget,
            "calculated_labor_spend": calculated_spend,
            "budget_remaining": max(0.0, project.budget - calculated_spend),
            "variance_percentage": round((calculated_spend / project.budget * 100.0), 2) if project.budget > 0 else 0.0
        }