import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, List

plt.style.use("dark_background")

class AdminAnalyticsService:
    """
    Admin-level Operational & Project Intelligence Engine.
    Tracks platform performance, project success rates, SLA breach ratios,
    and Autonomous Recovery Engine intervention statistics.
    """

    def __init__(self):
        self.primary_color = "#3B82F6"
        self.success_color = "#10B981"
        self.warning_color = "#F59E0B"
        self.danger_color = "#EF4444"
        self.purple_color = "#8B5CF6"

    def calculate_admin_telemetry(self, logs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculates project success ratios, agent recovery rates, and platform throughput."""
        if not logs:
            logs = self._generate_mock_admin_logs()

        df_projects = pd.DataFrame(logs["projects"])
        df_recovery = pd.DataFrame(logs["recovery_events"])
        df_tasks = pd.DataFrame(logs["task_executions"])

        # Calculations
        total_projects = len(df_projects)
        completed_projects = len(df_projects[df_projects["status"] == "SUCCESSFUL"])
        success_rate = (completed_projects / total_projects * 100) if total_projects > 0 else 0.0

        avg_sla_hours = df_projects["avg_completion_hours"].mean()
        on_time_projects = len(df_projects[df_projects["delayed"] == False])
        on_time_rate = (on_time_projects / total_projects * 100) if total_projects > 0 else 0.0

        # Human vs Autonomous Agent Split
        total_tasks = len(df_tasks)
        agent_completed = len(df_tasks[df_tasks["executed_by"] == "AUTONOMOUS_AGENT"])
        human_completed = len(df_tasks[df_tasks["executed_by"] == "HUMAN_OPERATOR"])
        agent_execution_ratio = (agent_completed / total_tasks * 100) if total_tasks > 0 else 0.0

        # Recovery Engine Interventions
        total_interventions = len(df_recovery)
        warning_stage_count = len(df_recovery[df_recovery["stage"] == "OVERDUE_WARNING"])
        escalation_stage_count = len(df_recovery[df_recovery["stage"] == "ESCALATED"])
        fallback_agent_spins = len(df_recovery[df_recovery["stage"] == "FALLBACK_PROCESSING"])

        return {
            "metrics": {
                "total_projects": total_projects,
                "project_success_rate": success_rate,
                "on_time_delivery_rate": on_time_rate,
                "avg_sla_hours": avg_sla_hours,
                "total_tasks_processed": total_tasks,
                "agent_execution_ratio": agent_execution_ratio,
                "recovery_interventions": total_interventions,
                "fallback_agents_triggered": fallback_agent_spins
            },
            "dataframes": {
                "projects": df_projects,
                "recovery": df_recovery,
                "tasks": df_tasks
            }
        }

    def generate_admin_charts(self, telemetry: Dict[str, Any]) -> List[plt.Figure]:
        """Generates 4 operational visual analytics charts for the system admin."""
        df_p = telemetry["dataframes"]["projects"]
        df_r = telemetry["dataframes"]["recovery"]
        df_t = telemetry["dataframes"]["tasks"]
        figs = []

        # -------------------------------------------------------------
        # CHART 1: Project Success Rate & Completion Velocity Trends
        # -------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(10, 4.5), facecolor="#0E1117")
        ax1.set_facecolor("#161B22")

        weekly_df = df_p.groupby("week").agg(
            total=("id", "count"),
            successful=("status", lambda s: (s == "SUCCESSFUL").sum())
        ).reset_index()
        weekly_df["rate"] = (weekly_df["successful"] / weekly_df["total"]) * 100

        ax1.plot(weekly_df["week"], weekly_df["rate"], color=self.success_color, marker="o", linewidth=2.5, label="Success Rate (%)")
        ax1.axhline(90, color=self.warning_color, linestyle="--", alpha=0.7, label="Target SLA Threshold (90%)")

        ax1.set_title("Project Success Rate Trajectory (Weekly)", fontsize=13, pad=12, fontweight="bold", color="#FFFFFF")
        ax1.set_ylim(50, 105)
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax1.legend(facecolor="#161B22", edgecolor="none")
        ax1.grid(True, linestyle="--", alpha=0.2)
        fig1.tight_layout()
        figs.append(fig1)

        # -------------------------------------------------------------
        # CHART 2: Execution Split (Human Operator vs Autonomous Agent)
        # -------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 4.5), facecolor="#0E1117")
        ax2.set_facecolor("#161B22")

        exec_counts = df_t["executed_by"].value_counts()
        labels = ["Human Operator", "Autonomous Agent"]
        colors = [self.primary_color, self.purple_color]

        ax2.pie(
            exec_counts, 
            labels=labels, 
            autopct="%1.1f%%", 
            colors=colors, 
            startangle=90,
            textprops=dict(color="#FFFFFF", fontweight="bold"),
            wedgeprops=dict(width=0.45, edgecolor="#0E1117", linewidth=2)
        )

        ax2.set_title("Workforce Allocation: Human vs AI Agent Deliverables", fontsize=13, pad=12, fontweight="bold", color="#FFFFFF")
        fig2.tight_layout()
        figs.append(fig2)

        # -------------------------------------------------------------
        # CHART 3: Recovery Engine State Interventions
        # -------------------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=(10, 4.5), facecolor="#0E1117")
        ax3.set_facecolor("#161B22")

        stage_counts = df_r["stage"].value_counts()
        colors = [self.warning_color, self.danger_color, self.purple_color]

        bars = ax3.bar(stage_counts.index, stage_counts.values, color=colors, width=0.5, alpha=0.85)
        ax3.set_title("Autonomous Recovery Engine Intervention Frequency", fontsize=13, pad=12, fontweight="bold", color="#FFFFFF")
        ax3.set_ylabel("Incident Count", color="#D0D0D0")
        ax3.grid(axis="y", linestyle="--", alpha=0.2)

        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{int(yval)}", ha="center", color="#FFFFFF", fontweight="bold")

        fig3.tight_layout()
        figs.append(fig3)

        # -------------------------------------------------------------
        # CHART 4: Task Bottlenecks & Category SLA Latency
        # -------------------------------------------------------------
        fig4, ax4 = plt.subplots(figsize=(10, 4.5), facecolor="#0E1117")
        ax4.set_facecolor("#161B22")

        category_sla = df_t.groupby("category")["duration_hours"].mean().sort_values()

        ax4.barh(category_sla.index, category_sla.values, color=self.primary_color, height=0.5, alpha=0.85)
        ax4.set_title("Average Task SLA Duration by Capability Domain", fontsize=13, pad=12, fontweight="bold", color="#FFFFFF")
        ax4.set_xlabel("Hours to Complete", color="#D0D0D0")
        ax4.grid(axis="x", linestyle="--", alpha=0.2)
        fig4.tight_layout()
        figs.append(fig4)

        return figs

    def _generate_mock_admin_logs(self) -> Dict[str, Any]:
        """Fallback dataset for tracking platform health and project deliverables."""
        return {
            "projects": [
                {"id": "PRJ-01", "week": "W1", "status": "SUCCESSFUL", "avg_completion_hours": 12.5, "delayed": False},
                {"id": "PRJ-02", "week": "W1", "status": "SUCCESSFUL", "avg_completion_hours": 14.0, "delayed": False},
                {"id": "PRJ-03", "week": "W2", "status": "FAILED", "avg_completion_hours": 28.0, "delayed": True},
                {"id": "PRJ-04", "week": "W2", "status": "SUCCESSFUL", "avg_completion_hours": 11.0, "delayed": False},
                {"id": "PRJ-05", "week": "W3", "status": "SUCCESSFUL", "avg_completion_hours": 9.5, "delayed": False},
                {"id": "PRJ-06", "week": "W3", "status": "SUCCESSFUL", "avg_completion_hours": 10.2, "delayed": False},
                {"id": "PRJ-07", "week": "W4", "status": "SUCCESSFUL", "avg_completion_hours": 8.0, "delayed": False},
                {"id": "PRJ-08", "week": "W4", "status": "SUCCESSFUL", "avg_completion_hours": 13.1, "delayed": True},
            ],
            "recovery_events": [
                {"task_id": "TSK-101", "stage": "OVERDUE_WARNING"},
                {"task_id": "TSK-102", "stage": "OVERDUE_WARNING"},
                {"task_id": "TSK-103", "stage": "ESCALATED"},
                {"task_id": "TSK-104", "stage": "FALLBACK_PROCESSING"},
                {"task_id": "TSK-105", "stage": "FALLBACK_PROCESSING"},
            ],
            "task_executions": [
                {"task_id": "TSK-01", "executed_by": "HUMAN_OPERATOR", "category": "Data Analysis", "duration_hours": 4.2},
                {"task_id": "TSK-02", "executed_by": "AUTONOMOUS_AGENT", "category": "Code Audit", "duration_hours": 0.8},
                {"task_id": "TSK-03", "executed_by": "HUMAN_OPERATOR", "category": "Strategy Brief", "duration_hours": 8.5},
                {"task_id": "TSK-04", "executed_by": "AUTONOMOUS_AGENT", "category": "Data Analysis", "duration_hours": 1.2},
                {"task_id": "TSK-05", "executed_by": "AUTONOMOUS_AGENT", "category": "Financial Reporting", "duration_hours": 0.5},
                {"task_id": "TSK-06", "executed_by": "HUMAN_OPERATOR", "category": "Code Audit", "duration_hours": 5.1},
            ]
        }