import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List

# Set dark theme matching Streamlit dashboard aesthetics
plt.style.use("dark_background")

class ExecutiveFinanceService:
    """
    Real-time Executive Financial Intelligence Engine.
    Calculates P&L, Cash Flow horizons, Budget Adherence, Forecasts,
    and generates high-density operational visual dashboards.
    """

    def __init__(self):
        self.primary_color = "#3B82F6"      # Tech Blue
        self.success_color = "#10B981"      # Emerald Green
        self.warning_color = "#F59E0B"      # Amber
        self.danger_color = "#EF4444"       # Crimson Red
        self.accent_color = "#8B5CF6"       # Purple

    def analyze_executive_finance(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Processes financial datasets or generates realistic executive telemetry if raw ledger is empty.
        Returns key KPIs, financial health scores, and metrics.
        """
        if not data:
            data = self._generate_mock_financial_data()

        df_monthly = pd.DataFrame(data["monthly_records"])
        df_budgets = pd.DataFrame(data["budget_allocations"])
        df_receivables = pd.DataFrame(data["outstanding_invoices"])

        # Core Metrics Calculations
        total_revenue = df_monthly["revenue"].sum()
        total_expenses = df_monthly["expenses"].sum()
        net_profit = total_revenue - total_expenses
        profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0

        avg_monthly_burn = df_monthly["expenses"].mean() - df_monthly["revenue"].mean()
        burn_rate = max(0, avg_monthly_burn)
        
        cash_balance = data.get("current_cash_reserve", 1_250_000)
        runway_months = (cash_balance / burn_rate) if burn_rate > 0 else 999.0

        total_outstanding = df_receivables["amount"].sum()
        overdue_receivables = df_receivables[df_receivables["status"] == "OVERDUE"]["amount"].sum()

        # Forecasted Revenue (3-Month Moving Average + 8% projected growth)
        recent_avg_rev = df_monthly["revenue"].tail(3).mean()
        forecasted_q_revenue = recent_avg_rev * 3 * 1.08

        # Budget Utilization Calculation
        df_budgets["utilization_pct"] = (df_budgets["spent"] / df_budgets["allocated"]) * 100
        overall_budget_utilization = (df_budgets["spent"].sum() / df_budgets["allocated"].sum()) * 100

        # Algorithmic Financial Health Score (0 - 100)
        health_score = self._calculate_health_score(
            margin_pct=profit_margin,
            runway_months=runway_months,
            budget_utilization=overall_budget_utilization,
            overdue_ratio=(overdue_receivables / total_outstanding if total_outstanding > 0 else 0)
        )

        return {
            "kpis": {
                "gross_revenue": total_revenue,
                "net_profit": net_profit,
                "profit_margin_pct": profit_margin,
                "cash_reserve": cash_balance,
                "burn_rate": burn_rate,
                "runway_months": runway_months,
                "forecasted_q_revenue": forecasted_q_revenue,
                "outstanding_payments": total_outstanding,
                "overdue_payments": overdue_receivables,
                "budget_utilization_pct": overall_budget_utilization,
                "health_score": health_score
            },
            "dataframes": {
                "monthly": df_monthly,
                "budgets": df_budgets,
                "receivables": df_receivables
            }
        }

    def _calculate_health_score(self, margin_pct: float, runway_months: float, budget_utilization: float, overdue_ratio: float) -> int:
        """Weighted algorithm producing an executive health index from 0 to 100."""
        score = 50.0  # Baseline

        # Margin Weight (+/- 20 pts)
        score += np.clip(margin_pct * 0.8, -20, 20)

        # Runway Weight (+/- 20 pts)
        if runway_months >= 18:
            score += 20
        elif runway_months >= 12:
            score += 15
        elif runway_months >= 6:
            score += 5
        else:
            score -= 20

        # Budget Adherence (+/- 10 pts)
        if 80 <= budget_utilization <= 100:
            score += 10
        elif budget_utilization > 105:
            score -= 15

        # Overdue Penalty (-10 pts max)
        score -= min(10, overdue_ratio * 20)

        return int(np.clip(score, 0, 100))

    def generate_executive_charts(self, metrics: Dict[str, Any]) -> List[plt.Figure]:
        """Generates 4 tailored financial charts for executive decision-making."""
        df_m = metrics["dataframes"]["monthly"]
        df_b = metrics["dataframes"]["budgets"]
        df_r = metrics["dataframes"]["receivables"]
        figs = []

        # -------------------------------------------------------------
        # CHART 1: Profit & Loss + Revenue Trend
        # -------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(10, 4.5), facecolor="#0E1117")
        ax1.set_facecolor("#161B22")
        
        x = np.arange(len(df_m["month"]))
        width = 0.35

        rects1 = ax1.bar(x - width/2, df_m["revenue"] / 1000, width, label="Revenue ($K)", color=self.primary_color, alpha=0.9)
        rects2 = ax1.bar(x + width/2, df_m["expenses"] / 1000, width, label="Expenses ($K)", color=self.danger_color, alpha=0.85)

        net_profit_k = (df_m["revenue"] - df_m["expenses"]) / 1000
        ax1.plot(x, net_profit_k, color=self.success_color, marker="o", linewidth=2.5, label="Net Profit ($K)")

        ax1.set_title("Executive Profit & Loss (P&L) Trajectory", fontsize=13, pad=12, fontweight="bold", color="#FFFFFF")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_m["month"], color="#D0D0D0")
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}K"))
        ax1.legend(facecolor="#161B22", edgecolor="none")
        ax1.grid(axis="y", linestyle="--", alpha=0.2)
        fig1.tight_layout()
        figs.append(fig1)

        # -------------------------------------------------------------
        # CHART 2: Cash Flow & Projected Revenue Horizon
        # -------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 4.5), facecolor="#0E1117")
        ax2.set_facecolor("#161B22")

        hist_months = list(df_m["month"])
        proj_months = ["Month +1", "Month +2", "Month +3"]
        all_months = hist_months + proj_months

        hist_rev = list(df_m["revenue"] / 1000)
        proj_rev = [hist_rev[-1] * (1.03 ** i) for i in range(1, 4)]
        full_rev = hist_rev + proj_rev

        ax2.plot(hist_months, hist_rev, color=self.primary_color, marker="o", linewidth=2.5, label="Historical Revenue")
        ax2.plot(all_months[len(hist_months)-1:], full_rev[len(hist_months)-1:], color=self.accent_color, marker="s", linestyle="--", linewidth=2.5, label="Forecasted Growth")
        ax2.fill_between(all_months[len(hist_months)-1:], full_rev[len(hist_months)-1:], color=self.accent_color, alpha=0.15)

        ax2.set_title("Cash Flow Forecast Horizon (3-Month Outlook)", fontsize=13, pad=12, fontweight="bold", color="#FFFFFF")
        ax2.set_xticks(range(len(all_months)))
        ax2.set_xticklabels(all_months, rotation=30, color="#D0D0D0")
        ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}K"))
        ax2.legend(facecolor="#161B22", edgecolor="none")
        ax2.grid(True, linestyle="--", alpha=0.2)
        fig2.tight_layout()
        figs.append(fig2)

        # -------------------------------------------------------------
        # CHART 3: Departmental Budget Utilization
        # -------------------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=(10, 4.5), facecolor="#0E1117")
        ax3.set_facecolor("#161B22")

        y_pos = np.arange(len(df_b["department"]))
        allocated_k = df_b["allocated"] / 1000
        spent_k = df_b["spent"] / 1000

        ax3.barh(y_pos - 0.18, allocated_k, height=0.35, align="center", label="Allocated Budget", color="#4B5563", alpha=0.7)
        ax3.barh(y_pos + 0.18, spent_k, height=0.35, align="center", label="Spent to Date", color=self.warning_color, alpha=0.9)

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(df_b["department"], color="#D0D0D0")
        ax3.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}K"))
        ax3.set_title("Departmental Budget Allocation vs Spent", fontsize=13, pad=12, fontweight="bold", color="#FFFFFF")
        ax3.legend(facecolor="#161B22", edgecolor="none")
        ax3.grid(axis="x", linestyle="--", alpha=0.2)
        fig3.tight_layout()
        figs.append(fig3)

        # -------------------------------------------------------------
        # CHART 4: Accounts Receivable / Outstanding Payments Aging
        # -------------------------------------------------------------
        fig4, ax4 = plt.subplots(figsize=(10, 4.5), facecolor="#0E1117")
        ax4.set_facecolor("#161B22")

        status_grouped = df_r.groupby("status")["amount"].sum() / 1000
        colors = [self.success_color if s == "CURRENT" else self.warning_color if s == "PENDING" else self.danger_color for s in status_grouped.index]

        wedges, texts, autotexts = ax4.pie(
            status_grouped, 
            labels=status_grouped.index, 
            autopct="%1.1f%%", 
            colors=colors, 
            startangle=140,
            textprops=dict(color="#FFFFFF", fontweight="bold"),
            wedgeprops=dict(width=0.4, edgecolor="#0E1117", linewidth=2)  # Donut chart style
        )

        ax4.set_title("Outstanding Invoices & Receivables Risk Split", fontsize=13, pad=12, fontweight="bold", color="#FFFFFF")
        fig4.tight_layout()
        figs.append(fig4)

        return figs

    def _generate_mock_financial_data(self) -> Dict[str, Any]:
        """Fallback mock dataset generator for testing without an attached SQL ledger."""
        return {
            "current_cash_reserve": 1_850_000,
            "monthly_records": [
                {"month": "Jan", "revenue": 210000, "expenses": 160000},
                {"month": "Feb", "revenue": 240000, "expenses": 175000},
                {"month": "Mar", "revenue": 225000, "expenses": 170000},
                {"month": "Apr", "revenue": 290000, "expenses": 195000},
                {"month": "May", "revenue": 310000, "expenses": 205000},
                {"month": "Jun", "revenue": 340000, "expenses": 210000},
            ],
            "budget_allocations": [
                {"department": "Engineering", "allocated": 400000, "spent": 380000},
                {"department": "Marketing", "allocated": 200000, "spent": 215000},
                {"department": "Operations", "allocated": 150000, "spent": 120000},
                {"department": "Sales", "allocated": 250000, "spent": 230000},
            ],
            "outstanding_invoices": [
                {"client": "Alpha Corp", "amount": 45000, "status": "CURRENT"},
                {"client": "Beta LLC", "amount": 28000, "status": "PENDING"},
                {"client": "Gamma Inc", "amount": 15000, "status": "OVERDUE"},
                {"client": "Delta Co", "amount": 62000, "status": "CURRENT"},
                {"client": "Epsilon Group", "amount": 22000, "status": "OVERDUE"},
            ]
        }