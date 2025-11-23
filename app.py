import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

st.set_page_config(page_title="Abacum Outreach Scoring", page_icon="📈", layout="wide")

# Styling
st.markdown("""
<style>
.stApp { background-color: #FAFBFC; color: #1a1a2e; font-family: "Inter", sans-serif; }
[data-testid="stSidebar"] { background-color: #FFF; border-right: 1px solid #E5E7EB; }
h1 { color: #1a1a2e; border-bottom: 2px solid #7C3AED; padding-bottom: 0.5rem; display: inline-block; }
[data-testid="stMetric"] { background: #FFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1rem; }
[data-testid="stMetricValue"] { color: #7C3AED !important; font-weight: 700; }
.stButton > button { color: #FFF; background: linear-gradient(135deg, #7C3AED, #9333EA); border: none; border-radius: 8px; }
.score-driver { background: #F3F4F6; padding: 10px 14px; border-radius: 8px; margin: 6px 0; border-left: 3px solid #7C3AED; }
.outreach-angle { background: #F5F3FF; padding: 14px 18px; border-radius: 10px; border: 1px solid #DDD6FE; }
</style>
""", unsafe_allow_html=True)

# ICP config - define your ideal customer profile

# Industries that match Abacum's ICP (and how well they fit)
industry_fit = {
    "saas": 1.0,        # Perfect fit - ARR tracking, cohort analysis
    "fintech": 0.9,     # Strong fit - transaction modeling, unit economics
    "finops": 0.9,      # Strong fit - financial operations focus
    "sportstech": 0.7,  # Good fit - subscription metrics (like Strava)
    "healthtech": 0.6,  # Moderate fit - operational KPIs
    "hr tech": 0.7,     # Good fit - headcount planning
    "media": 0.6,       # Moderate fit - content ROI
}

# Stages that match Abacum's mid-market focus
stage_fit = {
    "series a": 0.7,    # Growing
    "series b": 1.0,    # Sweet spot 
    "series c": 0.9,    # Strong fit 
    "series d": 0.7,    # May have existing solutions
    "seed": 0.3,        # Too early
}

# Tools Abacum integrates with (for tech overlap scoring)
integration_partners = ["campfire", "netsuite", "quickbooks", "salesforce", "hubspot", "sage"]


# Sidebar
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload CRM + enrichment dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "mock_prospects.csv"))

st.sidebar.header("Scoring Weights")
w_industry = st.sidebar.slider("Industry Fit", 0.0, 1.0, 0.25, 0.05)
w_stage = st.sidebar.slider("Stage Fit", 0.0, 1.0, 0.15, 0.05)
w_growth = st.sidebar.slider("Growth Momentum", 0.0, 1.0, 0.25, 0.05)
w_tech = st.sidebar.slider("Tech Overlap", 0.0, 1.0, 0.20, 0.05)
w_engagement = st.sidebar.slider("Engagement Readiness", 0.0, 1.0, 0.15, 0.05)

# Auto-normalize weights to sum to 1.0
total = w_industry + w_stage + w_growth + w_tech + w_engagement
w_industry, w_stage, w_growth, w_tech, w_engagement = (
    w_industry/total, w_stage/total, w_growth/total, w_tech/total, w_engagement/total
)
st.sidebar.caption("Weights auto-normalized to sum to 1.0")


# Scoring functions
def score_industry(industry):
    #Score industry based on ICP fit. Returns 0-1.
    industry_lower = industry.lower().strip()
    for key, score in industry_fit.items():
        if key in industry_lower:
            return score
    return 0.3  # Default for unknown industries

def score_stage(stage):
    #Score funding stage based on ICP fit. Returns 0-1.
    stage_lower = stage.lower().strip()
    for key, score in stage_fit.items():
        if key in stage_lower:
            return score
    return 0.3  # Default for unknown stages

def score_tech_overlap(tech_stack):
    #Score based on how many integration partners they use. Returns 0-1.
    if pd.isna(tech_stack) or tech_stack == "":
        return 0.0
    
    tools = [t.strip().lower() for t in str(tech_stack).split(";")]
    matches = sum(1 for tool in tools if any(partner in tool for partner in integration_partners))
    
    # Score based on number of matches (0=0, 1=0.5, 2=0.75, 3+=1.0)
    if matches >= 3:
        return 1.0
    elif matches == 2:
        return 0.75
    elif matches == 1:
        return 0.5
    return 0.0

def normalize_growth(growth_series):
    #Normalize headcount growth to 0-1 scale.
    return (growth_series - growth_series.min()) / (growth_series.max() - growth_series.min() + 1e-9)


# Helpers
def get_score_drivers(row):
    #Generate plain-English explanations for what's driving the score.
    drivers = []
    
    factors = [
        ("industry_score", "Industry Fit", w_industry),
        ("stage_score", "Stage Fit", w_stage),
        ("growth_score", "Growth", w_growth),
        ("tech_score", "Tech Overlap", w_tech),
        ("engagement_score", "Engagement", w_engagement)
    ]
    
    for col, name, weight in factors:
        val = row[col]
        contribution = val * weight * 100
        
        if val >= 0.7:
            drivers.append(f"✓ {name}: Strong signal (+{contribution:.0f}pts)")
        elif val <= 0.3 and weight > 0.1:
            drivers.append(f"✗ {name}: Low signal")
    
    return drivers


def get_outreach_angle(row):
    #Generate tailored outreach recommendations based on industry and stage.
    industry = row.get("industry", "").lower()
    stage = row.get("stage", "").lower()
    tech_stack = str(row.get("tech_stack", "")).lower()
    
    angles = {
        "saas": ("ARR & Revenue Forecasting", "Automate ARR tracking, cohort analysis, and churn scenarios."),
        "fintech": ("Transaction & Payment Drivers", "Model transaction volume, payment costs, and unit economics."),
        "finops": ("Financial Operations", "Automate reporting and consolidate financial KPIs."),
        "sportstech": ("Subscription & Engagement", "Track subscriber LTV, engagement-driven revenue, and seasonal planning."),
        "healthtech": ("Operational KPIs", "Centralize clinical and financial KPIs with compliance visibility."),
        "hr tech": ("Headcount Planning & OPEX", "Streamline headcount planning and compensation modeling."),
        "media": ("Content ROI & Ad Revenue", "Model content ROI, ad revenue scenarios, and subscriber economics."),
    }
    
    # Find matching angle or use default
    focus, pitch = "FP&A Automation", "Consolidate KPIs and automate reporting."
    for key, (f, p) in angles.items():
        if key in industry:
            focus, pitch = f, p
            break
    
    # Add stage-specific modifier
    if "series a" in stage or "series b" in stage:
        pitch += " Scale finance ops without adding headcount."
    elif "series c" in stage or "series d" in stage:
        pitch += " Board-ready reporting and multi-entity consolidation."
    
    # Add tech stack integration angle if relevant
    tech_note = ""
    if "campfire" in tech_stack:
        tech_note = "Uses Campfire — pitch end-to-end AI accounting → FP&A workflow."
    elif "netsuite" in tech_stack:
        tech_note = "Uses NetSuite — emphasize seamless ERP data sync."
    elif "quickbooks" in tech_stack:
        tech_note = "Uses QuickBooks — easy migration path to scalable FP&A."
    
    return {"focus": focus, "pitch": pitch, "tech_note": tech_note}


def get_component_note(name, val, row):
    #Generate a plain English note explaining why this score is high/medium/low.
    if name == "Industry Fit":
        industry = row.get("industry", "Unknown")
        if val >= 0.7:
            return f"{industry} is a strong ICP match — similar to clients like Strava, Kajabi"
        elif val >= 0.4:
            return f"{industry} has moderate fit with our target verticals"
        else:
            return f"{industry} is outside our typical ICP"
    
    elif name == "Stage Fit":
        stage = row.get("stage", "Unknown")
        if val >= 0.7:
            return f"{stage} is our sweet spot — scaling fast, needs FP&A"
        elif val >= 0.4:
            return f"{stage} is acceptable — may need tailored approach"
        else:
            return f"{stage} is outside primary target range"
    
    elif name == "Growth":
        growth = row.get("headcount_growth_6mo", 0)
        if val >= 0.7:
            return f"{growth}% headcount growth — likely scaling finance ops"
        elif val >= 0.4:
            return f"{growth}% growth — stable but not urgent need"
        else:
            return f"{growth}% growth — may not prioritize FP&A investment"
    
    elif name == "Tech Overlap":
        tech = row.get("tech_stack", "")
        if val >= 0.7:
            return f"Uses multiple integration partners — easy adoption path"
        elif val >= 0.4:
            return f"Some compatible tools in stack — integration possible"
        else:
            return f"Limited tech overlap — may need more onboarding"
    
    else: 
        eng = row.get("engagement_score", 0)
        if val >= 0.7:
            return f"High engagement ({eng:.0%}) — recent site visits, content downloads"
        elif val >= 0.4:
            return f"Moderate engagement ({eng:.0%}) — aware of Abacum"
        else:
            return f"Low engagement ({eng:.0%}) — cold outreach needed"


# Score calculation
df = df.copy()

# Calculate scores from raw data
df["industry_score"] = df["industry"].apply(score_industry)
df["stage_score"] = df["stage"].apply(score_stage)
df["tech_score"] = df["tech_stack"].apply(score_tech_overlap)
df["growth_score"] = normalize_growth(df["headcount_growth_6mo"].astype(float))
# assuming engagement_score already comes as 0-1 from CRM

# Calculate weighted outreach score (0-100)
df["score"] = (
    w_industry * df["industry_score"] +
    w_stage * df["stage_score"] +
    w_growth * df["growth_score"] +
    w_tech * df["tech_score"] +
    w_engagement * df["engagement_score"]
) * 100

# Classify priority
def classify_priority(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Warm"
    else:
        return "Low"

df["priority"] = df["score"].apply(classify_priority)

# Generate drivers and angles for each prospect
df["drivers"] = df.apply(get_score_drivers, axis=1)
df["angle"] = df.apply(get_outreach_angle, axis=1)


# Main UI
st.title("Outreach Scoring Dashboard")
st.caption("Rank prospects using CRM insights, growth signals, tech stack fit, and enrichment data.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Prospects", len(df))
col2.metric("Avg Score", f"{df['score'].mean():.1f}")
col3.metric("High Priority", (df["priority"] == "High").sum())
col4.metric("Warm Leads", (df["priority"] == "Warm").sum())


# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Deep Dive", "📈 Analytics"])

# Tab 1: Overview Table
with tab1:
    st.subheader("Prospect Priority Overview")
    
    display_cols = ["company_name", "score", "priority", "industry", "stage", 
                    "headcount_growth_6mo", "tech_stack", "engagement_score"]
    
    st.dataframe(
        df[display_cols].sort_values("score", ascending=False),
        use_container_width=True,
        column_config={
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "company_name": "Company",
            "headcount_growth_6mo": "6mo Growth %",
            "tech_stack": "Tech Stack",
            "engagement_score": "Engagement"
        }
    )

# Tab 2: Deep Dive 
with tab2:
    st.subheader("Prospect Deep Dive")
    
    sorted_df = df.sort_values("score", ascending=False)
    selected = st.selectbox("Select Prospect", sorted_df["company_name"].tolist())
    prospect = sorted_df[sorted_df["company_name"] == selected].iloc[0]
    
    left_col, right_col = st.columns(2)
    
    # Left column: Score breakdown
    with left_col:
        st.markdown(f"### Why This Score?")
        st.markdown(f"**Overall: {prospect['score']:.0f}/100** ({prospect['priority']})")
        
        # Score components with visual bars and explanations
        components = [
            ("Industry Fit", "industry_score", w_industry),
            ("Stage Fit", "stage_score", w_stage),
            ("Growth", "growth_score", w_growth),
            ("Tech Overlap", "tech_score", w_tech),
            ("Engagement", "engagement_score", w_engagement)
        ]
        
        for name, col, weight in components:
            val = prospect[col]
            pts = val * weight * 100
            color = "#22C55E" if val >= 0.7 else "#F59E0B" if val >= 0.4 else "#EF4444"
            note = get_component_note(name, val, prospect)
            
            st.markdown(f"""
            <div style="margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;margin-bottom:2px">
                    <span><strong>{name}</strong></span>
                    <span style="color:{color};font-weight:600">+{pts:.1f}pts</span>
                </div>
                <div style="font-size:12px;color:#6B7280;margin-bottom:4px">{note}</div>
                <div style="background:#E5E7EB;border-radius:6px;height:8px">
                    <div style="background:{color};width:{val*100}%;height:100%;border-radius:6px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Key drivers summary
        st.markdown("#### Key Drivers")
        for driver in prospect["drivers"]:
            st.markdown(f"<div class='score-driver'>{driver}</div>", unsafe_allow_html=True)
    
    # Right column: Outreach angle
    with right_col:
        angle = prospect["angle"]
        
        st.markdown("### Outreach Angle")
        st.markdown(f"**Focus:** {angle['focus']}")
        st.markdown(f"<div class='outreach-angle'><strong>Pitch:</strong><br>{angle['pitch']}</div>", unsafe_allow_html=True)
        
        if angle["tech_note"]:
            st.info(angle["tech_note"])
        
        st.markdown("#### Snapshot")
        st.markdown(f"""
        - **Industry:** {prospect['industry']}
        - **Stage:** {prospect['stage']}
        - **6mo Growth:** {prospect['headcount_growth_6mo']:.0f}%
        - **Tech Stack:** {prospect['tech_stack']}
        """)

# Tab 3: Analytics Chart 
with tab3:
    st.subheader("Top Prospects by Score")
    
    top = df.sort_values("score", ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FAFBFC')
    
    # Colour by priority tier
    colors = ["#00C853" if s >= 75 else "#FFC300" if s >= 50 else "#D7263D" for s in top["score"]]
    bars = ax.bar(top["company_name"], top["score"], color=colors)
    
    ax.set_xticklabels(top["company_name"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 105)
    
    # Add score labels on bars
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
                f"{bar.get_height():.0f}", ha='center', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", alpha=0.7)
    
    legend_elements = [
        Patch(facecolor='#7C3AED', label='High (75+)'),
        Patch(facecolor='#A78BFA', label='Warm (50-74)'),
        Patch(facecolor='#DDD6FE', label='Low (<50)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    st.pyplot(fig)


# Download
st.markdown("---")

export_df = df.copy()
export_df["outreach_focus"] = export_df["angle"].apply(lambda x: x["focus"])
export_df["outreach_pitch"] = export_df["angle"].apply(lambda x: x["pitch"])
export_df["score_explanation"] = export_df["drivers"].apply(lambda x: " | ".join(x))

export_cols = ["company_name", "score", "priority", "score_explanation", "outreach_focus",
               "outreach_pitch", "industry", "stage", "headcount_growth_6mo", "tech_stack", "engagement_score"]

st.download_button(
    "Download CSV",
    export_df[export_cols].to_csv(index=False),
    "abacum_prospects.csv",
    "text/csv"
)

st.caption("Prototype for Abacum GTM/Revops.")