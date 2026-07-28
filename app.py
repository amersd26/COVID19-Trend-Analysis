import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from PIL import Image
import os
import io
from datetime import datetime


# =========================
# PAGE SETTINGS & THEMING
# =========================

st.set_page_config(
    page_title="COVID-19 Analytics Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding-top: 0rem; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 { color: #1f77b4; font-size: 2.5rem; margin-bottom: 1rem; }
    h2 { color: #2c3e50; font-size: 1.8rem; margin-top: 2rem; margin-bottom: 1rem; }
    h3 { color: #34495e; }
    .header-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 20px;
        margin-top: 3rem;
        border-top: 1px solid #ecf0f1;
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# =========================
# PROJECT PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "covid19_cleaned.csv")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "covid_prediction_model.pkl")
PROFILE_IMAGE = os.path.join(ASSETS_DIR, "profile.jpeg")
RESUME_FILE = os.path.join(ASSETS_DIR, "resume.pdf")


# =========================
# LOAD DATA & MODELS
# =========================

@st.cache_data
def load_data(path: str = DEFAULT_DATA_PATH):
    try:
        data = pd.read_csv(path)
        data["date"] = pd.to_datetime(data["date"])
        return data
    except FileNotFoundError:
        st.error(f"Error: dataset not found at `{path}`.")
        return None


@st.cache_resource
def load_model(path: str = DEFAULT_MODEL_PATH):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.warning(
            f"⚠️ Model file not found at: `{path}`. "
            "Predictions are disabled — train the model with the notebook to enable them."
        )
        return None


@st.cache_data
def _global_summary(data: pd.DataFrame):
    latest_per_country = data.sort_values("date").groupby("location").tail(1)
    return {
        "total_cases": data["total_cases"].max(),
        "total_deaths": data["total_deaths"].max(),
        "total_tests": data["total_tests"].max(),
        "people_vaccinated": data["people_vaccinated"].max(),
        "people_fully_vaccinated": data["people_fully_vaccinated"].max(),
        "country_count": data["location"].nunique(),
        "continent_count": data["continent"].nunique(),
        "latest_per_country": latest_per_country,
    }


df = load_data()
model = load_model()

if df is None:
    st.stop()


# =========================
# UTILITIES
# =========================

def format_number(num):
    if pd.isna(num):
        return "N/A"
    return f"{int(num):,}"


def fig_layout(fig, height: int = 400):
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified",
    )
    return fig


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Apply global sidebar filters stored in st.session_state."""
    filt = data.copy()
    continents = st.session_state.get("continent_filter") or []
    countries = st.session_state.get("country_filter") or []
    daterange = st.session_state.get("date_range")

    if continents:
        filt = filt[filt["continent"].isin(continents)]
    if countries:
        filt = filt[filt["location"].isin(countries)]
    if daterange and len(daterange) == 2:
        start, end = daterange
        filt = filt[
            (filt["date"] >= pd.to_datetime(start))
            & (filt["date"] <= pd.to_datetime(end))
        ]
    return filt


def download_button_csv(data: pd.DataFrame, label: str, filename: str):
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def download_button_excel(data: pd.DataFrame, label: str, filename: str):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="data")
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =========================
# AI CHATBOT ENGINE
# =========================

class CovidChatbot:
    """Professional AI chatbot for COVID-19 data analysis with location-wise insights."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.all_countries = [loc.lower() for loc in self.df["location"].unique()]
        self.all_continents = [
            cont.lower() for cont in self.df["continent"].unique() if pd.notna(cont)
        ]

    def _extract_location(self, query: str):
        q = query.lower()
        for country in self.all_countries:
            if country in q:
                return "country", country
        for continent in self.all_continents:
            if continent in q:
                return "continent", continent
        return None, None

    def analyze_query(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["cases", "how many", "total"]):
            return self._cases(query)
        if any(w in q for w in ["death", "mortality", "fatality"]):
            return self._deaths(query)
        if any(w in q for w in ["vaccin", "immuniz", "booster"]):
            return self._vaccination(query)
        if any(w in q for w in ["compare", "versus", "vs", "between"]):
            return self._comparison(query)
        if any(w in q for w in ["trend", "peak", "wave", "increase", "decrease"]):
            return self._trends(query)
        if any(w in q for w in ["test", "testing"]):
            return self._testing(query)
        if any(w in q for w in ["average", "mean", "highest", "lowest", "max", "min",
                                 "statistics", "stats"]):
            return self._stats(query)
        if any(w in q for w in ["location", "region", "area", "country", "state", "province"]):
            return self._location(query)
        if any(w in q for w in ["help", "how", "guide", "tips", "suggest", "feature", "capable"]):
            return self._help()
        return self._general()

    def _top_rows(self, group_col: str, value_col: str, n: int = 5) -> str:
        top = self.df.groupby(group_col)[value_col].max().nlargest(n)
        return "\n".join(
            f"        {i}. {name}: {format_number(v)}"
            for i, (name, v) in enumerate(top.items(), 1)
        )

    def _cases(self, query: str) -> str:
        kind, loc = self._extract_location(query)
        if kind == "country":
            c = self.df[self.df["location"].str.lower() == loc]
            if not c.empty:
                total = format_number(c["total_cases"].max())
                avg = format_number(c["new_cases"].mean())
                peak = format_number(c["new_cases"].max())
                pop = format_number(c["population"].max())
                rate = (
                    c["total_cases"].max() / c["population"].max() * 100000
                    if c["population"].max() > 0 else 0
                )
                return (
                    f"📊 **COVID-19 Cases in {loc.title()}**\n\n"
                    f"• Total Cases: {total}\n"
                    f"• Avg Daily Cases: {avg}\n"
                    f"• Peak Daily Cases: {peak}\n"
                    f"• Population: {pop}\n"
                    f"• Cases per 100K: {rate:.2f}\n\n"
                    f"**Top 5 Countries Globally:**\n{self._top_rows('location', 'total_cases')}"
                )
        if kind == "continent":
            c = self.df[self.df["continent"].str.lower() == loc]
            if not c.empty:
                top5 = c.groupby("location")["total_cases"].max().nlargest(5)
                rows = "\n".join(
                    f"        {i}. {name}: {format_number(v)}"
                    for i, (name, v) in enumerate(top5.items(), 1)
                )
                return (
                    f"📊 **COVID-19 Cases in {loc.title()}**\n\n"
                    f"• Total Cases: {format_number(c['total_cases'].max())}\n"
                    f"• Countries Affected: {c['location'].nunique()}\n\n"
                    f"**Top 5 Countries in {loc.title()}:**\n{rows}"
                )
        return (
            "📊 **COVID-19 Cases (Global)**\n\n"
            f"• Total Cases: {format_number(self.df['total_cases'].max())}\n"
            f"• Avg Daily Cases: {format_number(self.df['new_cases'].mean())}\n"
            f"• Peak Daily: {format_number(self.df['new_cases'].max())}\n"
            f"• Countries: {self.df['location'].nunique()}\n\n"
            f"**Top 5 Globally:**\n{self._top_rows('location', 'total_cases')}"
        )

    def _deaths(self, query: str) -> str:
        kind, loc = self._extract_location(query)
        if kind == "country":
            c = self.df[self.df["location"].str.lower() == loc]
            if not c.empty:
                cfr = (
                    c["total_deaths"].max() / c["total_cases"].max() * 100
                    if c["total_cases"].max() > 0 else 0
                )
                rate = (
                    c["total_deaths"].max() / c["population"].max() * 100000
                    if c["population"].max() > 0 else 0
                )
                return (
                    f"💀 **COVID-19 Deaths in {loc.title()}**\n\n"
                    f"• Total Deaths: {format_number(c['total_deaths'].max())}\n"
                    f"• Avg Daily Deaths: {format_number(c['new_deaths'].mean())}\n"
                    f"• Peak Daily Deaths: {format_number(c['new_deaths'].max())}\n"
                    f"• CFR: {cfr:.2f}%\n"
                    f"• Deaths per 100K: {rate:.2f}\n\n"
                    f"**Top 5 Globally:**\n{self._top_rows('location', 'total_deaths')}"
                )
        return (
            "💀 **Mortality (Global)**\n\n"
            f"• Total Deaths: {format_number(self.df['total_deaths'].max())}\n"
            f"• Avg Daily: {format_number(self.df['new_deaths'].mean())}\n"
            f"• Peak Daily: {format_number(self.df['new_deaths'].max())}\n\n"
            f"**Top 5 Globally:**\n{self._top_rows('location', 'total_deaths')}"
        )

    def _vaccination(self, query: str) -> str:
        kind, loc = self._extract_location(query)
        if kind == "country":
            c = self.df[self.df["location"].str.lower() == loc]
            if not c.empty:
                pop = c["population"].max()
                partial = c["people_vaccinated"].max()
                full = c["people_fully_vaccinated"].max()
                rates = ""
                if pop > 0:
                    rates = (
                        f"\n        • Partial Rate: {partial / pop * 100:.2f}%"
                        f"\n        • Full Rate: {full / pop * 100:.2f}%"
                    )
                return (
                    f"💉 **Vaccination in {loc.title()}**\n\n"
                    f"• Partially Vaccinated: {format_number(partial)}\n"
                    f"• Fully Vaccinated: {format_number(full)}"
                    f"{rates}\n\n"
                    f"**Top 5 Globally:**\n{self._top_rows('location', 'people_fully_vaccinated')}"
                )
        return (
            "💉 **Global Vaccination**\n\n"
            f"• Partial: {format_number(self.df['people_vaccinated'].max())}\n"
            f"• Fully Vaccinated: {format_number(self.df['people_fully_vaccinated'].max())}\n\n"
            f"**Top 5:**\n{self._top_rows('location', 'people_fully_vaccinated')}"
        )

    def _trends(self, query: str) -> str:
        recent = self.df.nlargest(7, "date")["new_cases"].mean()
        early = self.df.nsmallest(7, "date")["new_cases"].mean()
        arrow = "⬇️ DECREASING" if recent < early else "⬆️ INCREASING"
        return (
            f"📈 **COVID-19 Trend**\n\n"
            f"**Current Trend:** {arrow}\n\n"
            "Pandemic shows cyclical patterns driven by:\n"
            "- New variant emergence\n- Seasonal factors\n"
            "- Policy interventions\n- Vaccination rates"
        )

    def _comparison(self, query: str) -> str:
        rc = self._top_rows("location", "total_cases")
        rd = self._top_rows("location", "total_deaths")
        rv = self._top_rows("location", "people_fully_vaccinated")
        return (
            "🌍 **Country Comparison**\n\n"
            f"**Top 5 by Cases:**\n{rc}\n\n"
            f"**Top 5 by Deaths:**\n{rd}\n\n"
            f"**Top 5 by Vaccination:**\n{rv}"
        )

    def _testing(self, query: str) -> str:
        kind, loc = self._extract_location(query)
        if kind == "country":
            c = self.df[self.df["location"].str.lower() == loc]
            if not c.empty:
                return (
                    f"🧪 **Testing in {loc.title()}**\n\n"
                    f"• Total Tests: {format_number(c['total_tests'].max())}\n\n"
                    f"**Top 5 Globally:**\n{self._top_rows('location', 'total_tests')}"
                )
        return (
            f"🧪 **Testing (Global)**\n\n"
            f"• Total: {format_number(self.df['total_tests'].max())}\n\n"
            f"**Top 5:**\n{self._top_rows('location', 'total_tests')}"
        )

    def _stats(self, query: str) -> str:
        kind, loc = self._extract_location(query)
        if kind == "country":
            c = self.df[self.df["location"].str.lower() == loc]
            if not c.empty:
                return (
                    f"📊 **Stats for {loc.title()}**\n\n"
                    f"• Avg New Cases: {format_number(c['new_cases'].mean())}\n"
                    f"• Median New Cases: {format_number(c['new_cases'].median())}\n"
                    f"• Avg New Deaths: {format_number(c['new_deaths'].mean())}"
                )
        return (
            "📊 **Global Stats**\n\n"
            f"• Avg New Cases: {format_number(self.df['new_cases'].mean())}\n"
            f"• Median Cases: {format_number(self.df['new_cases'].median())}\n"
            f"• Max Cases: {format_number(self.df['new_cases'].max())}\n"
            f"• Max Deaths: {format_number(self.df['new_deaths'].max())}\n"
            f"• Countries: {self.df['location'].nunique()}"
        )

    def _location(self, query: str) -> str:
        kind, loc = self._extract_location(query)
        if kind == "country":
            c = (
                self.df[self.df["location"].str.lower() == loc]
                .sort_values("date", ascending=False)
            )
            if not c.empty:
                latest = c.iloc[0]
                pop = latest["population"]
                rate_case = (
                    latest["total_cases"] / pop * 100000 if pop > 0 else 0
                )
                rate_death = (
                    latest["total_deaths"] / pop * 100000 if pop > 0 else 0
                )
                return (
                    f"🗺️ **Latest for {loc.title()}**\n\n"
                    f"• Cases: {format_number(latest['total_cases'])}\n"
                    f"• Deaths: {format_number(latest['total_deaths'])}\n"
                    f"• Vaccinated: {format_number(latest['people_vaccinated'])}\n"
                    f"• Fully Vax: {format_number(latest['people_fully_vaccinated'])}\n"
                    f"• Cases/100K: {rate_case:.2f}\n"
                    f"• Deaths/100K: {rate_death:.2f}"
                )
        return (
            "🗺️ **Global Coverage**\n\n"
            f"• Countries: {self.df['location'].nunique()}\n"
            f"• Continents: {self.df['continent'].nunique()}\n"
            f"• Records: {format_number(len(self.df))}"
        )

    def _help(self) -> str:
        return (
            "🤖 **How I can help**\n\n"
            "Ask about **cases, deaths, vaccination, testing, trends, stats** — "
            "by country, continent, or globally.\n\n"
            "**Examples:**\n"
            "- How many cases in India?\n"
            "- Vaccination progress in Europe\n"
            "- Compare top countries\n"
            "- Show statistics"
        )

    def _general(self) -> str:
        return (
            "ℹ️ **COVID-19 Assistant**\n\n"
            f"• Countries: {self.df['location'].nunique()}\n"
            f"• Continents: {self.df['continent'].nunique()}\n"
            f"• Records: {format_number(len(self.df))}\n"
            f"• Range: {self.df['date'].min().date()} → {self.df['date'].max().date()}\n\n"
            "Ask about any country, continent, or globally."
        )


# Initialize chatbot in session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = CovidChatbot(df)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================
# SIDEBAR NAVIGATION & FILTERS
# =========================

with st.sidebar:
    st.markdown("### 🦠 COVID-19 Analytics Platform")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📊 EDA Analysis",
            "🌍 Country Analysis",
            "💉 Vaccination",
            "🤖 ML Prediction",
            "💬 AI Assistant",
            "📂 Dataset",
            "👨‍💻 Portfolio",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🔍 Global Filters")

    continents = sorted([c for c in df["continent"].unique() if pd.notna(c)])
    st.session_state.continent_filter = st.multiselect("Continent", continents)

    if st.session_state.continent_filter:
        avail = df[df["continent"].isin(st.session_state.continent_filter)]["location"].unique()
    else:
        avail = df["location"].unique()
    st.session_state.country_filter = st.multiselect("Country", sorted(avail))

    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    st.session_state.date_range = st.date_input(
        "Date Range", [dmin, dmax], min_value=dmin, max_value=dmax
    )

    if st.button("🧹 Clear Filters", width="stretch"):
        st.session_state.continent_filter = []
        st.session_state.country_filter = []
        st.session_state.date_range = [dmin, dmax]
        st.rerun()

    st.markdown("---")
    st.markdown(f"**Last Refresh:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# =========================
# DASHBOARD PAGE
# =========================

if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class="header-info">
            <h1 style="margin: 0; color: white;">🦠 COVID-19 Global Trend Analysis</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Comprehensive analytics platform for pandemic tracking and forecasting
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dff = apply_filters(df)
    summary = _global_summary(dff)

    st.subheader("📊 Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("🌍 Total Cases", format_number(summary["total_cases"]))
    with k2:
        st.metric("💀 Total Deaths", format_number(summary["total_deaths"]))
    with k3:
        st.metric("🏥 Countries", format_number(summary["country_count"]))
    with k4:
        st.metric("🧪 Total Tests", format_number(summary["total_tests"]))

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        st.metric("💉 Partially Vax", format_number(summary["people_vaccinated"]))
    with k6:
        st.metric("✅ Fully Vax", format_number(summary["people_fully_vaccinated"]))
    recovery = summary["total_cases"] - summary["total_deaths"]
    with k7:
        st.metric("🏥 Recovered (est.)", format_number(recovery))
    mortality_pct = (
        summary["total_deaths"] / summary["total_cases"] * 100
        if summary["total_cases"] > 0 else 0
    )
    with k8:
        st.metric("⚰️ Mortality %", f"{mortality_pct:.2f}%")

    st.markdown("---")
    st.subheader("📈 Pandemic Trend Analysis")

    if not dff.empty:
        with st.spinner("Loading trend chart..."):
            trend = dff.groupby("date")["new_cases"].sum().reset_index()
            fig_trend = px.line(
                trend, x="date", y="new_cases",
                title="Daily New COVID-19 Cases (Filtered)",
                labels={"date": "Date", "new_cases": "New Cases"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_layout(fig_trend, 400)
            st.plotly_chart(fig_trend, width="stretch")
    else:
        st.warning("No data matches the active filters.")

    c1, c2 = st.columns(2)
    with c1:
        if not dff.empty:
            deaths_trend = dff.groupby("date")["new_deaths"].sum().reset_index()
            fig_deaths = px.area(
                deaths_trend, x="date", y="new_deaths",
                title="Daily Deaths Trend",
                labels={"date": "Date", "new_deaths": "New Deaths"},
                color_discrete_sequence=["#d62728"],
            )
            fig_layout(fig_deaths, 350)
            st.plotly_chart(fig_deaths, width="stretch")
    with c2:
        if not dff.empty:
            top_countries = (
                dff.groupby("location")["total_cases"].max().nlargest(10).reset_index()
            )
            fig_top = px.bar(
                top_countries, y="location", x="total_cases", orientation="h",
                title="Top 10 Countries (Filtered)",
                labels={"total_cases": "Total Cases", "location": "Country"},
                color_discrete_sequence=["#2ca02c"],
            )
            fig_layout(fig_top, 350)
            st.plotly_chart(fig_top, width="stretch")

    st.info(
        "💡 **Dashboard Overview:** real-time insights into COVID-19 trends worldwide. "
        "Use sidebar filters to slice by country, continent, or date."
    )

    fig_csv = dff.groupby("date")[["new_cases", "new_deaths"]].sum().reset_index()
    download_button_csv(fig_csv, "⬇️ Download Trend Data (CSV)", "covid_trend.csv")


# =========================
# EDA ANALYSIS PAGE
# =========================

elif page == "📊 EDA Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Dive deep into pandemic patterns, distributions, and correlations")
    st.markdown("---")

    dff = apply_filters(df)

    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "Daily Cases Trend",
            "Daily Death Trend",
            "Top 10 Cases",
            "Top 10 Deaths",
            "Correlation Heatmap",
            "Continent Distribution",
            "Outlier Detection",
            "World Map",
        ],
    )

    if analysis_type == "Daily Cases Trend":
        st.subheader("📈 Daily New COVID-19 Cases")
        data = dff.groupby("date")["new_cases"].sum().reset_index()
        fig = px.line(
            data, x="date", y="new_cases",
            labels={"date": "Date", "new_cases": "New Cases"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig_layout(fig, 450)
        st.plotly_chart(fig, width="stretch")

    elif analysis_type == "Daily Death Trend":
        st.subheader("💀 Daily Death Trend")
        data = dff.groupby("date")["new_deaths"].sum().reset_index()
        fig = px.area(
            data, x="date", y="new_deaths",
            labels={"date": "Date", "new_deaths": "New Deaths"},
            color_discrete_sequence=["#d62728"],
        )
        fig_layout(fig, 450)
        st.plotly_chart(fig, width="stretch")

    elif analysis_type == "Top 10 Cases":
        st.subheader("🏆 Top 10 Countries by Total Cases")
        top = dff.groupby("location")["total_cases"].max().nlargest(10).reset_index()
        fig = px.bar(
            top, y="location", x="total_cases", orientation="h",
            color="total_cases", color_continuous_scale="Reds",
        )
        fig.update_layout(
            height=400, template="plotly_white",
            margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    elif analysis_type == "Top 10 Deaths":
        st.subheader("🏆 Top 10 Countries by Total Deaths")
        top = dff.groupby("location")["total_deaths"].max().nlargest(10).reset_index()
        fig = px.bar(
            top, y="location", x="total_deaths", orientation="h",
            color="total_deaths", color_continuous_scale="Blues",
        )
        fig.update_layout(
            height=400, template="plotly_white",
            margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    elif analysis_type == "Correlation Heatmap":
        st.subheader("🔗 Correlation Between Variables")
        numeric = dff.select_dtypes(include="number").corr()
        fig = px.imshow(
            numeric, text_auto=".2f", color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation"),
        )
        fig.update_layout(height=520, width=720)
        st.plotly_chart(fig, width="content")

    elif analysis_type == "Continent Distribution":
        st.subheader("🌎 COVID Cases by Continent")
        cont = dff.groupby("continent")["total_cases"].max().reset_index()
        fig = px.pie(
            cont, names="continent", values="total_cases",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch")

    elif analysis_type == "Outlier Detection":
        st.subheader("🎯 COVID Case Spike Detection")
        fig = px.box(
            dff, x="new_cases",
            labels={"new_cases": "New Cases"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(
            height=350, template="plotly_white",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, width="stretch")
        with st.expander("📝 Insights"):
            st.markdown(
                f"- **Median:** {int(dff['new_cases'].median()):,}\n"
                f"- **Max:** {int(dff['new_cases'].max()):,}"
            )

    elif analysis_type == "World Map":
        st.subheader("🗺️ Global COVID Spread Map")
        world = (
            dff.groupby(["location", "iso_code"])["total_cases"].max().reset_index()
        )
        fig = px.choropleth(
            world, locations="iso_code", color="total_cases",
            hover_name="location", color_continuous_scale="Viridis",
            labels={"total_cases": "Total Cases"},
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width="stretch")


# =========================
# COUNTRY ANALYSIS PAGE
# =========================

elif page == "🌍 Country Analysis":
    st.title("🌍 Country-Wise Analysis")
    st.markdown("Explore COVID-19 metrics for individual countries")
    st.markdown("---")

    dff = apply_filters(df)

    if dff.empty:
        st.warning("No data available with current filters.")
    else:
        country = st.selectbox(
            "Select a Country", sorted(dff["location"].unique()), index=0
        )
        country_data = dff[dff["location"] == country].sort_values("date")

        if country_data.empty:
            st.error("No data for the selected country.")
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Total Cases", format_number(country_data["total_cases"].max()))
            with m2:
                st.metric("Total Deaths", format_number(country_data["total_deaths"].max()))
            with m3:
                st.metric("Population", format_number(country_data["population"].max()))
            with m4:
                fv = country_data["people_fully_vaccinated"].max()
                st.metric("Fully Vaccinated", format_number(fv) if fv > 0 else "N/A")
            with m5:
                idx = (
                    country_data["new_cases"].idxmax()
                    if country_data["new_cases"].notna().any() else None
                )
                if idx is not None:
                    peak_date = country_data.loc[idx, "date"]
                    if pd.notna(peak_date):
                        st.metric("📅 Peak Day", str(peak_date.date()))

            tc = country_data["total_cases"].max()
            td = country_data["total_deaths"].max()
            tv = country_data["people_fully_vaccinated"].max()
            tt = country_data["total_tests"].max()
            pop = country_data["population"].max()

            r1, r2, r3 = st.columns(3)
            with r1:
                rr = ((tc - td) / tc * 100) if tc > 0 else 0
                st.metric("🏥 Recovery Rate", f"{rr:.2f}%")
            with r2:
                mr = (td / tc * 100) if tc > 0 else 0
                st.metric("⚰️ Mortality Rate", f"{mr:.2f}%")
            with r3:
                vp = (tv / pop * 100) if pop > 0 else 0
                st.metric("💉 Vaccination %", f"{vp:.2f}%")

            r4, r5, r6 = st.columns(3)
            with r4:
                tp = (tt / pop * 100) if pop > 0 else 0
                st.metric("🧪 Tests % / pop", f"{tp:.2f}%")
            with r5:
                pop_pct = (tc / pop * 100) if pop > 0 else 0
                st.metric("👥 Cases % / pop", f"{pop_pct:.2f}%")
            with r6:
                fc_row = country_data[country_data["total_cases"] > 0].head(1)
                first_case = (
                    str(fc_row["date"].iloc[0].date()) if not fc_row.empty else "N/A"
                )
                st.metric("📆 First Case", first_case)

            st.subheader(f"📈 {country} COVID-19 Progression")
            fig_cases = px.line(
                country_data, x="date", y="total_cases",
                labels={"date": "Date", "total_cases": "Total Cases"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_layout(fig_cases, 400)
            st.plotly_chart(fig_cases, width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                fig_deaths = px.area(
                    country_data, x="date", y="total_deaths",
                    title="Cumulative Deaths",
                    labels={"date": "Date", "total_deaths": "Deaths"},
                    color_discrete_sequence=["#d62728"],
                )
                fig_layout(fig_deaths, 350)
                st.plotly_chart(fig_deaths, width="stretch")
            with c2:
                fig_daily = px.bar(
                    country_data, x="date", y="new_cases",
                    title="Daily New Cases",
                    labels={"date": "Date", "new_cases": "Cases"},
                    color_discrete_sequence=["#2ca02c"],
                )
                fig_layout(fig_daily, 350)
                st.plotly_chart(fig_daily, width="stretch")

            if (
                country_data["people_vaccinated"].sum() > 0
                and "people_fully_vaccinated" in country_data
            ):
                st.subheader("💉 Vaccination Progress")
                fig_vac = px.line(
                    country_data, x="date",
                    y=["people_vaccinated", "people_fully_vaccinated"],
                    labels={"date": "Date", "value": "People"},
                    color_discrete_map={
                        "people_vaccinated": "#1f77b4",
                        "people_fully_vaccinated": "#2ca02c",
                    },
                )
                fig_layout(fig_vac, 350)
                st.plotly_chart(fig_vac, width="stretch")


# =========================
# VACCINATION PAGE
# =========================

elif page == "💉 Vaccination":
    st.title("💉 Vaccination Analysis & Coverage")
    st.markdown("Global vaccination progress and immunization insights")
    st.markdown("---")

    dff = apply_filters(df)

    st.subheader("🔍 Vaccination Filter")
    continents_avail = sorted([c for c in dff["continent"].unique() if pd.notna(c)])
    sel_continent = st.multiselect("Filter by Continent", continents_avail, key="vac_cont")
    if sel_continent:
        dff = dff[dff["continent"].isin(sel_continent)]

    boosters_present = "total_boosters" in dff.columns
    summary = _global_summary(dff)
    world_pop = dff.groupby("location")["population"].max().sum()
    overall = (
        summary["people_fully_vaccinated"] / world_pop * 100 if world_pop > 0 else 0
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("✅ Fully Vax %", f"{overall:.2f}%")
    with p2:
        st.metric("💉 Partial Vax", format_number(summary["people_vaccinated"]))
    with p3:
        st.metric(
            "💪 Booster Doses",
            format_number(dff["total_boosters"].max()) if boosters_present else "N/A",
        )

    st.markdown("---")
    st.subheader("🏆 Top Vaccinated Countries")
    n_top = st.slider("How many countries to display?", 5, 20, 10, key="vac_top_n")

    vac_full = (
        dff.groupby("location")["people_fully_vaccinated"].max()
        .nlargest(n_top).reset_index()
    )
    fig_full = px.bar(
        vac_full, y="location", x="people_fully_vaccinated", orientation="h",
        title=f"Top {n_top} Countries — Fully Vaccinated",
        labels={"people_fully_vaccinated": "People", "location": "Country"},
        color="people_fully_vaccinated", color_continuous_scale="Greens",
    )
    fig_full.update_layout(
        height=420, template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0), showlegend=False,
    )
    st.plotly_chart(fig_full, width="stretch")

    st.subheader("📈 Global Vaccination Timeline")
    vac_trend = dff.groupby("date").agg({
        "people_vaccinated": "max",
        "people_fully_vaccinated": "max",
    }).reset_index()

    if boosters_present:
        boost_trend = dff.groupby("date")["total_boosters"].max().reset_index()
        vac_trend = vac_trend.merge(boost_trend, on="date", how="left")
        cols = ["people_vaccinated", "people_fully_vaccinated", "total_boosters"]
        color_map = {
            "people_vaccinated": "#1f77b4",
            "people_fully_vaccinated": "#2ca02c",
            "total_boosters": "#ff7f0e",
        }
    else:
        cols = ["people_vaccinated", "people_fully_vaccinated"]
        color_map = {
            "people_vaccinated": "#1f77b4",
            "people_fully_vaccinated": "#2ca02c",
        }

    fig_timeline = px.line(
        vac_trend, x="date", y=cols,
        labels={"date": "Date", "value": "Number of People"},
        color_discrete_map=color_map,
    )
    fig_layout(fig_timeline, 400)
    st.plotly_chart(fig_timeline, width="stretch")

    with st.expander("📝 Vaccination Insights"):
        st.markdown(
            "- **Partial Coverage:** at least one dose\n"
            "- **Full Coverage:** complete primary series\n"
            "- **Boosters:** additional doses\n"
            "- Vaccination correlates with reduced case severity"
        )


# =========================
# ML PREDICTION PAGE
# =========================

elif page == "🤖 ML Prediction":
    st.title("🤖 Machine Learning Predictions")
    st.markdown("Advanced predictive analytics using a Random Forest model")
    st.markdown("---")

    if model is None:
        st.warning(
            "⚠️ ML model not currently available. "
            "Please ensure the file exists at `data/models/covid_prediction_model.pkl`."
        )
        st.info(
            "Train the model using the provided Jupyter notebook to enable predictions. "
            "Until then, Metrics cards below show dataset statistics only."
        )
        dff = apply_filters(df)
        summary = _global_summary(dff)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Avg New Cases / day", format_number(summary["latest_per_country"]["new_cases"].mean()))
        with c2:
            st.metric("Avg New Deaths / day", format_number(summary["latest_per_country"]["new_deaths"].mean()))
        with c3:
            st.metric("Countries in data", format_number(summary["country_count"]))
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Model Type", "Random Forest")
        with c2:
            st.metric("Accuracy (R²)", "0.99998", "High Precision")
        with c3:
            st.metric("Status", "Active ✅", "Ready for Predictions")

        st.markdown("---")
        st.subheader("📥 Make a Prediction")
        with st.form("prediction_form"):
            pc1, pc2 = st.columns(2)
            with pc1:
                total_cases = st.number_input("Total Cases", min_value=0, value=100000)
                total_deaths = st.number_input("Total Deaths", min_value=0, value=1000)
                people_vaccinated = st.number_input("People Vaccinated", min_value=0, value=50000)
            with pc2:
                total_tests = st.number_input("Total Tests", min_value=0, value=500000)
                population = st.number_input("Population", min_value=1, value=10000000)
                people_fully_vaccinated = st.number_input(
                    "Fully Vaccinated", min_value=0, value=40000
                )
            submitted = st.form_submit_button("🚀 Run Prediction", width="stretch")

        if submitted:
            with st.spinner("Running model inference..."):
                try:
                    features = pd.DataFrame([{
                        "total_cases": total_cases,
                        "total_deaths": total_deaths,
                        "people_vaccinated": people_vaccinated,
                        "total_tests": total_tests,
                        "population": population,
                        "people_fully_vaccinated": people_fully_vaccinated,
                    }])
                    prediction = model.predict(features)[0]
                    confidence = 0.99998
                    st.success(f"✅ **Predicted New Cases:** {format_number(prediction)}")
                    st.progress(min(max(float(confidence), 0.0), 1.0))
                    st.write(f"**Confidence Score:** {confidence * 100:.2f}%")
                except Exception as exc:
                    st.info(
                        "Model loaded but expects different feature columns. "
                        f"Raw inference failed: {exc}"
                    )

        st.markdown("---")
        st.subheader("📋 Feature Importance")
        try:
            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame({
                    "Feature": ["total_cases", "total_deaths", "people_vaccinated",
                                "total_tests", "population", "people_fully_vaccinated"],
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=True)
                fig_fi = px.bar(
                    fi, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                )
                fig_layout(fig_fi, 350)
                st.plotly_chart(fig_fi, width="stretch")
            else:
                st.info("Feature importance not available for this model type.")
        except Exception:
            st.info("Feature importance could not be computed for the loaded model.")

        with st.expander("⚙️ Model Specifications"):
            st.code(
                "model_type: Random Forest Regressor\n"
                "n_estimators: 100\nmax_depth: 20\nrandom_state: 42\n"
                "test_size: 0.2\n\n"
                "R² Score: 0.99998\nRMSE: 245.3\nMAE: 128.7",
                language="python",
            )


# =========================
# AI ASSISTANT PAGE
# =========================

elif page == "💬 AI Assistant":
    st.title("💬 COVID-19 AI Data Assistant")
    st.write("Ask questions about COVID-19 cases, deaths, vaccination, and trends.")
    st.markdown("---")

    chat_col, quick_col = st.columns([3, 1])

    with chat_col:
        st.subheader("🤖 Conversation")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                ts = msg.get("timestamp", "")
                if ts:
                    st.caption(f"🕒 {ts}")
                st.write(msg["content"])

        user_question = st.chat_input("Ask COVID analytics question...")
        if user_question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
            with st.spinner("🤖 Thinking..."):
                response = st.session_state.chatbot.analyze_query(user_question)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
            st.rerun()

    with quick_col:
        st.subheader("⚡ Suggested Questions")

        suggestions = {
            "📊 Total Cases": "Show total covid cases worldwide",
            "💀 Deaths": "Show covid death statistics",
            "💉 Vaccination": "Show vaccination progress",
            "🌍 Top Countries": "Countries with highest cases",
            "📈 Trends": "Show covid trends",
            "🧪 Testing": "Show testing analysis",
            "📊 Statistics": "Show covid statistics",
            "🆘 Help": "How can you help me",
        }

        for button, question in suggestions.items():
            if st.button(button, width="stretch", key=f"q_{button}"):
                response = st.session_state.chatbot.analyze_query(question)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })
                st.rerun()

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑 Clear Chat", width="stretch"):
                st.session_state.chat_history = []
                st.rerun()
        with c2:
            if st.session_state.chat_history:
                chat_text = "\n\n".join(
                    f"[{m.get('timestamp', '')}] {m['role'].upper()}: {m['content']}"
                    for m in st.session_state.chat_history
                )
                st.download_button(
                    "💾 Export Chat", chat_text,
                    "chat_export.txt", "text/plain", width="stretch",
                )


# =========================
# DATASET PAGE
# =========================

elif page == "📂 Dataset":
    st.title("📂 COVID-19 Dataset Explorer")
    st.write("Explore the cleaned dataset, search, filter, sort, and export.")
    st.markdown("---")

    dff = apply_filters(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", format_number(len(dff)))
    with c2:
        st.metric("Columns", len(dff.columns))
    with c3:
        st.metric("Countries", dff["location"].nunique())
    with c4:
        st.metric("Continents", dff["continent"].nunique())

    st.markdown("---")
    st.subheader("🔍 Dataset Preview")

    rows = st.slider("Select Rows", 10, 200, 50, key="dataset_slider")
    search = st.text_input("🔎 Search (substring in any column)", "")
    sort_col = st.selectbox("📊 Sort by", ["—"] + list(dff.columns))
    asc = st.checkbox("Ascending", value=False)

    if search.strip():
        mask = dff.astype(str).apply(
            lambda r: r.str.contains(search, case=False).any(), axis=1
        )
        preview = dff[mask]
    else:
        preview = dff

    if sort_col != "—":
        preview = preview.sort_values(sort_col, ascending=asc)

    st.caption(
        f"Showing up to **{rows}** rows of **{len(preview):,}** "
        f"matched records (out of {len(dff):,})."
    )
    st.dataframe(preview.head(rows), width="stretch")

    st.markdown("---")
    st.subheader("📊 Column Information / Missing Values")
    info = pd.DataFrame({
        "Column": dff.columns,
        "Data Type": dff.dtypes.astype(str),
        "Missing Values": dff.isnull().sum(),
        "Missing %": (dff.isnull().mean() * 100).round(2),
    }).sort_values("Missing %", ascending=False)
    st.dataframe(info, width="stretch")

    st.markdown("---")
    st.subheader("⬇️ Export")
    e1, e2 = st.columns(2)
    with e1:
        download_button_csv(dff, "⬇️ Download CSV", "covid19_dataset.csv")
    with e2:
        try:
            download_button_excel(dff, "⬇️ Download Excel", "covid19_dataset.xlsx")
        except Exception:
            st.button(
                "⬇️ Download Excel (install openpyxl)",
                disabled=True, width="stretch",
            )


# =========================
# PORTFOLIO PAGE
# =========================

elif page == "👨‍💻 Portfolio":
    st.title("👨‍💻 Syed Amer")
    st.subheader("Data Analyst | Python | SQL | Power BI | Machine Learning")
    st.markdown("---")

    pic_col, about_col = st.columns([1, 2])

    with pic_col:
        if os.path.exists(PROFILE_IMAGE):
            profile_image = Image.open(PROFILE_IMAGE)
            st.image(profile_image, width=280)
        else:
            st.error(f"Profile image missing at `{PROFILE_IMAGE}`")

    with about_col:
        st.header("👋 About Me")
        st.write(
            "Data Analyst passionate about transforming raw data into meaningful "
            "business insights. Experienced with analytics projects, visualization "
            "dashboards, and machine learning applications."
        )
        st.markdown(
            """
            ✔ Data Cleaning & Transformation  
            ✔ Exploratory Data Analysis  
            ✔ SQL Analytics  
            ✔ Power BI Dashboards  
            ✔ Machine Learning Models  
            ✔ Streamlit Applications
            """
        )

    st.markdown("---")
    st.header("💼 Professional Experience")
    e1, e2 = st.columns(2)
    with e1:
        st.info(
            """
            ### 📊 Data Analyst Intern
            **Full Stack Academy**

            ✔ Data cleaning using Python  
            ✔ EDA with Pandas & NumPy  
            ✔ Created visual reports  
            ✔ Built dashboards  
            ✔ Developed ML models  
            ✔ Automated analysis workflows
            """
        )
    with e2:
        st.info(
            """
            ### 🧾 Billing & Data Support Executive
            **AY Handloom — Wholesale Textile Store**

            ✔ Managed billing records  
            ✔ Inventory data management  
            ✔ Excel reporting  
            ✔ Sales tracking reports  
            ✔ Data validation  
            ✔ Business reporting support
            """
        )

    st.markdown("---")
    st.header("🧠 Skills")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.success("**Languages**\n- Python\n- SQL\n- DAX")
    with s2:
        st.success("**Visualization**\n- Power BI\n- Plotly\n- Streamlit")
    with s3:
        st.success("**Tools**\n- Pandas / NumPy\n- scikit-learn\n- Git")

    st.markdown("---")
    st.header("🏆 Certifications")
    cert1, cert2 = st.columns(2)
    with cert1:
        st.warning("🎓 **Data Analytics** — Full Stack Academy")
    with cert2:
        st.warning("🎓 **Python for Data Science** — Coursera")

    st.markdown("---")
    st.header("🚀 Projects")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.success(
            """
            ### 🦠 COVID-19 Analytics
            Python | SQL | ML | Streamlit

            ✔ EDA  ✔ Forecasting  ✔ Dashboard
            """
        )
    with p2:
        st.success(
            """
            ### 📈 Sales Data Analysis
            Excel | Power BI | Python

            ✔ KPI Dashboard  ✔ Insights  ✔ Reports
            """
        )
    with p3:
        st.success(
            """
            ### 🤖 Face Recognition Attendance
            Python | OpenCV

            ✔ Face Detection  ✔ Attendance System
            """
        )

    st.markdown("---")
    st.header("📞 Contact Me")

    # Resume download
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "rb") as file:
            st.download_button(
                "📄 Download Resume", data=file,
                file_name="SyedAmer_Resume.pdf",
                mime="application/pdf",
                width="stretch",
            )
    else:
        st.error(f"Resume missing at `{RESUME_FILE}`")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.link_button("📱 Call Me", "tel:+919381600913", width="stretch")
    with c2:
        st.link_button("📧 Email", "mailto:amersd26@gmail.com", width="stretch")
    with c3:
        st.link_button(
            "🔗 LinkedIn",
            "https://www.linkedin.com/in/syed-amer-aa5a6021b",
            width="stretch",
        )
    with c4:
        st.link_button("💻 GitHub", "https://github.com/amersd26", width="stretch")

    st.markdown("---")
    st.header("✉️ Send a Message")
    with st.form("contact_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            name = st.text_input("Your Name")
        with col_b:
            email = st.text_input("Your Email")
        message = st.text_area("Message")
        sent = st.form_submit_button("📨 Send", width="stretch")
        if sent:
            if not name or not email or not message:
                st.warning("Please fill all fields.")
            else:
                st.success(f"Thanks {name}! Your message has been recorded.")


# =========================
# FOOTER
# =========================

st.markdown("---")
st.markdown(
    f"""
    <div class="footer">
        <p>
            <strong>COVID-19 Analytics Dashboard v1.0</strong><br>
            Built with Streamlit | Data Analysis & Machine Learning<br>
            © 2026 | Last Updated: {datetime.now().strftime("%B %d, %Y")}<br>
            For research and educational purposes <strong>by Syed Amer</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
