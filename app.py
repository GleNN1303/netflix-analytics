# app.py
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional WordCloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except Exception:
    WORDCLOUD_OK = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- THEME / CSS --------------------
st.markdown(
    """
    <style>
    /* Global */
    .main { background-color: #0f1116; color: #e6e6e6; }
    h1, h2, h3, h4 { color: #e50914; }
    /* KPI cards */
    .metric-card { background: #171a21; border: 1px solid #2a2f3a; border-radius: 16px; padding: 16px; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0b0d12; }
    /* Plotly containers */
    .stPlotlyChart { background: #0f1116; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- DATA LOAD --------------------
@st.cache_data
def load_data(path="netflix_titles.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Dates
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(
            df["date_added"].astype(str).str.strip(), errors="coerce"
        )
        df["year_added"] = df["date_added"].dt.year
        df["month_added"] = df["date_added"].dt.month
    else:
        df["year_added"] = pd.NA
        df["month_added"] = pd.NA

    # Release year numeric
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")

    # Multi-value helpers
    df["country"] = df.get("country", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
    df["listed_in"] = df.get("listed_in", pd.Series([""] * len(df))).fillna("")

    df["country_list"] = df["country"].apply(
        lambda x: [i.strip() for i in str(x).split(",")] if pd.notnull(x) else ["Unknown"]
    )
    df["genre_list"] = df["listed_in"].apply(
        lambda x: [i.strip() for i in str(x).split(",")] if pd.notnull(x) else []
    )

    # Duration parsing
    df["duration"] = df.get("duration", pd.Series([""] * len(df))).fillna("")
    # Extract minutes for movies
    df["movie_minutes"] = (
        df["duration"].str.extract(r"(\d+)\s*min", expand=False).astype(float)
    )
    # Extract seasons for TV shows
    df["seasons"] = (
        df["duration"].str.extract(r"(\d+)\s*Season", expand=False).astype(float)
    )

    # Fill important text columns
    for col in ["type", "title", "rating", "director", "cast"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df

df = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
    use_container_width=True,
)
st.sidebar.markdown("### 🎬 Netflix Analytics — Controls")

# Choose time base
use_year_added = df["year_added"].notna().any()
if use_year_added:
    ymin, ymax = int(np.nanmin(df["year_added"])), int(np.nanmax(df["year_added"]))
    yr_label = "Year Added (to Netflix)"
else:
    ymin = int(df["release_year"].dropna().min()) if df["release_year"].notna().any() else 1950
    ymax = int(df["release_year"].dropna().max()) if df["release_year"].notna().any() else 2021
    yr_label = "Release Year (fallback)"

year_range = st.sidebar.slider(yr_label, ymin, ymax, (max(ymin, ymax - 8), ymax))

types = sorted(df["type"].dropna().unique().tolist()) if "type" in df.columns else []
type_sel = st.sidebar.multiselect("Type", types, default=types)

# Build base for country/genre choices from the selected time/type
base = df.copy()
if type_sel:
    base = base[base["type"].isin(type_sel)]
if use_year_added:
    base = base[base["year_added"].between(year_range[0], year_range[1], inclusive="both")]
else:
    base = base[base["release_year"].between(year_range[0], year_range[1], inclusive="both")]

country_counts = (
    base.explode("country_list")["country_list"].replace("", "Unknown").value_counts()
)
country_sel = st.sidebar.multiselect(
    "Countries (optional)",
    options=country_counts.index.tolist()[:100],
    default=[],
)

genre_counts = (
    base.explode("genre_list")["genre_list"].replace("", "Unknown").value_counts()
)
genre_options = [g for g in genre_counts.index.tolist() if g and g != "Unknown"]
genre_sel = st.sidebar.multiselect(
    "Genres (optional)",
    options=genre_options[:100],
    default=[],
)

rating_options = sorted(df["rating"].dropna().unique().tolist()) if "rating" in df.columns else []
rating_sel = st.sidebar.multiselect("Ratings (optional)", options=rating_options, default=[])

query = st.sidebar.text_input("Search (title / cast / director)")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 💾 Export")
export_name = st.sidebar.text_input("Filename (CSV)", value="netflix_filtered.csv")

# -------------------- FILTER APPLICATION --------------------
f = df.copy()
if type_sel:
    f = f[f["type"].isin(type_sel)]
if use_year_added:
    f = f[f["year_added"].between(year_range[0], year_range[1], inclusive="both")]
else:
    f = f[f["release_year"].between(year_range[0], year_range[1], inclusive="both")]

if country_sel:
    f = f[f["country_list"].apply(lambda lst: any(c in lst for c in country_sel))]
if genre_sel:
    f = f[f["genre_list"].apply(lambda lst: any(g in lst for g in genre_sel))]
if rating_sel and "rating" in f.columns:
    f = f[f["rating"].isin(rating_sel)]

if query.strip():
    q = query.lower().strip()
    f = f[
        f["title"].str.lower().str.contains(q, na=False)
        | f["cast"].str.lower().str.contains(q, na=False)
        | f["director"].str.lower().str.contains(q, na=False)
    ]

# -------------------- HEADER & KPIs --------------------
st.title("🎬 Netflix Analytics Dashboard — Advanced")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Titles", f"{len(f):,}")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    movies = int(f[f["type"] == "Movie"].shape[0]) if "type" in f.columns else 0
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Movies", f"{movies:,}")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    shows = int(f[f["type"] == "TV Show"].shape[0]) if "type" in f.columns else 0
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("TV Shows", f"{shows:,}")
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    top_genre = (
        f.explode("genre_list")["genre_list"]
        .replace("", np.nan).dropna()
        .value_counts().idxmax()
        if len(f) and f["genre_list"].astype(bool).any()
        else "—"
    )
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Top Genre", top_genre)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- TABS --------------------
tab_overview, tab_trends, tab_regions, tab_people, tab_words, tab_table = st.tabs(
    ["📊 Overview", "📈 Trends", "🌍 Regions", "🎭 People", "☁ Genres (WordCloud)", "🗂 Data"]
)

# ---------- OVERVIEW ----------
with tab_overview:
    c1, c2 = st.columns([1.1, 1])
    # Type split
    if "type" in f.columns and len(f):
        type_counts = f["type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig = px.pie(
            type_counts, values="count", names="type",
            hole=0.45, title="Movies vs TV Shows"
        )
        c1.plotly_chart(fig, use_container_width=True)
    else:
        c1.info("No data for type distribution.")

    # Top genres
    gcounts = (
        f.explode("genre_list")["genre_list"]
        .replace("", np.nan).dropna()
        .value_counts().head(15).reset_index()
    )
    gcounts.columns = ["genre", "count"]
    if len(gcounts):
        fig = px.bar(
            gcounts, x="count", y="genre", orientation="h",
            title="Top 15 Genres", color="count", color_continuous_scale="Reds"
        )
        c2.plotly_chart(fig, use_container_width=True)
    else:
        c2.info("No genre data for current filters.")

# ---------- TRENDS ----------
with tab_trends:
    c1, c2 = st.columns([1.2, 1])

    # Year trend by type
    if use_year_added and f["year_added"].notna().any():
        trend = f.groupby(["year_added", "type"]).size().reset_index(name="count")
        fig = px.line(trend, x="year_added", y="count", color="type",
                      markers=True, title="Titles Added Over Time (Year Added)")
        c1.plotly_chart(fig, use_container_width=True)
    elif f["release_year"].notna().any():
        trend = f.groupby(["release_year", "type"]).size().reset_index(name="count")
        fig = px.line(trend, x="release_year", y="count", color="type",
                      markers=True, title="Titles by Release Year")
        c1.plotly_chart(fig, use_container_width=True)
    else:
        c1.info("No temporal data for current filters.")

    # Month x Year heatmap (needs date_added)
    if use_year_added and f["year_added"].notna().any() and f["month_added"].notna().any():
        heat = f.dropna(subset=["year_added", "month_added"])
        heat = heat.groupby(["year_added", "month_added"]).size().reset_index(name="count")
        fig = px.density_heatmap(
            heat, x="month_added", y="year_added", z="count",
            title="Heatmap: Titles Added by Year & Month",
            nbinsx=12, color_continuous_scale="Blues"
        )
        c2.plotly_chart(fig, use_container_width=True)
    else:
        c2.info("Heatmap requires valid 'date_added'.")

    # Movie duration distribution
    if f["movie_minutes"].notna().any():
        fig = px.histogram(
            f.dropna(subset=["movie_minutes"]) ,
            x="movie_minutes", nbins=40, marginal="box",
            title="Movie Duration Distribution (minutes)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No movie duration data available.")

# ---------- REGIONS ----------
with tab_regions:
    c1, c2 = st.columns([1.2, 1.1])

    c_exp = f.explode("country_list")
    c_exp["country_list"] = c_exp["country_list"].replace("", "Unknown")

    top_c = c_exp["country_list"].value_counts().head(20).reset_index()
    top_c.columns = ["country", "count"]

    if len(top_c):
        fig = px.bar(
            top_c, x="count", y="country", orientation="h",
            title="Top 20 Countries by Titles",
            color="count", color_continuous_scale="Teal"
        )
        c1.plotly_chart(fig, use_container_width=True)

        # Choropleth
        world = c_exp.groupby("country_list").size().reset_index(name="count")
        fig = px.choropleth(
            world, locations="country_list", locationmode="country names",
            color="count", color_continuous_scale="Reds",
            title="Global Distribution of Titles"
        )
        c2.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No country data for current filters.")

# ---------- PEOPLE ----------
with tab_people:
    c1, c2 = st.columns(2)

    # Actors
    casts = (
        f["cast"].dropna().astype(str).str.split(", ").explode().str.strip()
    )
    actors = casts[casts != ""].value_counts().head(20).reset_index()
    actors.columns = ["actor", "count"]

    if len(actors):
        fig = px.bar(
            actors, x="count", y="actor", orientation="h",
            title="Top 20 Actors", color="count", color_continuous_scale="Viridis"
        )
        c1.plotly_chart(fig, use_container_width=True)
    else:
        c1.info("No cast data for current filters.")

    # Directors
    directors = (
        f["director"].dropna().astype(str).str.split(", ").explode().str.strip()
    )
    directors = directors[directors != ""].value_counts().head(20).reset_index()
    directors.columns = ["director", "count"]

    if len(directors):
        fig = px.bar(
            directors, x="count", y="director", orientation="h",
            title="Top 20 Directors", color="count", color_continuous_scale="Plasma"
        )
        c2.plotly_chart(fig, use_container_width=True)
    else:
        c2.info("No director data for current filters.")

# ---------- WORDCLOUD ----------
with tab_words:
    if not WORDCLOUD_OK:
        st.warning("`wordcloud` not installed. Run `pip install wordcloud` to enable this tab.")
    else:
        # Build genre text
        genres_text = " ".join(
            f.explode("genre_list")["genre_list"].dropna().astype(str).tolist()
        )
        if genres_text.strip():
            wc = WordCloud(width=1400, height=500, background_color="black", colormap="Reds").generate(genres_text)
            st.image(wc.to_array(), caption="Genre WordCloud", use_container_width=True)
        else:
            st.info("No genre text available for WordCloud with current filters.")

# ---------- DATA TABLE + DOWNLOAD ----------
with tab_table:
    st.markdown("**Filtered Dataset Preview**")
    cols_show = [
        c for c in ["type", "title", "director", "cast", "country", "release_year", "rating", "duration", "date_added"]
        if c in f.columns
    ]
    st.dataframe(f[cols_show].reset_index(drop=True), use_container_width=True, height=480)

    csv_bytes = f[cols_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered CSV",
        data=csv_bytes,
        file_name=export_name or "netflix_filtered.csv",
        mime="text/csv",
    )
