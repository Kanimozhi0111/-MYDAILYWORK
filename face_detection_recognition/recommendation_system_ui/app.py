import pandas as pd
import streamlit as st


@st.cache_data
def load_items() -> pd.DataFrame:
    df = pd.read_csv("data/items.csv")
    return df


def score_items(
    df: pd.DataFrame,
    preferred_genres: list[str],
    selected_mood: str,
    preferred_language: str,
    min_rating: float,
    year_range: tuple[int, int],
    runtime_range: tuple[int, int],
    platforms: list[str],
) -> pd.DataFrame:
    working = df.copy()
    working["score"] = 0.0

    if preferred_genres:
        working["genre_match"] = working["genre"].apply(
            lambda value: len(set(value.split("|")) & set(preferred_genres))
        )
        working["score"] += working["genre_match"] * 4.0
    else:
        working["genre_match"] = 0

    working["mood_match"] = (working["mood"] == selected_mood).astype(int)
    working["language_match"] = (working["language"] == preferred_language).astype(int)
    working["score"] += working["mood_match"] * 2.2
    working["score"] += working["language_match"] * 1.6
    working["score"] += working["rating"] * 1.3

    filtered = working[
        (working["rating"] >= min_rating)
        & (working["year"] >= year_range[0])
        & (working["year"] <= year_range[1])
        & (working["duration_min"] >= runtime_range[0])
        & (working["duration_min"] <= runtime_range[1])
    ]

    if platforms:
        filtered = filtered[filtered["platform"].isin(platforms)]

    return filtered.sort_values(by=["score", "rating"], ascending=False)


def recommendation_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["genre_match"] > 0:
        reasons.append("strong genre match")
    if row["mood_match"] == 1:
        reasons.append("matches your mood")
    if row["language_match"] == 1:
        reasons.append("preferred language")
    reasons.append("high viewer rating")
    return ", ".join(reasons).capitalize() + "."


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f5f7fb 0%, #eef2f9 100%);
        }
        .block-container {
            padding-top: 1.5rem;
        }
        .hero-box {
            background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 100%);
            padding: 1.3rem 1.4rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1rem;
        }
        .reason-pill {
            background: #e0e7ff;
            color: #1e3a8a;
            padding: 0.3rem 0.55rem;
            border-radius: 999px;
            display: inline-block;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Recommendation Studio", page_icon="🎬", layout="wide")
    apply_custom_style()
    items = load_items()

    st.markdown(
        """
        <div class="hero-box">
            <h2 style="margin: 0;">Recommendation Studio</h2>
            <p style="margin: 0.35rem 0 0 0;">
                Professional movie recommendation app with advanced filters,
                ranking transparency, and live catalog insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Personalization")
        all_genres = sorted(
            {genre for value in items["genre"] for genre in value.split("|")}
        )
        preferred_genres = st.multiselect(
            "Preferred genres",
            all_genres,
            default=["Drama", "Sci-Fi"],
        )
        selected_mood = st.selectbox("Current mood", sorted(items["mood"].unique()))
        preferred_language = st.selectbox(
            "Preferred language", sorted(items["language"].unique())
        )
        min_rating = st.slider("Minimum rating", 1.0, 5.0, 4.0, 0.1)
        year_range = st.slider(
            "Release year range",
            int(items["year"].min()),
            int(items["year"].max()),
            (2010, int(items["year"].max())),
        )
        runtime_range = st.slider("Runtime (minutes)", 80, 210, (90, 180))
        platforms = st.multiselect(
            "Watch platforms",
            sorted(items["platform"].unique()),
            default=["Netflix", "Prime Video"],
        )
        top_n = st.slider("Number of recommendations", 3, 15, 8)

    recommended = score_items(
        items,
        preferred_genres=preferred_genres,
        selected_mood=selected_mood,
        preferred_language=preferred_language,
        min_rating=min_rating,
        year_range=year_range,
        runtime_range=runtime_range,
        platforms=platforms,
    ).head(top_n)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Catalog Size", f"{len(items)} titles")
    metric_col2.metric("Recommendations", f"{len(recommended)} found")
    metric_col3.metric("Avg Catalog Rating", f"{items['rating'].mean():.2f}")
    metric_col4.metric("Languages", items["language"].nunique())

    tab_reco, tab_insights, tab_catalog = st.tabs(
        ["Top Picks", "Insights", "Full Catalog"]
    )

    with tab_reco:
        if recommended.empty:
            st.warning("No items match this profile. Relax one or more filters.")
        else:
            for _, row in recommended.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([3.2, 1.8])
                    c1.markdown(f"### {row['title']} ({row['year']})")
                    c1.caption(row["description"])
                    c1.markdown(
                        f"<span class='reason-pill'>{recommendation_reason(row)}</span>",
                        unsafe_allow_html=True,
                    )
                    c2.write(
                        f"**Genre**: {row['genre']}  \n"
                        f"**Mood**: {row['mood']}  \n"
                        f"**Language**: {row['language']}  \n"
                        f"**Duration**: {row['duration_min']} min  \n"
                        f"**Platform**: {row['platform']}  \n"
                        f"**Rating**: ⭐ {row['rating']}  \n"
                        f"**Score**: {row['score']:.2f}"
                    )

    with tab_insights:
        st.subheader("Catalog Trends")
        genre_counts = (
            items.assign(primary_genre=items["genre"].str.split("|").str[0])[
                "primary_genre"
            ]
            .value_counts()
            .rename_axis("Genre")
            .reset_index(name="Titles")
        )
        st.bar_chart(genre_counts.set_index("Genre"))

        lang_counts = (
            items["language"].value_counts().rename_axis("Language").reset_index(name="Titles")
        )
        st.bar_chart(lang_counts.set_index("Language"))

    with tab_catalog:
        st.dataframe(
            items.sort_values(by="rating", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
