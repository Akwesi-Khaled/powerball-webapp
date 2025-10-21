import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import powerball_auto_analysis as pba

# -------------------- APP CONFIG --------------------
st.set_page_config(
    page_title="Powerball Analyzer",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- SIDEBAR --------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/99/Powerball_logo.svg", width=150)
st.sidebar.title("🎯 Powerball Analyzer")
st.sidebar.markdown("Analyze historical Powerball draws and visualize data trends — for fun!")

st.sidebar.markdown("---")
st.sidebar.info("**Disclaimer:** This app is for **educational and entertainment** purposes only — not for predicting or gambling.")
st.sidebar.markdown("Made with ❤️ by **Akwesi Khaled | Cinemagic Studios**")

# -------------------- MAIN CONTENT --------------------
st.title("🎰 Powerball Data Analyzer Dashboard")
st.markdown("Welcome! Click below to fetch real Powerball results and explore data-driven insights, trends, and visual patterns.")

if st.button("🚀 Fetch & Analyze Data"):
    with st.spinner("Fetching and analyzing Powerball data... ⏳"):
        try:
            # Fetch + preprocess + analyze
            df_raw = pba.fetch_powerball_data()
            df = pba.preprocess(df_raw)
            results, fig = pba.analyze(df)

            st.success("✅ Analysis complete!")

            # ---------- FREQUENCY OVERVIEW ----------
            st.subheader("📊 Frequency Overview")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### White Ball Frequencies")
                st.bar_chart(results["white_freq_series"])
            with col2:
                st.markdown("### Powerball Frequencies")
                st.bar_chart(results["red_freq_series"])

            # ---------- CUSTOM VISUAL: SIDE-BY-SIDE ----------
            st.subheader("🎨 Combined Frequency Visualization")

            freq_fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(results["white_freq_series"].index, results["white_freq_series"].values, label="White Balls", alpha=0.7)
            ax.bar(
                results["red_freq_series"].index,
                results["red_freq_series"].values / max(results["red_freq_series"].values) * max(results["white_freq_series"].values),
                label="Powerballs (scaled)",
                alpha=0.7,
                color="red"
            )
            ax.legend()
            ax.set_title("White Balls vs Powerballs Frequency Comparison")
            ax.set_xlabel("Ball Number")
            ax.set_ylabel("Frequency (scaled)")
            st.pyplot(freq_fig)

            # ---------- TRENDS OVER TIME ----------
            st.subheader("📈 Frequency Trend Over Time")

            white_numbers = df[['white1', 'white2', 'white3', 'white4', 'white5']].melt(value_name='number')
            white_trend = white_numbers['number'].value_counts().sort_index().reset_index()
            white_trend.columns = ['Ball', 'Frequency']

            trend_fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=white_trend, x='Ball', y='Frequency', color='blue', ax=ax)
            ax.set_title("Trend of White Ball Frequencies")
            ax.set_xlabel("White Ball Number (1–69)")
            ax.set_ylabel("Occurrences")
            st.pyplot(trend_fig)

            # ---------- HEATMAP ----------
            st.subheader("🔥 Hot vs Cold Heatmap")

            heat_data = pd.DataFrame({
                "White Balls": results["white_freq_series"],
                "Powerballs": results["red_freq_series"].reindex(range(1, 70), fill_value=0)
            })

            heat_fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(heat_data.T, cmap="coolwarm", cbar=True, ax=ax)
            ax.set_title("Heatmap of Hot (Red) and Cold (Blue) Numbers")
            st.pyplot(heat_fig)

            # ---------- RANDOMNESS TEST ----------
            st.subheader("🎲 Randomness Check")
            st.markdown(
                f"""
                **White Balls:** χ² = `{results['chi_square_white']['statistic']:.2f}` | p = `{results['chi_square_white']['pvalue']:.4f}`  
                **Powerballs:** χ² = `{results['chi_square_red']['statistic']:.2f}` | p = `{results['chi_square_red']['pvalue']:.4f}`  
                """
            )
            st.caption("Lower p-values suggest non-random patterns — higher means more random draws.")

            # ---------- HOT & COLD ----------
            st.subheader("🔥 Hot & 🧊 Cold Numbers")

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Top 5 Hottest White Balls:**")
                st.write(results["hot_whites"])
                st.markdown("**Top 3 Hottest Powerballs:**")
                st.write(results["hot_reds"])
            with col4:
                st.markdown("**Top 5 Coldest White Balls:**")
                st.write(results["cold_whites"])
                st.markdown("**Top 3 Coldest Powerballs:**")
                st.write(results["cold_reds"])

            # ---------- WEIGHTED PICKS ----------
st.subheader("🎯 Weighted Number Picks for Upcoming Draws")
st.caption("Generated using frequency-weighted probabilities (for fun only).")

draw_days = ["Monday", "Wednesday", "Saturday"]
today = datetime.date.today()
today_name = today.strftime("%A")

# Determine next draw day
def next_draw_day(today_name):
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    idx_today = days_order.index(today_name)
    for offset in range(1, 8):
        next_day = days_order[(idx_today + offset) % 7]
        if next_day in draw_days:
            return next_day
    return "Monday"

next_draw = next_draw_day(today_name)

# Prepare weights
white_weights = results["white_freq_series"].values
white_numbers = results["white_freq_series"].index
power_weights = results["red_freq_series"].values
power_numbers = results["red_freq_series"].index

# Normalize weights
white_weights = white_weights / white_weights.sum()
power_weights = power_weights / power_weights.sum()

# Generate fresh picks for each draw day
for day in draw_days:
    if day == next_draw:
        st.markdown(f"### 🗓 <span style='color:limegreen'>**{day} Draw Picks (NEXT DRAW)**</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"### 🗓 {day} Draw Picks")

    for i in range(1, 6):
        whites = sorted(
            list(np.random.choice(white_numbers, size=5, replace=False, p=white_weights))
        )
        powerball = int(np.random.choice(power_numbers, size=1, replace=False, p=power_weights))
        st.markdown(f"**Pick {i}:** 🎱 Whites → {whites} | 🔴 Powerball → {powerball}")
    st.markdown("---")

st.caption(f"🕓 Picks generated on **{today.strftime('%A, %B %d, %Y')}** at **{datetime.datetime.now().strftime('%I:%M %p')}**")


            # ---------- VISUAL RECAP ----------
            st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error during analysis: {e}")

else:
    st.info("Click the **'🚀 Fetch & Analyze Data'** button to begin.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center>© 2025 Cinemagic Studios | Built by Akwesi Khaled 🎬</center>",
    unsafe_allow_html=True,
)
