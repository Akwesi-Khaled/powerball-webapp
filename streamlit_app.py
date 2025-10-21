# streamlit_app.py (complete file)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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
st.markdown("Welcome! Click the button below to fetch real Powerball results and explore data-driven insights, trends, and visual patterns.")

# Button to run analysis
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

            # ---------- COMBINED FREQUENCY VISUAL ----------
            st.subheader("🎨 Combined Frequency Visualization")
            freq_fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(results["white_freq_series"].index, results["white_freq_series"].values, label="White Balls", alpha=0.7)
            # scale powerballs to white ball max for visual comparison
            red_vals = results["red_freq_series"].reindex(range(1, 70), fill_value=0).values
            if red_vals.max() > 0:
                scaled_red = red_vals / red_vals.max() * results["white_freq_series"].values.max()
            else:
                scaled_red = red_vals
            ax.bar(range(1, 70), scaled_red, label="Powerballs (scaled)", alpha=0.6, color="red")
            ax.legend()
            ax.set_title("White Balls vs Powerballs Frequency Comparison (Powerballs scaled)")
            ax.set_xlabel("Ball Number")
            ax.set_ylabel("Frequency (scaled)")
            st.pyplot(freq_fig)
            plt.close(freq_fig)

            # ---------- TREND OVER TIME ----------
            st.subheader("📈 Frequency Trend Over Time")
            # Build trend (simple frequency line of white numbers)
            white_numbers = df[['white1', 'white2', 'white3', 'white4', 'white5']].melt(value_name='number')
            white_trend = white_numbers['number'].value_counts().sort_index().reset_index()
            white_trend.columns = ['Ball', 'Frequency']

            trend_fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=white_trend, x='Ball', y='Frequency', marker='o', ax=ax)
            ax.set_title("Trend of White Ball Frequencies (by number)")
            ax.set_xlabel("White Ball Number (1–69)")
            ax.set_ylabel("Occurrences")
            st.pyplot(trend_fig)
            plt.close(trend_fig)

            # ---------- HEATMAP ----------
            st.subheader("🔥 Hot vs Cold Heatmap")
            heat_data = pd.DataFrame({
                "White Balls": results["white_freq_series"],
                "Powerballs": results["red_freq_series"].reindex(range(1, 70), fill_value=0)
            })
            # Transpose so rows = categories, cols = numbers
            heat_fig, ax = plt.subplots(figsize=(12, 3))
            sns.heatmap(heat_data.T, cmap="coolwarm", cbar=True, ax=ax)
            ax.set_title("Heatmap of White Balls (left) and Powerballs (right)")
            st.pyplot(heat_fig)
            plt.close(heat_fig)

            # ---------- RANDOMNESS TEST ----------
            st.subheader("🎲 Randomness Check")
            st.markdown(
                f"**White Balls:** χ² = `{results['chi_square_white']['statistic']:.2f}` | p = `{results['chi_square_white']['pvalue']:.4f}`  \n"
                f"**Powerballs:** χ² = `{results['chi_square_red']['statistic']:.2f}` | p = `{results['chi_square_red']['pvalue']:.4f}`"
            )
            st.caption("Lower p-values suggest deviations from uniformity; higher p-values are consistent with randomness.")

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

            # ---------- WEIGHTED PICKS (unique per draw day) ----------
            st.subheader("🎯 Weighted Number Picks for Upcoming Draws")
            st.caption("Generated using frequency-weighted probabilities (for fun only).")

            # Define draw days & compute next draw
            draw_days = ["Monday", "Wednesday", "Saturday"]
            today = datetime.date.today()
            today_name = today.strftime("%A")

            def next_draw_day(today_name):
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                idx_today = days_order.index(today_name)
                for offset in range(1, 8):
                    next_day = days_order[(idx_today + offset) % 7]
                    if next_day in draw_days:
                        return next_day
                return "Monday"

            next_draw = next_draw_day(today_name)

            # Prepare probability arrays
            white_numbers_idx = results["white_freq_series"].index.to_numpy()
            white_weights = results["white_freq_series"].values.astype(float)
            power_numbers_idx = results["red_freq_series"].index.to_numpy()
            power_weights = results["red_freq_series"].values.astype(float)

            # Safety: if any sum is zero, replace with uniform weights
            if white_weights.sum() <= 0:
                white_weights = np.ones_like(white_weights)
            if power_weights.sum() <= 0:
                power_weights = np.ones_like(power_weights)

            # Normalize for np.random.choice 'p' argument
            white_p = white_weights / white_weights.sum()
            power_p = power_weights / power_weights.sum()

            rng = np.random.default_rng()  # use new Generator for better randomness

            for day in draw_days:
                if day == next_draw:
                    st.markdown(f"### 🗓 <span style='color:limegreen'>**{day} Draw Picks (NEXT DRAW)**</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"### 🗓 {day} Draw Picks")

                # Generate 5 unique picks for this draw day
                picks_for_day = []
                for i in range(5):
                    # sample 5 unique white numbers using weighted probabilities
                    whites = []
                    # draw iteratively without replacement using weights:
                    available_numbers = white_numbers_idx.copy()
                    available_probs = white_p.copy()
                    # Select 5 numbers without replacement
                    selected = []
                    for _ in range(5):
                        choice = rng.choice(available_numbers, p=available_probs / available_probs.sum())
                        selected.append(int(choice))
                        # remove choice
                        mask = available_numbers != choice
                        available_numbers = available_numbers[mask]
                        available_probs = available_probs[mask]
                        if available_probs.size == 0:
                            break
                    whites = sorted(selected)

                    # powerball pick (one number)
                    powerball = int(rng.choice(power_numbers_idx, p=power_p))
                    picks_for_day.append({'whites': whites, 'powerball': powerball})
                    st.markdown(f"**Pick {i+1}:** 🎱 Whites → {whites} | 🔴 Powerball → {powerball}")

                st.markdown("---")

            # Timestamp
            st.caption(f"🕓 Picks generated on **{today.strftime('%A, %B %d, %Y')}** at **{datetime.datetime.now().strftime('%I:%M %p')}**")

            # ---------- VISUAL RECAP ----------
            st.subheader("Visual Recap")
            st.pyplot(fig)
            plt.close(fig)

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
