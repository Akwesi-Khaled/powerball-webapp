import streamlit as st
import matplotlib.pyplot as plt
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
st.sidebar.markdown("Analyze historical Powerball draws and generate data-driven picks — for fun!")

st.sidebar.markdown("---")
st.sidebar.info("**Disclaimer:** This app is for **educational and entertainment** purposes only — not for predicting or gambling.")
st.sidebar.markdown("Made with ❤️ by **Akwesi Khaled**")

# -------------------- MAIN CONTENT --------------------
st.title("🎰 Powerball Data Analyzer Dashboard")
st.markdown("Welcome! Click below to fetch real Powerball results and explore frequency insights, randomness tests, and weighted picks.")

if st.button("🚀 Fetch & Analyze Data"):
    with st.spinner("Fetching and analyzing Powerball data... ⏳"):
        try:
            # Fetch + preprocess + analyze
            df_raw = pba.fetch_powerball_data()
            df = pba.preprocess(df_raw)
            results, fig = pba.analyze(df)

            st.success("✅ Analysis complete!")

            # -------------- DISPLAY RESULTS --------------
            st.subheader("📊 Frequency Overview")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### White Balls (1–69)")
                st.bar_chart(results["white_freq_series"])
            with col2:
                st.markdown("### Powerballs (1–26)")
                st.bar_chart(results["red_freq_series"])

            # -------------- CHI-SQUARE TEST --------------
            st.subheader("🎲 Randomness Check")
            st.markdown(
                f"""
                **White Balls:** χ² = `{results['chi_square_white']['statistic']:.2f}` | p = `{results['chi_square_white']['pvalue']:.4f}`  
                **Powerballs:** χ² = `{results['chi_square_red']['statistic']:.2f}` | p = `{results['chi_square_red']['pvalue']:.4f}`  
                """
            )
            st.caption("Lower p-values suggest non-random patterns — higher means more random draws.")

            # -------------- HOT & COLD --------------
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

            # -------------- WEIGHTED PICKS --------------
            st.subheader("🎯 Weighted Number Picks")
            st.caption("Generated using frequency-weighted probabilities (for fun only).")
            for i, ticket in enumerate(results["weighted_tickets"], 1):
                st.markdown(
                    f"**Pick {i}:** 🎱 Whites → {ticket['whites']} | 🔴 Powerball → {ticket['powerball']}"
                )

            # -------------- VISUAL --------------
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
