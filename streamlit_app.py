import streamlit as st
import io
import matplotlib.pyplot as plt
import powerball_auto_analysis as pba

st.set_page_config(page_title="Powerball Analyzer", layout="wide")

st.title("Powerball Analyzer — Frequency + Weighted Picks")
st.markdown(
    "Fetch Powerball historical data, analyze number frequencies, and generate weighted picks (for fun only)."
)

if st.button("Fetch data and run analysis"):
    with st.spinner("Fetching data and analyzing..."):
        try:
            df_raw = pba.fetch_powerball_data()
            df = pba.preprocess(df_raw)

            import sys, io as sysio
            buf = sysio.StringIO()
            sys_stdout = sys.stdout
            sys.stdout = buf
            try:
                pba.analyze(df)
            finally:
                sys.stdout = sys_stdout
            output_text = buf.getvalue()

            st.text(output_text)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("This app is for educational and entertainment purposes only.")
