# Note: This script requires Streamlit. If you're running in an environment without it,
# you must switch to a local Python environment where `streamlit` is installed.

try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as e:
    raise ImportError("This script requires the 'streamlit' package. Please run it in a local environment where Streamlit is installed.") from e

st.set_page_config(page_title="BNPL Risk Monitoring Dashboard", layout="wide")
st.title("📊 BNPL Risk Monitoring Dashboard")

# --- 1. File Upload ---
st.sidebar.header("Upload Data Files")
segment_file = st.sidebar.file_uploader("Upload segment_score_summary.csv", type=["csv"])
override_file = st.sidebar.file_uploader("Upload override_df.csv", type=["csv"])

if segment_file and override_file:
    segment_df = pd.read_csv(segment_file)
    override_df = pd.read_csv(override_file)

    # --- 2. Plot Default Rate by Score Bin ---
    st.subheader("1. 📈 Default Rate by Score Bin")
    fig, ax = plt.subplots()
    sns.lineplot(
        data=segment_df,
        x="score_bin",
        y="default_rate",
        hue="risk_segment",
        style="risk_segment",
        markers=True,
        dashes=False,
        ax=ax
    )
    ax.set_ylabel("Default Rate")
    ax.set_xlabel("Score Bin")
    ax.set_title("Default Rate by Score Bin and Risk Segment")
    st.pyplot(fig)

    # --- 3. Policy Trigger Status ---
    st.subheader("2. 📌 Policy Trigger Status")
    policy_status = segment_df["policy_trigger"].iloc[0]
    if policy_status:
        st.success("✅ Adaptive Policy Triggered")
    else:
        st.info("🟢 No Policy Triggered")

    # --- 4. Anomalies Table ---
    st.subheader("3. 🔍 Anomaly Detection")
    anomaly_table = segment_df[segment_df["low_risk_anomaly"] == True][["score_bin", "risk_segment", "default_rate"]]
    st.dataframe(anomaly_table, use_container_width=True)

    # --- 5. Override Simulation Table ---
    st.subheader("4. ⚙️ Override Simulation")
    st.dataframe(override_df, use_container_width=True)

    # --- 6. Optional: Strategy Brief Toggle ---
    with st.expander("🧠 Strategy Brief"):
        st.markdown("""
        - Several low-risk segments show higher default rates than high-risk ones.
        - Adaptive override has been simulated for affected bins.
        - Recommended action: retrain model or review risk thresholds.
        """)
else:
    st.warning("⬅️ Please upload both data files to proceed.")
