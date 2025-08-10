import os
import requests
import pandas as pd
import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Headline Sentiment Scorer", page_icon="📰", layout="wide")

# ---------------------------
# Config (backend)
# ---------------------------
DEFAULT_BACKEND = "http://localhost:8019"
BACKEND_URL = st.sidebar.text_input("Backend URL", os.getenv("BACKEND_URL", DEFAULT_BACKEND))

def ping_backend(url: str):
    try:
        r = requests.get(f"{url}/status", timeout=3)
        if r.status_code == 200:
            return True, "Backend OK"
        else:
            return False, f"Unexpected status code: {r.status_code}"
    except Exception as e:
        return False, f"Cannot reach backend: {e}"

# Sidebar status check
ok, msg = ping_backend(BACKEND_URL)
if ok:
    st.sidebar.success(f"Connected to backend: {BACKEND_URL}")
else:
    st.sidebar.error(msg)

# ---------------------------
# Load sample headlines
# ---------------------------
def load_samples():
    return pd.DataFrame({"headline": [
        "Stocks surge after strong earnings",
        "Rising unemployment fuels recession fears",
        "Fed holds rates steady amid uncertainty"
    ]})

if "df" not in st.session_state:
    st.session_state.df = load_samples()

# Sidebar button to load samples
if st.sidebar.button("Load sample headlines"):
    st.session_state.df = load_samples()

# ---------------------------
# Main UI
# ---------------------------
st.title("📰 Headline Sentiment Scorer")
st.caption(f"Edit, add, or delete headlines. Click **Score Headlines** to get labels. Streamlit: 9019 → FastAPI: 8019")

# Editable table
edited_df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={"headline": "Headline"},
    key="headline_table"
)

st.session_state.df = edited_df

# ---------------------------
# Score Headlines
# ---------------------------
if st.button("🚀 Score Headlines"):
    headlines = [h for h in edited_df["headline"].tolist() if h.strip()]
    if not headlines:
        st.warning("Please enter at least one headline.")
    else:
        try:
            res = requests.post(
                f"{BACKEND_URL}/score_headlines",
                json={"headlines": headlines}
            )
            if res.status_code == 200:
                labels = res.json().get("labels", [])
                results_df = pd.DataFrame({"Headline": headlines, "Label": labels})
                st.success("✅ Scoring complete!")
                st.dataframe(results_df, use_container_width=True)

                # Chart
                chart_df = results_df["Label"].value_counts().reset_index()
                chart_df.columns = ["Sentiment", "Count"]
                st.bar_chart(chart_df.set_index("Sentiment"))

                # Download CSV
                csv = results_df.to_csv(index=False)
                st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

            else:
                st.error(f"Error from backend: {res.status_code}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
