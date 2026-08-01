import os

import pandas as pd
import requests
import streamlit as st

from theme import statusbar, stylesheet, titlebar

API_URL = os.environ.get("API_URL", "http://localhost:8080")
TIMEOUT = 10
UI_SCALE = float(os.environ.get("UI_SCALE", "1.0"))

EMBARKED_OPTIONS = [None, "S", "C", "Q"]
SEX_OPTIONS = ["male", "female"]

BATCH_SEED = pd.DataFrame([
    {"Pclass": 1, "Name": "Allen, Mr. William Henry", "Sex": "male",
     "Age": 35.0, "SibSp": 0, "Parch": 0, "Ticket": "373450",
     "Fare": 8.05, "Cabin": "C85", "Embarked": "S"},
    {"Pclass": 3, "Name": "Braund, Mr. Owen Harris", "Sex": "male",
     "Age": 22.0, "SibSp": 1, "Parch": 0, "Ticket": "A/5 21171",
     "Fare": 7.25, "Cabin": None, "Embarked": None},
    {"Pclass": 2, "Name": "Nasser, Mrs. Nicholas", "Sex": "female",
     "Age": None, "SibSp": 1, "Parch": 0, "Ticket": "237736",
     "Fare": None, "Cabin": None, "Embarked": "C"},
])


def call_api(endpoint: str, payload):
    """POST to the prediction API. Returns (data, error_message)."""
    try:
        response = requests.post(
            f"{API_URL}{endpoint}", json=payload, timeout=TIMEOUT
        )
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}"
    except requests.exceptions.Timeout:
        return None, f"The API did not respond within {TIMEOUT}s"

    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text[:500]}"

    return response.json(), None


def blank_to_none(value):
    """Empty / whitespace-only strings and NaN become None (an absent optional field)."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def rows_from_editor(df: pd.DataFrame) -> list[dict]:
    """Convert the editable table into the JSON shape PassengerSchema expects."""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Pclass": int(r["Pclass"]),
            "Name": str(r["Name"]),
            "Sex": str(r["Sex"]),
            "Age": None if pd.isna(r["Age"]) else float(r["Age"]),
            "SibSp": int(r["SibSp"]),
            "Parch": int(r["Parch"]),
            "Ticket": str(r["Ticket"]),
            "Fare": None if pd.isna(r["Fare"]) else float(r["Fare"]),
            "Cabin": blank_to_none(r["Cabin"]),
            "Embarked": blank_to_none(r["Embarked"]),
        })
    return rows


def render_verdict(result: dict):
    survived = bool(result["survived"])
    proba = float(result["proba"])

    left, right = st.columns([1, 2])
    with left:
        st.metric("Prediction", "Survived" if survived else "Did not survive")
    with right:
        st.metric("Survival probability", f"{proba:.1%}")
        st.progress(proba)


st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")
st.markdown(stylesheet(UI_SCALE), unsafe_allow_html=True)
st.markdown(titlebar("Titanic Survival Prediction"), unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Status")
    if st.button("Check API", width="stretch"):
        try:
            probe = requests.post(f"{API_URL}/predict_batch", json=[], timeout=TIMEOUT)
            if probe.status_code == 200:
                st.success("The API is responding")
            else:
                st.error(f"HTTP {probe.status_code}")
        except requests.exceptions.RequestException:
            st.error("No connection")

    st.divider()
    st.caption(
        "Empty Age, Fare, Cabin and Embarked fields are sent as null — "
        "the model is trained to handle missing values."
    )

single_tab, batch_tab = st.tabs(["Single passenger", "Batch"])

with single_tab:
    with st.form("single"):
        st.subheader("Required fields")
        c1, c2, c3 = st.columns(3)
        pclass = c1.selectbox("Pclass", [1, 2, 3])
        sex = c2.selectbox("Sex", SEX_OPTIONS)
        ticket = c3.text_input("Ticket", value="373450")
        name = st.text_input("Name", value="Allen, Mr. William Henry")
        c4, c5 = st.columns(2)
        sibsp = c4.number_input("SibSp", min_value=0, step=1, value=0)
        parch = c5.number_input("Parch", min_value=0, step=1, value=0)

        st.subheader("Optional fields")
        st.caption("Leave a field empty if the value is unknown")
        c6, c7, c8, c9 = st.columns(4)
        age = c6.number_input("Age", min_value=0.0, max_value=120.0, value=35.0)
        fare = c7.number_input("Fare", min_value=0.0, value=8.05)
        cabin = c8.text_input("Cabin", value="C85")
        embarked = c9.selectbox(
            "Embarked",
            EMBARKED_OPTIONS,
            format_func=lambda v: "not specified" if v is None else v,
        )

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        payload = {
            "Pclass": int(pclass),
            "Name": name,
            "Sex": sex,
            "Age": None if age is None else float(age),
            "SibSp": int(sibsp),
            "Parch": int(parch),
            "Ticket": ticket,
            "Fare": None if fare is None else float(fare),
            "Cabin": blank_to_none(cabin),
            "Embarked": embarked,
        }

        data, error = call_api("/predict", payload)
        if error:
            st.error(error)
        else:
            render_verdict(data)
            with st.expander("Request and response"):
                st.json({"request": payload, "response": data})

with batch_tab:
    st.subheader("Passengers")
    st.caption("Rows can be added and removed. An empty cell means a missing value.")

    edited = st.data_editor(
        BATCH_SEED,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Pclass": st.column_config.SelectboxColumn(
                "Pclass", options=[1, 2, 3], required=True, width="small"),
            "Name": st.column_config.TextColumn("Name", required=True, width="medium"),
            "Sex": st.column_config.SelectboxColumn(
                "Sex", options=SEX_OPTIONS, required=True, width="small"),
            "Age": st.column_config.NumberColumn(
                "Age", min_value=0, max_value=120, width="small"),
            "SibSp": st.column_config.NumberColumn(
                "SibSp", min_value=0, step=1, required=True, width="small"),
            "Parch": st.column_config.NumberColumn(
                "Parch", min_value=0, step=1, required=True, width="small"),
            "Ticket": st.column_config.TextColumn("Ticket", required=True, width="small"),
            "Fare": st.column_config.NumberColumn("Fare", min_value=0, width="small"),
            "Cabin": st.column_config.TextColumn("Cabin", width="small"),
            "Embarked": st.column_config.SelectboxColumn(
                "Embarked", options=["S", "C", "Q"], width="small"),
        },
    )

    if st.button("Predict batch", type="primary"):
        try:
            rows = rows_from_editor(edited)
        except (ValueError, TypeError) as exc:
            st.error(f"Could not read the table: {exc}")
        else:
            data, error = call_api("/predict_batch", rows)
            if error:
                st.error(error)
            elif not data:
                st.info("The API returned an empty result — the table has no rows.")
            else:
                results = pd.DataFrame(data)
                results.insert(0, "Name", [r["Name"] for r in rows])
                results["survived"] = results["survived"].astype(bool)

                st.dataframe(
                    results,
                    width="stretch",
                    column_config={
                        "survived": st.column_config.CheckboxColumn("Survived"),
                        "proba": st.column_config.ProgressColumn(
                            "Probability", min_value=0.0, max_value=1.0, format="%.3f"
                        ),
                    },
                )
                with st.expander("Request and response"):
                    st.json({"request": rows, "response": data})

st.markdown(
    statusbar(f"API endpoint: {API_URL}", "Ready"),
    unsafe_allow_html=True,
)
