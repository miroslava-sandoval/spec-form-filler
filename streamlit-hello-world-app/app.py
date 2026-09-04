import os
import io
import json
import time
import requests
import streamlit as st
import pandas as pd
from datetime import datetime
from databricks import sql
import uuid
from pathlib import Path

# ---------------- Page config ----------------
st.set_page_config(
    page_title="AI-Driven MRS Form Filling",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Session state defaults ----------------
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "review_df" not in st.session_state:
    st.session_state["review_df"] = None

# ---------------- Constants ----------------
APP_VERSION = os.environ.get("APP_VERSION", "v1.0.0")

KNOWN_SPEC_SUGGESTIONS = [
    "SPEC-246246 (Stopper)",
    "SPEC-238357 (Diluent)",
    "SPEC-246247 (Container Glass)",
    "SPEC-246302 (Filter/ Membrane)",
    "SPEC-242456 (Auxiliary)",
    "SPEC-243026 (Chromatography Resin)",
    "SPEC-234283 (Filter Aid)",
    "SPEC-251228 (Bag)",
    "VAL-610832 (Bag)",
]

# ---------------- Env validation ----------------
REQUIRED_ENV_VARS = [
    "DATABRICKS_HOST",
    "DATABRICKS_ENDPOINT",
    "DATABRICKS_WAREHOUSE_ID",
    "TEXT_TABLE",
    "SCHEMAS_TABLE",
    "TTP_TABLE",
    "FEEDBACK_DELTA_TABLE",
]

missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]

if missing:
    st.error(
        f"Missing required environment variables: {', '.join(missing)}"
    )
    st.stop()


# ---------------- Env ----------------
HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
SERVING_ENDPOINT = os.environ["DATABRICKS_ENDPOINT"]
WAREHOUSE_ID = os.environ["DATABRICKS_WAREHOUSE_ID"]

SQL_HTTP_PATH = (
    os.environ.get("DATABRICKS_SQL_HTTP_PATH")
    or f"/sql/1.0/warehouses/{WAREHOUSE_ID}"
)

TEXT_TABLE = os.environ["TEXT_TABLE"]
SCHEMAS_TABLE = os.environ["SCHEMAS_TABLE"]
TTP_TABLE = os.environ["TTP_TABLE"]

INVOCATIONS_URL = (
    f"{HOST}/serving-endpoints/{SERVING_ENDPOINT}/invocations"
)

SERVER_HOSTNAME = (
    HOST.replace("https://", "")
    .replace("http://", "")
)

FEEDBACK_DELTA_TABLE = os.environ["FEEDBACK_DELTA_TABLE"]
# ---------------- User authentication ----------------
def get_user_token():
    token = st.context.headers.get("X-Forwarded-Access-Token")

    if not token:
        raise RuntimeError(
            "No user access token was provided by Databricks Apps."
        )

    return token


# ---------------- SQL connection helper ----------------
def get_sql_connection():
    return sql.connect(
        server_hostname=SERVER_HOSTNAME,
        http_path=SQL_HTTP_PATH,
        access_token=get_user_token(),
    )
# ---------------- Title ----------------
st.title("AI-Driven MRS Form Filling")
st.caption(
    "This app reads the required form structure, material specification and tech transfer plan, "
    "then uses AI to populate the Material Requirement Specification form fields."
)

# ---------------- Helpers: downloads ----------------
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="mrs_extract")
    return output.getvalue()


def sanitize(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
    )


def render_spec_not_found(spec_id: str):
    st.error(f"❌ Material Specification not found: `{spec_id}`")
    st.markdown(
        "**Try one of these suggestions:**\n\n"
        + "\n".join(f"- {s}" for s in KNOWN_SPEC_SUGGESTIONS)
    )


def append_feedback_to_jsonl(events: list[dict]) -> Path:
    Path(FEEDBACK_BUFFER_DIR).mkdir(parents=True, exist_ok=True)

    buf_path = Path(FEEDBACK_BUFFER_DIR) / "feedback_buffer.jsonl"

    with buf_path.open("a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return buf_path

def save_feedback_to_delta(events: list[dict]) -> None:
    if not events:
        return

    df = pd.DataFrame(events)

    wanted = [
        "run_id",
        "user_email",
        "spec_id",
        "mrs_category",
        "field_id",
        "field_label",
        "llm_answer",
        "review_flag",
        "correct",
        "feedback",
        "timestamp_utc",
        "saved_at_utc",
        "serving_endpoint",
        "app_version",
        "elapsed_seconds",
        "selected_ttp_ids",
    ]

    df = df[[c for c in wanted if c in df.columns]]

    if df.empty:
        return

    insert_cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(insert_cols))
    col_list = ", ".join(insert_cols)

    q = f"""
    INSERT INTO {FEEDBACK_DELTA_TABLE}
    ({col_list})
    VALUES ({placeholders})
    """

    data = [tuple(r) for r in df.itertuples(index=False, name=None)]

    with get_sql_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(q, data)

# ---------------- SQL helpers ----------------
@st.cache_data(ttl=300)
def get_spec_info_from_sql(spec_id: str) -> tuple[bool, str]:
    ID_COLUMN = "Document_Number"
    TITLE_COLUMN = "Title"

    q = f"""
    SELECT {TITLE_COLUMN}
    FROM {TEXT_TABLE}
    WHERE {ID_COLUMN} = ?
    LIMIT 1
    """
    try:
        with get_sql_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(q, (spec_id,))
                row = cur.fetchone()

        if row is None:
            return False, spec_id

        title = (row[0] if row[0] is not None else "").strip()
        return True, title if title else spec_id

    except Exception:
        return False, ""


@st.cache_data(ttl=300)
def load_mrs_categories_from_sql() -> list[str]:
    COLUMN_CANDIDATES = ["form_name", "mrs_category", "MRS_category", "category"]

    last_err = None
    for col in COLUMN_CANDIDATES:
        q = f"""
        SELECT DISTINCT {col} AS form_name
        FROM {SCHEMAS_TABLE}
        WHERE {col} IS NOT NULL AND TRIM({col}) <> ''
        ORDER BY form_name
        """
        try:
            with get_sql_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(q)
                    rows = cur.fetchall()

            vals = [r[0] for r in rows if r and r[0]]
            if vals:
                return vals

        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Could not load MRS categories from {SCHEMAS_TABLE}. "
        f"Tried columns {COLUMN_CANDIDATES}. Last error: {repr(last_err)}"
    )


@st.cache_data(ttl=300)
def load_titles_from_sql() -> list[tuple]:
    ID_COL = "TTPD_id"
    TITLE_COL = "Title"
    FILTER_COL = "Manufacturing_Process_Overview"

    q = f"""
    SELECT {ID_COL}, {TITLE_COL}
    FROM {TTP_TABLE}
    WHERE {FILTER_COL} IS NOT NULL
      AND {TITLE_COL} IS NOT NULL
      AND TRIM({TITLE_COL}) <> ''
    ORDER BY {TITLE_COL}
    """
    try:
        with get_sql_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(q)
                rows = cur.fetchall()

        return [
            (r[0], r[1])
            for r in rows
            if r and r[0] is not None and r[1] is not None
        ]

    except Exception as e:
        raise RuntimeError(f"Failed to load titles from SQL: {e}")


@st.cache_data(ttl=300)
def load_overviews_from_sql(spec_ids: list[str]) -> pd.DataFrame:
    if not spec_ids:
        return pd.DataFrame(columns=["Document_Number", "Title", "Manufacturing_Process_Overview"])

    ID_COL = "TTPD_id"
    TITLE_COL = "Title"
    OVERVIEW_COL = "Manufacturing_Process_Overview"

    placeholders = ", ".join(["?"] * len(spec_ids))

    q = f"""
    SELECT {ID_COL}, {TITLE_COL}, {OVERVIEW_COL}
    FROM {TTP_TABLE}
    WHERE {ID_COL} IN ({placeholders})
      AND {OVERVIEW_COL} IS NOT NULL
    ORDER BY {TITLE_COL}
    """

    try:
        with get_sql_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(q, tuple(spec_ids))
                rows = cur.fetchall()

        return pd.DataFrame(
            rows,
            columns=["Document_Number", "Title", "Manufacturing_Process_Overview"],
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load Manufacturing_Process_Overview from SQL: {e}")



# ---------------- Serving call ----------------
def call_serving(
    spec_id: str,
    mrs_category: str,
    ttp_ids: str = "",
    timeout: int = 240,
) -> pd.DataFrame:
    payload = {
        "dataframe_split": {
            "columns": ["spec_id", "mrs_category", "ttp_ids"],
            "data": [[spec_id, mrs_category, ttp_ids]],
        }
    }

    token = get_user_token()

    st.write("User token received:", bool(token))
    st.write(
        "Forwarded user:",
        st.context.headers.get("X-Forwarded-Preferred-Username")
    )

    r = requests.post(
        INVOCATIONS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=timeout,
    )


    if not r.ok:
        raise RuntimeError(f"{r.status_code} — {r.text[:2000]}")

    resp = r.json()
    preds = resp.get("predictions")

    if not isinstance(preds, list):
        raise ValueError(f"Unexpected response format: keys={list(resp.keys())}")

    return pd.DataFrame(preds)


# ---------------- Load dropdown options ----------------
try:
    mrs_options = load_mrs_categories_from_sql()
except Exception as e:
    st.error("Failed to load MRS categories from SQL Warehouse.")
    st.exception(e)
    st.stop()


# ---------------- Load document titles for multi-select ----------------
try:
    title_rows = load_titles_from_sql()
    title_options = [title for _, title in title_rows]
    title_to_id = {title: ttpd_id for ttpd_id, title in title_rows}

except Exception as e:
    st.error("Failed to load document titles from SQL.")
    st.exception(e)
    title_options = []
    title_to_id = {}


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Input")

    user_email = st.text_input(
        "Email (required to download + save feedback)",
        key="user_email",
        placeholder="name@takeda.com",
    ).strip().lower()

    if not user_email:
        st.warning("Enter your email to enable downloads.")

    spec_id = st.text_input("SPEC ID", value="SPEC-243026")
    spec_id = spec_id.strip().upper().replace("\u00A0", " ")

    spec_exists = False
    spec_title = ""

    if spec_id:
        try:
            spec_exists, spec_title = get_spec_info_from_sql(spec_id)

            if spec_exists:
                st.success(f"Material Specification: {spec_title}")
            else:
                st.error("Specification ID not found, click Run Extraction for known suggestions")

        except Exception as e:
            st.warning("Could not retrieve SPEC title.")
            st.exception(e)

    mrs_category = st.selectbox(
        "MRS Category",
        options=mrs_options,
        index=mrs_options.index("Stopper") if "Stopper" in mrs_options else 0,
    )

    selected_titles = st.multiselect(
        "Tech Transfer Plan Documents",
        options=title_options,
        default=[],
    )

    selected_ttp_ids = [
        title_to_id[label]
        for label in selected_titles
        if label in title_to_id
    ]

    ttp_ids_str = ",".join(selected_ttp_ids)

    if selected_ttp_ids:
        st.caption(f"Selected TTP IDs: {ttp_ids_str}")
    else:
        st.caption("No TTP documents selected.")

    run_btn = st.button("Run Extraction", type="primary")

    clear_btn = st.button("Clear results")

    if clear_btn:
        for key in [
            "results_df",
            "review_df",
            "run_id",
            "elapsed_seconds",
            "selected_ttp_ids",
            "serving_endpoint",
            "app_version",
        ]:
            st.session_state.pop(key, None)

        st.rerun()


today_str = datetime.utcnow().strftime("%Y%m%d")
mrs_part = mrs_category or "UNKNOWN"
file_base = f"{sanitize(spec_id)}_{sanitize(mrs_part)}_{today_str}"


# ---------------- Run ----------------
if run_btn:

    if not mrs_category or not mrs_category.strip():
        st.error("Select an MRS category before running extraction.")
        st.stop()
        
    if selected_ttp_ids:

        try:
            ov_df = load_overviews_from_sql(selected_ttp_ids)
        except Exception as e:
            st.error("Failed to load Manufacturing_Process_Overview for selected documents.")
            st.exception(e)
            ov_df = pd.DataFrame()

    if not spec_exists:
        render_spec_not_found(spec_id)
        st.stop()

    st.info(f"Filling form {mrs_category} with Material Specification {spec_title}")

    try:
        start_time = time.perf_counter()

        with st.spinner("Running extraction..."):
            df_out = call_serving(
                spec_id,
                mrs_category,
                ttp_ids=ttp_ids_str,
            ).reset_index(drop=True)

        elapsed_seconds = time.perf_counter() - start_time

        st.session_state["results_df"] = df_out
        st.session_state.pop("review_df", None)

        st.session_state["run_id"] = str(uuid.uuid4())
        st.session_state["elapsed_seconds"] = elapsed_seconds
        st.session_state["selected_ttp_ids"] = ttp_ids_str
        st.session_state["serving_endpoint"] = SERVING_ENDPOINT
        st.session_state["app_version"] = APP_VERSION

        st.success(f"Extraction complete — {len(df_out)} rows | ⏱ {elapsed_seconds:.2f}s")
        st.caption(f"Endpoint: {SERVING_ENDPOINT}")

    except requests.exceptions.ReadTimeout:
        st.error(
            "The extraction exceeded the 240-second wait limit. "
            "The serving endpoint may be starting from zero or processing a large request. "
            "Please wait a minute and try again."
        )
    except requests.exceptions.RequestException as e:
        st.error("Could not reach the model serving endpoint.")
        st.exception(e)
    except Exception as e:
        st.error("Extraction failed")
        st.exception(e)


# ---------------- Review UI ----------------
if st.session_state.get("results_df") is None:
    st.info("Run an extraction to see results.")

else:
    df_out = st.session_state["results_df"].copy().reset_index(drop=True)

    df_disp = df_out.copy().reset_index(drop=True)
    df_disp = df_disp.rename(columns={"Source": "Source Type"})

    desired_df_disp_cols = [
        "Field Category",
        "Field",
        "LLM Answer",
        "Strongly suggested to check LLM Answer",
        "Source Type",
        "Excerpt",
        "SPEC_ID",
        "MRS_category",
        "timestamp_utc",
    ]

    df_disp = df_disp[[c for c in desired_df_disp_cols if c in df_disp.columns]]

    df_disp = df_disp.reset_index(drop=True).copy()
    df_disp["_row_id"] = df_disp.index

    if (
        st.session_state.get("review_df") is None
        or len(st.session_state["review_df"]) != len(df_disp)
        or "_row_id" not in st.session_state["review_df"].columns
    ):
        st.session_state["review_df"] = pd.DataFrame({
            "_row_id": df_disp["_row_id"],
            "Correct": [False] * len(df_disp),
            "Feedback (required if unchecked)": [""] * len(df_disp),
        })

    st.markdown("### Review & feedback")
    elapsed_seconds = st.session_state.get("elapsed_seconds")
    serving_endpoint = st.session_state.get("serving_endpoint")

    meta_parts = []

    if elapsed_seconds is not None:
        meta_parts.append(f"Extraction time: {elapsed_seconds:.2f}s")

    if serving_endpoint:
        meta_parts.append(f"Endpoint: {serving_endpoint}")

    if meta_parts:
        st.caption(" | ".join(meta_parts))

    selected_ttp_ids_display = st.session_state.get("selected_ttp_ids", "")

    if selected_ttp_ids_display:
        st.caption(f"Selected TTP IDs: {selected_ttp_ids_display}")
    else:
        st.caption("Selected TTP IDs: none")
    st.caption("Rule: Check at least one row as correct. Feedback is required for every unchecked row.")

    b1, b2, b3 = st.columns([1, 1, 2])

    with b1:
        if st.button("✅ Check all"):
            if st.session_state.get("review_df") is not None:
                st.session_state["review_df"]["Correct"] = True
                st.session_state["review_df"]["Feedback (required if unchecked)"] = (
                    st.session_state["review_df"]["Feedback (required if unchecked)"]
                    .astype(str)
                    .fillna("")
                )
                st.rerun()
            else:
                st.warning("Review table not initialized yet.")

    with b2:
        if st.button("⬜ Uncheck all"):
            if st.session_state.get("review_df") is not None:
                st.session_state["review_df"]["Correct"] = False
                st.session_state["review_df"]["Feedback (required if unchecked)"] = (
                    st.session_state["review_df"]["Feedback (required if unchecked)"]
                    .astype(str)
                    .fillna("")
                )
                st.rerun()
            else:
                st.warning("Review table not initialized yet.")

    with b3:
        if st.button("📝 Fill missing feedback for unchecked (placeholder)"):
            rdf = st.session_state["review_df"].copy()
            unchecked = ~rdf["Correct"].astype(bool)
            missing = rdf["Feedback (required if unchecked)"].astype(str).str.strip() == ""
            rdf.loc[unchecked & missing, "Feedback (required if unchecked)"] = "Needs review / incorrect"
            st.session_state["review_df"] = rdf
            st.rerun()

    review_cols = ["Correct", "Feedback (required if unchecked)"]

    rdf = st.session_state["review_df"].copy()
    combined = df_disp.merge(rdf, on="_row_id", how="left")

    cols = list(combined.columns)
    base_cols = [c for c in cols if c not in review_cols]

    anchor_col = "Strongly suggested to check LLM Answer"

    if anchor_col in base_cols:
        i = base_cols.index(anchor_col) + 1
        combined = combined[base_cols[:i] + review_cols + base_cols[i:]]
    else:
        combined = combined[base_cols + review_cols]

    edited = st.data_editor(
        combined,
        key="combined_editor",
        height=650,
        use_container_width=True,
        column_config={
            "_row_id": None,
            "Correct": st.column_config.CheckboxColumn("Correct"),
            "Feedback (required if unchecked)": st.column_config.TextColumn(
                "Feedback (required if unchecked)",
                width="large",
            ),
        },
        disabled=[c for c in combined.columns if c not in review_cols],
    ).reset_index(drop=True)

    st.session_state["review_df"] = edited[["_row_id", *review_cols]].copy()

    export_df = edited.drop(columns=["_row_id"])

    rdf = st.session_state["review_df"].copy().reset_index(drop=True)
    correct = rdf["Correct"].astype(bool)
    feedback = rdf["Feedback (required if unchecked)"].astype(str).str.strip()

    at_least_one_correct = bool(correct.any())
    unchecked_mask = ~correct
    unchecked_feedback_ok = (feedback[unchecked_mask] != "").all() if unchecked_mask.any() else True

    ready = at_least_one_correct and unchecked_feedback_ok

    if not unchecked_feedback_ok:
        missing_count = int((feedback[unchecked_mask] == "").sum())
        st.warning(f"Feedback is required for every unchecked row, {missing_count} row(s) missing feedback.")

    if ready:
        st.success("Review complete — downloads enabled.")

    user_email = (st.session_state.get("user_email") or "").strip().lower()

    with st.container():
        user_email = (st.session_state.get("user_email") or "").strip().lower()
        run_id = st.session_state.get("run_id")

        if not ready:
            st.button("⬇️ Download Excel (with feedback)", disabled=True)

        elif not user_email:
            st.button("⬇️ Download Excel (with feedback)", disabled=True)
            st.info("Enter your email in the sidebar to enable download + saving feedback.")

        else:
            clicked = st.download_button(
                label="⬇️ Download Excel (with feedback)",
                data=df_to_excel_bytes(export_df),
                file_name=f"{file_base}_with_feedback.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            if clicked:
                if not run_id:
                    st.error("No run ID found. Please run the extraction again.")
                    st.stop()

                now = datetime.utcnow().isoformat()

                elapsed_seconds = st.session_state.get("elapsed_seconds")
                selected_ttp_ids = st.session_state.get("selected_ttp_ids", "")
                serving_endpoint = st.session_state.get("serving_endpoint", SERVING_ENDPOINT)
                app_version = st.session_state.get("app_version", APP_VERSION)

                rdf = st.session_state["review_df"].copy()
                events = []

                for _, r in rdf.iterrows():
                    rid = r.get("_row_id")
                    left = df_disp.loc[df_disp["_row_id"] == rid]
                    left_row = left.iloc[0] if len(left) else {}

                    events.append({
                        "run_id": run_id,
                        "user_email": user_email,
                        "spec_id": spec_id,
                        "mrs_category": mrs_category,
                        "field_id": str(rid),
                        "field_label": left_row.get("Field"),
                        "llm_answer": left_row.get("LLM Answer"),
                        "review_flag": left_row.get("Strongly suggested to check LLM Answer"),
                        "correct": bool(r.get("Correct", False)),
                        "feedback": str(r.get("Feedback (required if unchecked)", "")),
                        "timestamp_utc": left_row.get("timestamp_utc"),
                        "saved_at_utc": now,
                        "serving_endpoint": serving_endpoint,
                        "app_version": app_version,
                        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else None,
                        "selected_ttp_ids": selected_ttp_ids,
                    })

                try:
                    save_feedback_to_delta(events)
                    st.success("Feedback saved.")

                except Exception as e:
                    st.error("Download worked, but saving feedback failed.")
                    st.exception(e)