# ============================================================
# SANKHYAKI
# Demography & Employment Data Assistant
# Clean Streamlit Frontend
# ============================================================

import json
import os
import re
import sqlite3
import hashlib
import hmac
import base64
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from backend import ask_demography


# ============================================================
# 1. PROJECT PATHS + PAGE CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
DATA_DIR = PROJECT_DIR / "data"
USER_DB_FILE = PROJECT_DIR / "users.db"
FACT_FILE = DATA_DIR / "demography_long_semantic.csv"

# Keep the existing logo asset for now. The final Sankhyaki logo can
# replace this file later without changing the application code.
LOGO_FILE = ASSETS_DIR / "sankhyaki_logo.png"
# Canonical high-definition Sankhyaki logo supplied for the final interface.
# Save that image as assets/sankhyaki_logo.png.

st.set_page_config(
    page_title="Sankhyaki",
    page_icon=str(LOGO_FILE) if LOGO_FILE.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. SESSION STATE
# ============================================================

def initialise_session_state():
    defaults = {
        "authenticated": False,
        "user": None,
        "auth_page": "login",
        "current_page": "New Notebook",
        "messages": [],
        "notebook_title": "Untitled Notebook",
        "notebook_id": None,
        "processing_question": False,
        "theme": "Light",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_state()


# ============================================================
# 3. DATABASE
# ============================================================

def get_database_connection():
    connection = sqlite3.connect(USER_DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialise_database():
    connection = get_database_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS notebooks (
            notebook_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """
    )

    connection.commit()
    connection.close()


initialise_database()


# ============================================================
# 4. AUTHENTICATION HELPERS
# ============================================================

def hash_password(password):
    salt = os.urandom(16)
    derived_key = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 200_000
    )
    return f"{salt.hex()}:{derived_key.hex()}"


def verify_password(password, stored_hash):
    try:
        salt_hex, key_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        supplied_key = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, 200_000
        )
        return hmac.compare_digest(supplied_key, expected_key)
    except Exception:
        return False


def normalise_email(email):
    return str(email).strip().lower()


def valid_email(email):
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.match(pattern, email))


def create_user(name, email, password):
    name = str(name).strip()
    email = normalise_email(email)

    if not name:
        return {"success": False, "message": "Please enter your name."}
    if not valid_email(email):
        return {"success": False, "message": "Please enter a valid email address."}
    if len(password) < 8:
        return {
            "success": False,
            "message": "Password must contain at least 8 characters.",
        }

    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO users (name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, email, hash_password(password), datetime.now().isoformat()),
        )
        connection.commit()
        return {
            "success": True,
            "user": {
                "user_id": cursor.lastrowid,
                "name": name,
                "email": email,
            },
        }
    except sqlite3.IntegrityError:
        return {
            "success": False,
            "message": "An account already exists for this email address.",
        }
    finally:
        connection.close()


def authenticate_user(email, password):
    email = normalise_email(email)
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT user_id, name, email, password_hash
        FROM users
        WHERE email = ?
        """,
        (email,),
    )
    record = cursor.fetchone()
    connection.close()

    if record is None or not verify_password(password, record["password_hash"]):
        return {"success": False, "message": "Invalid email or password."}

    return {
        "success": True,
        "user": {
            "user_id": record["user_id"],
            "name": record["name"],
            "email": record["email"],
        },
    }


def login_user(user):
    st.session_state.authenticated = True
    st.session_state.user = user
    st.session_state.current_page = "New Notebook"
    st.session_state.messages = []
    st.session_state.notebook_title = "Untitled Notebook"
    st.session_state.notebook_id = None


def logout_user():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.auth_page = "login"
    st.session_state.current_page = "New Notebook"
    st.session_state.messages = []
    st.session_state.notebook_title = "Untitled Notebook"
    st.session_state.notebook_id = None
    st.session_state.processing_question = False


# ============================================================
# 5. NOTEBOOK PERSISTENCE
# ============================================================

def _serialisable_messages(messages):
    cleaned = []
    for message in messages:
        item = {
            "role": message.get("role"),
            "content": message.get("content", ""),
        }
        result = message.get("result")
        if isinstance(result, dict):
            safe_result = {
                key: value
                for key, value in result.items()
                if key not in {"chart", "figure"}
            }
            item["result"] = safe_result
        cleaned.append(item)
    return cleaned


def save_current_notebook():
    if not st.session_state.authenticated or not st.session_state.messages:
        return

    user_id = st.session_state.user["user_id"]
    title = st.session_state.notebook_title or "Untitled Notebook"
    messages_json = json.dumps(
        _serialisable_messages(st.session_state.messages),
        ensure_ascii=False,
        default=str,
    )
    now = datetime.now().isoformat()

    connection = get_database_connection()
    cursor = connection.cursor()

    if st.session_state.notebook_id is None:
        cursor.execute(
            """
            INSERT INTO notebooks
                (user_id, title, messages_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, title, messages_json, now, now),
        )
        st.session_state.notebook_id = cursor.lastrowid
    else:
        cursor.execute(
            """
            UPDATE notebooks
            SET title = ?, messages_json = ?, updated_at = ?
            WHERE notebook_id = ? AND user_id = ?
            """,
            (
                title,
                messages_json,
                now,
                st.session_state.notebook_id,
                user_id,
            ),
        )

    connection.commit()
    connection.close()


def get_user_notebooks():
    if not st.session_state.authenticated:
        return []

    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT notebook_id, title, created_at, updated_at
        FROM notebooks
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (st.session_state.user["user_id"],),
    )
    rows = cursor.fetchall()
    connection.close()
    return [dict(row) for row in rows]


def load_notebook(notebook_id):
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT notebook_id, title, messages_json
        FROM notebooks
        WHERE notebook_id = ? AND user_id = ?
        """,
        (notebook_id, st.session_state.user["user_id"]),
    )
    row = cursor.fetchone()
    connection.close()

    if row is None:
        return False

    try:
        messages = json.loads(row["messages_json"])
    except Exception:
        messages = []

    st.session_state.notebook_id = row["notebook_id"]
    st.session_state.notebook_title = row["title"]
    st.session_state.messages = messages
    st.session_state.current_page = "New Notebook"
    return True


def delete_notebook(notebook_id):
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute(
        "DELETE FROM notebooks WHERE notebook_id = ? AND user_id = ?",
        (notebook_id, st.session_state.user["user_id"]),
    )
    connection.commit()
    connection.close()

    if st.session_state.notebook_id == notebook_id:
        start_new_notebook(save_existing=False)


def start_new_notebook(save_existing=True):
    if save_existing:
        save_current_notebook()
    st.session_state.messages = []
    st.session_state.notebook_title = "Untitled Notebook"
    st.session_state.notebook_id = None
    st.session_state.processing_question = False
    st.session_state.current_page = "New Notebook"


# ============================================================
# 6. GENERAL UI HELPERS
# ============================================================

def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def get_first_name():
    user = st.session_state.user or {}
    name = str(user.get("name", "there")).strip()
    return name.split()[0] if name else "there"


def navigate(page_name):
    save_current_notebook()
    st.session_state.current_page = page_name


EMBEDDED_LOGO_B64 = "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"

def logo_data_uri():
    if LOGO_FILE.exists():
        try:
            data = base64.b64encode(LOGO_FILE.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{data}"
        except Exception:
            pass
    return "data:image/png;base64," + EMBEDDED_LOGO_B64

def render_logo(width=58):
    st.markdown(
        f"<img src='{logo_data_uri()}' style='width:{width}px;height:{width}px;object-fit:contain;display:block;'>",
        unsafe_allow_html=True,
    )


# ============================================================
# 7. GLOBAL CSS + OPTIONAL LIGHT/DARK THEME
# ============================================================

def apply_theme_css():
    dark = False  # Sankhyaki uses a consistent light theme

    bg = "#0D0D0D" if dark else "#FBF5E9"
    sidebar = "#101010" if dark else "#F8ECD7"
    surface = "#171717" if dark else "#FFFCF7"
    surface_hover = "#1D1D1D" if dark else "#FFF8EE"
    text = "#F7F1E8" if dark else "#241B16"
    muted = "#C9C0B5" if dark else "#6D6259"
    border = "#343434" if dark else "#E7D3B2"
    gold = "#E7AE3A" if dark else "#D98212"
    composer = "#171717" if dark else "#FFFFFF"

    st.markdown(f"""
    <style>
    #MainMenu, footer, [data-testid="stStatusWidget"] {{display:none !important;}}
    header[data-testid="stHeader"] {{background:transparent !important;}}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{background:{bg} !important; color:{text} !important;}}
    .block-container {{max-width:1220px; padding-top:2.0rem; padding-bottom:7.2rem;}}

    /* Sidebar */
    [data-testid="stSidebar"] {{background:{sidebar} !important; border-right:1px solid {border};}}
    [data-testid="stSidebar"] > div:first-child {{padding:1.45rem 1rem 1rem;}}
    [data-testid="stSidebar"] .stButton > button {{
        width:100%; min-height:54px; justify-content:flex-start !important; text-align:left !important;
        padding:0 1.15rem !important; border:1px solid {border}; border-radius:10px;
        background:{surface}; color:{text}; box-shadow:none; font-size:0.92rem; font-weight:500;
    }}
    [data-testid="stSidebar"] .stButton > button > div,
    [data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {{
        width:100% !important; display:flex !important; justify-content:flex-start !important; text-align:left !important;
    }}
    [data-testid="stSidebar"] .stButton > button p {{
        width:100% !important; margin:0 !important; text-align:left !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{border-color:{gold}; background:{surface_hover}; color:{text};}}
    [data-testid="stSidebar"] .stButton:first-of-type > button {{border-color:{gold}; background:{"#261F10" if dark else "#FBE7CC"};}}
    .sidebar-brand {{display:flex; align-items:center; gap:.62rem; margin:0 0 1.35rem;}}
    .sidebar-brand img {{width:58px; height:58px; object-fit:contain;}}
    .sidebar-product-name {{font-size:1.22rem; font-weight:760; line-height:1.05; color:{text};}}
    .sidebar-product-subtitle {{font-size:.62rem; line-height:1.2; color:{muted}; margin-top:.25rem;}}
    .profile-card {{margin-top:1.5rem; padding-top:1rem; border-top:1px solid {border};}}
    .profile-name {{font-size:.82rem; font-weight:700; color:{text};}}
    .profile-email {{font-size:.66rem; color:{muted};}}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{color:{text} !important;}}

    /* Auth */
    .auth-brand-name {{font-size:2.35rem; font-weight:800; color:#241B16; line-height:1;}}
    .auth-brand-sub {{font-size:1rem; color:#332923; line-height:1.2; margin-top:.45rem;}}
    .hero-heading {{font-family:Georgia,'Times New Roman',serif; color:#241B16; font-size:2.65rem; line-height:1.06; font-weight:700; margin-top:2.7rem;}}
    .hero-description {{max-width:390px; color:#4F4A46; font-size:1.02rem; line-height:1.45; margin-top:1.35rem;}}
    .top-tagline {{text-align:right; color:#9B5A20; font-size:.82rem; padding-bottom:.55rem; border-bottom:1px solid #C58B51; display:inline-block; float:right;}}
    .login-shell {{background:rgba(255,255,255,.82); border:1px solid #EBD9BF; border-radius:14px; padding:1.65rem 1.7rem 1.45rem; box-shadow:0 12px 32px rgba(105,72,32,.06);}}
    .login-title {{font-family:Georgia,'Times New Roman',serif; text-align:center; color:#241B16; font-size:1.72rem; font-weight:700;}}
    .login-subtitle {{text-align:center; color:#514A45; font-size:.88rem; margin:.38rem 0 1.1rem;}}
    .auth-divider {{display:flex; align-items:center; gap:.75rem; color:#6B625B; font-size:.8rem; margin:.9rem 0;}}
    .auth-divider:before,.auth-divider:after {{content:''; flex:1; height:1px; background:#DED5CA;}}
    .account-question {{text-align:center; color:#5C544E; font-size:.78rem; margin-top:.85rem;}}
    .account-question span {{color:#C76512;}}

    /* Chat home */
    .chat-home {{max-width:850px; margin:1.4vh auto 0; text-align:center;}}
    .chat-mark img {{width:82px; height:82px; object-fit:contain; margin-bottom:.45rem;}}
    .chat-title {{font-family:Georgia,'Times New Roman',serif; font-size:1.72rem; line-height:1.15; font-weight:700; color:{text};}}
    .chat-subtitle {{margin:.55rem auto 1.25rem; color:{muted}; font-size:.78rem;}}
    .examples-label {{display:none;}}

    /* Suggestion cards: large, full text, arrow at lower-left */
    .st-key-example_1 button, .st-key-example_2 button, .st-key-example_3 button {{
        height:112px !important; min-height:112px !important; white-space:normal !important;
        display:flex !important; align-items:flex-start !important; justify-content:flex-start !important;
        text-align:left !important; padding:1rem 1rem 2.3rem !important; position:relative !important;
        border:1px solid {border} !important; border-radius:10px !important; background:{surface} !important;
        color:{text} !important; font-size:.82rem !important; line-height:1.35 !important; font-weight:500 !important;
        box-shadow:0 2px 8px rgba(0,0,0,.03) !important;
    }}
    .st-key-example_1 button:after, .st-key-example_2 button:after, .st-key-example_3 button:after {{
        content:'→'; position:absolute; left:1rem; bottom:.72rem; color:{gold}; font-size:1.35rem; line-height:1;
    }}
    .st-key-example_1 button p, .st-key-example_2 button p, .st-key-example_3 button p {{
        white-space:normal !important; overflow:visible !important; text-overflow:clip !important;
        display:block !important; width:100% !important; line-height:1.42 !important; text-align:left !important;
    }}
    .st-key-example_1 button div, .st-key-example_2 button div, .st-key-example_3 button div {{
        overflow:visible !important; width:100% !important;
    }}
    .st-key-example_1 button:hover, .st-key-example_2 button:hover, .st-key-example_3 button:hover {{border-color:{gold} !important;}}

    /* Prototype-sized composer */
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {{
        background:{bg} !important; border:0 !important;
    }}
    [data-testid="stBottom"] {{padding:0 0 18px !important;}}
    [data-testid="stBottomBlockContainer"] {{
        max-width:820px !important; width:calc(100% - 48px) !important; margin:0 auto !important; padding:0 !important;
    }}
    [data-testid="stChatInput"] {{
        width:100% !important; max-width:820px !important; height:60px !important; min-height:60px !important;
        margin:0 auto !important; padding:0 8px 0 12px !important; overflow:hidden !important;
        border:1px solid {border} !important; border-radius:15px !important;
        background:{composer} !important; box-shadow:none !important;
    }}
    [data-testid="stChatInput"] > div {{
        height:58px !important; min-height:58px !important; background:{composer} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        height:56px !important; min-height:56px !important; max-height:56px !important;
        padding-top:17px !important; padding-bottom:12px !important;
        background:{composer} !important; color:{text} !important; line-height:1.25 !important;
    }}
    [data-testid="stChatInput"] button {{
        width:42px !important; height:42px !important; min-height:42px !important;
        margin:8px 3px 8px 0 !important; background:{gold} !important; color:#2A2118 !important;
        border-radius:999px !important;
    }}

    /* Composer - dark stays dark all the way to viewport bottom */

    .page-header {{font-size:1.65rem; font-weight:700; color:{text};}}
    .page-description {{color:{muted}; margin:.35rem 0 1.4rem;}}
    .source-line {{margin-top:.8rem; padding:.7rem .85rem; background:{surface}; border:1px solid {border}; border-radius:9px; color:{muted}; font-size:.78rem;}}
    .source-label {{font-weight:700; color:{text};}}

    @media (max-width:900px) {{.block-container{{padding-left:1rem;padding-right:1rem;}} .hero-heading{{font-size:2rem;}}}}
    </style>
    """, unsafe_allow_html=True)


apply_theme_css()
st.markdown("""<style>
.sidebar-brand {gap:.25rem !important;margin-bottom:1.05rem !important;}
.sidebar-brand img {width:68px !important;height:68px !important;}
[data-testid="stBottom"],[data-testid="stBottomBlockContainer"] {background:#FBF5E9 !important;}
[data-testid="stBottomBlockContainer"] {max-width:1120px !important;width:calc(100% - 24px) !important;}
[data-testid="stChatInput"],[data-testid="stChatInput"]>div,
[data-testid="stChatInput"] textarea {background:#FBF5E9 !important;}
[data-testid="stChatInput"] button {display:flex !important;align-items:center !important;justify-content:center !important;padding:0 !important;overflow:visible !important;margin:7px 4px 7px 0 !important;flex-shrink:0 !important;}
[data-testid="stChatInput"] button svg {display:block !important;margin:auto !important;transform:none !important;}
[data-testid="stChatMessage"] {max-width:100% !important;}
@media(max-width:650px){[data-testid="stBottomBlockContainer"]{width:calc(100% - 12px) !important;}}
</style>""",unsafe_allow_html=True)



# Targeted polish: preserve the original landing proportions and fix only defects.
st.markdown("""<style>
/* Keep the original large login card and logo; provide bottom breathing room. */
.st-key-create_account_from_login {margin-bottom:1.65rem !important;}
.auth-brand-row {gap:7px !important;}
.auth-brand-row .brand-subtitle {margin-top:.08rem !important;}
/* Sidebar: restore generous branding and keep the profile clear of sign-out. */
.sidebar-brand {margin-bottom:1.45rem !important;gap:.5rem !important;}
.sidebar-brand img {width:70px !important;height:70px !important;}
.profile-card {margin-top:1.25rem !important;padding-top:1rem !important;padding-bottom:.8rem !important;}
.profile-email {margin-bottom:.45rem !important;overflow-wrap:anywhere !important;}
.st-key-nav_signout {margin-top:.65rem !important;}
/* Neutral keyboard focus instead of a distracting red input border. */
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] > div:focus-within,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
  border-color:#D9A34E !important;
  outline:none !important;
  box-shadow:0 0 0 1px rgba(217,163,78,.25) !important;
}
/* Keep native focus indication for keyboard accessibility on controls. */
</style>""",unsafe_allow_html=True)

# ============================================================
# 7A. FINAL VIEWPORT / CHAT COMPOSER POLISH (2026-10-09)
# These are intentionally narrow overrides: do not resize the logo,
# login form, sidebar or alter any application/business logic.
# ============================================================
st.markdown("""<style>
/* Give the registration button real breathing room without shrinking it.
   The additional room comes from a small upward shift of the auth content,
   not from changing the original form or logo dimensions. */
.st-key-create_account_from_login {
    margin-bottom: 2.35rem !important;
}
/* Keep the login content clear of the lower-right Cloud owner overlay. */
@media (min-width: 901px) and (min-height: 650px) {
    [data-testid="stMainBlockContainer"]:has(.auth-brand-row) {
        padding-top: 1.15rem !important;
        padding-bottom: 3.5rem !important;
    }
}

/* Composer: cover the full bottom width in the same beige as the app,
   and lift the input enough to clear Streamlit's floating owner control. */
[data-testid="stBottom"] {
    background: #FBF5E9 !important;
    padding: 0 0 56px !important;
    border: 0 !important;
}
[data-testid="stBottomBlockContainer"] {
    background: #FBF5E9 !important;
    max-width: none !important;
    width: 100% !important;
    padding: 0 24px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}
[data-testid="stBottom"]:before,
[data-testid="stBottom"]:after {
    background: #FBF5E9 !important;
}
/* Only the outer composer gets a border, including on focus. */
[data-testid="stChatInput"] {
    background: #FBF5E9 !important;
    border: 1px solid #E7D3B2 !important;
    border-radius: 15px !important;
    outline: none !important;
    box-shadow: none !important;
    overflow: visible !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #D9A34E !important;
    outline: none !important;
    box-shadow: 0 0 0 1px rgba(217,163,78,.20) !important;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div:focus-within,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
    background: #FBF5E9 !important;
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
/* BaseWeb often draws a second border using an inner div/pseudo-element. */
[data-testid="stChatInput"] [data-baseweb="textarea"]::before,
[data-testid="stChatInput"] [data-baseweb="textarea"]::after,
[data-testid="stChatInput"] [data-baseweb="base-input"]::before,
[data-testid="stChatInput"] [data-baseweb="base-input"]::after {
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
}
@media (max-width: 650px) {
    [data-testid="stBottom"] {padding-bottom: 28px !important;}
    [data-testid="stBottomBlockContainer"] {padding: 0 10px !important;}
}
</style>""", unsafe_allow_html=True)

# ============================================================
# 7B. CHAT COMPOSER OUTLINE + SEND ARROW ALIGNMENT
# Deliberately scoped to the chat composer; all other UI remains unchanged.
# ============================================================
st.markdown("""<style>
/* Inset border remains continuous even when Streamlit's internal layers
   paint over the outer edge; clipping follows the same rounded silhouette. */
[data-testid="stChatInput"] {
    box-sizing: border-box !important;
    position: relative !important;
    border: 1px solid #D9A34E !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: inset 0 0 0 1px #D9A34E !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #C88C29 !important;
    box-shadow: inset 0 0 0 1px #C88C29 !important;
}
/* Leave the outer ring intact while removing inner outlines. */
[data-testid="stChatInput"] > div {
    box-sizing: border-box !important;
    width: 100% !important;
    min-width: 0 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
/* Center the circular send control within the 60px input height. */
[data-testid="stChatInput"] button {
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    min-height: 42px !important;
    padding: 0 !important;
    margin: auto 10px auto 0 !important;
    align-self: center !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
    line-height: 1 !important;
    border-radius: 50% !important;
    box-sizing: border-box !important;
}
[data-testid="stChatInput"] button svg {
    display: block !important;
    width: 20px !important;
    height: 20px !important;
    margin: 0 !important;
    transform: none !important;
    flex-shrink: 0 !important;
}
</style>""", unsafe_allow_html=True)

# ============================================================
# 8. AUTHENTICATION PAGES
# ============================================================

def _landing_background_css():
    """Full-viewport, aspect-ratio-preserving art for login and registration."""
    bg_file = ASSETS_DIR / "sankhyaki_landing_bg.webp"
    if not bg_file.exists():
        return
    encoded = base64.b64encode(bg_file.read_bytes()).decode("ascii")
    st.markdown(f"""<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color:#FBF5E9 !important;
        background-image:linear-gradient(rgba(255,250,242,.25),rgba(255,250,242,.25)),url('data:image/webp;base64,{encoded}') !important;
        background-size:cover !important; background-position:center center !important;
        background-repeat:no-repeat !important; background-attachment:fixed !important;
    }}
    [data-testid="stMainBlockContainer"] {{min-height:calc(100dvh - 4rem);}}
    .auth-brand-row {{display:flex;align-items:center;gap:10px;}}
    .auth-brand-row img {{width:98px;height:98px;object-fit:contain;flex-shrink:0;}}
    .auth-brand-row .brand-name {{font-size:2.15rem;font-weight:850;line-height:1.05;color:#241B16;letter-spacing:-.035em;}}
    .auth-brand-row .brand-subtitle {{font-size:.95rem;line-height:1.18;margin-top:.18rem;color:#55483b;}}
    [data-testid="stForm"] {{background:rgba(255,252,247,.91);padding:1.25rem;border-radius:16px;border:1px solid #ead7bb;}}
    @media(max-width:800px) {{
       .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"] {{background-position:52% center !important;}}
       .auth-brand-row img {{width:72px;height:72px;}}
       .auth-brand-row .brand-name {{font-size:1.75rem;}}
    }}
    </style>""",unsafe_allow_html=True)


def _hide_sidebar_for_auth():
    _landing_background_css()
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] {display:none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login_page():
    _hide_sidebar_for_auth()

    # Keep the current landing-page concept frozen for now.
    brand_area, tagline_area = st.columns([3, 1], vertical_alignment="center")
    with brand_area:
        st.markdown(
            f"<div class='auth-brand-row'><img src='{logo_data_uri()}' alt='Sankhyaki logo'>"
            "<div><div class='brand-name'>Sankhyaki</div>"
            "<div class='brand-subtitle'>Demography &amp; Employment<br>Data Assistant</div></div></div>",
            unsafe_allow_html=True,
        )
    with tagline_area:
        st.markdown(
            "<div class='top-tagline'>Insights for a Stronger India</div>",
            unsafe_allow_html=True,
        )

    left_col, centre_col, right_col = st.columns(
        [1.15, 0.72, 1.05], gap="medium", vertical_alignment="center"
    )

    with left_col:
        st.markdown(
            "<div class='hero-heading'>India's State Data,<br>Now in Conversation</div>"
            "<div class='hero-description'>Get accurate, source-backed insights on "
            "demography and employment across Indian states using natural language.</div>",
            unsafe_allow_html=True,
        )

    with centre_col:
        st.markdown("<div class='hero-art-clearance' style='min-height:230px'></div>", unsafe_allow_html=True)

    with right_col:
        st.markdown(
            "<div class='login-title'>Welcome to Sankhyaki</div>"
            "<div class='login-subtitle'>Sign in to continue</div>",
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            email = st.text_input(
                "Email address", placeholder="Enter your email address"
            )
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password"
            )
            submitted = st.form_submit_button(
                "Sign in →", use_container_width=True
            )

        if submitted:
            result = authenticate_user(email=email, password=password)
            if result["success"]:
                login_user(result["user"])
                st.rerun()
            else:
                st.error(result["message"])

        st.markdown(
            "<div class='account-question'>Don't have an account?</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Create an account", key="create_account_from_login", use_container_width=True
        ):
            st.session_state.auth_page = "register"
            st.rerun()

    # Background image includes the decorative bottom waves.


def render_register_page():
    _hide_sidebar_for_auth()

    left, centre, right = st.columns([1.2, 1, 1.2])
    with centre:
        render_logo(58)
        st.markdown(
            "<div class='login-title'>Create your Sankhyaki account</div>"
            "<div class='login-subtitle'>Create an account to continue</div>",
            unsafe_allow_html=True,
        )

        with st.form("register_form"):
            name = st.text_input("Full name", placeholder="Enter your full name")
            email = st.text_input(
                "Email address", placeholder="Enter your email address"
            )
            password = st.text_input(
                "Password", type="password", placeholder="Minimum 8 characters"
            )
            confirm_password = st.text_input(
                "Confirm password", type="password", placeholder="Re-enter your password"
            )
            submitted = st.form_submit_button(
                "Create account →", use_container_width=True
            )

        if submitted:
            if password != confirm_password:
                st.error("The passwords do not match.")
            else:
                result = create_user(name=name, email=email, password=password)
                if result["success"]:
                    st.success("Account created successfully. Please sign in.")
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error(result["message"])

        st.markdown(
            "<div class='account-question'>Already have an account?</div>",
            unsafe_allow_html=True,
        )
        if st.button("← Back to sign in", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()


def render_authentication():
    if st.session_state.auth_page == "register":
        render_register_page()
    else:
        render_login_page()


# ============================================================
# 9. SIDEBAR
# ============================================================

def render_sidebar():
    user = st.session_state.user or {}
    with st.sidebar:
        # Canonical logo + product name. Navigation is intentionally left-aligned.
        st.markdown(
            f"<div class='sidebar-brand'><img src='{logo_data_uri()}' alt='Sankhyaki'>"
            "<div><div class='sidebar-product-name'>Sankhyaki</div>"
            "<div class='sidebar-product-subtitle'>Demography &amp; Employment<br>Data Assistant</div>"
            "</div></div>", unsafe_allow_html=True,
        )

        if st.button("＋   New Notebook", key="nav_new", use_container_width=True):
            start_new_notebook(save_existing=True); st.rerun()
        if st.button("🖥   Workspace", key="nav_workspace", use_container_width=True):
            navigate("Workspace"); st.rerun()
        if st.button("↗   Alt Data", key="nav_alt", use_container_width=True):
            navigate("Alt Data"); st.rerun()
        if st.button("ⓘ   Help", key="nav_help", use_container_width=True):
            navigate("Help"); st.rerun()

        st.markdown(
            f"<div class='profile-card'><div class='profile-name'>● &nbsp; {user.get('name','')}</div>"
            f"<div class='profile-email'>{user.get('email','')}</div></div>", unsafe_allow_html=True)

        if st.button("Sign out", key="nav_signout", use_container_width=True):
            save_current_notebook(); logout_user(); st.rerun()


# ============================================================
# 10. BACKEND RESULT + PROVENANCE RENDERERS
# ============================================================

def render_provenance(provenance):
    if not isinstance(provenance, dict) or not provenance:
        return

    source = provenance.get("source")
    indicator = provenance.get("indicator")
    operation = provenance.get("operation")
    data_status = provenance.get("data_status")
    calculation = provenance.get("calculation")
    warnings = provenance.get("warnings") or []

    if source:
        st.markdown(
            f"<div class='source-line'><span class='source-label'>Source</span>"
            f"{source}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("View data provenance", expanded=False):
        if indicator:
            st.markdown(f"**Indicator:** {indicator}")
        if operation:
            st.markdown(f"**Operation:** {str(operation).title()}")
        if data_status:
            st.markdown(f"**Data status:** {data_status}")
        if calculation:
            st.markdown(f"**Calculation:** {calculation}")
        if warnings:
            st.markdown("**Warnings:**")
            for warning in warnings:
                st.markdown(f"- {warning}")


def render_backend_result(result):
    if not isinstance(result, dict):
        st.error("Sankhyaki received an unexpected response from the backend.")
        return

    status = result.get("status", "error")
    answer = result.get("answer")
    message = result.get("message")
    chart = result.get("chart")
    provenance = result.get("provenance")
    validation = result.get("validation")

    if status == "success":
        if answer:
            st.markdown(answer)
        if chart is not None:
            try:
                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )
            except Exception:
                st.warning(
                    "The answer was retrieved successfully, but the chart could not be displayed."
                )
        render_provenance(provenance)
        return

    if status == "no_data":
        st.info(
            answer
            or message
            or "No verified observation is available for this request."
        )
        render_provenance(provenance)
        return

    if status == "clarification_required":
        clarification = None
        if isinstance(validation, dict):
            clarification = validation.get("clarification")
        clarification = (
            clarification
            or result.get("clarification")
            or message
            or answer
            or "Your question matches more than one possible indicator. Please be more specific."
        )
        st.info(clarification)
        return

    if status == "invalid":
        st.warning(
            message
            or answer
            or "I could not construct a valid analytical request from that question."
        )
        return

    if status in {"parser_error", "api_error", "rate_limit_or_quota"}:
        st.warning(
            message
            or "The natural-language interpretation service is temporarily unavailable. "
            "The website is still running; please try again later."
        )
        return

    st.error(message or answer or "Sankhyaki could not complete this request.")


# ============================================================
# 11. CONVERSATIONAL PAGE
# ============================================================

def make_notebook_title(question):
    question = " ".join(str(question).split())
    return question if len(question) <= 55 else question[:52].rstrip() + "..."


def submit_question(question):
    question = str(question).strip()
    if not question or st.session_state.processing_question:
        return

    if st.session_state.notebook_title == "Untitled Notebook":
        st.session_state.notebook_title = make_notebook_title(question)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.processing_question = True

    try:
        result = ask_demography(question)
        if not isinstance(result, dict):
            result = {
                "status": "error",
                "answer": None,
                "message": "The analytical backend returned an unexpected response.",
                "chart": None,
                "provenance": None,
            }
    except Exception as exc:
        result = {
            "status": "error",
            "answer": None,
            "message": (
                "Sankhyaki could not complete the request because the analytical "
                "service encountered an unexpected error."
            ),
            "chart": None,
            "provenance": None,
            "debug_error": str(exc),
        }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("answer") or "",
            "result": result,
        }
    )
    st.session_state.processing_question = False
    save_current_notebook()
    st.rerun()


def render_chat_page():
    if not st.session_state.messages:
        # Canonical mark above greeting; embedded fallback means it cannot disappear.
        logo_html = f"<div class='chat-mark'><img src='{logo_data_uri()}'></div>"
        st.markdown(
            f"<div class='chat-home'>{logo_html}<div class='chat-title'>{get_greeting()}, {get_first_name()}</div>"
            "<div class='chat-subtitle'>Ask about India’s state-level demography and employment data.</div></div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3, gap="small")
        examples = [
            (c1, "What is the population of Bihar in 2026?", "example_1"),
            (c2, "Show the unemployment rate trend for Maharashtra", "example_2"),
            (c3, "Compare labour force participation across states", "example_3"),
        ]
        for col, question, key in examples:
            with col:
                if st.button(question, key=key, use_container_width=True):
                    submit_question(question)
    else:
        st.markdown(f"<div class='page-header'>{st.session_state.notebook_title}</div>", unsafe_allow_html=True)
        st.caption("Sankhyaki · Demography & Employment Data Assistant")
        for message in st.session_state.messages:
            role = message.get("role", "assistant")
            with st.chat_message(role):
                if role == "assistant" and message.get("result"):
                    render_backend_result(message["result"])
                elif message.get("content"):
                    st.markdown(message["content"])
                # Streamlit's native code widget provides a reliable clipboard button.
                # Same rendering path is used for active and reopened saved notebooks.
                copy_text = str(message.get("content") or "")
                if role == "assistant" and isinstance(message.get("result"), dict):
                    copy_text = str(message["result"].get("answer") or message["result"].get("response") or copy_text)
                if copy_text.strip():
                    with st.popover("⧉ Copy", help="Copy this message to your clipboard"):
                        st.code(copy_text, language=None, wrap_lines=True)

    prompt = st.chat_input("Ask Sankhyaki...")
    if prompt:
        submit_question(prompt)


# ============================================================
# 12. WORKSPACE
# ============================================================

def render_workspace():
    st.markdown("<div class='page-header'>Workspace</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-description'>Open, continue, or delete your saved Sankhyaki notebooks.</div>",
        unsafe_allow_html=True,
    )

    notebooks = get_user_notebooks()
    if not notebooks:
        st.info("You do not have any saved notebooks yet. Start a New Notebook and ask a question.")
        return

    for notebook in notebooks:
        with st.container(border=True):
            title_col, open_col, delete_col = st.columns([6, 1.2, 1.2])
            with title_col:
                st.markdown(f"**{notebook['title']}**")
                try:
                    updated = datetime.fromisoformat(notebook["updated_at"])
                    st.caption(f"Updated {updated.strftime('%d %b %Y, %I:%M %p')}")
                except Exception:
                    st.caption("Saved notebook")
            with open_col:
                if st.button(
                    "Open",
                    key=f"open_notebook_{notebook['notebook_id']}",
                    use_container_width=True,
                ):
                    load_notebook(notebook["notebook_id"])
                    st.rerun()
            with delete_col:
                if st.button(
                    "Delete",
                    key=f"delete_notebook_{notebook['notebook_id']}",
                    use_container_width=True,
                ):
                    delete_notebook(notebook["notebook_id"])
                    st.rerun()


# ============================================================
# 13. ALT DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_fact_data():
    if not FACT_FILE.exists():
        return None
    try:
        df = pd.read_csv(FACT_FILE)
    except Exception:
        return None

    required = {"State", "Year", "Indicator", "Value"}
    if not required.issubset(df.columns):
        return None

    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["State", "Indicator", "Year", "Value"])
    df["Year"] = df["Year"].astype(int)
    return df


def render_alt_data():
    st.markdown("<div class='page-header'>Alt Data</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-description'>Explore curated charts directly from the structured dataset.</div>",
        unsafe_allow_html=True,
    )

    df = load_fact_data()
    if df is None or df.empty:
        st.warning(
            "The Alt Data dashboard could not load data/demography_long_semantic.csv. "
            "The chatbot can still be used if backend.py has loaded its data successfully."
        )
        return

    indicators = sorted(df["Indicator"].dropna().astype(str).unique().tolist())
    states = sorted(df["State"].dropna().astype(str).unique().tolist())

    tab1, tab2, tab3 = st.tabs(["State trend", "State comparison", "Coverage"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            state = st.selectbox("State", states, key="alt_trend_state")
        with c2:
            indicator = st.selectbox(
                "Indicator", indicators, key="alt_trend_indicator"
            )

        subset = df[(df["State"] == state) & (df["Indicator"] == indicator)].sort_values("Year")
        if subset.empty:
            st.info("No observations are available for this state-indicator combination.")
        else:
            fig = px.line(subset, x="Year", y="Value", markers=True, title=f"{indicator} — {state}")
            fig.update_layout(xaxis_title="Year", yaxis_title=indicator)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(subset[["State", "Year", "Indicator", "Value"]], use_container_width=True, hide_index=True)

    with tab2:
        indicator2 = st.selectbox(
            "Indicator", indicators, key="alt_compare_indicator"
        )
        available_years = sorted(
            df.loc[df["Indicator"] == indicator2, "Year"].unique().tolist()
        )
        if available_years:
            year = st.selectbox(
                "Year", available_years, index=len(available_years) - 1, key="alt_compare_year"
            )
            subset = df[(df["Indicator"] == indicator2) & (df["Year"] == year)].copy()
            subset = subset[subset["State"] != "India"].sort_values("Value", ascending=False)
            top_n = st.slider("Number of states", 5, min(29, max(5, len(subset))), min(10, max(5, len(subset)))) if len(subset) >= 5 else len(subset)
            display = subset.head(top_n)
            if display.empty:
                st.info("No observations are available for this indicator and year.")
            else:
                fig = px.bar(display.sort_values("Value"), x="Value", y="State", orientation="h", title=f"{indicator2} — {year}")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(display[["State", "Year", "Value"]], use_container_width=True, hide_index=True)

    with tab3:
        coverage = (
            df.groupby("Indicator")
            .agg(
                Observations=("Value", "count"),
                States=("State", "nunique"),
                First_Year=("Year", "min"),
                Last_Year=("Year", "max"),
            )
            .reset_index()
            .sort_values("Indicator")
        )
        st.dataframe(coverage, use_container_width=True, hide_index=True)


# ============================================================
# 14. HELP
# ============================================================

def render_help():
    st.markdown("<div class='page-header'>Help</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-description'>How to use Sankhyaki effectively.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("What can I ask?", expanded=True):
        st.markdown(
            """
            Sankhyaki supports individual values, state comparisons, historical trends,
            rankings, percentage growth, absolute change, data coverage, definitions,
            sources and related metadata.

            Example questions:
            - What was Bihar's population density in 2026?
            - Compare Bihar and India population density in 2011 and 2026.
            - Show the trend in Bihar's population density over time.
            - Which states had the highest population density in 2026?
            - What is the source of population density data?
            """
        )

    with st.expander("How are numerical answers produced?"):
        st.markdown(
            """
            The language model interprets the question, but numerical retrieval and
            calculations are performed against the structured dataset. Missing values
            are not treated as zero, and ambiguous indicators should trigger a
            clarification request rather than a silent assumption.
            """
        )

    with st.expander("What happens if Gemini/API access is unavailable?"):
        st.markdown(
            """
            Sankhyaki displays a controlled error message instead of crashing the
            Streamlit website. This keeps the user interface separate from the
            natural-language interpretation service and deterministic analytical layer.
            """
        )


# ============================================================
# 15. ROUTERS + APPLICATION ENTRY POINT
# ============================================================

def render_authenticated_page():
    page = st.session_state.current_page

    if page == "New Notebook":
        render_chat_page()
    elif page == "Workspace":
        render_workspace()
    elif page == "Alt Data":
        render_alt_data()
    elif page == "Help":
        render_help()
    else:
        st.session_state.current_page = "New Notebook"
        render_chat_page()


def main():
    if not st.session_state.authenticated:
        render_authentication()
        return

    render_sidebar()
    render_authenticated_page()


if __name__ == "__main__":
    main()
