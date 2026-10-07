# Sankhyaki

Sankhyaki is a Streamlit-based Demography & Employment Data Assistant for Indian state-level statistics.

## Local run

1. Create and activate a Python environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` if Gemini fallback is required.
4. Run: `streamlit run app.py`

## Streamlit Community Cloud

Deploy `app.py` from the repository root. Add `GEMINI_API_KEY` (and optionally `GEMINI_PARSER_MODEL` and `GEMINI_REQUEST_TIMEOUT_SECONDS`) in the app's Secrets settings rather than committing a `.env` file.

## Data

Runtime data files are under `data/`. Missing observations are not converted to zero.

## Important persistence note

The current demo uses a local SQLite file (`users.db`) for accounts and notebook history. This is suitable for local/demo use but should not be treated as durable production storage on ephemeral cloud hosting. For persistent multi-user deployment, replace it with a managed database.
