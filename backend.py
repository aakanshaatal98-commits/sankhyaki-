# ============================================================
# DEMOGRAPHY AND EMPLOYMENT DATA ASSISTANT
# Production Backend
# ============================================================

# ============================================================
# 1. IMPORTS AND PROJECT PATHS
# ============================================================

import os
import re
import json
import math
import time
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import numpy as np
import pandas as pd
import duckdb

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from google import genai
from google.genai import types

import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------
# General configuration
# ------------------------------------------------------------

warnings.filterwarnings("ignore")

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.max_colwidth",
    None
)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_DIR = Path(
    __file__
).resolve().parent

DATA_DIR = (
    PROJECT_DIR
    / "data"
)

ENV_FILE = (
    PROJECT_DIR
    / ".env"
)


# ------------------------------------------------------------
# Backend-ready data files
# ------------------------------------------------------------

FACT_FILE = (
    DATA_DIR
    / "demography_long_semantic.csv"
)

INDICATOR_CATALOGUE_FILE = (
    DATA_DIR
    / "semantic_indicator_catalogue.csv"
)

GEOGRAPHY_CATALOGUE_FILE = (
    DATA_DIR
    / "semantic_geography_catalogue.csv"
)

# ============================================================
# 2. ENVIRONMENT AND GEMINI CONFIGURATION
# ============================================================

if ENV_FILE.exists():

    load_dotenv(
        dotenv_path=ENV_FILE
    )

else:

    # Deployment environments may provide secrets directly
    # rather than through a local .env file.
    load_dotenv()


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


# ------------------------------------------------------------
# Gemini client
# ------------------------------------------------------------

if GEMINI_API_KEY:

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

else:

    gemini_client = None


# ------------------------------------------------------------
# Parser configuration
# ------------------------------------------------------------

# Keep model configuration centralized so it can be changed
# without modifying the parser logic later.

GEMINI_PARSER_MODEL = os.getenv(
    "GEMINI_PARSER_MODEL",
    "gemini-3.8-flash"
)

# Hard network timeout for Gemini fallback requests. This prevents one
# external API call from blocking the chatbot/evaluation indefinitely.
GEMINI_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "15")
)

# ============================================================
# 3. LOAD BACKEND-READY DATA FILES
# ============================================================

REQUIRED_DATA_FILES = {
    "fact": FACT_FILE,
    "indicator_catalogue": INDICATOR_CATALOGUE_FILE,
    "geography_catalogue": GEOGRAPHY_CATALOGUE_FILE
}


missing_files = [
    str(path)
    for path in REQUIRED_DATA_FILES.values()
    if not path.exists()
]


if missing_files:

    raise FileNotFoundError(
        "Required backend data file(s) were not found:\n"
        + "\n".join(missing_files)
    )


# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------

fact_df = pd.read_csv(
    FACT_FILE
)

semantic_indicator_catalogue = pd.read_csv(
    INDICATOR_CATALOGUE_FILE
)

semantic_geography_catalogue = pd.read_csv(
    GEOGRAPHY_CATALOGUE_FILE
)

# ============================================================
# 4. BUILD ANALYTICAL FACT TABLE
# ============================================================

REQUIRED_FACT_COLUMNS = [
    "State",
    "Year",
    "Indicator",
    "Value"
]


missing_fact_columns = [
    column
    for column in REQUIRED_FACT_COLUMNS
    if column not in fact_df.columns
]


if missing_fact_columns:

    raise ValueError(
        "Fact dataset is missing required column(s): "
        + ", ".join(missing_fact_columns)
    )


# ------------------------------------------------------------
# Select analytical fields
# ------------------------------------------------------------

analytical_fact_table = (
    fact_df[
        REQUIRED_FACT_COLUMNS
    ]
    .copy()
)


# ------------------------------------------------------------
# Standardize analytical fields
# ------------------------------------------------------------

analytical_fact_table["State"] = (
    analytical_fact_table["State"]
    .astype(str)
    .str.strip()
)


analytical_fact_table["Indicator"] = (
    analytical_fact_table["Indicator"]
    .astype(str)
    .str.strip()
)


analytical_fact_table["Year"] = (
    pd.to_numeric(
        analytical_fact_table["Year"],
        errors="coerce"
    )
)


analytical_fact_table["Value"] = (
    pd.to_numeric(
        analytical_fact_table["Value"],
        errors="coerce"
    )
)


# ------------------------------------------------------------
# Remove invalid identifiers
# ------------------------------------------------------------

analytical_fact_table = (
    analytical_fact_table
    .dropna(
        subset=[
            "State",
            "Year",
            "Indicator"
        ]
    )
    .copy()
)


analytical_fact_table["Year"] = (
    analytical_fact_table["Year"]
    .astype(int)
)


# ------------------------------------------------------------
# Missing values are never converted to zero
# ------------------------------------------------------------

analytical_fact_table = (
    analytical_fact_table
    .dropna(
        subset=[
            "Value"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# Validate unique State × Year × Indicator observations
# ------------------------------------------------------------

duplicate_fact_rows = (
    analytical_fact_table
    .duplicated(
        subset=[
            "State",
            "Year",
            "Indicator"
        ],
        keep=False
    )
)


if duplicate_fact_rows.any():

    duplicate_count = int(
        duplicate_fact_rows.sum()
    )

    raise ValueError(
        "Analytical fact table contains "
        f"{duplicate_count} duplicate "
        "State-Year-Indicator rows."
    )

# ============================================================
# 5. SEMANTIC VOCABULARY AND ALIASES
# ============================================================


# ------------------------------------------------------------
# Canonical geographies
# ------------------------------------------------------------

CANONICAL_STATES = sorted(
    analytical_fact_table[
        "State"
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)


# ------------------------------------------------------------
# Canonical indicators
# ------------------------------------------------------------

CANONICAL_INDICATORS = sorted(
    analytical_fact_table[
        "Indicator"
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)


# ------------------------------------------------------------
# Normalization helper
# ------------------------------------------------------------

def normalize_text(value):
    """
    Normalize text for deterministic semantic matching.
    """

    if value is None:
        return ""

    value = str(
        value
    ).strip().lower()

    value = re.sub(
        r"[_\-]+",
        " ",
        value
    )

    value = re.sub(
        r"[^\w\s%]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ------------------------------------------------------------
# Canonical lookup dictionaries
# ------------------------------------------------------------

CANONICAL_STATE_LOOKUP = {
    normalize_text(state): state
    for state in CANONICAL_STATES
}


CANONICAL_INDICATOR_LOOKUP = {
    normalize_text(indicator): indicator
    for indicator in CANONICAL_INDICATORS
}


# ------------------------------------------------------------
# Collision safeguards
# ------------------------------------------------------------

if (
    len(CANONICAL_STATE_LOOKUP)
    !=
    len(CANONICAL_STATES)
):

    raise ValueError(
        "Canonical state normalization produced "
        "duplicate lookup keys."
    )


if (
    len(CANONICAL_INDICATOR_LOOKUP)
    !=
    len(CANONICAL_INDICATORS)
):

    raise ValueError(
        "Canonical indicator normalization produced "
        "duplicate lookup keys."
    )

# ============================================================
# 5A. STATE ALIASES
# ============================================================

RAW_STATE_ALIASES = {

    "andhra": "Andhra Pradesh",
    "ap": "Andhra Pradesh",

    "arunachal": "Arunachal Pradesh",

    "assam": "Assam",

    "bihar": "Bihar",

    "chhattisgarh": "Chhattisgarh",
    "chattisgarh": "Chhattisgarh",

    "goa": "Goa",

    "gujarat": "Gujarat",

    "haryana": "Haryana",

    "himachal": "Himachal Pradesh",
    "hp": "Himachal Pradesh",

    "jharkhand": "Jharkhand",

    "karnataka": "Karnataka",

    "kerala": "Kerala",

    "madhya": "Madhya Pradesh",
    "madhya pradesh": "Madhya Pradesh",
    "mp": "Madhya Pradesh",

    "maharashtra": "Maharashtra",

    "manipur": "Manipur",

    "meghalaya": "Meghalaya",

    "mizoram": "Mizoram",

    "nagaland": "Nagaland",

    "odisha": "Odisha",
    "orissa": "Odisha",

    "punjab": "Punjab",

    "rajasthan": "Rajasthan",

    "sikkim": "Sikkim",

    "tamil nadu": "Tamil Nadu",
    "tn": "Tamil Nadu",

    "telangana": "Telangana",

    "tripura": "Tripura",

    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",

    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",

    "west bengal": "West Bengal",
    "wb": "West Bengal",

    "india": "India"
}


# Keep only aliases whose targets actually exist in the data.

STATE_ALIASES = {
    normalize_text(alias): canonical
    for alias, canonical in RAW_STATE_ALIASES.items()
    if canonical in CANONICAL_STATES
}

# ============================================================
# 5B. INDICATOR ALIASES
# ============================================================

# The deterministic vocabulary is data-driven.  The JSON file is optional
# for backwards compatibility, but when present it is the authoritative
# human-friendly alias catalogue shipped with the project.
INDICATOR_ALIAS_FILE = DATA_DIR / "indicator_aliases.json"

RAW_INDICATOR_ALIASES = {
    "population density": "Population Density (per square km)",
    "density": "Population Density (per square km)",
    "dependency ratio": "Dependency Ratio (%)",
    "urban population share": "Urban Population as a Share of Total State Population (%)",
    "urban population percentage": "Urban Population as a Share of Total State Population (%)",
    "urbanisation": "Urban Population as a Share of Total State Population (%)",
    "urbanization": "Urban Population as a Share of Total State Population (%)",
}

if INDICATOR_ALIAS_FILE.exists():
    try:
        with open(INDICATOR_ALIAS_FILE, "r", encoding="utf-8") as handle:
            external_aliases = json.load(handle)
        if isinstance(external_aliases, dict):
            RAW_INDICATOR_ALIASES.update(external_aliases)
    except Exception as exc:
        warnings.warn(f"Could not load indicator_aliases.json: {exc}")

# Every canonical indicator is automatically a deterministic alias as well.
# This guarantees that the exact dataset vocabulary never requires Gemini.
for canonical_indicator in CANONICAL_INDICATORS:
    RAW_INDICATOR_ALIASES.setdefault(canonical_indicator, canonical_indicator)

# Keep only aliases whose canonical targets actually exist in the data.
INDICATOR_ALIASES = {
    normalize_text(alias): canonical
    for alias, canonical in RAW_INDICATOR_ALIASES.items()
    if canonical in CANONICAL_INDICATORS
}

# Useful short forms that can be derived unambiguously from the catalogue.
def _add_unique_indicator_alias(alias, required_terms):
    candidates = [
        indicator for indicator in CANONICAL_INDICATORS
        if all(term in normalize_text(indicator) for term in required_terms)
    ]
    if len(candidates) == 1:
        INDICATOR_ALIASES[normalize_text(alias)] = candidates[0]

_add_unique_indicator_alias("female unemployment", ["female", "unemployment"])
_add_unique_indicator_alias("population", ["population", "000s"])
_add_unique_indicator_alias("lfpr", ["labour force participation rate", "15 years and above"])
_add_unique_indicator_alias("unemployment rate", ["unemployment rate", "15 years and above"])

# Generic employment phrases must resolve to the overall (not sex/residence-specific)
# indicators.  These are the phrases used by the UI suggestion cards.
_GENERIC_EMPLOYMENT_ALIASES = {
    "unemployment rate": "Unemployment Rate, Age 15 Years and Above (%)",
    "unemployment": "Unemployment Rate, Age 15 Years and Above (%)",
    "employment": "Unemployment Rate, Age 15 Years and Above (%)",
    "employment rate": "Unemployment Rate, Age 15 Years and Above (%)",
    "employment situation": "Unemployment Rate, Age 15 Years and Above (%)",
    "labour force participation": "Labour Force Participation Rate, Age 15 Years and Above (%)",
    "labour force participation rate": "Labour Force Participation Rate, Age 15 Years and Above (%)",
    "labor force participation": "Labour Force Participation Rate, Age 15 Years and Above (%)",
    "labor force participation rate": "Labour Force Participation Rate, Age 15 Years and Above (%)",
    "lfpr": "Labour Force Participation Rate, Age 15 Years and Above (%)",
}
for _alias, _canonical in _GENERIC_EMPLOYMENT_ALIASES.items():
    if _canonical in CANONICAL_INDICATORS:
        INDICATOR_ALIASES[normalize_text(_alias)] = _canonical

# ============================================================
# 6. STATE AND INDICATOR RESOLUTION
# ============================================================

def make_resolution_result(
    status,
    query,
    canonical=None,
    candidates=None
):
    """
    Standard result returned by semantic resolvers.
    """

    return {
        "status": status,
        "query": query,
        "canonical": canonical,
        "candidates": (
            candidates
            if candidates is not None
            else []
        )
    }


def resolve_state(
    state
):
    """
    Resolve a geography to a canonical dataset geography.
    """

    if state is None:

        return make_resolution_result(
            status="unresolved",
            query=state
        )


    normalized = normalize_text(
        state
    )


    # Exact canonical match
    if normalized in CANONICAL_STATE_LOOKUP:

        return make_resolution_result(
            status="resolved",
            query=state,
            canonical=(
                CANONICAL_STATE_LOOKUP[
                    normalized
                ]
            )
        )


    # Alias match
    if normalized in STATE_ALIASES:

        return make_resolution_result(
            status="resolved",
            query=state,
            canonical=(
                STATE_ALIASES[
                    normalized
                ]
            )
        )


    return make_resolution_result(
        status="unresolved",
        query=state
    )

def resolve_indicator_exact(
    indicator
):
    """
    Resolve exact canonical indicator names and
    deterministic indicator aliases.
    """

    if indicator is None:

        return make_resolution_result(
            status="unresolved",
            query=indicator
        )


    normalized = normalize_text(
        indicator
    )


    # Exact canonical match
    if normalized in CANONICAL_INDICATOR_LOOKUP:

        return make_resolution_result(
            status="resolved",
            query=indicator,
            canonical=(
                CANONICAL_INDICATOR_LOOKUP[
                    normalized
                ]
            )
        )


    # Alias match
    if normalized in INDICATOR_ALIASES:

        return make_resolution_result(
            status="resolved",
            query=indicator,
            canonical=(
                INDICATOR_ALIASES[
                    normalized
                ]
            )
        )


    return make_resolution_result(
        status="unresolved",
        query=indicator
    )

def find_indicator_candidates(
    indicator
):
    """
    Find plausible canonical indicators without
    automatically selecting among ambiguous matches.
    """

    if indicator is None:
        return []


    normalized_query = normalize_text(
        indicator
    )


    if not normalized_query:
        return []


    query_tokens = set(
        normalized_query.split()
    )


    candidates = []


    for canonical in CANONICAL_INDICATORS:

        normalized_canonical = (
            normalize_text(
                canonical
            )
        )

        canonical_tokens = set(
            normalized_canonical.split()
        )


        # Direct substring relationship
        if (
            normalized_query
            in normalized_canonical
            or
            normalized_canonical
            in normalized_query
        ):

            candidates.append(
                canonical
            )

            continue


        # Token overlap
        overlap = (
            query_tokens
            &
            canonical_tokens
        )


        if (
            query_tokens
            and
            len(overlap)
            ==
            len(query_tokens)
        ):

            candidates.append(
                canonical
            )


    return sorted(
        set(candidates)
    )

def resolve_indicator(
    indicator
):
    """
    Resolve an indicator while preserving ambiguity.

    The function never silently chooses among multiple
    plausible canonical indicators.
    """

    exact_result = (
        resolve_indicator_exact(
            indicator
        )
    )


    if (
        exact_result["status"]
        ==
        "resolved"
    ):

        return exact_result


    candidates = (
        find_indicator_candidates(
            indicator
        )
    )


    if len(candidates) == 1:

        return make_resolution_result(
            status="resolved",
            query=indicator,
            canonical=candidates[0],
            candidates=candidates
        )


    if len(candidates) > 1:

        return make_resolution_result(
            status="ambiguous",
            query=indicator,
            candidates=candidates
        )


    return make_resolution_result(
        status="unresolved",
        query=indicator
    )

# ============================================================
# 7. STRUCTURED QUERY PLAN
# ============================================================

MetadataRequestType = Literal[
    "definition",
    "source",
    "link",
    "procurement",
    "update",
    "calculation_notes",
    "coverage",
    "general"
]


OperationType = Literal[
    "lookup",
    "compare",
    "trend",
    "change",
    "growth",
    "rank",
    "metadata"
]


class QueryPlan(
    BaseModel
):
    """
    Structured representation of a user's analytical request.

    Gemini may propose this structure, but all entities and
    operation requirements are validated deterministically
    before analytical execution.
    """

    operation: OperationType

    states: List[str] = Field(
        default_factory=list
    )

    years: List[int] = Field(
        default_factory=list
    )

    indicator: Optional[str] = None

    metadata_request: Optional[
        MetadataRequestType
    ] = None

    top_n: Optional[int] = Field(
        default=None,
        ge=1,
        le=100
    )

# ============================================================
# BACKEND PART 1 — DEVELOPMENT AUDIT
# ============================================================

def backend_part1_audit():
    """
    Lightweight development audit for Backend Part 1.
    """

    print("Backend Part 1 loaded successfully.")
    print("Project directory:", PROJECT_DIR)
    print("Fact observations:", len(analytical_fact_table))
    print("Canonical geographies:", len(CANONICAL_STATES))
    print("Canonical indicators:", len(CANONICAL_INDICATORS))
    print(
        "Indicator metadata records:",
        len(semantic_indicator_catalogue)
    )
    print(
        "Geography catalogue records:",
        len(semantic_geography_catalogue)
    )
    print(
        "Gemini configured:",
        gemini_client is not None
    )


# ============================================================
# RUN DEVELOPMENT TESTS
# ============================================================

if __name__ == "__main__":

    backend_part1_audit()

    print("\nState resolution test:")
    print(
        resolve_state("Bihar")
    )

    print("\nState alias test:")
    print(
        resolve_state("UP")
    )

    print("\nIndicator resolution test:")
    print(
        resolve_indicator(
            "population density"
        )
    )

    print("\nAmbiguity test:")
    print(
        resolve_indicator(
            "sex ratio"
        )
    )

    print("\nQueryPlan test:")

    test_plan = QueryPlan(
        operation="compare",
        states=[
            "Bihar",
            "India"
        ],
        years=[
            2011,
            2026
        ],
        indicator="population density"
    )

    print(
        test_plan.model_dump()
    )

# ============================================================
# 8. QUERY VALIDATION AND CANONICALISATION
# ============================================================

def make_validation_result(
    valid,
    plan=None,
    errors=None,
    warnings=None,
    clarification=None
):
    """
    Standard result returned by deterministic query validation.
    """

    return {
        "valid": bool(valid),
        "plan": plan,
        "errors": errors or [],
        "warnings": warnings or [],
        "clarification": clarification
    }

def canonicalize_states(states):
    """
    Resolve requested geography names to canonical dataset names.

    Returns:
        canonical_states
        unresolved_states
    """

    canonical_states = []
    unresolved_states = []

    for state in states or []:

        result = resolve_state(state)

        if result["status"] == "resolved":

            canonical = result["canonical"]

            if canonical not in canonical_states:
                canonical_states.append(canonical)

        else:
            unresolved_states.append(state)

    return canonical_states, unresolved_states

def canonicalize_indicator(indicator):
    """
    Resolve a requested indicator while preserving ambiguity.
    """

    if indicator is None:
        return {
            "status": "unresolved",
            "canonical": None,
            "candidates": []
        }

    result = resolve_indicator(indicator)

    return {
        "status": result["status"],
        "canonical": result.get("canonical"),
        "candidates": result.get("candidates", [])
    }

def build_indicator_clarification(
    indicator,
    candidates
):
    """
    Construct a user-facing clarification message
    for ambiguous indicator terminology.
    """

    if not candidates:

        return (
            f"I could not identify a supported indicator "
            f"for '{indicator}'."
        )

    candidate_text = "\n".join(
        f"- {candidate}"
        for candidate in candidates
    )

    return (
        f"The term '{indicator}' matches more than one "
        "indicator. Please specify one of the following:\n"
        f"{candidate_text}"
    )

OPERATION_REQUIREMENTS = {

    "lookup": {
        "states": True,
        "years": True,
        "indicator": True
    },

    "compare": {
        "states": False,
        "years": False,
        "indicator": True
    },

    "trend": {
        "states": True,
        "years": False,
        "indicator": True
    },

    "change": {
        "states": True,
        "years": True,
        "indicator": True
    },

    "growth": {
        "states": True,
        "years": True,
        "indicator": True
    },

    "rank": {
        "states": False,
        "years": True,
        "indicator": True
    },

    "metadata": {
        "states": False,
        "years": False,
        "indicator": True
    }
}

def validate_operation_requirements(plan):
    """
    Validate whether required fields are present
    for the requested analytical operation.
    """

    errors = []

    requirements = OPERATION_REQUIREMENTS.get(
        plan.operation
    )

    if requirements is None:

        return [
            f"Unsupported operation: {plan.operation}"
        ]

    if (
        requirements["states"]
        and
        not plan.states
    ):
        errors.append(
            f"Operation '{plan.operation}' "
            "requires at least one geography."
        )

    if (
        requirements["years"]
        and
        not plan.years
    ):
        errors.append(
            f"Operation '{plan.operation}' "
            "requires at least one year."
        )

    if (
        requirements["indicator"]
        and
        not plan.indicator
    ):
        errors.append(
            f"Operation '{plan.operation}' "
            "requires an indicator."
        )

    return errors

def validate_operation_cardinality(plan):
    """
    Validate operation-specific numbers of states and years.
    """

    errors = []

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    if plan.operation == "lookup":

        if not plan.states:
            errors.append(
                "Lookup requires at least one geography."
            )

        if not plan.years:
            errors.append(
                "Lookup requires at least one year."
            )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    elif plan.operation == "compare":

        if (
            len(plan.states) < 2
            and
            len(plan.years) < 2
        ):
            errors.append(
                "Comparison requires at least two "
                "geographies, two years, or both."
            )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    elif plan.operation == "trend":

        if len(plan.states) != 1:
            errors.append(
                "Trend analysis requires exactly "
                "one geography."
            )

    # --------------------------------------------------------
    # Change / Growth
    # --------------------------------------------------------

    elif plan.operation in {
        "change",
        "growth"
    }:

        if len(plan.states) < 1:
            errors.append(
                f"Operation '{plan.operation}' requires "
                "at least one geography."
            )

        if len(plan.years) != 2:
            errors.append(
                f"Operation '{plan.operation}' requires "
                "exactly two years."
            )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    elif plan.operation == "rank":

        if len(plan.years) != 1:
            errors.append(
                "Ranking requires exactly one year."
            )

    return errors

def validate_year_values(years):
    """
    Validate and normalize requested years.
    """

    canonical_years = []
    errors = []

    for year in years or []:

        try:
            numeric_year = int(year)

        except (TypeError, ValueError):

            errors.append(
                f"Invalid year: {year}"
            )

            continue

        # Broad structural safeguard.
        # Availability is checked separately against the data.
        if (
            numeric_year < 1900
            or
            numeric_year > 2200
        ):

            errors.append(
                f"Year {numeric_year} is outside "
                "the supported validation range."
            )

            continue

        if numeric_year not in canonical_years:
            canonical_years.append(
                numeric_year
            )

    return canonical_years, errors

def canonicalize_query_plan(plan):
    """
    Resolve a QueryPlan into canonical dataset terminology.

    This function performs semantic canonicalisation only.
    It does not retrieve numerical observations.
    """

    if not isinstance(plan, QueryPlan):

        try:
            plan = QueryPlan.model_validate(
                plan
            )

        except ValidationError as exc:

            return make_validation_result(
                valid=False,
                errors=[
                    f"QueryPlan validation failed: {exc}"
                ]
            )

    errors = []
    warnings = []

    # --------------------------------------------------------
    # States
    # --------------------------------------------------------

    canonical_states, unresolved_states = (
        canonicalize_states(
            plan.states
        )
    )

    if unresolved_states:

        errors.append(
            "Unsupported geography/geographies: "
            + ", ".join(
                str(state)
                for state in unresolved_states
            )
        )

    # --------------------------------------------------------
    # Indicator
    # --------------------------------------------------------

    canonical_indicator = None

    if plan.indicator is not None:

        indicator_result = (
            canonicalize_indicator(
                plan.indicator
            )
        )

        if (
            indicator_result["status"]
            ==
            "resolved"
        ):

            canonical_indicator = (
                indicator_result[
                    "canonical"
                ]
            )

        elif (
            indicator_result["status"]
            ==
            "ambiguous"
        ):

            clarification = (
                build_indicator_clarification(
                    plan.indicator,
                    indicator_result[
                        "candidates"
                    ]
                )
            )

            return make_validation_result(
                valid=False,
                plan=None,
                errors=[],
                warnings=warnings,
                clarification=clarification
            )

        else:

            errors.append(
                "Unsupported indicator: "
                f"{plan.indicator}"
            )

    # --------------------------------------------------------
    # Years
    # --------------------------------------------------------

    canonical_years, year_errors = (
        validate_year_values(
            plan.years
        )
    )

    errors.extend(
        year_errors
    )

    if errors:

        return make_validation_result(
            valid=False,
            errors=errors,
            warnings=warnings
        )

    # --------------------------------------------------------
    # Canonical plan
    # --------------------------------------------------------

    canonical_plan = plan.model_copy(
        update={
            "states": canonical_states,
            "years": canonical_years,
            "indicator": canonical_indicator
        }
    )

    return make_validation_result(
        valid=True,
        plan=canonical_plan,
        warnings=warnings
    )

def validate_query_plan(plan):
    """
    Complete deterministic validation pipeline.
    """

    canonical_result = (
        canonicalize_query_plan(
            plan
        )
    )

    if not canonical_result["valid"]:

        return canonical_result

    canonical_plan = (
        canonical_result["plan"]
    )

    errors = []

    errors.extend(
        validate_operation_requirements(
            canonical_plan
        )
    )

    errors.extend(
        validate_operation_cardinality(
            canonical_plan
        )
    )

    # --------------------------------------------------------
    # Metadata-specific requirement
    # --------------------------------------------------------

    if (
        canonical_plan.operation
        ==
        "metadata"
        and
        canonical_plan.metadata_request
        is None
    ):

        canonical_plan = (
            canonical_plan.model_copy(
                update={
                    "metadata_request":
                    "general"
                }
            )
        )

    if errors:

        return make_validation_result(
            valid=False,
            plan=canonical_plan,
            errors=errors,
            warnings=canonical_result[
                "warnings"
            ]
        )

    return make_validation_result(
        valid=True,
        plan=canonical_plan,
        warnings=canonical_result[
            "warnings"
        ]
    )

# ============================================================
# 9. AVAILABILITY HELPERS
# ============================================================

availability_index = (
    analytical_fact_table[
        [
            "State",
            "Year",
            "Indicator"
        ]
    ]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)

def is_data_available(
    state,
    year,
    indicator
):
    """
    Check whether an exact verified observation exists.
    """

    match = availability_index[
        (
            availability_index["State"]
            ==
            state
        )
        &
        (
            availability_index["Year"]
            ==
            int(year)
        )
        &
        (
            availability_index["Indicator"]
            ==
            indicator
        )
    ]

    return not match.empty

def get_available_years(
    state,
    indicator
):
    """
    Return all available years for a
    State × Indicator combination.
    """

    years = (
        availability_index.loc[
            (
                availability_index["State"]
                ==
                state
            )
            &
            (
                availability_index["Indicator"]
                ==
                indicator
            ),
            "Year"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    return sorted(
        years
    )

def get_available_states(
    indicator,
    year=None,
    include_india=True
):
    """
    Return geographies with verified observations
    for an indicator and optional year.
    """

    subset = availability_index[
        availability_index["Indicator"]
        ==
        indicator
    ].copy()

    if year is not None:

        subset = subset[
            subset["Year"]
            ==
            int(year)
        ]

    states = (
        subset["State"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not include_india:

        states = [
            state
            for state in states
            if state != "India"
        ]

    return sorted(
        states
    )

def get_available_indicators(
    state=None,
    year=None
):
    """
    Return indicators available for optional
    geography/year restrictions.
    """

    subset = (
        availability_index.copy()
    )

    if state is not None:

        subset = subset[
            subset["State"]
            ==
            state
        ]

    if year is not None:

        subset = subset[
            subset["Year"]
            ==
            int(year)
        ]

    return sorted(
        subset["Indicator"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

# ============================================================
# 10. DETERMINISTIC ANALYTICAL OPERATIONS
# ============================================================

def to_python_scalar(value):
    """
    Convert NumPy/Pandas scalar values into
    JSON-safe native Python values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            np.integer,
        )
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
        )
    ):
        if np.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        np.bool_
    ):
        return bool(value)

    if pd.isna(value):
        return None

    return value

def records_to_python(records):
    """
    Convert record dictionaries to JSON-safe values.
    """

    safe_records = []

    for record in records:

        safe_record = {
            key: to_python_scalar(value)
            for key, value in record.items()
        }

        safe_records.append(
            safe_record
        )

    return safe_records

def dataframe_to_records(
    dataframe
):
    """
    Convert a DataFrame to JSON-safe record dictionaries.
    """

    if dataframe is None:
        return []

    if dataframe.empty:
        return []

    records = dataframe.to_dict(
        orient="records"
    )

    return records_to_python(
        records
    )

def make_execution_result(
    success,
    operation,
    data=None,
    message=None,
    warnings=None,
    calculation=None,
    metadata=None
):
    """
    Standard result returned by deterministic operations.
    """

    return {
        "success": bool(success),
        "operation": operation,
        "data": data or [],
        "message": message,
        "warnings": warnings or [],
        "calculation": calculation,
        "metadata": metadata
    }

def filter_fact_table(
    states=None,
    years=None,
    indicator=None,
    data=analytical_fact_table
):
    """
    Deterministically filter the analytical fact table.
    """

    subset = data.copy()

    if states:

        subset = subset[
            subset["State"].isin(
                states
            )
        ]

    if years:

        subset = subset[
            subset["Year"].isin(
                years
            )
        ]

    if indicator is not None:

        subset = subset[
            subset["Indicator"]
            ==
            indicator
        ]

    return subset.copy()

def execute_lookup(plan):
    """
    Retrieve exact requested observations.

    Missing observations are reported explicitly
    and are never replaced with zero.
    """

    rows = []
    warnings = []

    for state in plan.states:

        for year in plan.years:

            match = filter_fact_table(
                states=[state],
                years=[year],
                indicator=plan.indicator
            )

            if match.empty:

                rows.append({
                    "State": state,
                    "Year": year,
                    "Indicator": plan.indicator,
                    "Value": None,
                    "Data_Status": "Unavailable",
                    "Available_Years":
                        get_available_years(
                            state,
                            plan.indicator
                        )
                })

                warnings.append(
                    f"No verified observation was "
                    f"available for {state}, "
                    f"{year}, {plan.indicator}."
                )

            else:

                record = (
                    match.iloc[0]
                    .to_dict()
                )

                rows.append({
                    "State":
                        to_python_scalar(
                            record["State"]
                        ),

                    "Year":
                        to_python_scalar(
                            record["Year"]
                        ),

                    "Indicator":
                        to_python_scalar(
                            record["Indicator"]
                        ),

                    "Value":
                        to_python_scalar(
                            record["Value"]
                        ),

                    "Data_Status":
                        "Available",

                    "Available_Years":
                        get_available_years(
                            state,
                            plan.indicator
                        )
                })

    available_rows = [
        row
        for row in rows
        if row["Data_Status"]
        ==
        "Available"
    ]

    success = (
        len(available_rows)
        >
        0
    )

    message = (
        None
        if success
        else
        "No requested observations were available."
    )

    return make_execution_result(
        success=success,
        operation="lookup",
        data=rows,
        message=message,
        warnings=warnings
    )

def execute_compare(plan):
    """
    Compare an indicator across requested
    states and/or years.
    """

    subset = filter_fact_table(
        states=(
            plan.states
            if plan.states
            else None
        ),
        years=(
            plan.years
            if plan.years
            else None
        ),
        indicator=plan.indicator
    )

    if subset.empty:

        return make_execution_result(
            success=False,
            operation="compare",
            message=(
                "No requested observations "
                "were available."
            )
        )

    subset = (
        subset[
            [
                "State",
                "Year",
                "Indicator",
                "Value"
            ]
        ]
        .sort_values(
            by=[
                "Year",
                "State"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    warnings = []

    # --------------------------------------------------------
    # Identify requested combinations that are unavailable
    # when both states and years were explicitly supplied.
    # --------------------------------------------------------

    if (
        plan.states
        and
        plan.years
    ):

        for state in plan.states:

            for year in plan.years:

                exists = (
                    (
                        subset["State"]
                        ==
                        state
                    )
                    &
                    (
                        subset["Year"]
                        ==
                        year
                    )
                ).any()

                if not exists:

                    warnings.append(
                        f"No verified observation was "
                        f"available for {state}, {year}."
                    )

    return make_execution_result(
        success=True,
        operation="compare",
        data=dataframe_to_records(
            subset
        ),
        warnings=warnings
    )

def execute_trend(plan):
    """
    Retrieve an indicator across time for one geography.
    """

    state = plan.states[0]

    subset = filter_fact_table(
        states=[state],
        years=(
            plan.years
            if plan.years
            else None
        ),
        indicator=plan.indicator
    )

    if subset.empty:

        return make_execution_result(
            success=False,
            operation="trend",
            message=(
                "No trend observations were "
                "available for the requested query."
            )
        )

    subset = (
        subset[
            [
                "State",
                "Year",
                "Indicator",
                "Value"
            ]
        ]
        .sort_values(
            "Year"
        )
        .reset_index(
            drop=True
        )
    )

    return make_execution_result(
        success=True,
        operation="trend",
        data=dataframe_to_records(
            subset
        )
    )

def execute_change(plan):
    """
    Calculate deterministic absolute change between
    exactly two years for one or more geographies.
    """

    start_year, end_year = sorted(
        plan.years
    )

    results = []
    warnings = []

    for state in plan.states:

        start_match = filter_fact_table(
            states=[state],
            years=[start_year],
            indicator=plan.indicator
        )

        end_match = filter_fact_table(
            states=[state],
            years=[end_year],
            indicator=plan.indicator
        )

        if (
            start_match.empty
            or
            end_match.empty
        ):

            warnings.append(
                f"Change could not be calculated "
                f"for {state} because one or both "
                "requested observations were unavailable."
            )

            continue

        start_value = float(
            start_match.iloc[0][
                "Value"
            ]
        )

        end_value = float(
            end_match.iloc[0][
                "Value"
            ]
        )

        absolute_change = (
            end_value
            -
            start_value
        )

        results.append({
            "State": state,
            "Indicator": plan.indicator,
            "Start_Year": start_year,
            "End_Year": end_year,
            "Start_Value": start_value,
            "End_Value": end_value,
            "Absolute_Change":
                absolute_change
        })

    if not results:

        return make_execution_result(
            success=False,
            operation="change",
            message=(
                "The requested change could not "
                "be calculated from available data."
            ),
            warnings=warnings,
            calculation=(
                "Absolute Change = "
                "End Value - Start Value"
            )
        )

    return make_execution_result(
        success=True,
        operation="change",
        data=records_to_python(
            results
        ),
        warnings=warnings,
        calculation=(
            "Absolute Change = "
            "End Value - Start Value"
        )
    )

def execute_growth(plan):
    """
    Calculate deterministic percentage growth between
    exactly two years for one or more geographies.
    """

    start_year, end_year = sorted(
        plan.years
    )

    results = []
    warnings = []

    for state in plan.states:

        start_match = filter_fact_table(
            states=[state],
            years=[start_year],
            indicator=plan.indicator
        )

        end_match = filter_fact_table(
            states=[state],
            years=[end_year],
            indicator=plan.indicator
        )

        if (
            start_match.empty
            or
            end_match.empty
        ):

            warnings.append(
                f"Growth could not be calculated "
                f"for {state} because one or both "
                "requested observations were unavailable."
            )

            continue

        start_value = float(
            start_match.iloc[0][
                "Value"
            ]
        )

        end_value = float(
            end_match.iloc[0][
                "Value"
            ]
        )

        # ----------------------------------------------------
        # Zero-baseline safeguard
        # ----------------------------------------------------

        if start_value == 0:

            warnings.append(
                f"Percentage growth could not be "
                f"calculated for {state} because "
                f"the starting value in {start_year} "
                "was zero."
            )

            continue

        growth_percent = (
            (
                end_value
                -
                start_value
            )
            /
            start_value
        ) * 100

        results.append({
            "State": state,
            "Indicator": plan.indicator,
            "Start_Year": start_year,
            "End_Year": end_year,
            "Start_Value": start_value,
            "End_Value": end_value,
            "Growth_Percent":
                growth_percent
        })

    if not results:

        return make_execution_result(
            success=False,
            operation="growth",
            message=(
                "The requested growth rate could not "
                "be calculated from available data."
            ),
            warnings=warnings,
            calculation=(
                "Growth (%) = "
                "((End Value - Start Value) / "
                "Start Value) × 100"
            )
        )

    return make_execution_result(
        success=True,
        operation="growth",
        data=records_to_python(
            results
        ),
        warnings=warnings,
        calculation=(
            "Growth (%) = "
            "((End Value - Start Value) / "
            "Start Value) × 100"
        )
    )

def execute_rank(plan):
    """
    Rank states by a requested indicator for one year.

    India is excluded because it is a national aggregate,
    not a state.
    """

    year = int(
        plan.years[0]
    )

    ranking_df = filter_fact_table(
        years=[year],
        indicator=plan.indicator
    )

    # --------------------------------------------------------
    # State-ranking universe safeguard
    # --------------------------------------------------------

    ranking_df = ranking_df[
        ranking_df["State"]
        !=
        "India"
    ].copy()

    # --------------------------------------------------------
    # Optional state restriction
    # --------------------------------------------------------

    if plan.states:

        requested_states = [
            state
            for state in plan.states
            if state != "India"
        ]

        ranking_df = ranking_df[
            ranking_df["State"].isin(
                requested_states
            )
        ].copy()

    ranking_df["Value"] = (
        pd.to_numeric(
            ranking_df["Value"],
            errors="coerce"
        )
    )

    ranking_df = (
        ranking_df
        .dropna(
            subset=[
                "State",
                "Value"
            ]
        )
        .copy()
    )

    if ranking_df.empty:

        return make_execution_result(
            success=False,
            operation="rank",
            message=(
                "No verified state observations "
                "were available for the requested "
                "indicator and year."
            )
        )

    # --------------------------------------------------------
    # Highest value = Rank 1
    # --------------------------------------------------------

    ranking_df = (
        ranking_df
        .sort_values(
            by="Value",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    ranking_df["Rank"] = (
        ranking_df["Value"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    top_n = (
        plan.top_n
        if plan.top_n is not None
        else 5
    )

    ranking_df = (
        ranking_df
        .head(top_n)
        .copy()
    )

    ranking_df = ranking_df[
        [
            "State",
            "Year",
            "Indicator",
            "Value",
            "Rank"
        ]
    ]

    return make_execution_result(
        success=True,
        operation="rank",
        data=dataframe_to_records(
            ranking_df
        ),
        calculation=(
            "States ranked in descending order "
            "of verified indicator values. "
            "India is excluded from the "
            "state-ranking universe."
        )
    )

METADATA_REQUEST_MAP = {

    "definition": [
        "Definitions"
    ],

    "source": [
        "Data Source"
    ],

    "link": [
        "Data Links"
    ],

    "procurement": [
        "Procured From"
    ],

    "update": [
        "Previous Update",
        "Current Update"
    ],

    "calculation_notes": [
        "Calculations Applied/Notes"
    ],

    "general": [
        "Data Source",
        "Data Links",
        "Definitions",
        "Procured From",
        "Previous Update",
        "Current Update",
        "Calculations Applied/Notes"
    ]
}

def get_indicator_metadata(
    indicator
):
    """
    Retrieve verified metadata for a canonical indicator.
    """

    if (
        "Indicator"
        not in
        semantic_indicator_catalogue.columns
    ):

        return None

    match = (
        semantic_indicator_catalogue[
            semantic_indicator_catalogue[
                "Indicator"
            ]
            ==
            indicator
        ]
    )

    if match.empty:
        return None

    record = (
        match.iloc[0]
        .to_dict()
    )

    return {
        key: to_python_scalar(value)
        for key, value in record.items()
    }

def execute_metadata(plan):
    """
    Retrieve verified metadata or coverage information.
    """

    request_type = (
        plan.metadata_request
        or
        "general"
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    if request_type == "coverage":

        indicator_subset = (
            availability_index[
                availability_index[
                    "Indicator"
                ]
                ==
                plan.indicator
            ]
        )

        if indicator_subset.empty:

            return make_execution_result(
                success=False,
                operation="metadata",
                message=(
                    "No coverage information was "
                    "available for the requested indicator."
                )
            )

        states = sorted(
            indicator_subset[
                "State"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        years = sorted(
            indicator_subset[
                "Year"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        coverage_record = {
            "Indicator":
                plan.indicator,

            "Available_States":
                states,

            "Available_Years":
                years,

            "State_Count":
                len(states),

            "Year_Count":
                len(years),

            "Earliest_Year":
                min(years)
                if years
                else None,

            "Latest_Year":
                max(years)
                if years
                else None
        }

        return make_execution_result(
            success=True,
            operation="metadata",
            metadata={
                "Indicator":
                    plan.indicator,

                "Metadata_Request":
                    "coverage",

                "Result":
                    coverage_record
            }
        )

    # --------------------------------------------------------
    # Other metadata
    # --------------------------------------------------------

    metadata_record = (
        get_indicator_metadata(
            plan.indicator
        )
    )

    if metadata_record is None:

        return make_execution_result(
            success=False,
            operation="metadata",
            message=(
                "No verified metadata were available "
                "for the requested indicator."
            )
        )

    requested_fields = (
        METADATA_REQUEST_MAP.get(
            request_type
        )
    )

    if requested_fields is None:

        return make_execution_result(
            success=False,
            operation="metadata",
            message=(
                f"Unsupported metadata request: "
                f"{request_type}"
            )
        )

    result_metadata = {
        "Indicator":
            plan.indicator,

        "Metadata_Request":
            request_type
    }

    for field in requested_fields:

        result_metadata[field] = (
            to_python_scalar(
                metadata_record.get(
                    field
                )
            )
        )

    available_values = [
        value
        for key, value
        in result_metadata.items()
        if (
            key
            not in {
                "Indicator",
                "Metadata_Request"
            }
            and
            value is not None
        )
    ]

    if not available_values:

        return make_execution_result(
            success=False,
            operation="metadata",
            metadata=result_metadata,
            message=(
                "The requested metadata field "
                "is not documented for this indicator."
            )
        )

    return make_execution_result(
        success=True,
        operation="metadata",
        metadata=result_metadata
    )

# ============================================================
# 11. QUERY DISPATCHER
# ============================================================

QUERY_EXECUTORS = {
    "lookup": execute_lookup,
    "compare": execute_compare,
    "trend": execute_trend,
    "change": execute_change,
    "growth": execute_growth,
    "rank": execute_rank,
    "metadata": execute_metadata
}

def execute_query_plan(plan):
    """
    Execute an already validated canonical QueryPlan.

    No LLM/API call occurs here.
    """

    executor = QUERY_EXECUTORS.get(
        plan.operation
    )

    if executor is None:

        return make_execution_result(
            success=False,
            operation=plan.operation,
            message=(
                f"Unsupported operation: "
                f"{plan.operation}"
            )
        )

    return executor(
        plan
    )

def process_query_plan(plan):
    """
    Validate, canonicalise and execute a QueryPlan
    entirely deterministically.

    No Gemini/API call occurs in this function.
    """

    validation = (
        validate_query_plan(
            plan
        )
    )

    # --------------------------------------------------------
    # Clarification required
    # --------------------------------------------------------

    if validation.get(
        "clarification"
    ):

        return {
            "status":
                "clarification_required",

            "validation":
                validation,

            "execution":
                None
        }

    # --------------------------------------------------------
    # Invalid plan
    # --------------------------------------------------------

    if not validation[
        "valid"
    ]:

        return {
            "status":
                "invalid",

            "validation":
                validation,

            "execution":
                None
        }

    # --------------------------------------------------------
    # Deterministic execution
    # --------------------------------------------------------

    canonical_plan = (
        validation["plan"]
    )

    execution = (
        execute_query_plan(
            canonical_plan
        )
    )

    status = (
        "success"
        if execution["success"]
        else
        "no_data"
    )

    return {
        "status":
            status,

        "validation":
            validation,

        "plan":
            canonical_plan,

        "execution":
            execution
    }

# ============================================================
# BACKEND PART 2 — DEVELOPMENT TEST SUITE
# ============================================================

def print_test_result(
    test_name,
    result
):
    """
    Compact terminal display for deterministic tests.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        test_name
    )

    print(
        "=" * 70
    )

    print(
        "Status:",
        result.get(
            "status"
        )
    )

    execution = result.get(
        "execution"
    )

    if execution is not None:

        print(
            "Operation:",
            execution.get(
                "operation"
            )
        )

        print(
            "Success:",
            execution.get(
                "success"
            )
        )

        print(
            "Message:",
            execution.get(
                "message"
            )
        )

        print(
            "Data:"
        )

        print(
            execution.get(
                "data"
            )
        )

        if execution.get(
            "metadata"
        ) is not None:

            print(
                "Metadata:"
            )

            print(
                execution.get(
                    "metadata"
                )
            )

        if execution.get(
            "warnings"
        ):

            print(
                "Warnings:"
            )

            print(
                execution.get(
                    "warnings"
                )
            )

    else:

        print(
            "Validation:",
            result.get(
                "validation"
            )
        )

if __name__ == "__main__":

    backend_part1_audit()

    print(
        "\nRunning Backend Part 2 "
        "deterministic tests..."
    )

    # --------------------------------------------------------
    # TEST 1 — LOOKUP
    # --------------------------------------------------------

    lookup_test = process_query_plan(
        QueryPlan(
            operation="lookup",
            states=["Bihar"],
            years=[2026],
            indicator="population density"
        )
    )

    print_test_result(
        "TEST 1 — LOOKUP",
        lookup_test
    )

    # --------------------------------------------------------
    # TEST 2 — COMPARISON
    # --------------------------------------------------------

    comparison_test = process_query_plan(
        QueryPlan(
            operation="compare",
            states=[
                "Bihar",
                "India"
            ],
            years=[
                2011,
                2026
            ],
            indicator="population density"
        )
    )

    print_test_result(
        "TEST 2 — COMPARISON",
        comparison_test
    )

    # --------------------------------------------------------
    # TEST 3 — TREND
    # --------------------------------------------------------

    trend_test = process_query_plan(
        QueryPlan(
            operation="trend",
            states=["Bihar"],
            years=[],
            indicator="population density"
        )
    )

    print_test_result(
        "TEST 3 — TREND",
        trend_test
    )

    # --------------------------------------------------------
    # TEST 4 — RANKING
    # --------------------------------------------------------

    ranking_test = process_query_plan(
        QueryPlan(
            operation="rank",
            states=[],
            years=[2026],
            indicator="population density",
            top_n=10
        )
    )

    print_test_result(
        "TEST 4 — RANKING",
        ranking_test
    )

    # Ranking safeguard
    if (
        ranking_test.get(
            "execution"
        )
        is not None
    ):

        ranked_states = [
            row["State"]
            for row
            in ranking_test[
                "execution"
            ].get(
                "data",
                []
            )
        ]

        print(
            "India included in "
            "state ranking:",
            "India"
            in ranked_states
        )

    # --------------------------------------------------------
    # TEST 5 — MISSING DATA
    # --------------------------------------------------------

    missing_test = process_query_plan(
        QueryPlan(
            operation="lookup",
            states=["Bihar"],
            years=[2099],
            indicator="population density"
        )
    )

    print_test_result(
        "TEST 5 — MISSING DATA",
        missing_test
    )

    # --------------------------------------------------------
    # TEST 6 — METADATA
    # --------------------------------------------------------

    metadata_test = process_query_plan(
        QueryPlan(
            operation="metadata",
            indicator="population density",
            metadata_request="source"
        )
    )

    print_test_result(
        "TEST 6 — METADATA",
        metadata_test
    )

    # --------------------------------------------------------
    # TEST 7 — GROWTH
    # --------------------------------------------------------

    growth_test = process_query_plan(
        QueryPlan(
            operation="growth",
            states=["Bihar"],
            years=[
                2011,
                2026
            ],
            indicator="population density"
        )
    )

    print_test_result(
        "TEST 7 — GROWTH",
        growth_test
    )

    # --------------------------------------------------------
    # TEST 8 — CHANGE
    # --------------------------------------------------------

    change_test = process_query_plan(
        QueryPlan(
            operation="change",
            states=["Bihar"],
            years=[
                2011,
                2026
            ],
            indicator="population density"
        )
    )

    print_test_result(
        "TEST 8 — CHANGE",
        change_test
    )

    # --------------------------------------------------------
    # TEST 9 — AMBIGUITY SAFEGUARD
    # --------------------------------------------------------

    ambiguity_test = process_query_plan(
        QueryPlan(
            operation="lookup",
            states=["Bihar"],
            years=[2011],
            indicator="sex ratio"
        )
    )

    print_test_result(
        "TEST 9 — AMBIGUITY SAFEGUARD",
        ambiguity_test
    )

# ============================================================
# BACKEND PART 1 — DEVELOPMENT AUDIT
# ============================================================

def backend_part1_audit():
    """
    Lightweight development audit for Backend Part 1.
    """

    print("Backend Part 1 loaded successfully.")
    print("Project directory:", PROJECT_DIR)
    print("Fact observations:", len(analytical_fact_table))
    print("Canonical geographies:", len(CANONICAL_STATES))
    print("Canonical indicators:", len(CANONICAL_INDICATORS))

    print(
        "Indicator metadata records:",
        len(semantic_indicator_catalogue)
    )

    print(
        "Geography catalogue records:",
        len(semantic_geography_catalogue)
    )

    print(
        "Gemini configured:",
        gemini_client is not None
    )


# ============================================================
# BACKEND PART 2 — DEVELOPMENT TEST SUITE
# ============================================================

def print_test_result(
    test_name,
    result
):
    """
    Compact terminal display for deterministic tests.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(test_name)

    print(
        "=" * 70
    )

    print(
        "Status:",
        result.get("status")
    )

    execution = result.get("execution")

    if execution is not None:

        print(
            "Operation:",
            execution.get("operation")
        )

        print(
            "Success:",
            execution.get("success")
        )

        print(
            "Message:",
            execution.get("message")
        )

        print("Data:")
        print(
            execution.get("data")
        )

        if execution.get("metadata") is not None:

            print("Metadata:")
            print(
                execution.get("metadata")
            )

        if execution.get("warnings"):

            print("Warnings:")
            print(
                execution.get("warnings")
            )

    else:

        print(
            "Validation:",
            result.get("validation")
        )


# ============================================================
# RUN BACKEND DEVELOPMENT TESTS
# ============================================================

if __name__ == "__main__":

    backend_part1_audit()

    print(
        "\nRunning Backend Part 2 "
        "deterministic tests..."
    )

    # Your TEST 1, TEST 2, ... TEST 9 code
    # continues here, indented under this block.

# ============================================================
# 12. GEMINI NATURAL-LANGUAGE PARSER
# ============================================================

PARSER_GEOGRAPHY_VOCABULARY = CANONICAL_STATES
PARSER_INDICATOR_VOCABULARY = CANONICAL_INDICATORS


def build_parser_vocabulary_context():
    """
    Build the controlled vocabulary supplied to Gemini.

    Gemini interprets the question, but the deterministic
    backend remains responsible for canonical validation.
    """

    geography_text = ", ".join(
        PARSER_GEOGRAPHY_VOCABULARY
    )

    indicator_text = "\n".join(
        f"- {indicator}"
        for indicator
        in PARSER_INDICATOR_VOCABULARY
    )

    return (
        "SUPPORTED GEOGRAPHIES:\n"
        f"{geography_text}\n\n"
        "SUPPORTED INDICATORS:\n"
        f"{indicator_text}"
    )


PARSER_VOCABULARY_CONTEXT = (
    build_parser_vocabulary_context()
)

GEMINI_PARSER_INSTRUCTIONS = """
You are the natural-language query parser for an Indian
state-level Demography and Employment Data Assistant.

Your ONLY task is to translate the user's question into the
provided QueryPlan structure.

You do NOT answer the user's question.
You do NOT retrieve data.
You do NOT calculate numerical values.
You do NOT estimate missing observations.
You do NOT invent statistics.
You do NOT generate SQL.
You do NOT determine whether an observation exists.

SUPPORTED OPERATIONS

lookup:
Retrieve requested observations.

compare:
Compare one indicator across multiple geographies,
multiple years, or both.

trend:
Retrieve one geography's indicator across time.

change:
Calculate absolute change between exactly two years.

growth:
Calculate percentage growth between exactly two years.

rank:
Rank states by an indicator for one year.

metadata:
Retrieve documentation about an indicator.

SUPPORTED METADATA REQUESTS

definition
source
link
procurement
update
calculation_notes
coverage
general

INTERPRETATION RULES

- Put geography names in states.
- Put explicitly requested years in years.
- Put the requested indicator terminology in indicator.
- Extract top_n from expressions such as "top 5".
- Never invent a geography, year, indicator, or value.
- Do not silently resolve ambiguous indicators.
- For trend questions without explicit years, leave years empty.
- "What does X mean?" is metadata / definition.
- "What is the source of X?" is metadata / source.
- Questions asking for the data link are metadata / link.
- Questions asking where data were procured from are
  metadata / procurement.
- Questions about data updates are metadata / update.
- Questions about calculations or methodology are
  metadata / calculation_notes.
- Questions asking which years or geographies are available
  are metadata / coverage.

Return only a QueryPlan conforming to the supplied schema.
"""
def build_gemini_parser_prompt(
    question
):
    """
    Build the parser prompt for one user question.
    """

    return f"""
{GEMINI_PARSER_INSTRUCTIONS}

{PARSER_VOCABULARY_CONTEXT}

USER QUESTION:
{question}
"""
def build_gemini_parser_prompt(
    question
):
    """
    Build the parser prompt for one user question.
    """

    return f"""
{GEMINI_PARSER_INSTRUCTIONS}

{PARSER_VOCABULARY_CONTEXT}

USER QUESTION:
{question}
"""


def parse_question_with_gemini(
    question
):
    """
    Convert a natural-language question into a QueryPlan.

    IMPORTANT:
    This function performs interpretation only.
    It never executes analytical calculations.
    """

    if gemini_client is None:

        raise RuntimeError(
            "Gemini is not configured."
        )

    if question is None:

        raise ValueError(
            "Question cannot be empty."
        )

    question = str(
        question
    ).strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    prompt = (
        build_gemini_parser_prompt(
            question
        )
    )

    interaction = (
        gemini_client
        .interactions
        .create(
            model=GEMINI_PARSER_MODEL,
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,

            input=prompt,

            response_format={
                "type": "text",
                "mime_type":
                    "application/json",
                "schema":
                    QueryPlan
                    .model_json_schema()
            }
        )
    )

    response_text = (
        interaction.output_text
    )

    if not response_text:

        raise ValueError(
            "Gemini returned an empty "
            "parser response."
        )

    return (
        QueryPlan
        .model_validate_json(
            response_text
        )
    )

def classify_gemini_error(
    error
):
    """
    Convert common external API errors into a compact,
    application-safe category.
    """

    message = str(
        error
    ).lower()

    if (
        "429" in message
        or
        "resource_exhausted" in message
        or
        "rate limit" in message
        or
        "quota" in message
    ):

        return "rate_limit_or_quota"

    if (
        "timeout" in message
        or
        "timed out" in message
    ):

        return "timeout"

    if (
        "401" in message
        or
        "403" in message
        or
        "api key" in message
    ):

        return "authentication"

    if (
        "404" in message
        or
        "not found" in message
        or
        "model" in message
    ):

        return "model_or_endpoint"

    return "external_api_error"

def safe_parse_question_with_gemini(
    question
):
    """
    Safe production wrapper around the external parser.

    API failures are returned as structured results rather
    than crashing the deterministic backend.
    """

    start_time = (
        time.perf_counter()
    )

    try:

        plan = (
            parse_question_with_gemini(
                question
            )
        )

        latency = (
            time.perf_counter()
            -
            start_time
        )

        return {
            "success": True,
            "plan": plan,
            "error": None,
            "error_type": None,
            "latency_seconds": latency
        }

    except ValidationError as exc:

        latency = (
            time.perf_counter()
            -
            start_time
        )

        return {
            "success": False,
            "plan": None,
            "error": (
                "Gemini returned a response "
                "that did not satisfy QueryPlan."
            ),
            "error_type":
                "schema_validation",
            "technical_error":
                str(exc),
            "latency_seconds":
                latency
        }

    except Exception as exc:

        latency = (
            time.perf_counter()
            -
            start_time
        )

        return {
            "success": False,
            "plan": None,
            "error": (
                "The natural-language parser "
                "is temporarily unavailable."
            ),
            "error_type":
                classify_gemini_error(
                    exc
                ),
            "technical_error":
                str(exc),
            "latency_seconds":
                latency
        }

# ============================================================
# 12A. DETERMINISTIC FAST-PATH PARSER
# ============================================================

BACKEND_BUILD = "2E-2026-10-08"

def _phrase_in_question(question_normalized, phrase):
    """Token-boundary phrase matching for deterministic parsing."""
    phrase_normalized = normalize_text(phrase)
    if not phrase_normalized:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(phrase_normalized) + r"(?!\w)", question_normalized))


def _extract_states_from_question(question):
    """Extract canonical geographies using only the controlled state vocabulary."""
    q = normalize_text(question)
    matches = []

    # Prefer longer aliases first (e.g. Uttar Pradesh before UP).
    vocabulary = []
    for alias, canonical in STATE_ALIASES.items():
        vocabulary.append((alias, canonical))
    for canonical in CANONICAL_STATES:
        vocabulary.append((normalize_text(canonical), canonical))

    seen_terms = set()
    for term, canonical in sorted(vocabulary, key=lambda x: len(x[0]), reverse=True):
        if term in seen_terms:
            continue
        seen_terms.add(term)
        if _phrase_in_question(q, term) and canonical not in matches:
            matches.append(canonical)
    return matches


def _remove_indicator_phrase_from_question(question, indicator):
    """Return question text with the resolved indicator phrase masked out.

    This prevents words or years that are part of an indicator label from
    being reinterpreted as query instructions. For example, "growth" in
    "Population Growth Rate (Decadal & Annual)" must not force a growth
    operation, and "2011" in an indicator such as "...(2011 Census)" must
    not be treated as a requested observation year.
    """
    text = str(question)
    if not indicator:
        return text

    # The deterministic resolver returns either the exact alias found in the
    # question or the canonical indicator label. Remove only that literal
    # phrase, case-insensitively, preserving the rest of the user's wording.
    pattern = re.compile(re.escape(str(indicator)), flags=re.IGNORECASE)
    return pattern.sub(" ", text, count=1)


def _extract_years_from_question(question):
    """Extract explicit four-digit years without inferring unmentioned years."""
    years = []
    for token in re.findall(r"(?<!\d)(?:19|20|21)\d{2}(?!\d)", str(question)):
        year = int(token)
        if year not in years:
            years.append(year)
    return years


def _extract_indicator_from_question(question):
    """Resolve a high-confidence indicator phrase while preserving ambiguity."""
    q = normalize_text(question)

    # Exact controlled aliases first.
    alias_hits = []
    for alias, canonical in sorted(INDICATOR_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if _phrase_in_question(q, alias):
            alias_hits.append((alias, canonical))
    if alias_hits:
        return alias_hits[0][0]

    # Canonical indicator text if the user uses it directly.
    canonical_hits = []
    for canonical in CANONICAL_INDICATORS:
        term = normalize_text(canonical)
        if term and _phrase_in_question(q, term):
            canonical_hits.append(canonical)
    if canonical_hits:
        # Gold/evaluation questions often contain the full canonical indicator
        # name. Some indicator names are substrings of longer indicators, so
        # requiring exactly one hit can unnecessarily force a Gemini fallback.
        # The longest full canonical phrase present in the question is the
        # most specific deterministic match.
        return max(
            canonical_hits,
            key=lambda value: len(normalize_text(value))
        )

    # Deliberately preserve known ambiguous language for deterministic clarification.
    ambiguity_phrases = ["sex ratio"]
    for phrase in ambiguity_phrases:
        if _phrase_in_question(q, phrase):
            return phrase

    return None


def _extract_top_n_from_question(question):
    q = normalize_text(question)
    match = re.search(r"\btop\s+(\d{1,3})\b", q)
    if match:
        return max(1, min(100, int(match.group(1))))
    return None


def try_deterministic_question_plan(question):
    """
    Build a QueryPlan locally for routine, high-confidence requests.

    Returns a structured parser result. If confidence is insufficient, the
    caller falls back to Gemini. No numerical retrieval or calculation occurs
    in this function.
    """
    start = time.perf_counter()
    q = normalize_text(question)
    states = _extract_states_from_question(question)
    indicator = _extract_indicator_from_question(question)
    top_n = _extract_top_n_from_question(question)

    if not indicator:
        return {
            "success": False, "plan": None, "reason": "indicator_not_resolved",
            "parser_mode": "deterministic_fast_path",
            "latency_seconds": time.perf_counter() - start,
        }

    # Round 2C: once the indicator has been resolved, remove its literal text
    # before extracting years or interpreting intent keywords. Indicator labels
    # are data vocabulary, not user instructions.
    # Work in normalized text for masking.  The resolved indicator may itself
    # be a normalized alias (e.g. punctuation removed), so literal masking of
    # the raw question is not reliable.  Removing the normalized matched phrase
    # prevents embedded words/years from leaking into intent detection.
    q_normalized = normalize_text(question)
    indicator_normalized = normalize_text(indicator)
    q_for_intent = re.sub(
        r"(?<!\w)" + re.escape(indicator_normalized) + r"(?!\w)",
        " ",
        q_normalized,
        count=1,
    )
    years = _extract_years_from_question(q_for_intent)

    metadata_request = None
    operation = None

    # Metadata intent has priority over analytical keywords.
    if any(phrase in q_for_intent for phrase in ["source of", "data source", "source for", "where does", "where is the data from"]):
        operation, metadata_request = "metadata", "source"
    elif any(phrase in q_for_intent for phrase in ["definition", "what does", "what is the meaning", "define "]):
        operation, metadata_request = "metadata", "definition"
    elif any(phrase in q_for_intent for phrase in ["data link", "link for", "link to"]):
        operation, metadata_request = "metadata", "link"
    elif any(phrase in q_for_intent for phrase in ["procured from", "procurement"]):
        operation, metadata_request = "metadata", "procurement"
    elif any(phrase in q_for_intent for phrase in ["last updated", "current update", "previous update", "updated"]):
        operation, metadata_request = "metadata", "update"
    elif any(phrase in q_for_intent for phrase in ["calculation notes", "methodology", "calculated"]):
        operation, metadata_request = "metadata", "calculation_notes"
    elif any(phrase in q_for_intent for phrase in ["coverage", "available years", "years available", "available states", "states available"]):
        operation, metadata_request = "metadata", "coverage"
    elif any(phrase in q_for_intent for phrase in ["percentage growth", "percent growth", "growth rate", "grew by", "growth in"]):
        operation = "growth"
    elif any(phrase in q_for_intent for phrase in ["absolute change", "how much did", "change between", "change from", "changed between"]):
        operation = "change"
    elif any(phrase in q_for_intent for phrase in ["highest", "lowest", "rank", "ranking", "top "]):
        operation = "rank"
    elif any(phrase in q_for_intent for phrase in ["trend", "over time", "time series"]):
        operation = "trend"
    elif any(phrase in q_for_intent for phrase in ["compare", "comparison", "compared to", "compared with", "versus", " vs "]):
        operation = "compare"
    else:
        # Any simple state + indicator request is safely a lookup.
        # If a year is omitted, deterministic validation/answer logic handles
        # the missing year rather than sending a routine question to Gemini.
        if states:
            operation = "lookup"

    if operation is None:
        return {
            "success": False, "plan": None, "reason": "intent_not_high_confidence",
            "parser_mode": "deterministic_fast_path",
            "latency_seconds": time.perf_counter() - start,
        }

    # Natural-language "across states" means all 28 states (India excluded).
    # When no year is supplied, use the latest year for which this indicator
    # has observations across the state panel.  This keeps comparison cards
    # deterministic and prevents a Gemini fallback / oversized all-year query.
    if operation == "compare" and not states and any(
        phrase in q_for_intent for phrase in ["across states", "all states", "between states"]
    ):
        states = [state for state in CANONICAL_STATES if state != "India"]
        if not years:
            _canonical_indicator = INDICATOR_ALIASES.get(normalize_text(indicator), indicator)
            _available = fact_df.loc[
                (fact_df["Indicator"] == _canonical_indicator)
                & (fact_df["State"].isin(states))
                & fact_df["Value"].notna(),
                ["State", "Year"]
            ].copy()
            if not _available.empty:
                _available["Year"] = pd.to_numeric(_available["Year"], errors="coerce")
                _available = _available.dropna(subset=["Year"])
                if not _available.empty:
                    _coverage = _available.groupby("Year")["State"].nunique()
                    _full_years = _coverage[_coverage == len(states)].index.tolist()
                    _chosen_year = max(_full_years) if _full_years else _available["Year"].max()
                    years = [int(_chosen_year)]

    plan = QueryPlan(
        operation=operation,
        states=states,
        years=years,
        indicator=indicator,
        metadata_request=metadata_request,
        top_n=top_n,
    )

    return {
        "success": True,
        "plan": plan,
        "error": None,
        "error_type": None,
        "parser_mode": "deterministic_fast_path",
        "latency_seconds": time.perf_counter() - start,
    }

# ============================================================
# 13. QUESTION-UNDERSTANDING LAYER
# ============================================================

_UNSUPPORTED_GEO_NAMES = {
    "mumbai": "Mumbai is a city; the dataset contains Maharashtra at state level",
    "delhi": "Delhi is not included among the 28 states or the India aggregate in this dataset",
    "new delhi": "New Delhi is not a separate geography in this state-level dataset",
    "chennai": "Chennai is a city; the dataset contains Tamil Nadu at state level",
    "kolkata": "Kolkata is a city; the dataset contains West Bengal at state level",
    "bengaluru": "Bengaluru is a city; the dataset contains Karnataka at state level",
    "bangalore": "Bangalore is a city; the dataset contains Karnataka at state level",
    "hyderabad": "Hyderabad is a city; the dataset contains Telangana at state level",
    "pune": "Pune is a city; the dataset contains Maharashtra at state level",
}

def _preflight_clarification(question):
    q = normalize_text(question)
    unsupported = []
    for name, description in _UNSUPPORTED_GEO_NAMES.items():
        if _phrase_in_question(q, name):
            unsupported.append((name, description))
    # Deduplicate overlapping New Delhi / Delhi mentions.
    if any(name == "new delhi" for name, _ in unsupported):
        unsupported = [(name, desc) for name, desc in unsupported if name != "delhi"]
    if unsupported:
        return ("Unsupported geography: " + "; ".join(desc for _, desc in unsupported)
                + ". Please choose one of the 28 supported states or India. "
                + "I will not substitute a city with its state without your confirmation.")
    indicator = _extract_indicator_from_question(question)
    if not indicator:
        states = _extract_states_from_question(question)
        if states:
            return ("Which indicator would you like to analyse for "
                    + ", ".join(states) + "? For example, unemployment rate, "
                    + "population density, or forest cover. Please specify the indicator.")
        return ("Sankhyaki supports verified state-level demography and employment "
                "indicators for 28 states and India. Please specify a supported "
                "geography and indicator (for example, Bihar's unemployment rate).")
    return None

def _latest_common_comparison_year(plan):
    if plan.operation != "compare" or plan.years or len(plan.states) < 2:
        return plan, None
    resolved = [resolve_state(state) for state in plan.states]
    if any(r["status"] != "resolved" for r in resolved):
        return plan, None
    indicator_resolution = resolve_indicator(plan.indicator)
    if indicator_resolution["status"] != "resolved":
        return plan, None
    canonical = indicator_resolution["canonical"]
    states = list(dict.fromkeys(r["canonical"] for r in resolved))
    common = None
    for state in states:
        years = set(get_available_years(state, canonical))
        common = years if common is None else common & years
    if not common:
        return plan, ("There is no common available year for " + canonical
                      + " across " + ", ".join(states)
                      + ". I cannot make a like-for-like comparison from the available data.")
    return plan.model_copy(update={"states": states, "indicator": canonical,
                                   "years": [max(common)]}), None

def understand_question(
    question
):
    """
    Hybrid interpretation layer.

    Routine, high-confidence requests are parsed locally first. Gemini is used
    only when deterministic interpretation is not sufficiently confident.
    Every candidate plan still passes through the same deterministic validator.
    """
    clarification = _preflight_clarification(question)
    if clarification:
        return {"status": "clarification_required", "question": question,
                "plan": None, "validation": None,
                "parser": {"parser_mode": "deterministic_preflight"},
                "message": clarification}

    local_result = try_deterministic_question_plan(question)

    if local_result.get("success"):
        parser_result = local_result
    else:
        parser_result = safe_parse_question_with_gemini(question)
        parser_result["parser_mode"] = "gemini_fallback"
        parser_result["fast_path_reason"] = local_result.get("reason")
        parser_result["fast_path_latency_seconds"] = local_result.get("latency_seconds")

    if not parser_result.get("success"):
        return {
            "status": "parser_error",
            "question": question,
            "plan": None,
            "validation": None,
            "parser": parser_result,
            "message": parser_result.get("error"),
        }

    candidate_plan = parser_result["plan"]

    # --------------------------------------------------------
    # Natural lookup default: latest available observation
    # --------------------------------------------------------
    # A user asking "What is [indicator] for [state]?" is making a
    # complete lookup request even when no year is stated. For a single
    # geography, resolve that request deterministically to the latest
    # verified year available for that state-indicator pair. This keeps
    # ordinary lookups off the LLM fallback path and never fabricates data.
    if (
        candidate_plan.operation == "lookup"
        and len(candidate_plan.states) == 1
        and not candidate_plan.years
        and candidate_plan.indicator
    ):
        state_resolution = resolve_state(candidate_plan.states[0])
        indicator_resolution = resolve_indicator(candidate_plan.indicator)
        lookup_state = (
            state_resolution.get("canonical")
            if state_resolution.get("status") == "resolved"
            else candidate_plan.states[0]
        )
        lookup_indicator = (
            indicator_resolution.get("canonical")
            if indicator_resolution.get("status") == "resolved"
            else candidate_plan.indicator
        )
        available_years = get_available_years(lookup_state, lookup_indicator)

        if available_years:
            resolved_year = max(available_years)
            year_resolution = "latest_available"
        else:
            # Keep a no-data lookup executable even when this state-indicator
            # pair has no observations at all. Use the latest documented year
            # for the indicator elsewhere in the dataset solely as the lookup
            # frame; execution will correctly return Data_Status=Unavailable.
            indicator_years = (
                fact_df.loc[fact_df["Indicator"] == lookup_indicator, "Year"]
                .dropna()
                .astype(int)
                .tolist()
            )
            resolved_year = max(indicator_years) if indicator_years else None
            year_resolution = "latest_indicator_year_no_state_data"

        if resolved_year is not None:
            candidate_plan = candidate_plan.model_copy(
                update={
                    "states": [lookup_state],
                    "indicator": lookup_indicator,
                    "years": [resolved_year],
                }
            )
            parser_result["plan"] = candidate_plan
            parser_result["year_resolution"] = year_resolution
            parser_result["resolved_year"] = resolved_year

    candidate_plan, comparison_issue = _latest_common_comparison_year(candidate_plan)
    if comparison_issue:
        return {"status": "clarification_required", "question": question,
                "plan": candidate_plan, "validation": None,
                "parser": parser_result, "message": comparison_issue}
    parser_result["plan"] = candidate_plan

    validation = validate_query_plan(candidate_plan)

    if validation.get("clarification"):
        return {
            "status": "clarification_required",
            "question": question,
            "plan": candidate_plan,
            "validation": validation,
            "parser": parser_result,
            "message": validation["clarification"],
        }

    if not validation["valid"]:
        error_text = "; ".join(validation["errors"]) if validation["errors"] else (
            "The question could not be converted into a valid query."
        )
        return {
            "status": "invalid",
            "question": question,
            "plan": candidate_plan,
            "validation": validation,
            "parser": parser_result,
            "message": error_text,
        }

    return {
        "status": "validated",
        "question": question,
        "plan": validation["plan"],
        "validation": validation,
        "parser": parser_result,
        "message": None,
    }

# ============================================================
# 14. EVIDENCE PACKAGE
# ============================================================

def get_indicator_source(
    indicator
):
    """
    Retrieve the documented source for an indicator.
    """

    metadata = (
        get_indicator_metadata(
            indicator
        )
    )

    if metadata is None:
        return None

    return (
        metadata.get(
            "Data Source"
        )
    )

def build_evidence_package(
    question,
    plan,
    execution
):
    """
    Construct the verified evidence object used by
    response formatting and visualisation.

    Only deterministic execution results enter this package.
    """

    source = (
        get_indicator_source(
            plan.indicator
        )
        if plan.indicator
        else None
    )

    data_records = (
        execution.get(
            "data",
            []
        )
        if execution
        else []
    )

    metadata = (
        execution.get(
            "metadata"
        )
        if execution
        else None
    )

    evidence = {

        "question":
            question,

        "operation":
            plan.operation,

        "indicator":
            plan.indicator,

        "states":
            list(
                plan.states
            ),

        "years":
            list(
                plan.years
            ),

        "top_n":
            plan.top_n,

        "metadata_request":
            plan.metadata_request,

        "records":
            data_records,

        "metadata":
            metadata,

        "source":
            to_python_scalar(
                source
            ),

        "calculation":
            (
                execution.get(
                    "calculation"
                )
                if execution
                else None
            ),

        "warnings":
            (
                execution.get(
                    "warnings",
                    []
                )
                if execution
                else []
            ),

        "execution_success":
            bool(
                execution
                and
                execution.get(
                    "success"
                )
            ),

        "execution_message":
            (
                execution.get(
                    "message"
                )
                if execution
                else None
            )
    }

    return evidence

# ============================================================
# 15. GROUNDED RESPONSE FORMATTING
# ============================================================

def format_number(
    value,
    decimals=2
):
    """
    Format numerical values for user-facing responses.
    """

    if value is None:
        return "Not available"

    try:

        numeric = float(
            value
        )

    except (TypeError, ValueError):

        return str(
            value
        )

    if numeric.is_integer():

        return (
            f"{int(numeric):,}"
        )

    return (
        f"{numeric:,.{decimals}f}"
        .rstrip("0")
        .rstrip(".")
    )

def format_lookup_response(
    evidence
):
    """
    Format exact lookup observations.
    """

    records = evidence.get(
        "records",
        []
    )

    if not records:

        return (
            evidence.get(
                "execution_message"
            )
            or
            "No requested observation "
            "was available."
        )

    lines = []

    for record in records:

        state = record.get(
            "State"
        )

        year = record.get(
            "Year"
        )

        value = record.get(
            "Value"
        )

        status = record.get(
            "Data_Status"
        )

        if (
            status == "Unavailable"
            or
            value is None
        ):

            available_years = (
                record.get(
                    "Available_Years",
                    []
                )
            )

            line = (
                f"{state}, {year}: "
                "data unavailable"
            )

            if available_years:

                line += (
                    ". Available years: "
                    +
                    ", ".join(
                        str(year_value)
                        for year_value
                        in available_years
                    )
                )

            lines.append(
                line
            )

        else:

            lines.append(
                f"{state}, {year}: "
                f"{format_number(value)}"
            )

    return (
        f"{evidence['indicator']}:\n"
        +
        "\n".join(
            f"- {line}"
            for line in lines
        )
    )

def format_multi_observation_response(
    evidence
):
    """
    Format comparison and trend observations.
    """

    records = evidence.get(
        "records",
        []
    )

    if not records:

        return (
            evidence.get(
                "execution_message"
            )
            or
            "No requested observations "
            "were available."
        )

    lines = []

    for record in records:

        lines.append(
            (
                f"{record.get('State')}, "
                f"{record.get('Year')}: "
                f"{format_number(record.get('Value'))}"
            )
        )

    return (
        f"{evidence['indicator']}:\n"
        +
        "\n".join(
            f"- {line}"
            for line in lines
        )
    )

def format_calculation_response(
    evidence
):
    """
    Format deterministic growth/change calculations.
    """

    operation = evidence.get(
        "operation"
    )

    records = evidence.get(
        "records",
        []
    )

    if not records:

        return (
            evidence.get(
                "execution_message"
            )
            or
            "The requested calculation "
            "could not be completed."
        )

    lines = []

    for result in records:

        state = result.get(
            "State",
            "Requested geography"
        )

        start_year = result.get(
            "Start_Year"
        )

        end_year = result.get(
            "End_Year"
        )

        start_value = result.get(
            "Start_Value"
        )

        end_value = result.get(
            "End_Value"
        )

        if operation == "growth":

            value = result.get(
                "Growth_Percent"
            )

            lines.append(
                (
                    f"{state}: "
                    f"{format_number(value, 2)}% growth "
                    f"between {start_year} and {end_year} "
                    f"({format_number(start_value)} → "
                    f"{format_number(end_value)})"
                )
            )

        elif operation == "change":

            value = result.get(
                "Absolute_Change"
            )

            lines.append(
                (
                    f"{state}: an absolute change of "
                    f"{format_number(value)} "
                    f"between {start_year} and {end_year} "
                    f"({format_number(start_value)} → "
                    f"{format_number(end_value)})"
                )
            )

    return (
        f"{evidence['indicator']}:\n"
        +
        "\n".join(
            f"- {line}"
            for line in lines
        )
    )

def format_rank_response(
    evidence
):
    """
    Format state-ranking results.
    """

    records = evidence.get(
        "records",
        []
    )

    if not records:

        return (
            evidence.get(
                "execution_message"
            )
            or
            "No ranking data were available."
        )

    lines = []

    for record in records:

        lines.append(
            (
                f"{record.get('Rank')}. "
                f"{record.get('State')}: "
                f"{format_number(record.get('Value'))}"
            )
        )

    year_text = ""

    if evidence.get(
        "years"
    ):

        year_text = (
            f", {evidence['years'][0]}"
        )

    return (
        f"{evidence['indicator']} — "
        f"State Ranking{year_text}:\n"
        +
        "\n".join(
            lines
        )
    )

def format_metadata_response(
    evidence
):
    """
    Format documented indicator metadata.
    """

    metadata = evidence.get(
        "metadata"
    )

    if not metadata:

        return (
            evidence.get(
                "execution_message"
            )
            or
            "No verified metadata were available."
        )

    request_type = (
        metadata.get(
            "Metadata_Request"
        )
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    if request_type == "coverage":

        coverage = metadata.get(
            "Result",
            {}
        )

        available_states = (
            coverage.get(
                "Available_States",
                []
            )
        )

        return (
            f"{evidence['indicator']} coverage:\n"
            f"- Earliest year: "
            f"{coverage.get('Earliest_Year')}\n"
            f"- Latest year: "
            f"{coverage.get('Latest_Year')}\n"
            f"- Number of available years: "
            f"{coverage.get('Year_Count')}\n"
            f"- Number of geographies: "
            f"{coverage.get('State_Count')}\n"
            f"- Geographies: "
            f"{', '.join(available_states)}"
        )

    # --------------------------------------------------------
    # Documented metadata fields
    # --------------------------------------------------------

    excluded = {
        "Indicator",
        "Metadata_Request"
    }

    lines = []

    for key, value in metadata.items():

        if (
            key not in excluded
            and
            value is not None
        ):

            lines.append(
                f"- {key}: {value}"
            )

    if not lines:

        return (
            "The requested metadata field "
            "is not documented."
        )

    return (
        f"{evidence['indicator']}:\n"
        +
        "\n".join(
            lines
        )
    )

def generate_grounded_response(
    evidence
):
    """
    Generate a deterministic user-facing response
    exclusively from verified backend evidence.
    """

    if not evidence.get(
        "execution_success"
    ):

        return (
            evidence.get(
                "execution_message"
            )
            or
            "No verified result was available "
            "for the requested query."
        )

    operation = evidence.get(
        "operation"
    )

    if operation == "lookup":

        return format_lookup_response(
            evidence
        )

    if operation in {
        "compare",
        "trend"
    }:

        return (
            format_multi_observation_response(
                evidence
            )
        )

    if operation in {
        "change",
        "growth"
    }:

        return (
            format_calculation_response(
                evidence
            )
        )

    if operation == "rank":

        return format_rank_response(
            evidence
        )

    if operation == "metadata":

        return format_metadata_response(
            evidence
        )

    return (
        "The analytical operation completed, "
        "but no response formatter was available."
    )

# ============================================================
# 16. PROVENANCE
# ============================================================

def build_provenance(
    evidence
):
    """
    Construct structured provenance for the verified answer.
    """

    records = evidence.get(
        "records",
        []
    )

    available_records = [
        record
        for record in records
        if (
            record.get(
                "Value"
            )
            is not None
            or
            evidence.get(
                "operation"
            )
            in {
                "growth",
                "change",
                "rank"
            }
        )
    ]

    return {
        "indicator":
            evidence.get(
                "indicator"
            ),

        "states":
            evidence.get(
                "states",
                []
            ),

        "years":
            evidence.get(
                "years",
                []
            ),

        "operation":
            evidence.get(
                "operation"
            ),

        "evidence_records":
            len(
                available_records
            ),

        "source":
            evidence.get(
                "source"
            ),

        "data_status":
            (
                "Available"
                if evidence.get(
                    "execution_success"
                )
                else
                "Unavailable"
            ),

        "calculation":
            evidence.get(
                "calculation"
            ),

        "warnings":
            evidence.get(
                "warnings",
                []
            )
    }

def execute_validated_chatbot_plan(
    question,
    plan
):
    """
    Execute a validated canonical QueryPlan and construct
    the complete deterministic chatbot result.

    Pipeline:
        QueryPlan
            -> deterministic execution
            -> evidence package
            -> grounded response
            -> automatic visualisation
            -> provenance

    No LLM generates or modifies numerical evidence here.
    """

    # --------------------------------------------------------
    # 1. Deterministic analytical execution
    # --------------------------------------------------------

    execution = execute_query_plan(
        plan
    )

    # --------------------------------------------------------
    # 2. Verified evidence package
    # --------------------------------------------------------

    evidence = build_evidence_package(
        question=question,
        plan=plan,
        execution=execution
    )

    # --------------------------------------------------------
    # 3. Deterministic grounded response
    # --------------------------------------------------------

    answer = generate_grounded_response(
        evidence
    )

    # --------------------------------------------------------
    # 4. Automatic visualisation
    # --------------------------------------------------------

    chart_result = generate_automatic_chart(
        evidence
    )

    # --------------------------------------------------------
    # 5. Provenance
    # --------------------------------------------------------

    provenance = build_provenance(
        evidence
    )

    # --------------------------------------------------------
    # 6. Final status
    # --------------------------------------------------------

    status = (
        "success"
        if execution.get("success")
        else "no_data"
    )

    return make_chatbot_result(
        status=status,
        question=question,
        answer=answer,
        plan=plan,
        execution=execution,
        evidence=evidence,
        chart=chart_result.get(
            "figure"
        ),
        chart_type=chart_result.get(
            "chart_type"
        ),
        provenance=provenance,
        warnings=execution.get(
            "warnings",
            []
        ),
        message=execution.get(
            "message"
        )
    )

def backend_part3_deterministic_test():
    """
    Test evidence, grounded response and provenance
    without calling Gemini.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BACKEND PART 3 — "
        "DETERMINISTIC RESPONSE TEST"
    )

    print(
        "=" * 70
    )

    test_plan = QueryPlan(
        operation="growth",
        states=["Bihar"],
        years=[
            2011,
            2026
        ],
        indicator=(
            "Population Density "
            "(per square km)"
        )
    )

    validation = (
        validate_query_plan(
            test_plan
        )
    )

    if not validation["valid"]:

        print(
            "FAILED:",
            validation
        )

        return

    result = (
        execute_validated_chatbot_plan(
            question=(
                "What was the percentage growth "
                "in Bihar's population density "
                "between 2011 and 2026?"
            ),
            plan=validation["plan"]
        )
    )

    print(
        "Status:",
        result["status"]
    )

    print(
        "\nAnswer:\n"
    )

    print(
        result["answer"]
    )

    print(
        "\nProvenance:\n"
    )

    print(
        result["provenance"]
    )


# ============================================================
# 17. AUTOMATIC DATA VISUALISATION
# ============================================================

def make_chart_result(
    chart_created,
    figure=None,
    chart_type=None,
    message=None
):
    """
    Standard result returned by the automatic
    visualisation layer.
    """

    return {
        "chart_created": bool(chart_created),
        "figure": figure,
        "chart_type": chart_type,
        "message": message
    }

def build_chart_title(
    evidence
):
    """
    Construct a concise, publication-style chart title.
    """

    operation = evidence.get(
        "operation"
    )

    indicator = evidence.get(
        "indicator"
    )

    states = evidence.get(
        "states",
        []
    )

    years = evidence.get(
        "years",
        []
    )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    if operation == "trend":

        state = (
            states[0]
            if states
            else "Selected Geography"
        )

        return (
            f"{indicator} in {state}"
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if operation == "compare":

        if len(years) == 1:

            return (
                f"{indicator} across "
                f"Selected Geographies, {years[0]}"
            )

        if len(states) == 1:

            return (
                f"{indicator} in "
                f"{states[0]} across Selected Years"
            )

        return (
            f"Comparison of {indicator}"
        )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    if operation == "rank":

        year_text = (
            f", {years[0]}"
            if years
            else ""
        )

        return (
            f"State Ranking by {indicator}"
            f"{year_text}"
        )

    # --------------------------------------------------------
    # Growth
    # --------------------------------------------------------

    if operation == "growth":

        if len(years) >= 2:

            return (
                f"Growth in {indicator}, "
                f"{min(years)}–{max(years)}"
            )

        return (
            f"Growth in {indicator}"
        )

    # --------------------------------------------------------
    # Change
    # --------------------------------------------------------

    if operation == "change":

        if len(years) >= 2:

            return (
                f"Change in {indicator}, "
                f"{min(years)}–{max(years)}"
            )

        return (
            f"Change in {indicator}"
        )

    return str(
        indicator
        or
        "Demography and Employment Data"
    )

def evidence_to_dataframe(
    evidence
):
    """
    Convert verified evidence records into a DataFrame
    suitable for plotting.

    No new values are calculated here.
    """

    records = evidence.get(
        "records",
        []
    )

    if not records:

        return pd.DataFrame()

    return pd.DataFrame(
        records
    )

def create_trend_chart(
    evidence
):
    """
    Create a line chart for a time-series trend.
    """

    chart_df = (
        evidence_to_dataframe(
            evidence
        )
    )

    if (
        chart_df.empty
        or
        "Year" not in chart_df.columns
        or
        "Value" not in chart_df.columns
    ):

        return make_chart_result(
            chart_created=False,
            message=(
                "Insufficient verified data "
                "for a trend chart."
            )
        )

    chart_df["Year"] = (
        pd.to_numeric(
            chart_df["Year"],
            errors="coerce"
        )
    )

    chart_df["Value"] = (
        pd.to_numeric(
            chart_df["Value"],
            errors="coerce"
        )
    )

    chart_df = (
        chart_df
        .dropna(
            subset=[
                "Year",
                "Value"
            ]
        )
        .sort_values(
            "Year"
        )
    )

    if chart_df.empty:

        return make_chart_result(
            chart_created=False,
            message=(
                "No plottable trend observations "
                "were available."
            )
        )

    fig = px.line(
        chart_df,
        x="Year",
        y="Value",
        markers=True,
        title=build_chart_title(
            evidence
        ),
        hover_data=[
            column
            for column in [
                "State",
                "Indicator"
            ]
            if column in chart_df.columns
        ]
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=evidence.get(
            "indicator"
        ),
        hovermode="x unified"
    )

    return make_chart_result(
        chart_created=True,
        figure=fig,
        chart_type="line"
    )

def create_comparison_chart(
    evidence
):
    """Create a true grouped bar chart with years treated as discrete categories."""
    chart_df = evidence_to_dataframe(evidence)
    if chart_df.empty or "Value" not in chart_df.columns:
        return make_chart_result(False, message="Insufficient verified data for a comparison chart.")

    chart_df = chart_df.copy()
    chart_df["Value"] = pd.to_numeric(chart_df["Value"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Value"])
    if chart_df.empty:
        return make_chart_result(False, message="No plottable comparison observations were available.")

    states = evidence.get("states", [])
    years = evidence.get("years", [])

    if len(states) >= 2 and len(years) >= 2:
        chart_df["Year_Label"] = chart_df["Year"].astype(int).astype(str)
        fig = px.bar(
            chart_df, x="State", y="Value", color="Year_Label", barmode="group",
            title=build_chart_title(evidence),
            labels={"Year_Label": "Year", "State": "Geography", "Value": evidence.get("indicator")},
            hover_data=[c for c in ["Indicator"] if c in chart_df.columns],
        )
        fig.update_layout(xaxis_title="Geography", yaxis_title=evidence.get("indicator"), legend_title_text="Year")
    elif len(states) == 1 and len(years) >= 2:
        chart_df["Year_Label"] = chart_df["Year"].astype(int).astype(str)
        fig = px.bar(
            chart_df, x="Year_Label", y="Value", title=build_chart_title(evidence),
            labels={"Year_Label": "Year", "Value": evidence.get("indicator")},
            hover_data=[c for c in ["State", "Indicator"] if c in chart_df.columns],
        )
        fig.update_layout(xaxis_title="Year", yaxis_title=evidence.get("indicator"))
    else:
        fig = px.bar(
            chart_df, x="State", y="Value", title=build_chart_title(evidence),
            labels={"State": "Geography", "Value": evidence.get("indicator")},
            hover_data=[c for c in ["Year", "Indicator"] if c in chart_df.columns],
        )
        fig.update_layout(xaxis_title="Geography", yaxis_title=evidence.get("indicator"))

    return make_chart_result(chart_created=True, figure=fig, chart_type="grouped_bar")

def create_ranking_chart(
    evidence
):
    """
    Create a horizontal bar chart from the verified
    state-ranking result.
    """

    chart_df = (
        evidence_to_dataframe(
            evidence
        )
    )

    if (
        chart_df.empty
        or
        "State" not in chart_df.columns
        or
        "Value" not in chart_df.columns
    ):

        return make_chart_result(
            chart_created=False,
            message=(
                "Insufficient verified data "
                "for a ranking chart."
            )
        )

    chart_df["Value"] = (
        pd.to_numeric(
            chart_df["Value"],
            errors="coerce"
        )
    )

    chart_df = (
        chart_df
        .dropna(
            subset=[
                "State",
                "Value"
            ]
        )
        .sort_values(
            "Value",
            ascending=True
        )
    )

    if chart_df.empty:

        return make_chart_result(
            chart_created=False,
            message=(
                "No plottable ranking "
                "observations were available."
            )
        )

    fig = px.bar(
        chart_df,
        x="Value",
        y="State",
        orientation="h",
        title=build_chart_title(
            evidence
        ),
        hover_data=[
            column
            for column in [
                "Rank",
                "Year",
                "Indicator"
            ]
            if column in chart_df.columns
        ]
    )

    fig.update_layout(
        xaxis_title=evidence.get(
            "indicator"
        ),
        yaxis_title="State"
    )

    return make_chart_result(
        chart_created=True,
        figure=fig,
        chart_type="horizontal_bar"
    )

def create_calculation_chart(
    evidence
):
    """
    Visualise growth/change using the verified start and end observations.

    The calculated growth/change remains in the grounded text and hover data;
    the chart itself shows the economically interpretable levels rather than a
    single oversized calculated bar.
    """
    records = evidence.get("records", []) or []
    if not records:
        return make_chart_result(False, message="Insufficient verified calculation results for visualisation.")

    operation = evidence.get("operation")
    rows = []
    for record in records:
        state = record.get("State")
        start_year, end_year = record.get("Start_Year"), record.get("End_Year")
        start_value, end_value = record.get("Start_Value"), record.get("End_Value")
        calculation_value = record.get("Growth_Percent") if operation == "growth" else record.get("Absolute_Change")
        calculation_label = "Growth (%)" if operation == "growth" else "Absolute Change"
        for year, value in [(start_year, start_value), (end_year, end_value)]:
            rows.append({
                "State": state,
                "Year": str(int(year)) if year is not None else "",
                "Value": value,
                calculation_label: calculation_value,
            })

    chart_df = pd.DataFrame(rows)
    chart_df["Value"] = pd.to_numeric(chart_df["Value"], errors="coerce")
    chart_df = chart_df.dropna(subset=["State", "Year", "Value"])
    if chart_df.empty:
        return make_chart_result(False, message="No plottable start/end observations were available.")

    calc_col = "Growth (%)" if operation == "growth" else "Absolute Change"
    fig = px.bar(
        chart_df,
        x="State",
        y="Value",
        color="Year",
        barmode="group",
        title=build_chart_title(evidence),
        labels={"State": "Geography", "Value": evidence.get("indicator")},
        hover_data={calc_col: True},
    )
    fig.update_layout(
        xaxis_title="Geography",
        yaxis_title=evidence.get("indicator"),
        legend_title_text="Year",
    )
    return make_chart_result(chart_created=True, figure=fig, chart_type="start_end_grouped_bar")

def generate_automatic_chart(
    evidence
):
    """
    Determine whether a chart is appropriate and,
    if so, generate it from verified evidence.

    Chart policy:
        lookup   -> no automatic chart
        metadata -> no automatic chart
        trend    -> line
        compare  -> grouped bar
        rank     -> horizontal bar
        growth   -> bar
        change   -> bar
    """

    if not evidence.get(
        "execution_success"
    ):

        return make_chart_result(
            chart_created=False,
            message=(
                "No chart generated because "
                "verified data were unavailable."
            )
        )

    operation = evidence.get(
        "operation"
    )

    if operation == "trend":

        return create_trend_chart(
            evidence
        )

    if operation == "compare":

        return create_comparison_chart(
            evidence
        )

    if operation == "rank":

        return create_ranking_chart(
            evidence
        )

    if operation in {
        "growth",
        "change"
    }:

        return create_calculation_chart(
            evidence
        )

    if operation in {
        "lookup",
        "metadata"
    }:

        return make_chart_result(
            chart_created=False,
            chart_type=None,
            message=(
                "An automatic chart is not required "
                f"for the '{operation}' operation."
            )
        )

    return make_chart_result(
        chart_created=False,
        message=(
            f"No visualisation rule exists "
            f"for operation '{operation}'."
        )
    )

def backend_part4_visualisation_test():
    """
    Test automatic visualisation without Gemini.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BACKEND PART 4 — "
        "AUTOMATIC VISUALISATION TEST"
    )

    print(
        "=" * 70
    )

    tests = {

        "Lookup": QueryPlan(
            operation="lookup",
            states=["Bihar"],
            years=[2026],
            indicator="population density"
        ),

        "Comparison": QueryPlan(
            operation="compare",
            states=[
                "Bihar",
                "India"
            ],
            years=[
                2011,
                2026
            ],
            indicator="population density"
        ),

        "Trend": QueryPlan(
            operation="trend",
            states=["Bihar"],
            years=[],
            indicator="population density"
        ),

        "Ranking": QueryPlan(
            operation="rank",
            states=[],
            years=[2026],
            indicator="population density",
            top_n=10
        ),

        "Missing Data": QueryPlan(
            operation="lookup",
            states=["Bihar"],
            years=[2099],
            indicator="population density"
        ),

        "Metadata": QueryPlan(
            operation="metadata",
            indicator="population density",
            metadata_request="source"
        ),

        "Growth": QueryPlan(
            operation="growth",
            states=["Bihar"],
            years=[
                2011,
                2026
            ],
            indicator="population density"
        ),

        "Change": QueryPlan(
            operation="change",
            states=["Bihar"],
            years=[
                2011,
                2026
            ],
            indicator="population density"
        )
    }

    summary = []

    for test_name, plan in tests.items():

        validation = (
            validate_query_plan(
                plan
            )
        )

        if not validation[
            "valid"
        ]:

            summary.append({
                "Test": test_name,
                "Status": "validation_failed",
                "Chart_Type": None
            })

            continue

        execution = (
            execute_query_plan(
                validation["plan"]
            )
        )

        evidence = (
            build_evidence_package(
                question=(
                    f"Development test: "
                    f"{test_name}"
                ),
                plan=validation[
                    "plan"
                ],
                execution=execution
            )
        )

        chart_result = (
            generate_automatic_chart(
                evidence
            )
        )

        status = (
            "success"
            if execution[
                "success"
            ]
            else
            "no_data"
        )

        summary.append({
            "Test":
                test_name,

            "Status":
                status,

            "Chart_Type":
                chart_result.get(
                    "chart_type"
                )
        })

    summary_df = (
        pd.DataFrame(
            summary
        )
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


# ============================================================
# 18. FINAL CHATBOT INTEGRATION
# ============================================================

def make_chatbot_result(
    status,
    question,
    answer=None,
    plan=None,
    execution=None,
    evidence=None,
    chart=None,
    chart_type=None,
    provenance=None,
    warnings=None,
    message=None,
    parser=None,
    timings=None
):
    """
    Construct the standardized result returned by
    ask_demography() to the frontend.

    The Streamlit application should consume this structure
    rather than interacting directly with internal backend
    functions.
    """

    return {
        "status": status,
        "question": question,
        "answer": answer,
        "plan": plan,
        "execution": execution,
        "evidence": evidence,
        "chart": chart,
        "chart_type": chart_type,
        "provenance": provenance,
        "warnings": warnings or [],
        "message": message,
        "parser": parser,
        "timings": timings or {}
    }




def _try_generic_sex_ratio_bundle(question):
    """
    Handle an unqualified natural-language request for "sex ratio" as a
    two-statistic bundle rather than forcing a clarification between Census
    and NFHS.  For each requested state, return the latest available Census
    Sex Ratio and the latest available NFHS Sex Ratio independently.
    """
    q = normalize_text(question)
    if not _phrase_in_question(q, "sex ratio"):
        return None

    # If the user explicitly names Census or NFHS, preserve the normal
    # single-indicator route.  The bundle is only for generic "sex ratio".
    if _phrase_in_question(q, "census sex ratio") or _phrase_in_question(q, "nfhs sex ratio"):
        return None

    states = _extract_states_from_question(question)
    if not states:
        return None

    indicators = [
        "Census Sex Ratio (females per 1000 males)",
        "NFHS Sex Ratio (females per 1000 males)",
    ]
    labels = {
        indicators[0]: "Census Sex Ratio",
        indicators[1]: "NFHS Sex Ratio",
    }

    answer_blocks = []
    evidence = []
    provenance = []

    for state in states:
        state_resolution = resolve_state(state)
        canonical_state = (
            state_resolution.get("canonical")
            if state_resolution.get("status") == "resolved"
            else state
        )
        lines = []
        for indicator in indicators:
            rows = fact_df.loc[
                (fact_df["State"] == canonical_state)
                & (fact_df["Indicator"] == indicator)
                & fact_df["Value"].notna()
            ].copy()
            if rows.empty:
                lines.append(f"- {labels[indicator]}: data not available.")
                evidence.append({
                    "State": canonical_state, "Indicator": indicator,
                    "Year": None, "Value": None, "Data_Status": "Unavailable"
                })
                continue

            rows["Year"] = pd.to_numeric(rows["Year"], errors="coerce")
            rows = rows.dropna(subset=["Year"]).sort_values("Year")
            latest = rows.iloc[-1]
            year = int(latest["Year"])
            value = to_python_scalar(latest["Value"])
            value_text = f"{float(value):,.0f}" if value is not None else "data not available"
            lines.append(
                f"- {labels[indicator]}: {value_text} females per 1,000 males ({year}, latest available)."
            )
            evidence.append({
                "State": canonical_state, "Indicator": indicator,
                "Year": year, "Value": value, "Data_Status": "Available"
            })
            metadata = get_indicator_metadata(indicator) or {}
            source = metadata.get("Data Source")
            if source and not pd.isna(source):
                provenance.append({
                    "Indicator": indicator, "Data Source": to_python_scalar(source)
                })

        answer_blocks.append(canonical_state + ":\n" + "\n".join(lines))

    # Deduplicate provenance records while preserving order.
    unique_provenance = []
    seen = set()
    for item in provenance:
        key = (item.get("Indicator"), item.get("Data Source"))
        if key not in seen:
            seen.add(key)
            unique_provenance.append(item)

    return make_chatbot_result(
        status="success",
        question=question,
        answer="Latest available sex-ratio statistics:\n\n" + "\n\n".join(answer_blocks),
        evidence=evidence,
        provenance=unique_provenance,
        parser={
            "success": True,
            "parser_mode": "deterministic_fast_path",
            "special_resolution": "generic_sex_ratio_returns_both_latest_series",
        },
        timings={},
    )

def ask_demography(
    question
):
    """Production entry point: local-first interpretation + deterministic analytics."""
    total_start = time.perf_counter()

    if question is not None and str(question).strip():
        sex_ratio_bundle = _try_generic_sex_ratio_bundle(str(question).strip())
        if sex_ratio_bundle is not None:
            sex_ratio_bundle["timings"]["total_seconds"] = time.perf_counter() - total_start
            return sex_ratio_bundle

    if question is None:
        return make_chatbot_result(
            status="invalid", question="",
            message="Please enter a question about the demography and employment dataset.",
            timings={"total_seconds": time.perf_counter() - total_start},
        )

    question = str(question).strip()
    if not question:
        return make_chatbot_result(
            status="invalid", question=question,
            message="Please enter a question about the demography and employment dataset.",
            timings={"total_seconds": time.perf_counter() - total_start},
        )

    interpretation_start = time.perf_counter()
    understanding = understand_question(question)
    interpretation_seconds = time.perf_counter() - interpretation_start
    parser_result = understanding.get("parser", {}) or {}
    timings = {
        "interpretation_seconds": interpretation_seconds,
        "parser_seconds": parser_result.get("latency_seconds"),
        "parser_mode": parser_result.get("parser_mode"),
    }

    understanding_status = understanding.get("status")
    if understanding_status == "parser_error":
        error_type = parser_result.get("error_type")
        messages = {
            "rate_limit_or_quota": "The natural-language parser is temporarily unavailable because its API quota or rate limit has been reached.",
            "timeout": "The natural-language parser took too long to respond. Please try again.",
            "authentication": "The natural-language parser is currently unavailable because its API configuration could not be authenticated.",
            "model_or_endpoint": "The configured language model or API endpoint is currently unavailable.",
        }
        timings["total_seconds"] = time.perf_counter() - total_start
        return make_chatbot_result(
            status="parser_error", question=question,
            message=messages.get(error_type, understanding.get("message") or "The natural-language parser is temporarily unavailable.") + " Please try a supported state, indicator and optional year; straightforward data queries can be answered without the language model.",
            parser=parser_result, timings=timings,
        )

    if understanding_status == "clarification_required":
        timings["total_seconds"] = time.perf_counter() - total_start
        return make_chatbot_result(
            status="clarification_required", question=question,
            plan=understanding.get("plan"), message=understanding.get("message"),
            parser=parser_result, timings=timings,
        )

    if understanding_status == "invalid":
        timings["total_seconds"] = time.perf_counter() - total_start
        return make_chatbot_result(
            status="invalid", question=question, plan=understanding.get("plan"),
            message=understanding.get("message"), parser=parser_result, timings=timings,
        )

    if understanding_status != "validated" or understanding.get("plan") is None:
        timings["total_seconds"] = time.perf_counter() - total_start
        return make_chatbot_result(
            status="invalid", question=question,
            message="The question could not be converted into a valid analytical request.",
            parser=parser_result, timings=timings,
        )

    downstream_start = time.perf_counter()
    result = execute_validated_chatbot_plan(question=question, plan=understanding["plan"])
    timings["deterministic_pipeline_seconds"] = time.perf_counter() - downstream_start
    timings["total_seconds"] = time.perf_counter() - total_start
    result["parser"] = parser_result
    result["timings"] = timings
    return result

def ask_demography_from_plan(
    question,
    plan
):
    """
    Development/testing entry point that bypasses Gemini.

    It validates the supplied QueryPlan and then uses the
    same deterministic downstream pipeline as the production
    chatbot.

    This function is useful for testing and evaluation.
    """

    validation = validate_query_plan(
        plan
    )

    # --------------------------------------------------------
    # Clarification
    # --------------------------------------------------------

    if validation.get(
        "clarification"
    ):

        return make_chatbot_result(
            status="clarification_required",
            question=question,
            plan=None,
            message=validation.get(
                "clarification"
            )
        )

    # --------------------------------------------------------
    # Invalid
    # --------------------------------------------------------

    if not validation.get(
        "valid"
    ):

        errors = validation.get(
            "errors",
            []
        )

        message = (
            "; ".join(errors)
            if errors
            else
            "The supplied query plan is invalid."
        )

        return make_chatbot_result(
            status="invalid",
            question=question,
            plan=validation.get(
                "plan"
            ),
            message=message
        )

    # --------------------------------------------------------
    # Execute canonical plan
    # --------------------------------------------------------

    return execute_validated_chatbot_plan(
        question=question,
        plan=validation[
            "plan"
        ]
    )

def backend_part5_integration_test():
    """
    Test the complete frontend-facing result contract
    without calling Gemini.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BACKEND PART 5 — "
        "FINAL INTEGRATION TEST"
    )

    print(
        "=" * 70
    )

    tests = {

        "Lookup": QueryPlan(
            operation="lookup",
            states=["Bihar"],
            years=[2026],
            indicator="population density"
        ),

        "Comparison": QueryPlan(
            operation="compare",
            states=[
                "Bihar",
                "India"
            ],
            years=[
                2011,
                2026
            ],
            indicator="population density"
        ),

        "Trend": QueryPlan(
            operation="trend",
            states=["Bihar"],
            years=[],
            indicator="population density"
        ),

        "Ranking": QueryPlan(
            operation="rank",
            states=[],
            years=[2026],
            indicator="population density",
            top_n=10
        ),

        "Missing Data": QueryPlan(
            operation="lookup",
            states=["Bihar"],
            years=[2099],
            indicator="population density"
        ),

        "Metadata": QueryPlan(
            operation="metadata",
            indicator="population density",
            metadata_request="source"
        ),

        "Growth": QueryPlan(
            operation="growth",
            states=["Bihar"],
            years=[
                2011,
                2026
            ],
            indicator="population density"
        ),

        "Change": QueryPlan(
            operation="change",
            states=["Bihar"],
            years=[
                2011,
                2026
            ],
            indicator="population density"
        )
    }

    summary = []

    for test_name, plan in tests.items():

        result = ask_demography_from_plan(
            question=(
                f"Development integration test: "
                f"{test_name}"
            ),
            plan=plan
        )

        summary.append({
            "Test":
                test_name,

            "Status":
                result.get(
                    "status"
                ),

            "Has_Answer":
                bool(
                    result.get(
                        "answer"
                    )
                ),

            "Chart_Type":
                result.get(
                    "chart_type"
                ),

            "Has_Provenance":
                (
                    result.get(
                        "provenance"
                    )
                    is not None
                )
        })

    summary_df = pd.DataFrame(
        summary
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

# ============================================================
# 20. PUBLIC BACKEND INTERFACE
# ============================================================

__all__ = [
    "ask_demography",
    "ask_demography_from_plan",
    "QueryPlan",
    "CANONICAL_STATES",
    "CANONICAL_INDICATORS",
    "get_available_years",
    "get_available_states",
    "get_available_indicators"
]

# ============================================================
# DEVELOPMENT RUNNER
# ============================================================

if __name__ == "__main__":

    backend_part1_audit()

    backend_part3_deterministic_test()

    backend_part4_visualisation_test()

    backend_part5_integration_test()
