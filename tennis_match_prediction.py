"""Pre-match ATP winner prediction with a chronological 2024 holdout.

Run from the project root:
    python tennis_match_prediction.py

The input is the yearly Jeff Sackmann-format ATP CSV files in data/.
The script cleans and combines seasons 1991-2024, builds pre-match features,
tunes three classifiers on 2022, then evaluates them on 2024.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BUILD_DIR = ROOT / "build_files" / ".codex-build"
FIRST_SEASON = 1991
LAST_SEASON = 2024
VALIDATION_SEASON = 2022
TEST_SEASON = 2024
HISTORY_GAP_DAYS = 14
RANDOM_STATE = 42

REQUIRED_COLUMNS = {
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num", "winner_id", "winner_hand", "winner_ht",
    "winner_age", "loser_id", "loser_hand", "loser_ht", "loser_age",
    "best_of", "round", "winner_rank", "winner_rank_points",
    "loser_rank", "loser_rank_points",
}

NUMERIC_FEATURES = [
    "rank_a", "rank_b", "rank_advantage_a", "log_rank_advantage_a",
    "points_a", "points_b", "points_advantage_a",
    "age_advantage_a", "height_advantage_a", "left_hand_advantage_a",
    "career_win_rate_advantage_a", "career_match_count_advantage_a",
    "last5_win_rate_advantage_a", "last10_win_rate_advantage_a",
    "last25_win_rate_advantage_a", "surface_win_rate_advantage_a",
    "surface_match_count_advantage_a", "head_to_head_win_rate_advantage_a",
    "head_to_head_match_count", "matches_last30d_advantage_a",
    "best_of", "draw_size",
]
CATEGORICAL_FEATURES = ["surface", "tourney_level", "round"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_clean_matches() -> tuple[pd.DataFrame, dict]:
    """Read, standardize, validate, and combine the annual raw files."""
    paths = sorted(DATA_DIR.glob("atp_matches_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No atp_matches_YYYY.csv files found in {DATA_DIR}. See README.md."
        )

    frames = []
    for path in paths:
        match = re.fullmatch(r"atp_matches_(\d{4})\.csv", path.name)
        if not match:
            continue
        file_season = int(match.group(1))
        if FIRST_SEASON <= file_season <= LAST_SEASON:
            frame = pd.read_csv(path, dtype={"tourney_date": "string"}, low_memory=False)
            missing = REQUIRED_COLUMNS - set(frame.columns)
            if missing:
                raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
            frame["source_season"] = file_season
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No source seasons from {FIRST_SEASON} through {LAST_SEASON} were found."
        )

    raw = pd.concat(frames, ignore_index=True, sort=False)
    rows_read = len(raw)
    raw.columns = [str(column).strip() for column in raw.columns]
    for column in raw.select_dtypes(include=["object", "string"]).columns:
        raw[column] = raw[column].astype("string").str.strip()

    raw["tourney_date"] = pd.to_datetime(
        raw["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    numeric_columns = [
        "winner_id", "loser_id", "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points", "winner_ht", "loser_ht",
        "winner_age", "loser_age", "best_of", "draw_size", "match_num",
    ]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    invalid_date = int(raw["tourney_date"].isna().sum())
    invalid_id = int(raw[["winner_id", "loser_id"]].isna().any(axis=1).sum())
    invalid_same_player = int((raw["winner_id"] == raw["loser_id"]).sum())
    raw = raw.dropna(subset=["tourney_date", "winner_id", "loser_id"])
    raw = raw.loc[raw["winner_id"] != raw["loser_id"]].copy()
    raw["winner_id"] = raw["winner_id"].astype("int64")
    raw["loser_id"] = raw["loser_id"].astype("int64")

    # The event key protects against a repeated source row without collapsing
    # legitimate matches in different rounds or tournaments.
    duplicate_key = ["tourney_id", "match_num", "winner_id", "loser_id"]
    duplicate_rows = int(raw.duplicated(duplicate_key, keep="first").sum())
    raw = raw.drop_duplicates(duplicate_key, keep="first").copy()
    # The annual file name defines the ATP season. Early-season events may
    # have a tournament start date in the prior calendar year (for example,
    # a 2024 event beginning on 2023-12-31).
    raw["season"] = raw["source_season"].astype("int64")
    raw = raw.drop(columns=["source_season"])
    raw = raw.loc[raw["season"].between(FIRST_SEASON, LAST_SEASON)].copy()
    raw = raw.sort_values(
        ["tourney_date", "tourney_id", "match_num", "winner_id", "loser_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    cleaning = {
        "source_files": len(frames),
        "source_seasons": [int(frame["source_season"].iloc[0]) for frame in frames],
        "source_rows_read": int(rows_read),
        "invalid_date_rows_removed": invalid_date,
        "missing_player_id_rows_removed": invalid_id,
        "same_player_rows_removed": invalid_same_player,
        "duplicate_rows_removed": duplicate_rows,
        "clean_rows": int(len(raw)),
        "season_range": [FIRST_SEASON, LAST_SEASON],
    }
    return raw, cleaning


def surface_name(value) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Unknown"
    return str(value).strip().title()


def numeric_value(value):
    return None if pd.isna(value) else float(value)


def safe_rate(wins: int, total: int, neutral: float = 0.5) -> float:
    return float(wins / total) if total else neutral


def add_player_history(history, match_row: dict, match_date: pd.Timestamp) -> None:
    """Add both outcomes to cumulative and rolling histories."""
    surface = surface_name(match_row.get("surface"))
    winner = int(match_row["winner_id"])
    loser = int(match_row["loser_id"])
    history["career"][winner][0] += 1
    history["career"][winner][1] += 1
    history["career"][loser][1] += 1
    history["surface"][winner][surface][0] += 1
    history["surface"][winner][surface][1] += 1
    history["surface"][loser][surface][1] += 1
    history["recent"][winner].append((match_date, 1, surface))
    history["recent"][loser].append((match_date, 0, surface))

    low, high = sorted((winner, loser))
    pair_key = (low, high)
    history["head_to_head"][pair_key][1] += 1
    if winner == low:
        history["head_to_head"][pair_key][0] += 1


def player_form(history, player_id: int, surface: str, current_date: pd.Timestamp) -> dict:
    career_wins, career_matches = history["career"][player_id]
    surface_wins, surface_matches = history["surface"][player_id][surface]
    recent = history["recent"][player_id]
    recent_results = [item[1] for item in recent]
    features = {
        "career_win_rate": safe_rate(career_wins, career_matches),
        "career_match_count": career_matches,
        "surface_win_rate": safe_rate(surface_wins, surface_matches),
        "surface_match_count": surface_matches,
    }
    for length in (5, 10, 25):
        previous = recent_results[-length:]
        features[f"last{length}_win_rate"] = safe_rate(sum(previous), len(previous))

    # tourney_date is the event start date, not a match date. To avoid using
    # results from an overlapping event, only tournaments that started more
    # than 14 days before this event enter the history snapshot.
    activity_start = current_date - pd.Timedelta(days=44)
    features["matches_last30d"] = sum(
        activity_start <= item[0] < current_date - pd.Timedelta(days=14)
        for item in recent
    )
    return features


def build_prematch_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create one row per match, with an ID-stable player A/B orientation.

    Player A is the player with the lower player ID. Events sharing a
    tournament start date are evaluated from the same prior-history snapshot.
    Results become eligible only after the 14-day event-start-date gap.
    """
    history = {
        "career": defaultdict(lambda: [0, 0]),
        "surface": defaultdict(lambda: defaultdict(lambda: [0, 0])),
        "recent": defaultdict(deque),
        "head_to_head": defaultdict(lambda: [0, 0]),
    }
    records = []
    # Filter Davis Cup from both training and history. Its team format is not
    # comparable to ordinary tour singles and it is not reliably date-stamped.
    is_davis_cup = matches["tourney_name"].fillna("").str.contains(
        "davis cup", case=False, regex=False
    )
    eligible = matches.loc[~is_davis_cup]
    date_groups = list(eligible.groupby("tourney_date", sort=True))
    history_group_index = 0

    for current_date, current_group in date_groups:
        cutoff = current_date - pd.Timedelta(days=HISTORY_GAP_DAYS)
        while (
            history_group_index < len(date_groups)
            and date_groups[history_group_index][0] < cutoff
        ):
            old_date, old_group = date_groups[history_group_index]
            for old_row in old_group.to_dict(orient="records"):
                add_player_history(history, old_row, old_date)
            history_group_index += 1

        for row in current_group.to_dict(orient="records"):
            winner_id = int(row["winner_id"])
            loser_id = int(row["loser_id"])
            player_a, player_b = sorted((winner_id, loser_id))
            a_is_winner = int(winner_id == player_a)
            a_is_winner_side = "winner" if a_is_winner else "loser"
            b_side = "loser" if a_is_winner else "winner"
            surface = surface_name(row.get("surface"))

            a_form = player_form(history, player_a, surface, current_date)
            b_form = player_form(history, player_b, surface, current_date)
            low, high = player_a, player_b
            h2h_low_wins, h2h_count = history["head_to_head"][(low, high)]
            h2h_advantage_a = (
                (2 * h2h_low_wins - h2h_count) / h2h_count if h2h_count else 0.0
            )
            def get_for_side(field: str, side: str):
                return numeric_value(row.get(f"{side}_{field}"))

            rank_a = get_for_side("rank", a_is_winner_side)
            rank_b = get_for_side("rank", b_side)
            points_a = get_for_side("rank_points", a_is_winner_side)
            points_b = get_for_side("rank_points", b_side)
            age_a = get_for_side("age", a_is_winner_side)
            age_b = get_for_side("age", b_side)
            height_a = get_for_side("ht", a_is_winner_side)
            height_b = get_for_side("ht", b_side)
            hand_a = row.get(f"{a_is_winner_side}_hand")
            hand_b = row.get(f"{b_side}_hand")
            left_a = None if pd.isna(hand_a) or not str(hand_a).strip() else int(str(hand_a).upper() == "L")
            left_b = None if pd.isna(hand_b) or not str(hand_b).strip() else int(str(hand_b).upper() == "L")

            level_value = row.get("tourney_level")
            round_value = row.get("round")
            record = {
                "match_date": current_date,
                "season": int(row["season"]),
                "tourney_id": str(row["tourney_id"]),
                "player_a_id": player_a,
                "player_b_id": player_b,
                "player_a_won": a_is_winner,
                "rank_a": rank_a,
                "rank_b": rank_b,
                "rank_advantage_a": None if rank_a is None or rank_b is None else rank_b - rank_a,
                "log_rank_advantage_a": None if rank_a is None or rank_b is None or rank_a <= 0 or rank_b <= 0 else math.log1p(rank_b) - math.log1p(rank_a),
                "points_a": points_a,
                "points_b": points_b,
                "points_advantage_a": None if points_a is None or points_b is None else points_a - points_b,
                "age_advantage_a": None if age_a is None or age_b is None else age_a - age_b,
                "height_advantage_a": None if height_a is None or height_b is None else height_a - height_b,
                "left_hand_advantage_a": None if left_a is None or left_b is None else left_a - left_b,
                "career_win_rate_advantage_a": a_form["career_win_rate"] - b_form["career_win_rate"],
                "career_match_count_advantage_a": math.log1p(a_form["career_match_count"]) - math.log1p(b_form["career_match_count"]),
                "last5_win_rate_advantage_a": a_form["last5_win_rate"] - b_form["last5_win_rate"],
                "last10_win_rate_advantage_a": a_form["last10_win_rate"] - b_form["last10_win_rate"],
                "last25_win_rate_advantage_a": a_form["last25_win_rate"] - b_form["last25_win_rate"],
                "surface_win_rate_advantage_a": a_form["surface_win_rate"] - b_form["surface_win_rate"],
                "surface_match_count_advantage_a": math.log1p(a_form["surface_match_count"]) - math.log1p(b_form["surface_match_count"]),
                "head_to_head_win_rate_advantage_a": h2h_advantage_a,
                "head_to_head_match_count": h2h_count,
                "matches_last30d_advantage_a": a_form["matches_last30d"] - b_form["matches_last30d"],
                "best_of": numeric_value(row.get("best_of")),
                "draw_size": numeric_value(row.get("draw_size")),
                "surface": surface,
                "tourney_level": "Unknown" if pd.isna(level_value) or not str(level_value).strip() else str(level_value).strip(),
                "round": "Unknown" if pd.isna(round_value) or not str(round_value).strip() else str(round_value).strip(),
            }
            records.append(record)

    features = pd.DataFrame.from_records(records)
    if features.empty:
        raise ValueError("No eligible matches remained after the Davis Cup filter.")
    return features


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    if scale_numeric:
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ])
    else:
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")


def model_candidates() -> list[tuple[str, dict, bool]]:
    """Return concise, predefined candidate grids for the three models."""
    grids = [
        ("Logistic regression", {"C": [0.01, 0.1, 1.0, 10.0]}, True),
        ("Linear SVM", {"C": [0.01, 0.1, 1.0, 10.0]}, True),
        ("Random forest", {
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 3],
        }, False),
    ]
    return grids


def create_estimator(name: str, params: dict):
    if name == "Logistic regression":
        return LogisticRegression(
            C=params["C"], solver="liblinear",
            max_iter=2000, random_state=RANDOM_STATE,
        )
    if name == "Linear SVM":
        return LinearSVC(
            C=params["C"], dual="auto", max_iter=12000,
            random_state=RANDOM_STATE,
        )
    if name == "Random forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {name}")


def pipeline_for(name: str, params: dict, scale_numeric: bool) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor(scale_numeric)),
        ("model", create_estimator(name, params)),
    ])


def get_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    prediction = model.predict(X)
    result = {
        "matches": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision_player_a": float(precision_score(y, prediction, zero_division=0)),
        "recall_player_a": float(recall_score(y, prediction, zero_division=0)),
        "f1_player_a": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }
    if hasattr(model, "decision_function"):
        try:
            result["roc_auc"] = float(roc_auc_score(y, model.decision_function(X)))
        except ValueError:
            result["roc_auc"] = None
    return result


def tune_models(features: pd.DataFrame, y: pd.Series) -> tuple[dict, dict]:
    train_mask = features["season"] <= VALIDATION_SEASON - 1
    validation_mask = features["season"] == VALIDATION_SEASON
    X_train = features.loc[train_mask, FEATURE_COLUMNS]
    y_train = y.loc[train_mask]
    X_valid = features.loc[validation_mask, FEATURE_COLUMNS]
    y_valid = y.loc[validation_mask]
    if X_train.empty or X_valid.empty:
        raise ValueError("The 2021 training or 2022 validation split is empty.")

    tuning = {}
    best_params = {}
    for name, grid, scale_numeric in model_candidates():
        candidates = []
        for params in ParameterGrid(grid):
            model = pipeline_for(name, params, scale_numeric)
            model.fit(X_train, y_train)
            prediction = model.predict(X_valid)
            score = float(accuracy_score(y_valid, prediction))
            candidates.append({"parameters": params, "validation_accuracy": score})
        candidates.sort(key=lambda item: (-item["validation_accuracy"], str(item["parameters"])))
        best_params[name] = candidates[0]["parameters"]
        tuning[name] = {
            "training_matches_through_2021": int(len(y_train)),
            "validation_matches_2022": int(len(y_valid)),
            "selection_metric": "accuracy",
            "trials": candidates,
            "selected_parameters": best_params[name],
            "selected_validation_accuracy": candidates[0]["validation_accuracy"],
        }
        print(f"{name}: best 2022 validation accuracy {candidates[0]['validation_accuracy']:.3%} with {best_params[name]}")
    return best_params, tuning


def evaluate_test_season(features: pd.DataFrame, y: pd.Series, best_params: dict) -> dict:
    train_mask = features["season"] <= TEST_SEASON - 1
    test_mask = features["season"] == TEST_SEASON
    X_train = features.loc[train_mask, FEATURE_COLUMNS]
    y_train = y.loc[train_mask]
    X_test = features.loc[test_mask, FEATURE_COLUMNS]
    y_test = y.loc[test_mask]
    test_features = features.loc[test_mask].reset_index(drop=True)
    if X_train.empty or X_test.empty:
        raise ValueError("The final training or 2024 test split is empty.")

    model_results = {}
    fitted_models = {}
    for name, _, scale_numeric in model_candidates():
        model = pipeline_for(name, best_params[name], scale_numeric)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        model_results[name] = {
            "selected_parameters_from_2022": best_params[name],
            "train_matches_through_2023": int(len(y_train)),
            "2024": get_scores(model, X_test, y_test),
        }

    # The rank-only benchmark is evaluated only where both pre-match ranks
    # exist; models are also rescored on this same subset for a fair comparison.
    rank_a = test_features["rank_a"]
    rank_b = test_features["rank_b"]
    rank_mask = rank_a.notna() & rank_b.notna()
    baseline_y = y_test.reset_index(drop=True).loc[rank_mask].to_numpy()
    baseline_prediction = (rank_a.loc[rank_mask].to_numpy() < rank_b.loc[rank_mask].to_numpy()).astype(int)
    rank_baseline = {
        "matches_with_both_ranks": int(rank_mask.sum()),
        "coverage_of_2024_matches": float(rank_mask.mean()),
        "accuracy": float(accuracy_score(baseline_y, baseline_prediction)),
        "confusion_matrix_labels_0_1": confusion_matrix(baseline_y, baseline_prediction, labels=[0, 1]).tolist(),
    }
    for name, model in fitted_models.items():
        rank_mask_array = rank_mask.to_numpy()
        X_rank = X_test.loc[rank_mask_array]
        y_rank = y_test.loc[rank_mask_array]
        model_results[name]["2024_same_matches_as_rank_baseline"] = get_scores(model, X_rank, y_rank)

    surface_results = {}
    test_rows = test_features.copy()
    test_rows["actual"] = y_test.reset_index(drop=True)
    for surface, group in test_rows.groupby("surface", dropna=False):
        X_group = group[FEATURE_COLUMNS]
        y_group = group["actual"]
        surface_results[str(surface)] = {
            "matches": int(len(group)),
            "models": {
                name: {"accuracy": float(accuracy_score(y_group, model.predict(X_group)))}
                for name, model in fitted_models.items()
            },
        }

    return {
        "training_matches_through_2023": int(len(y_train)),
        "test_matches_2024": int(len(y_test)),
        "rank_only_baseline": rank_baseline,
        "models": model_results,
        "results_by_surface": surface_results,
    }


def save_evaluation_plots(results: dict) -> list[Path]:
    """Save two simple plots from the final 2024 evaluation metrics."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # Confusion matrix for the model with the highest overall 2024 accuracy.
    matrix = results["models"]["Logistic regression"]["2024"]["confusion_matrix_labels_0_1"]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Player B won", "Player A won"])
    ax.set_yticks([0, 1], ["Player B won", "Player A won"])
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title("Logistic regression confusion matrix (2024)")
    threshold = max(max(row) for row in matrix) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(
                column_index, row_index, f"{value:,}",
                ha="center", va="center",
                color="white" if value > threshold else "#25313C",
                fontsize=12, fontweight="bold",
            )
    fig.colorbar(image, ax=ax, label="Number of matches")
    fig.tight_layout()
    path = BUILD_DIR / "confusion_matrix_2024.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)

    # Compare the same three models across the three court surfaces.
    by_surface = results["results_by_surface"]
    preferred_order = {"Clay": 0, "Grass": 1, "Hard": 2}
    surfaces = sorted(by_surface, key=lambda name: preferred_order.get(name, 99))
    model_names = ["Logistic regression", "Linear SVM", "Random forest"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_width = 0.24
    centers = list(range(len(surfaces)))
    colors = ["#3478A8", "#238A8D", "#E39B3B"]
    for model_index, model_name in enumerate(model_names):
        positions = [center + (model_index - 1) * bar_width for center in centers]
        accuracies = [
            by_surface[surface]["models"][model_name]["accuracy"] * 100
            for surface in surfaces
        ]
        ax.bar(positions, accuracies, bar_width, label=model_name, color=colors[model_index])
    ax.set_xticks(centers, surfaces)
    ax.set_ylim(0, 70)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Court surface")
    ax.set_title("2024 test accuracy by surface")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = BUILD_DIR / "accuracy_by_surface_2024.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)
    return saved_paths


def main() -> None:
    BUILD_DIR.mkdir(exist_ok=True)
    matches, cleaning = load_and_clean_matches()
    cleaned_csv = BUILD_DIR / "atp_matches_1991_2024_clean.csv"
    matches.to_csv(cleaned_csv, index=False, date_format="%Y-%m-%d")
    print(f"Cleaned match table: {len(matches):,} matches ({FIRST_SEASON}-{LAST_SEASON})")
    features = build_prematch_features(matches)
    y = features["player_a_won"].astype("int8")
    print(f"Prediction rows after excluding Davis Cup: {len(features):,}")

    best_params, tuning = tune_models(features, y)
    final_results = evaluate_test_season(features, y, best_params)
    plot_paths = save_evaluation_plots(final_results)
    output = {
        "project": "ATP tennis match winner prediction",
        "data_source": "Jeff Sackmann-format ATP match CSVs mirrored by farhadGithub/tennis-atp-data",
        "source_url": "https://github.com/farhadGithub/tennis-atp-data",
        "license": "CC BY-NC-SA 4.0; attribute Jeff Sackmann / Tennis Abstract; non-commercial share-alike",
        "cohort": {"first_season": FIRST_SEASON, "last_season": LAST_SEASON},
        "split": {
            "tuning_train": f"{FIRST_SEASON}-2021",
            "validation": "2022",
            "final_train": f"{FIRST_SEASON}-2023",
            "final_test": "2024",
        },
        "history_gap_days": HISTORY_GAP_DAYS,
        "history_gap_reason": "The source gives tournament start dates, not exact match dates; prior event results enter history only after a 14-day gap.",
        "feature_columns": FEATURE_COLUMNS,
        "excluded_from_prediction_features": [
            "winner and loser outcome for the current match",
            "current-match score, duration, aces, serve and break-point statistics",
            "Davis Cup matches",
        ],
        "cleaning": cleaning,
        "tuning": tuning,
        "final_evaluation": final_results,
    }
    results_path = BUILD_DIR / "project_results.json"
    results_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print("\n2024 test accuracy:")
    for name, result in final_results["models"].items():
        print(f"  {name}: {result['2024']['accuracy']:.3%}")
    print(f"  Higher-ranked baseline (both ranks known): {final_results['rank_only_baseline']['accuracy']:.3%}")
    print(f"Results saved to {results_path.relative_to(ROOT)}")
    print(f"Clean combined table saved to {cleaned_csv.relative_to(ROOT)}")
    for path in plot_paths:
        print(f"Plot saved to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
