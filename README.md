# ATP Tennis Match Prediction

This student project predicts ATP singles match winners from information that would be available before the match. It follows the broad feature and model ideas in the supplied senior reference, with a stricter chronological evaluation and explicit hyperparameter tuning.

## Project files

```text
data/                         Yearly source CSVs and one cleaned Excel workbook
deliverables/                 Two-page project report and presentation slides
tennis_match_prediction.ipynb  Main project notebook; charts display inline
tennis_match_prediction.py    Equivalent script for running from a terminal
requirements.txt              Python packages
README.md                     Project setup and method
build_files/                  Archived build leftovers; not needed for submission
```

Main hand-in files:

- `data/atp_matches_1991_2024.xlsx` — cleaned, combined dataset in one worksheet.
- `deliverables/project_summary.pdf` — two-page report.
- `deliverables/presentation_revised.pptx` — final presentation.

`build_files/` holds temporary chart/build output, Python cache, and archived presentation drafts and experiments. It is not needed for submission. Each run writes four PNG plots under `build_files/.codex-build/`: 2024 model accuracy comparison, confusion matrices for Logistic Regression/SVM/Random Forest, 2022 hyperparameter-tuning results, and 2024 accuracy by court surface. The terminal prints each plot's path when the run finishes.

## Dataset

The yearly ATP singles match CSVs are from the Jeff Sackmann format, obtained from the `farhadGithub/tennis-atp-data` mirror. The mirror contains annual files from 1968 through 2024. This project combines seasons 1991-2024 for the analysis workbook and model. The source files are kept separately in `data/`; the cleaned analysis dataset is also provided as one Excel workbook with one worksheet.

The data was collected by Jeff Sackmann / Tennis Abstract and is released under **CC BY-NC-SA 4.0**. Credit the original source, use it for non-commercial purposes, and keep redistributed adapted data under the same license. Source: <https://github.com/farhadGithub/tennis-atp-data>.

## Setup and run

Use Python 3.10 or newer. Install packages from the project root:

```bash
python -m pip install -r requirements.txt
```

Open `tennis_match_prediction.ipynb` in VS Code, select the project's Python environment as the notebook kernel, then choose **Run All**. The notebook reads `data/atp_matches_YYYY.csv`, cleans and combines 1991-2024, engineers pre-match features, tunes the models, fits through 2023, and evaluates on 2024. The run may take a few minutes because the random forest grid is trained on the full historical training set. The plots display inline and are also saved under `build_files/.codex-build/`. The equivalent terminal command is `python tennis_match_prediction.py`.

## Method

- **Cleaning:** combine the yearly files using the season in each filename; standardize the date and numeric fields; remove rows without valid dates/player IDs, self-matches, and duplicate match keys; retain missing optional values for training-only imputation. A few annual files include events whose start date falls in the prior December, so the file's year is used as the ATP season label.
- **Pre-match features:** rankings and points, age, height and handedness differences; career and recent win rates; surface record; prior head-to-head; tournament surface, level, round, best-of, and draw size.
- **Leakage control:** the source has tournament start dates, not exact match dates. Historical match results are therefore held back until their event start date is more than 14 days before the predicted event. Matches with the same start date share one history snapshot. Current match outcomes, scores, duration, aces, and serve statistics are not model inputs. Davis Cup is excluded from the model due to its team format and date ambiguity.
- **Validation and tuning:** train on seasons through 2021 and select settings on 2022 validation accuracy. Test the tuned logistic regression, linear SVM, and random forest after fitting through 2023 on 2024.
- **Baseline:** predict the higher-ranked player; evaluate this only for 2024 matches with both ranks recorded and compare models on that same subset as well.
- **Imputation and encoding:** fit median imputation, categorical imputation, one-hot encoding, and (where appropriate) scaling inside each training pipeline. Validation and test records do not set preprocessing values.

The 2024 test is a walk-forward simulation: the model is fitted only on seasons through 2023, but results from earlier, sufficiently completed 2024 events can contribute to the historical form features for later 2024 predictions. They never contribute to model fitting or hyperparameter selection.

## Results

The run prints the selected 2022 settings and the 2024 metrics for every model. In the completed run, logistic regression reached 65.34% accuracy, linear SVM 65.23%, and random forest 64.45% on the 2,816 eligible 2024 matches. The higher-ranked-player baseline scored 63.77% on the 2,810 matches with both rankings available.
