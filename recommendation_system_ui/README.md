# Recommendation System UI

This project is a professional and advanced movie recommendation web app built with Streamlit.
It supports richer filtering, score-based ranking, and catalog insights.

## Professional Features

- Elegant dashboard UI with custom styling
- Advanced personalization filters:
  - preferred genres
  - mood
  - language
  - minimum rating
  - release year range
  - runtime range
  - streaming platform
- Weighted recommendation score with ranking transparency
- Recommendation reason shown for every result
- Insights tab with visual charts
- Expanded dataset with multilingual and multi-platform titles

## Project Structure

```text
recommendation_system_ui/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── items.csv
```

## How to Run

1. Open terminal in `recommendation_system_ui`.
2. Create and activate virtual environment (optional but recommended):

   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

5. Open the local URL shown in terminal (usually `http://localhost:8501`).

## Notes

- The recommendation engine uses a weighted content-based scoring model.
- You can add more rows in `data/items.csv` to expand the catalog.
- If the app is already running, Streamlit auto-reloads after file changes.
