# Didaskalos Streamlit App

This folder contains the Streamlit web app for Didaskalos. I'm just testing out things at this point and a lot more is coming.

## What the app does

- Loads treebanks from the GitHub repository by default
- Loads lesson modules from the same GitHub repository by default
- Lets users paste one or more treebank URLs if they want to override the defaults
- Lets users upload XML treebanks and markdown lesson modules instead of using GitHub
- Builds a combined token dataframe
- Computes a frequency-based syllabus table
- Exports CSV, markdown, and HTML downloads

## GitHub sources used by default

- Treebanks: [treebanks/perseus](https://github.com/farnoosh-shamsian/didaskalos/tree/main/treebanks/perseus)
- Lesson modules: [lessons-no-decl](https://github.com/farnoosh-shamsian/didaskalos/tree/main/lessons-no-decl)
- App folder: [didaskalos_streamlit_app](https://github.com/farnoosh-shamsian/didaskalos/tree/main/didaskalos_streamlit_app)

## Project layout

- `app.py`: Streamlit UI
- `didaskalos_pipeline.py`: reusable data and export functions
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: Streamlit runtime and theme config
