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

- Treebank collections: listed in [treebanks/registry.json](https://github.com/farnoosh-shamsian/didaskalos/blob/main/treebanks/registry.json) (Perseus by default)
- Lesson modules: [lessons-no-decl](https://github.com/farnoosh-shamsian/didaskalos/tree/main/lessons-no-decl)
- App folder: [didaskalos_streamlit_app](https://github.com/farnoosh-shamsian/didaskalos/tree/main/didaskalos_streamlit_app)

## Adding a treebank collection

Treebanks are discovered from a manifest, `treebanks/registry.json`, so adding a new
collection needs no pipeline changes. Two supported formats:

- `agdt-xml` — Perseus / Ancient Greek Dependency Treebank XML (`<sentence>`/`<word>` with
  9-character postags). This is the format under `treebanks/perseus/`.
- `conllu` — CoNLL-U (Universal Dependencies / PROIEL). The parser normalizes UD morphology
  into the same 9-character postag the rest of the pipeline reads, preferring the original
  AGDT tag when the source carries it in the XPOS column.

To add a collection:

1. Put the files in a new folder, e.g. `treebanks/gorman/` (AGDT XML) or `treebanks/proiel/`
   (`*.conllu`).
2. Add an entry to `treebanks/registry.json`:

   ```json
   {
     "id": "gorman",
     "name": "Vanessa Gorman Treebanks",
     "path": "treebanks/gorman/",
     "format": "agdt-xml",
     "file_glob": "*.xml",
     "language": "grc",
     "license": "CC BY 4.0",
     "source_url": "https://github.com/vgorman1/Greek-Dependency-Trees"
   }
   ```

   Use `"format": "conllu"` and `"file_glob": "*.conllu"` for a CoNLL-U corpus. `name`,
   `author`, and `license` are shown in the treebank selector (and are the metadata source for
   CoNLL-U, which has no XML header).

3. Check that the files parse before committing:

   ```
   py -3 validate_treebank.py path/to/file.xml
   py -3 validate_treebank.py path/to/file.conllu --format conllu
   ```

   A healthy result shows non-zero sentences/tokens and a low "Undecodable postag" count.

4. Commit and push to `main`. Note: `.gcloudignore` excludes `treebanks/` from the Cloud Build
   upload, so the deployed app reads corpora from GitHub raw — a new collection goes live only
   once it is pushed to `main`. The local folder is used only when running the app locally.

A new *format* (beyond `agdt-xml`/`conllu`) is added once by writing an adapter in
`treebank_parsers.py` and registering it in `PARSERS`; every adapter must emit the same token
schema and normalize morphology into the 9-character postag.

## Project layout

- `app.py`: Streamlit UI and source discovery
- `didaskalos_pipeline.py`: reusable data and export functions
- `treebank_parsers.py`: pluggable per-format parser adapters + dispatcher
- `validate_treebank.py`: CLI to check a treebank file parses correctly
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: Streamlit runtime and theme config
