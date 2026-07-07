from __future__ import annotations

import base64
import os
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_stdio()

from didaskalos_pipeline import (
    build_combined_df,
    build_declension_summary,
    build_frequency_syllabus,
    generate_textbook_html,
    generate_textbook_markdown,
)


st.set_page_config(page_title="Didaskalos", page_icon="DB", layout="wide")
APP_DIR = Path(__file__).resolve().parent
HEADER_IMAGE_PATH = APP_DIR / "assets" / "electroplato.png"
LOGO_IMAGE_PATH = APP_DIR / "assets" / "englishdark.png"
header_image_html = ""
if HEADER_IMAGE_PATH.exists():
    encoded_image = base64.b64encode(HEADER_IMAGE_PATH.read_bytes()).decode("ascii")
    header_image_html = (
        f'<img src="data:image/png;base64,{encoded_image}" '
        'style="float: right; width: 34%; max-width: 360px; min-width: 200px; margin: 0 0 0.9rem 1.1rem; border-radius: 8px;" '
        'alt="Didaskalos header image" />'
    )

st.title("Didaskalos: Frequency-Based Greek Grammar Textbook Builder")
st.markdown(
    """
    <p style="font-size: 1.2rem; font-weight: 600; margin-top: -0.3rem; margin-bottom: 1rem;">
        A tool in the making for a frequency-based Ancient Greek grammar textbook from treebanks by someone obsessed with frequency.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="line-height: 1.6; font-size: 1rem; text-align: left;">
        {header_image_html}
        <p>Didaskalos builds a frequency-based Ancient Greek grammar textbook from treebanks and modular lessons. You can either use sample XML treebanks (like the ones I uploaded from the Perseus Digital Library) or upload your own.
            The app parses the treebanks, pulls out grammatical features, and ranks them by frequency. Based on that, it generates
            lessons that focus on the structures that actually show up most in the texts.
        </p>
        <p>
            Right now, the lessons are LLM-generated using a RAG setup with sources like Smyth's Greek Grammar and Crosby &amp; Schaefer.
            They're basically a first draft. I plan to clean them up manually later.
        </p>
        <p>
            You can download the generated textbook as Markdown or HTML, so it's easy to tweak or restyle. There are also generated
            exercises from your chosen texts, where the sentences are ranked by difficulty based on their length and how frequent their
            words are.
        </p>
        <p>
            You can now choose between two textbook types. The original case-based textbook explains nouns and adjectives
            case-by-case (like the accusative across all declensions). The new declension-based textbook first classifies every
            noun and adjective in your treebanks into declension classes (first declension feminine, second declension masculine,
            common third declension stem types, and so on), counts how often each class occurs, and arranges the lessons by that
            frequency. This is still an early version and mostly a starting point. More features coming soon!
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

GITHUB_OWNER = "farnoosh-shamsian"
GITHUB_REPO = "Didaskalos"
GITHUB_BRANCH = "main"
GITHUB_TREE_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}"
TREEBANK_PREFIX = "treebanks/perseus/"
LESSON_PREFIX = "lessons-no-decl/"
DECLENSION_LESSON_PREFIX = "lessons-decl/"
LESSON_PREFIXES = (LESSON_PREFIX, DECLENSION_LESSON_PREFIX)
REPO_ROOT = Path(__file__).resolve().parent.parent
STARTER_LESSON_FILES = [
    "about.md",
    "alphabet.md",
    "introduction_nouns.md",
    "introduction_adjectives.md",
    "introduction_verbs.md",
]


def _read_from_local_repo_if_available(source_url: str) -> bytes | None:
    # Fallback for Streamlit Cloud: read file from local checkout when HTTP fetch fails.
    try:
        parsed = urlparse(source_url)
        raw_prefix = f"/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"
        if not parsed.path.startswith(raw_prefix):
            return None

        repo_relative_path = parsed.path[len(raw_prefix):]
        local_path = (REPO_ROOT / repo_relative_path).resolve()

        if local_path.exists() and local_path.is_file():
            return local_path.read_bytes()
    except Exception:
        return None

    return None


def _unique_name(name: str, used_names: set[str]) -> str:
    base = Path(name).stem
    suffix = Path(name).suffix or ".xml"
    candidate = f"{base}{suffix}"
    counter = 2
    while candidate in used_names:
        candidate = f"{base}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def _extract_xml_metadata(xml_bytes: bytes) -> tuple[str | None, str | None]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None, None

    def _text_for(xpath: str) -> str | None:
        element = root.find(xpath)
        if element is None:
            return None
        value = " ".join(element.itertext()).strip()
        return value or None

    title = _text_for(".//title")
    author = _text_for(".//author")
    return title, author


def _parse_list_input(text: str) -> list[str]:
    parts: list[str] = []
    for line in (text or "").splitlines():
        parts.extend(item.strip() for item in line.split(","))
    urls = [item for item in parts if item]

    seen = set()
    deduped = []
    for item in urls:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _normalize_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url

    path = quote(parts.path, safe="/%")
    query = quote(parts.query, safe="=&?/%")
    fragment = quote(parts.fragment, safe="%")
    return urlunsplit((parts.scheme, parts.netloc, path, query, fragment))


def _download_url_records_to_dir(records: list[dict], suffix_dir_name: str) -> tuple[Path | None, list[dict]]:
    if not records:
        return None, []

    target_dir = Path(tempfile.mkdtemp(prefix=f"didaskalos_{suffix_dir_name}_"))
    enriched_records: list[dict] = []
    failed_records: list[dict] = []

    for item in records:
        source_url = _normalize_url(item["source_url"])
        try:
            request = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request) as response:
                payload = response.read()
                (target_dir / item["file"]).write_bytes(payload)
                title, author = _extract_xml_metadata(payload)
                enriched_records.append({**item, "title": title, "author": author})
        except (HTTPError, URLError, TimeoutError, ValueError):
            local_payload = _read_from_local_repo_if_available(source_url)
            if local_payload is not None:
                (target_dir / item["file"]).write_bytes(local_payload)
                title, author = _extract_xml_metadata(local_payload)
                enriched_records.append({**item, "title": title, "author": author})
            else:
                failed_records.append(item)
                continue

    if not enriched_records:
        return None, []

    return target_dir, enriched_records


@st.cache_data(show_spinner=False)
def load_github_tree_urls(prefix: str) -> list[str]:
    request = Request(GITHUB_TREE_API, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []

    tree_nodes = payload.get("tree") if isinstance(payload, dict) else None
    if not isinstance(tree_nodes, list):
        return []

    urls = []
    for node in tree_nodes:
        path = str(node.get("path", ""))
        if node.get("type") != "blob":
            continue
        if not path.startswith(prefix):
            continue
        if prefix == TREEBANK_PREFIX and not path.lower().endswith(".xml"):
            continue
        if prefix in LESSON_PREFIXES and not path.lower().endswith(".md"):
            continue
        urls.append(f"{GITHUB_RAW_BASE}/{path}")

    return sorted(urls)


def _build_records_from_urls(urls: list[str], extract_xml_metadata: bool = False) -> list[dict]:
    used_names = set()
    records = []
    for i, url in enumerate(urls, start=1):
        parsed = urlparse(url)
        file_name = Path(parsed.path).name or f"file_{i}"
        title = None
        author = None
        if extract_xml_metadata:
            try:
                request = Request(_normalize_url(url), headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(request) as response:
                    title, author = _extract_xml_metadata(response.read())
            except (HTTPError, URLError, TimeoutError, ET.ParseError):
                title, author = None, None
        records.append(
            {
                "file": _unique_name(file_name, used_names),
                "source_url": url,
                "title": title,
                "author": author,
            }
        )
    return records


def _ensure_starter_lesson_urls(urls: list[str]) -> list[str]:
    required_urls = [f"{GITHUB_RAW_BASE}/{LESSON_PREFIX}{filename}" for filename in STARTER_LESSON_FILES]
    combined = list(urls or []) + required_urls

    seen = set()
    unique_urls = []
    for url in combined:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def _list_local_lesson_urls(prefix: str = LESSON_PREFIX) -> list[str]:
    lesson_dir = REPO_ROOT / prefix
    if not lesson_dir.exists() or not lesson_dir.is_dir():
        return []

    urls: list[str] = []
    for path in sorted(lesson_dir.glob("*.md")):
        # Keep URL format consistent with GitHub raw sources for downstream handling.
        urls.append(f"{GITHUB_RAW_BASE}/{prefix}{path.name}")
    return urls


def _resolve_default_lesson_urls(syllabus_mode: str = "case") -> list[str]:
    # In declension mode, list the declension lesson modules first so they keep
    # their canonical filenames if a module exists in both folders.
    prefixes = [DECLENSION_LESSON_PREFIX, LESSON_PREFIX] if syllabus_mode == "declension" else [LESSON_PREFIX]

    merged: list[str] = []
    for prefix in prefixes:
        # Merge remote and local so transient GitHub API gaps do not hide available lessons.
        merged.extend(load_github_tree_urls(prefix))
        merged.extend(_list_local_lesson_urls(prefix))
    return _ensure_starter_lesson_urls(merged)


def _build_records_from_uploads(uploaded_files) -> list[dict]:
    used_names = set()
    records = []
    for i, uploaded_file in enumerate(uploaded_files or []):
        file_bytes = uploaded_file.getvalue()
        title, author = _extract_xml_metadata(file_bytes) if uploaded_file.name.lower().endswith(".xml") else (None, None)
        records.append(
            {
                "file": _unique_name(uploaded_file.name, used_names),
                "upload_index": i,
                "source_url": "uploaded",
                "title": title,
                "author": author,
            }
        )
    return records


def _materialize_uploaded_records(uploaded_files, selected_records: list[dict], suffix_dir_name: str) -> Path | None:
    if not uploaded_files or not selected_records:
        return None

    target_dir = Path(tempfile.mkdtemp(prefix=f"didaskalos_{suffix_dir_name}_"))
    for item in selected_records:
        uploaded_file = uploaded_files[item["upload_index"]]
        (target_dir / item["file"]).write_bytes(uploaded_file.getbuffer())
    return target_dir


def _build_treebank_display_table(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df[["file", "title", "author", "source_url"]]


def _format_treebank_option(file_name: str, records_df: pd.DataFrame) -> str:
    if records_df.empty or "file" not in records_df.columns:
        return file_name

    match = records_df[records_df["file"].eq(file_name)]
    if match.empty:
        return file_name

    row = match.iloc[0]
    title = row.get("title") or "Untitled"
    author = row.get("author") or "Unknown author"
    return f"{file_name} | {title} | {author}"


with st.sidebar:
    if LOGO_IMAGE_PATH.exists():
        st.image(str(LOGO_IMAGE_PATH), use_container_width=True)

    st.header("Inputs")

    textbook_type = st.radio(
        "Textbook type",
        options=["Case-based (original)", "Declension-based (new)"],
        index=0,
        help=(
            "Case-based arranges noun/adjective lessons by case (one lesson for accusative, one for genitive, ...). "
            "Declension-based first classifies every noun and adjective in the treebanks into declension classes "
            "(N1-N8, ADJ1-ADJ3) and arranges the lessons by how frequent each class is in the corpus."
        ),
    )
    syllabus_mode = "declension" if textbook_type.startswith("Declension") else "case"

    input_mode = st.radio(
        "Input source",
        options=["Use GitHub repo URLs", "Upload files"],
        index=0,
        help="Use the repository defaults or upload files directly.",
    )

    lesson_count = int(
        st.number_input(
            "Lesson count",
            min_value=1,
            max_value=200,
            value=35,
            step=1,
        )
    )

    if input_mode == "Use GitHub repo URLs":
        default_treebank_urls = load_github_tree_urls(TREEBANK_PREFIX)
        default_lesson_urls = _resolve_default_lesson_urls(syllabus_mode)

        treebank_url_input = st.text_area(
            "Treebank XML URL(s)",
            value="\n".join(default_treebank_urls),
            height=220,
            help="One URL per line or comma-separated. Defaults are loaded from Didaskalos/treebanks/perseus.",
        )

        treebank_records = _build_records_from_urls(_parse_list_input(treebank_url_input), extract_xml_metadata=True)
        lesson_records = _build_records_from_urls(default_lesson_urls)
        uploaded_treebanks = []
    else:
        uploaded_treebanks = st.file_uploader(
            "Upload treebank XML files",
            type=["xml"],
            accept_multiple_files=True,
            help="Upload one or more XML files.",
        )
        default_lesson_urls = _resolve_default_lesson_urls(syllabus_mode)
        treebank_records = _build_records_from_uploads(uploaded_treebanks)
        lesson_records = _build_records_from_urls(default_lesson_urls)


available_treebanks = _build_treebank_display_table(treebank_records)
available_lessons = pd.DataFrame(lesson_records)

if available_treebanks.empty:
    st.warning("No treebanks available. Provide treebank URLs or upload XML files.")
    st.stop()

st.subheader("Available Treebanks")
st.dataframe(available_treebanks, use_container_width=True, height=240)

selected_treebank_files = st.multiselect(
    "Select treebank files",
    options=available_treebanks["file"].tolist(),
    default=[],
    format_func=lambda file_name: _format_treebank_option(file_name, available_treebanks),
)

if available_lessons.empty:
    st.warning("No lesson modules available from the default lesson set.")
    st.stop()

selected_lesson_files = available_lessons["file"].tolist()

build_clicked = st.button("Build Syllabus", type="primary", use_container_width=True)

if build_clicked:
    if not selected_treebank_files:
        st.warning("Select at least one treebank file before building.")
        st.stop()

    selected_treebank_records = [row for row in treebank_records if row["file"] in selected_treebank_files]
    selected_lesson_records = [row for row in lesson_records if row["file"] in selected_lesson_files]

    with st.spinner("Preparing selected treebanks and lesson modules..."):
        if input_mode == "Use GitHub repo URLs":
            treebank_dir, selected_treebank_records = _download_url_records_to_dir(selected_treebank_records, "treebanks")
        else:
            treebank_dir = _materialize_uploaded_records(uploaded_treebanks, selected_treebank_records, "treebanks")

        lesson_dir, selected_lesson_records = _download_url_records_to_dir(selected_lesson_records, "lessons")

        if treebank_dir is None:
            st.error("Could not prepare the selected treebank files.")
            st.stop()
        if lesson_dir is None:
            st.error("Could not prepare the selected lesson modules.")
            st.stop()

    with st.spinner("Parsing treebanks and building the syllabus..."):
        combined_df = build_combined_df(treebank_dir, selected_treebank_files, syllabus_mode=syllabus_mode)
        frequency_syllabus = build_frequency_syllabus(combined_df)
        textbook_markdown = generate_textbook_markdown(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=lesson_dir,
            lesson_count=lesson_count,
            combined_df=combined_df,
            syllabus_mode=syllabus_mode,
        )
        textbook_html = generate_textbook_html(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=lesson_dir,
            lesson_count=lesson_count,
            combined_df=combined_df,
            syllabus_mode=syllabus_mode,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected treebanks", len(selected_treebank_files))
    c2.metric("Token rows", int(len(combined_df)))
    c3.metric("Frequency rows", int(len(frequency_syllabus)))

    st.subheader("Frequency Syllabus")
    st.dataframe(frequency_syllabus, use_container_width=True, height=420)

    if syllabus_mode == "declension":
        declension_summary = build_declension_summary(combined_df)
        st.subheader("Declension Categories")
        if declension_summary.empty:
            st.info("No nouns or adjectives could be classified in the selected treebanks.")
        else:
            st.dataframe(declension_summary, use_container_width=True, height=420)
            st.download_button(
                label="Download declension_summary.csv",
                data=declension_summary.to_csv(index=False).encode("utf-8"),
                file_name="declension_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.download_button(
        label="Download frequency_syllabus.csv",
        data=frequency_syllabus.to_csv(index=False).encode("utf-8"),
        file_name="frequency_syllabus.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label="Download combined_treebank_rows.csv",
        data=combined_df.to_csv(index=False).encode("utf-8"),
        file_name="combined_treebank_rows.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label="Download textbook.md",
        data=textbook_markdown.encode("utf-8"),
        file_name="textbook.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.download_button(
        label="Download textbook.html",
        data=textbook_html.encode("utf-8"),
        file_name="textbook.html",
        mime="text/html",
        use_container_width=True,
    )

    st.subheader("Textbook Markdown Preview")
    st.code(textbook_markdown[:6000], language="markdown")

    st.subheader("Textbook HTML Preview")
    components.html(textbook_html, height=800, scrolling=True)

st.markdown("---")
st.caption(
    "Didaskalos is an open-source project. Contributions and feedback are welcome! "
    "Visit the GitHub repository for more information."
)
