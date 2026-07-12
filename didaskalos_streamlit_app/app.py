from __future__ import annotations

import base64
import hashlib
import os
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
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
from i18n import AVAILABLE_LANGS, DEFAULT_LANG, LANG_NAMES, is_rtl, rtl_css, t


APP_DIR = Path(__file__).resolve().parent
FAVICON_PATH = APP_DIR / "assets" / "logo.png"

st.set_page_config(
    page_title="Didaskalos",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "DB",
    layout="wide",
)

# Active UI language. The selector widget (rendered in the sidebar) owns the
# "lang" session key; reading it here makes the choice available before the
# title is drawn, so the whole page renders in the selected language.
lang = st.session_state.get("lang", DEFAULT_LANG)
if is_rtl(lang):
    st.markdown(rtl_css(), unsafe_allow_html=True)

HEADER_IMAGE_PATH = APP_DIR / "assets" / "electroplato.png"
LOGO_IMAGE_PATHS = {
    "fa": APP_DIR / "assets" / "greek-d.png",
    "en": APP_DIR / "assets" / "english-d.png",
}
LOGO_IMAGE_PATH = LOGO_IMAGE_PATHS.get(lang, LOGO_IMAGE_PATHS[DEFAULT_LANG])
header_image_html = ""
if HEADER_IMAGE_PATH.exists():
    encoded_image = base64.b64encode(HEADER_IMAGE_PATH.read_bytes()).decode("ascii")
    header_image_html = (
        f'<img src="data:image/png;base64,{encoded_image}" '
        'style="float: right; width: 34%; max-width: 360px; min-width: 200px; margin: 0 0 0.9rem 1.1rem; border-radius: 8px;" '
        'alt="Didaskalos header image" />'
    )

st.title(t("app_title", lang))
st.markdown(
    f"""
    <p style="font-size: 1.2rem; font-weight: 600; margin-top: -0.3rem; margin-bottom: 1rem;">
        {t("subtitle", lang)}
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    t("intro_html", lang).replace("{header_image}", header_image_html),
    unsafe_allow_html=True,
)

GITHUB_OWNER = "farnoosh-shamsian"
GITHUB_REPO = "didaskalos"
GITHUB_BRANCH = "main"
GITHUB_TREE_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}"
TREEBANK_PREFIX = "treebanks/perseus/"
FETCH_TIMEOUT_SECONDS = 20
FETCH_MAX_WORKERS = 8
# Title/author live in the XML <header>, well within the first few KB even for
# the largest treebanks (Iliad ~20 MB), so a bounded range read is enough to
# populate the selector table without downloading whole files.
METADATA_HEADER_BYTES = 65536
LESSON_PREFIX = "lessons-no-decl/"
DECLENSION_LESSON_PREFIX = "lessons-decl/"
# Per-language lesson folders. Files keep the same names as the English
# originals so the pipeline's filename-based lesson lookup works unchanged; a
# translated file shadows its English counterpart, missing ones fall back.
LOCALIZED_LESSON_PREFIXES = {
    "fa": {
        LESSON_PREFIX: "lessons-no-decl-fa/",
        DECLENSION_LESSON_PREFIX: "lessons-decl-fa/",
    },
}
LESSON_PREFIXES = (LESSON_PREFIX, DECLENSION_LESSON_PREFIX) + tuple(
    localized_prefix
    for mapping in LOCALIZED_LESSON_PREFIXES.values()
    for localized_prefix in mapping.values()
)
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


def _extract_xml_metadata_from_header(header_bytes: bytes) -> tuple[str | None, str | None]:
    # A range read gives us a truncated (unclosed) document, so ET.fromstring
    # would raise. Feed the partial bytes to a pull parser instead and read the
    # end events for <title>/<author>; both close inside the header long before
    # the body, so they are fully available even from a small leading slice.
    parser = ET.XMLPullParser(events=("end",))
    title: str | None = None
    author: str | None = None
    try:
        parser.feed(header_bytes)
        for _event, element in parser.read_events():
            tag = element.tag
            local_tag = tag.rsplit("}", 1)[-1] if isinstance(tag, str) else tag
            if local_tag == "title" and title is None:
                title = (" ".join(element.itertext()).strip()) or None
            elif local_tag == "author" and author is None:
                author = (" ".join(element.itertext()).strip()) or None
            if title is not None and author is not None:
                break
    except ET.ParseError:
        pass
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


# The bytes cache holds every treebank/lesson fetched from GitHub raw for the
# lifetime of the server process, so a rerun (Streamlit re-executes the whole
# script on each widget interaction) never re-downloads a file. Failures raise
# instead of returning None so st.cache_data does not memoize transient errors.
@st.cache_data(show_spinner=False, max_entries=256)
def _fetch_url_bytes(url: str) -> bytes:
    source_url = _normalize_url(url)
    try:
        request = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError, ValueError):
        local_payload = _read_from_local_repo_if_available(source_url)
        if local_payload is not None:
            return local_payload
        raise


# Only the leading header bytes are needed for title/author, so this issues an
# HTTP Range request (raw.githubusercontent.com honors it with 206). If a server
# ignores Range and sends the whole file, read() still caps at max_bytes. Cached
# separately from _fetch_url_bytes so the tiny header slices never evict, and are
# never confused with, the full payloads used at Build time.
@st.cache_data(show_spinner=False, max_entries=256)
def _fetch_url_header_bytes(url: str, max_bytes: int = METADATA_HEADER_BYTES) -> bytes:
    source_url = _normalize_url(url)
    try:
        request = Request(
            source_url,
            headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes=0-{max_bytes - 1}"},
        )
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            return response.read(max_bytes)
    except (HTTPError, URLError, TimeoutError, ValueError):
        local_payload = _read_from_local_repo_if_available(source_url)
        if local_payload is not None:
            return local_payload[:max_bytes]
        raise


@st.cache_data(show_spinner=False)
def _fetch_xml_metadata(url: str) -> tuple[str | None, str | None]:
    return _extract_xml_metadata_from_header(_fetch_url_header_bytes(url))


def _prefetch_xml_metadata(urls: list[str]) -> dict[str, tuple[str | None, str | None]]:
    def fetch(url: str) -> tuple[str | None, str | None]:
        try:
            return _fetch_xml_metadata(url)
        except Exception:
            return None, None

    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        return dict(zip(urls, executor.map(fetch, urls)))


def _warm_url_cache(urls: list[str]) -> None:
    def fetch(url: str) -> None:
        try:
            _fetch_url_bytes(url)
        except Exception:
            pass

    if not urls:
        return
    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        list(executor.map(fetch, urls))


def _download_url_records_to_dir(records: list[dict], suffix_dir_name: str) -> tuple[Path | None, list[dict]]:
    if not records:
        return None, []

    target_dir = Path(tempfile.mkdtemp(prefix=f"didaskalos_{suffix_dir_name}_"))
    enriched_records: list[dict] = []
    failed_records: list[dict] = []

    _warm_url_cache([item["source_url"] for item in records])
    for item in records:
        try:
            payload = _fetch_url_bytes(item["source_url"])
        except Exception:
            failed_records.append(item)
            continue
        (target_dir / item["file"]).write_bytes(payload)
        title, author = _extract_xml_metadata(payload)
        enriched_records.append({**item, "title": title, "author": author})

    if not enriched_records:
        return None, []

    return target_dir, enriched_records


@st.cache_data(show_spinner=False)
def load_github_tree_urls(prefix: str) -> list[str]:
    request = Request(GITHUB_TREE_API, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
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


# Memoized on the URL set so a Streamlit rerun (any widget interaction) reuses
# the built records instead of re-orchestrating the parallel metadata prefetch.
@st.cache_data(show_spinner=False)
def _build_records_from_urls(urls: list[str], extract_xml_metadata: bool = False) -> list[dict]:
    used_names = set()
    records = []
    metadata_by_url = _prefetch_xml_metadata(urls) if extract_xml_metadata else {}
    for i, url in enumerate(urls, start=1):
        parsed = urlparse(url)
        file_name = Path(parsed.path).name or f"file_{i}"
        title, author = metadata_by_url.get(url, (None, None))
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


def _dedupe_lesson_urls_by_filename(urls: list[str]) -> list[str]:
    # First URL wins per filename, so a localized lesson listed earlier shadows
    # the English file of the same name and only one copy gets downloaded.
    seen_names: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        name = Path(urlparse(url).path).name
        if name not in seen_names:
            seen_names.add(name)
            deduped.append(url)
    return deduped


def _resolve_default_lesson_urls(syllabus_mode: str = "case", lang: str = DEFAULT_LANG) -> list[str]:
    # In declension mode, list the declension lesson modules first so they keep
    # their canonical filenames if a module exists in both folders.
    base_prefixes = [DECLENSION_LESSON_PREFIX, LESSON_PREFIX] if syllabus_mode == "declension" else [LESSON_PREFIX]

    # Localized folders come first so translated modules shadow the English
    # originals; untranslated modules still resolve via the English folders.
    localized = LOCALIZED_LESSON_PREFIXES.get(lang, {})
    prefixes = [localized[prefix] for prefix in base_prefixes if prefix in localized] + base_prefixes

    merged: list[str] = []
    for prefix in prefixes:
        # Merge remote and local so transient GitHub API gaps do not hide available lessons.
        merged.extend(load_github_tree_urls(prefix))
        merged.extend(_list_local_lesson_urls(prefix))
    return _dedupe_lesson_urls_by_filename(_ensure_starter_lesson_urls(merged))


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


# st.fragment (stable in 1.37, experimental in 1.33) lets the treebank grid rerun
# on its own. Resolve it defensively so an older Streamlit degrades to running the
# selector inline instead of crashing on a missing attribute.
st_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if st_fragment is None:
    def st_fragment(func):
        return func


@st_fragment
def render_treebank_selector(available_treebanks: pd.DataFrame, lang: str) -> None:
    # Isolated in a fragment so ticking a checkbox reruns only this grid, not the
    # whole script (which would re-run the sidebar's metadata prefetch). The
    # current selection is published to session state for the Build step to read.
    st.subheader(t("available_treebanks_header", lang))

    # Select all / Clear work by bumping the editor's widget key: st.data_editor
    # edits persist per key and cannot be mutated after instantiation, so a fresh
    # key drops stale edits and the base dataframe's "selected" default renders.
    if "treebank_editor_version" not in st.session_state:
        st.session_state["treebank_editor_version"] = 0
    if "treebank_select_default" not in st.session_state:
        st.session_state["treebank_select_default"] = False

    btn_all, btn_clear = st.columns(2)
    if btn_all.button(t("select_all_button", lang), key="treebank_select_all_btn", use_container_width=True):
        st.session_state["treebank_editor_version"] += 1
        st.session_state["treebank_select_default"] = True
    if btn_clear.button(t("clear_selection_button", lang), key="treebank_clear_btn", use_container_width=True):
        st.session_state["treebank_editor_version"] += 1
        st.session_state["treebank_select_default"] = False

    editor_df = available_treebanks.copy()
    editor_df.insert(0, "selected", st.session_state["treebank_select_default"])
    editor_df["title"] = editor_df["title"].fillna(t("untitled", lang))
    editor_df["author"] = editor_df["author"].fillna(t("unknown_author", lang))

    # Editor edits are stored by row position under the widget key; fingerprint
    # the row set into the key so a changed URL/upload list can never re-apply
    # stale checkmarks to shifted rows.
    files_fingerprint = hashlib.md5("\n".join(editor_df["file"]).encode("utf-8")).hexdigest()[:8]
    editor_key = f"treebank_editor_{st.session_state['treebank_editor_version']}_{files_fingerprint}"

    edited_treebanks = st.data_editor(
        editor_df,
        key=editor_key,
        hide_index=True,
        use_container_width=True,
        height=300,
        num_rows="fixed",
        disabled=["file", "title", "author", "source_url"],
        column_config={
            "selected": st.column_config.CheckboxColumn(t("select_column_label", lang), default=False),
            "file": st.column_config.TextColumn(t("treebank_col_file", lang)),
            "title": st.column_config.TextColumn(t("treebank_col_title", lang)),
            "author": st.column_config.TextColumn(t("treebank_col_author", lang)),
            "source_url": st.column_config.TextColumn(t("treebank_col_source", lang)),
        },
    )

    st.session_state["selected_treebank_files"] = edited_treebanks.loc[
        edited_treebanks["selected"].fillna(False), "file"
    ].tolist()


with st.sidebar:
    if LOGO_IMAGE_PATH.exists():
        st.image(str(LOGO_IMAGE_PATH), use_container_width=True)

    st.selectbox(
        t("language_label", lang),
        options=AVAILABLE_LANGS,
        format_func=lambda code: LANG_NAMES[code],
        key="lang",
    )

    st.header(t("sidebar_inputs", lang))

    # Stable option keys keep the branching logic language-independent; only the
    # displayed labels are translated via format_func.
    input_mode = st.radio(
        t("input_source_label", lang),
        options=["github", "upload"],
        index=0,
        format_func=lambda code: t(f"input_source_opt_{code}", lang),
        help=t("input_source_help", lang),
    )

    lesson_count = int(
        st.number_input(
            t("lesson_count_label", lang),
            min_value=1,
            max_value=200,
            value=35,
            step=1,
        )
    )

    syllabus_mode = st.radio(
        t("textbook_type_label", lang),
        options=["declension","case"],
        index=0,
        format_func=lambda code: t(f"textbook_type_opt_{code}", lang),
        help=t("textbook_type_help", lang),
    )

    if input_mode == "github":
        default_treebank_urls = load_github_tree_urls(TREEBANK_PREFIX)
        default_lesson_urls = _resolve_default_lesson_urls(syllabus_mode, lang)

        with st.expander(t("custom_treebank_urls_expander", lang), expanded=False):
            custom_treebank_url_input = st.text_area(
                t("custom_treebank_urls_label", lang),
                value="",
                height=120,
                help=t("custom_treebank_urls_help", lang),
                key="custom_treebank_urls",
            )

        # Defaults first, custom URLs appended; _parse_list_input handles
        # newline/comma splitting and order-preserving dedupe across both.
        treebank_urls = _parse_list_input(
            "\n".join(default_treebank_urls) + "\n" + custom_treebank_url_input
        )
        treebank_records = _build_records_from_urls(treebank_urls, extract_xml_metadata=True)
        lesson_records = _build_records_from_urls(default_lesson_urls)
        uploaded_treebanks = []
    else:
        uploaded_treebanks = st.file_uploader(
            t("upload_label", lang),
            type=["xml"],
            accept_multiple_files=True,
            help=t("upload_help", lang),
        )
        default_lesson_urls = _resolve_default_lesson_urls(syllabus_mode, lang)
        treebank_records = _build_records_from_uploads(uploaded_treebanks)
        lesson_records = _build_records_from_urls(default_lesson_urls)


available_treebanks = _build_treebank_display_table(treebank_records)
available_lessons = pd.DataFrame(lesson_records)

if available_treebanks.empty:
    st.warning(t("no_treebanks_warning", lang))
    st.stop()

render_treebank_selector(available_treebanks, lang)
selected_treebank_files = st.session_state.get("selected_treebank_files", [])

if available_lessons.empty:
    st.warning(t("no_lessons_warning", lang))
    st.stop()

selected_lesson_files = available_lessons["file"].tolist()

build_clicked = st.button(t("build_button", lang), type="primary", use_container_width=True)

if build_clicked:
    if not selected_treebank_files:
        st.warning(t("select_at_least_one_warning", lang))
        st.stop()

    selected_treebank_records = [row for row in treebank_records if row["file"] in selected_treebank_files]
    selected_lesson_records = [row for row in lesson_records if row["file"] in selected_lesson_files]

    with st.spinner(t("spinner_preparing", lang)):
        if input_mode == "github":
            treebank_dir, selected_treebank_records = _download_url_records_to_dir(selected_treebank_records, "treebanks")
        else:
            treebank_dir = _materialize_uploaded_records(uploaded_treebanks, selected_treebank_records, "treebanks")

        lesson_dir, selected_lesson_records = _download_url_records_to_dir(selected_lesson_records, "lessons")

        if treebank_dir is None:
            st.error(t("error_prepare_treebanks", lang))
            st.stop()
        if lesson_dir is None:
            st.error(t("error_prepare_lessons", lang))
            st.stop()

    with st.spinner(t("spinner_parsing", lang)):
        combined_df = build_combined_df(treebank_dir, selected_treebank_files, syllabus_mode=syllabus_mode)
        frequency_syllabus = build_frequency_syllabus(combined_df)
        textbook_markdown = generate_textbook_markdown(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=lesson_dir,
            lesson_count=lesson_count,
            combined_df=combined_df,
            syllabus_mode=syllabus_mode,
            lang=lang,
        )
        textbook_html = generate_textbook_html(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=lesson_dir,
            lesson_count=lesson_count,
            combined_df=combined_df,
            syllabus_mode=syllabus_mode,
            lang=lang,
            markdown_content=textbook_markdown,
        )

    declension_summary = build_declension_summary(combined_df) if syllabus_mode == "declension" else None

    # Results live in session state so later reruns (downloads, toggles, widget
    # changes) keep them on screen without rebuilding. CSV bytes are computed
    # once here instead of holding the full token DataFrame per session.
    st.session_state["build_result"] = {
        "treebank_count": len(selected_treebank_files),
        "token_rows": int(len(combined_df)),
        "frequency_rows": int(len(frequency_syllabus)),
        "frequency_syllabus": frequency_syllabus,
        "frequency_csv": frequency_syllabus.to_csv(index=False).encode("utf-8"),
        "combined_csv": combined_df.to_csv(index=False).encode("utf-8"),
        "declension_csv": (
            declension_summary.to_csv(index=False).encode("utf-8")
            if declension_summary is not None and not declension_summary.empty
            else None
        ),
        "textbook_markdown": textbook_markdown,
        "textbook_html": textbook_html,
    }

build_result = st.session_state.get("build_result")
if build_result:
    c1, c2, c3 = st.columns(3)
    c1.metric(t("metric_selected_treebanks", lang), build_result["treebank_count"])
    c2.metric(t("metric_token_rows", lang), build_result["token_rows"])
    c3.metric(t("metric_frequency_rows", lang), build_result["frequency_rows"])

    st.subheader(t("frequency_syllabus_header", lang))
    st.dataframe(build_result["frequency_syllabus"], use_container_width=True, height=420)

    if build_result["declension_csv"] is not None:
        st.download_button(
            label=t("download_declension_summary", lang),
            data=build_result["declension_csv"],
            file_name="declension_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.download_button(
        label=t("download_frequency_syllabus", lang),
        data=build_result["frequency_csv"],
        file_name="frequency_syllabus.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label=t("download_combined_rows", lang),
        data=build_result["combined_csv"],
        file_name="combined_treebank_rows.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label=t("download_textbook_md", lang),
        data=build_result["textbook_markdown"].encode("utf-8"),
        file_name="textbook.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.download_button(
        label=t("download_textbook_html", lang),
        data=build_result["textbook_html"].encode("utf-8"),
        file_name="textbook.html",
        mime="text/html",
        use_container_width=True,
    )

    st.subheader(t("textbook_md_preview_header", lang))
    st.code(build_result["textbook_markdown"][:6000], language="markdown")

    st.subheader(t("textbook_html_preview_header", lang))
    # The full textbook HTML can be several MB; only push it to the browser
    # when the user asks for it.
    if st.toggle(t("show_html_preview_label", lang), value=False, key="show_html_preview"):
        components.html(build_result["textbook_html"], height=800, scrolling=True)

st.markdown("---")
st.caption(t("footer_caption", lang))
