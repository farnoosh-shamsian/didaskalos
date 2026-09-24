import json
import xml.etree.ElementTree as ET
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
REGISTRY_PATH = REPO_ROOT / "treebanks" / "registry.json"
# Written beside the app so the Docker build context carries it into the image: discovery then needs no network at all, which is the only thing GitHub's rate limiting cannot take away.
MANIFEST_PATH = APP_DIR / "content_manifest.json"
LESSON_PREFIXES = ("lessons/en/", "lessons/fa/")
# Same slice the app used to range-request per file; enough for title/author and the first <sentence> start tag, which carries the document_id.
HEADER_BYTES = 65536


def _glob_suffix(file_glob: str) -> str:
    if file_glob and file_glob.startswith("*."):
        return file_glob[1:].lower()
    return ".xml"


def _files_under(prefix: str, suffix: str) -> list[str]:
    folder = REPO_ROOT / prefix
    if not folder.is_dir():
        return []
    return sorted(
        path.name
        for path in folder.iterdir()
        if path.is_file() and path.name.lower().endswith(suffix)
    )


def _header_metadata(path: Path) -> tuple[str | None, str | None, str | None]:
    parser = ET.XMLPullParser(events=("start", "end"))
    title: str | None = None
    author: str | None = None
    document_id: str | None = None
    try:
        parser.feed(path.read_bytes()[:HEADER_BYTES])
        for event, element in parser.read_events():
            tag = element.tag
            local_tag = tag.rsplit("}", 1)[-1] if isinstance(tag, str) else tag
            if event == "start":
                if local_tag == "sentence":
                    document_id = element.get("document_id") or None
                    break
                continue
            if local_tag == "title" and title is None:
                title = (" ".join(element.itertext()).strip()) or None
            elif local_tag == "author" and author is None:
                author = (" ".join(element.itertext()).strip()) or None
    except (ET.ParseError, OSError):
        pass
    return title, author, document_id


def build_manifest() -> dict:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    treebanks = {}
    for corpus in registry.get("corpora", []):
        prefix = corpus.get("path", "")
        if not prefix:
            continue
        suffix = _glob_suffix(corpus.get("file_glob", "*.xml"))
        entries = []
        for name in _files_under(prefix, suffix):
            # Picker labels come from the manifest, so the app needs no per-file fetch; only XML carries a header to read them from.
            title, author, document_id = (
                _header_metadata(REPO_ROOT / prefix / name) if name.lower().endswith(".xml")
                else (None, None, None)
            )
            entries.append(
                {"file": name, "title": title, "author": author, "document_id": document_id}
            )
        treebanks[prefix] = entries

    lessons = {prefix: _files_under(prefix, ".md") for prefix in LESSON_PREFIXES}
    return {"treebanks": treebanks, "lessons": lessons}


if __name__ == "__main__":
    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for prefix, entries in manifest["treebanks"].items():
        missing = sum(1 for entry in entries if not entry["document_id"])
        print(f"treebanks: {prefix} -> {len(entries)} files ({missing} without document_id)")
    for prefix, files in manifest["lessons"].items():
        print(f"lessons: {prefix} -> {len(files)} files")
    print(f"wrote {MANIFEST_PATH}")
