"""Human-friendly author/work names for the treebank picker.

The treebank files are named by their TLG/CTS reference (e.g.
``tlg0012.tlg001.perseus-grc1.tb.xml``). The XML ``<title>`` inside each file is
a bibliographic *edition* string ("Homeri Opera in five volumes."), not a clean
work name, and several works by one author share an identical title string
(e.g. four Lysias speeches). So we keep a small curated lookup from the TLG
``author.work`` id to a clean ``(author, work)`` pair, and fall back to a cleaned
XML title for anything not in the map (custom URLs / uploaded files).

To add a work, drop in a new ``"tlgAUTHOR.tlgWORK": ("Author", "Work")`` row.
For example, once Xenophon's treebanks are added to the corpus::

    "tlg0032.tlg006": ("Xenophon", "Hellenica"),
    "tlg0032.tlg007": ("Xenophon", "Anabasis"),

All names below were verified against the Perseus catalog
(https://catalog.perseus.org/).
"""
from __future__ import annotations

import re

# Keyed on the TLG "author.work" id (the first two dotted parts of the filename).
WORK_CATALOG: dict[str, tuple[str, str]] = {
    "tlg0003.tlg001": ("Thucydides", "The Peloponnesian War"),
    "tlg0007.tlg004": ("Plutarch", "Life of Lycurgus"),
    "tlg0007.tlg015": ("Plutarch", "Life of Alcibiades"),
    "tlg0008.tlg001": ("Athenaeus", "The Deipnosophists"),
    "tlg0011.tlg001": ("Sophocles", "Women of Trachis"),
    "tlg0011.tlg002": ("Sophocles", "Antigone"),
    "tlg0011.tlg003": ("Sophocles", "Ajax"),
    "tlg0011.tlg004": ("Sophocles", "Oedipus Tyrannus"),
    "tlg0011.tlg005": ("Sophocles", "Electra"),
    "tlg0012.tlg001": ("Homer", "Iliad"),
    "tlg0012.tlg002": ("Homer", "Odyssey"),
    "tlg0013.tlg002": ("Homeric Hymns", "Hymn to Demeter"),
    "tlg0016.tlg001": ("Herodotus", "Histories"),
    "tlg0020.tlg001": ("Hesiod", "Theogony"),
    "tlg0020.tlg002": ("Hesiod", "Works and Days"),
    "tlg0020.tlg003": ("Hesiod", "Shield of Heracles"),
    "tlg0059.tlg001": ("Plato", "Euthyphro"),
    "tlg0060.tlg001": ("Diodorus Siculus", "Library of History"),
    "tlg0085.tlg001": ("Aeschylus", "Suppliant Women"),
    "tlg0085.tlg002": ("Aeschylus", "Persians"),
    "tlg0085.tlg003": ("Aeschylus", "Prometheus Bound"),
    "tlg0085.tlg004": ("Aeschylus", "Seven Against Thebes"),
    "tlg0085.tlg005": ("Aeschylus", "Agamemnon"),
    "tlg0085.tlg006": ("Aeschylus", "Libation Bearers"),
    "tlg0085.tlg007": ("Aeschylus", "Eumenides"),
    "tlg0096.tlg002": ("Aesop", "Fables"),
    "tlg0540.tlg001": ("Lysias", "On the Murder of Eratosthenes"),
    "tlg0540.tlg014": ("Lysias", "Against Alcibiades 1"),
    "tlg0540.tlg015": ("Lysias", "Against Alcibiades 2"),
    "tlg0540.tlg023": ("Lysias", "Against Pancleon"),
    "tlg0543.tlg001": ("Polybius", "Histories"),
    "tlg0548.tlg001": ("Apollodorus", "The Library"),
    # --- Works added with the Gorman corpus (prose authors) ---
    "tlg0007.tlg086": ("Plutarch", "On the Fortune of the Romans"),
    "tlg0007.tlg087": ("Plutarch", "On the Fortune or Virtue of Alexander"),
    "tlg0014.tlg001": ("Demosthenes", "Olynthiac 1"),
    "tlg0014.tlg004": ("Demosthenes", "First Philippic"),
    "tlg0014.tlg018": ("Demosthenes", "On the Crown"),
    "tlg0014.tlg046": ("Demosthenes", "Against Stephanus 2"),
    "tlg0014.tlg047": ("Demosthenes", "Against Evergus and Mnesibulus"),
    "tlg0014.tlg049": ("Demosthenes", "Against Timotheus"),
    "tlg0014.tlg050": ("Demosthenes", "Against Polycles"),
    "tlg0014.tlg052": ("Demosthenes", "Against Callippus"),
    "tlg0014.tlg053": ("Demosthenes", "Against Nicostratus"),
    "tlg0014.tlg059": ("Demosthenes", "Against Neaera"),
    "tlg0026.tlg001": ("Aeschines", "Against Timarchus"),
    "tlg0028.tlg001": ("Antiphon", "Against the Stepmother for Poisoning"),
    "tlg0028.tlg002": ("Antiphon", "First Tetralogy"),
    "tlg0028.tlg005": ("Antiphon", "On the Murder of Herodes"),
    "tlg0028.tlg006": ("Antiphon", "On the Choreutes"),
    "tlg0032.tlg001": ("Xenophon", "Hellenica"),
    "tlg0032.tlg007": ("Xenophon", "Cyropaedia"),
    "tlg0032.tlg015": ("Xenophon", "Constitution of the Athenians"),
    "tlg0059.tlg002": ("Plato", "Apology"),
    "tlg0081.tlg001": ("Dionysius of Halicarnassus", "Roman Antiquities"),
    "tlg0086.tlg035": ("Aristotle", "Politics"),
    "tlg0526.tlg004": ("Josephus", "The Jewish War"),
    "tlg0540.tlg012": ("Lysias", "Against Eratosthenes"),
    "tlg0540.tlg013": ("Lysias", "Against Agoratus"),
    "tlg0540.tlg019": ("Lysias", "On the Property of Aristophanes"),
    "tlg0551.tlg017": ("Appian", "The Civil Wars"),
}


def _strip_extension(file_name: str) -> str:
    for suffix in (".tb.xml", ".xml", ".conllu", ".conll"):
        if file_name.lower().endswith(suffix):
            return file_name[: -len(suffix)]
    return file_name


def _tlg_key(file_name: str) -> str | None:
    """The ``tlgAUTHOR.tlgWORK`` id from a treebank filename, or None if it does
    not follow the TLG naming convention (custom URLs / arbitrary uploads)."""
    parts = _strip_extension(file_name).split(".")
    if len(parts) >= 2 and parts[0].startswith("tlg") and parts[1].startswith("tlg"):
        return f"{parts[0]}.{parts[1]}"
    return None


def _tlg_from_document_id(document_id: str | None) -> str | None:
    """Pull a ``tlgAUTHOR.tlgWORK`` id out of a CTS document_id / urn.

    The Gorman files are not TLG-named, but every sentence carries a
    ``document_id`` such as
    ``http://.../urn:cts:greekLit:tlg0540.tlg001.perseus-grc1`` — enough to
    identify the work and reuse the catalog below.
    """
    if not document_id:
        return None
    # Newer CTS urn form: "...urn:cts:greekLit:tlg0540.tlg001.perseus-grc1".
    match = re.search(r"(tlg\d+\.tlg\d+)", document_id)
    if match:
        return match.group(1)
    # Older Gorman form: "0014-046" (author-work) -> "tlg0014.tlg046".
    match = re.fullmatch(r"\s*(\d{1,4})-(\d{1,3})\s*", document_id)
    if match:
        return f"tlg{int(match.group(1)):04d}.tlg{int(match.group(2)):03d}"
    return None


def tlg_work_key(file_name: str, document_id: str | None = None) -> str | None:
    """The TLG ``author.work`` id for a file, from its name or its document_id.

    This is the key the picker groups on: every file that resolves to the same
    key is one *work* (so a work split across many passage files, like the
    Gorman texts, collapses into a single entry). ``None`` when neither source
    yields a TLG id (arbitrary uploads / custom URLs)."""
    return _tlg_key(file_name) or _tlg_from_document_id(document_id)


# Boilerplate fragments that clutter the raw XML <title> of texts not in the
# catalog. Cut the title at the first match, then trim leftover punctuation.
_TITLE_BOILERPLATE = re.compile(
    r"(,?\s*(with an English translation|with an English Translation|"
    r"ed\.|edited by|translated by|in (two|three|four|five|twelve) volumes)\b.*)$",
    re.IGNORECASE,
)


def _clean_title(xml_title: str | None) -> str | None:
    if not xml_title:
        return None
    cleaned = _TITLE_BOILERPLATE.sub("", xml_title).strip()
    cleaned = cleaned.strip(" .,:;-")
    return cleaned or xml_title.strip() or None


def resolve_author_work(
    file_name: str,
    xml_author: str | None,
    xml_title: str | None,
    document_id: str | None = None,
) -> tuple[str | None, str]:
    """Return ``(author, work)`` display names for a treebank file.

    Curated catalog entries win (matched on the TLG work key from the filename
    or the CTS ``document_id``); otherwise fall back to the (cleaned) XML author
    and title, and finally to the filename. ``author`` may be ``None`` (the
    caller buckets those under an "Unknown author" heading). No book/section
    suffix is added: files of one work are meant to collapse into a single
    whole-work entry.
    """
    entry = WORK_CATALOG.get(tlg_work_key(file_name, document_id) or "")
    if entry:
        return entry
    author = (xml_author or "").strip() or None
    work = _clean_title(xml_title) or _strip_extension(file_name)
    return author, work
