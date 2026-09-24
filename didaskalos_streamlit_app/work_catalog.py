# Author/work names for the treebank picker: files are named by TLG/CTS reference and the XML <title> inside is an edition string ("Homeri Opera in five volumes.") that several works by one author often share, so a curated lookup maps the TLG author.work id to a clean (author, work) pair and anything missing (custom URLs, uploads) falls back to a cleaned XML title; add a work with one "tlgAUTHOR.tlgWORK": ("Author", "Work") row, and note that the names below were checked against the Perseus catalog (https://catalog.perseus.org/).
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
    # Works added with the Gorman corpus (prose authors).
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
    # Lucian, added with the Harrington (Perseids) corpus.
    "tlg0062.tlg012": ("Lucian", "True Histories"),
    # New Testament books, added with the PROIEL corpus (TLG 0031).
    "tlg0031.tlg001": ("New Testament", "Gospel of Matthew"),
    "tlg0031.tlg002": ("New Testament", "Gospel of Mark"),
    "tlg0031.tlg003": ("New Testament", "Gospel of Luke"),
    "tlg0031.tlg004": ("New Testament", "Gospel of John"),
    "tlg0031.tlg005": ("New Testament", "Acts of the Apostles"),
    "tlg0031.tlg006": ("New Testament", "Romans"),
    "tlg0031.tlg007": ("New Testament", "1 Corinthians"),
    "tlg0031.tlg008": ("New Testament", "2 Corinthians"),
    "tlg0031.tlg009": ("New Testament", "Galatians"),
    "tlg0031.tlg010": ("New Testament", "Ephesians"),
    "tlg0031.tlg011": ("New Testament", "Philippians"),
    "tlg0031.tlg012": ("New Testament", "Colossians"),
    "tlg0031.tlg013": ("New Testament", "1 Thessalonians"),
    "tlg0031.tlg014": ("New Testament", "2 Thessalonians"),
    "tlg0031.tlg015": ("New Testament", "1 Timothy"),
    "tlg0031.tlg016": ("New Testament", "2 Timothy"),
    "tlg0031.tlg017": ("New Testament", "Titus"),
    "tlg0031.tlg018": ("New Testament", "Philemon"),
    "tlg0031.tlg019": ("New Testament", "Hebrews"),
    "tlg0031.tlg020": ("New Testament", "Epistle of James"),
    "tlg0031.tlg021": ("New Testament", "1 Peter"),
    "tlg0031.tlg025": ("New Testament", "3 John"),
    "tlg0031.tlg026": ("New Testament", "Jude"),
    "tlg0031.tlg027": ("New Testament", "Revelation"),
}


# Citation sigla, keyed as WORK_CATALOG is, each value being (siglum, leading_number): the LSJ-style abbreviation, and the fixed book/speech number that belongs to the canonical reference but is missing from the treebank subdoc (set for the orators only), which format_citation composes as "{siglum} {leading_number}.{subdoc}", dropping empty parts.
WORK_CITATION: dict[str, tuple[str, str | None]] = {
    "tlg0003.tlg001": ("Thuc.", None),
    "tlg0007.tlg004": ("Plut. Lyc.", None),
    "tlg0007.tlg015": ("Plut. Alc.", None),
    "tlg0007.tlg086": ("Plut. De fort. Rom.", None),
    "tlg0007.tlg087": ("Plut. De Alex. fort.", None),
    "tlg0008.tlg001": ("Ath.", None),
    "tlg0011.tlg001": ("Soph. Trach.", None),
    "tlg0011.tlg002": ("Soph. Ant.", None),
    "tlg0011.tlg003": ("Soph. Aj.", None),
    "tlg0011.tlg004": ("Soph. OT", None),
    "tlg0011.tlg005": ("Soph. El.", None),
    "tlg0012.tlg001": ("Hom. Il.", None),
    "tlg0012.tlg002": ("Hom. Od.", None),
    "tlg0013.tlg002": ("Hymn. Hom. Cer.", None),
    "tlg0016.tlg001": ("Hdt.", None),
    "tlg0020.tlg001": ("Hes. Theog.", None),
    "tlg0020.tlg002": ("Hes. Op.", None),
    "tlg0020.tlg003": ("Hes. Sc.", None),
    "tlg0059.tlg001": ("Pl. Euthphr.", None),
    "tlg0059.tlg002": ("Pl. Ap.", None),
    "tlg0060.tlg001": ("Diod. Sic.", None),
    "tlg0085.tlg001": ("Aesch. Supp.", None),
    "tlg0085.tlg002": ("Aesch. Pers.", None),
    "tlg0085.tlg003": ("Aesch. PV", None),
    "tlg0085.tlg004": ("Aesch. Sept.", None),
    "tlg0085.tlg005": ("Aesch. Ag.", None),
    "tlg0085.tlg006": ("Aesch. Cho.", None),
    "tlg0085.tlg007": ("Aesch. Eum.", None),
    "tlg0096.tlg002": ("Aesop", None),
    "tlg0540.tlg001": ("Lys.", "1"),
    "tlg0540.tlg012": ("Lys.", "12"),
    "tlg0540.tlg013": ("Lys.", "13"),
    "tlg0540.tlg014": ("Lys.", "14"),
    "tlg0540.tlg015": ("Lys.", "15"),
    "tlg0540.tlg019": ("Lys.", "19"),
    "tlg0540.tlg023": ("Lys.", "23"),
    "tlg0543.tlg001": ("Polyb.", None),
    "tlg0548.tlg001": ("Apollod.", None),
    "tlg0014.tlg001": ("Dem.", "1"),
    "tlg0014.tlg004": ("Dem.", "4"),
    "tlg0014.tlg018": ("Dem.", "18"),
    "tlg0014.tlg046": ("Dem.", "46"),
    "tlg0014.tlg047": ("Dem.", "47"),
    "tlg0014.tlg049": ("Dem.", "49"),
    "tlg0014.tlg050": ("Dem.", "50"),
    "tlg0014.tlg052": ("Dem.", "52"),
    "tlg0014.tlg053": ("Dem.", "53"),
    "tlg0014.tlg059": ("Dem.", "59"),
    "tlg0026.tlg001": ("Aeschin.", "1"),
    "tlg0028.tlg001": ("Antiph.", "1"),
    "tlg0028.tlg002": ("Antiph.", "2"),
    "tlg0028.tlg005": ("Antiph.", "5"),
    "tlg0028.tlg006": ("Antiph.", "6"),
    "tlg0032.tlg001": ("Xen. Hell.", None),
    "tlg0032.tlg007": ("Xen. Cyr.", None),
    "tlg0032.tlg015": ("Xen. Ath.", None),
    "tlg0081.tlg001": ("Dion. Hal. Ant. Rom.", None),
    "tlg0086.tlg035": ("Arist. Pol.", None),
    "tlg0526.tlg004": ("Joseph. BJ", None),
    "tlg0551.tlg017": ("App. BC", None),
    "tlg0062.tlg012": ("Luc. VH", None),
    "tlg0031.tlg001": ("Matt.", None),
    "tlg0031.tlg002": ("Mark", None),
    "tlg0031.tlg003": ("Luke", None),
    "tlg0031.tlg004": ("John", None),
    "tlg0031.tlg005": ("Acts", None),
    "tlg0031.tlg006": ("Rom.", None),
    "tlg0031.tlg007": ("1 Cor.", None),
    "tlg0031.tlg008": ("2 Cor.", None),
    "tlg0031.tlg009": ("Gal.", None),
    "tlg0031.tlg010": ("Eph.", None),
    "tlg0031.tlg011": ("Phil.", None),
    "tlg0031.tlg012": ("Col.", None),
    "tlg0031.tlg013": ("1 Thess.", None),
    "tlg0031.tlg014": ("2 Thess.", None),
    "tlg0031.tlg015": ("1 Tim.", None),
    "tlg0031.tlg016": ("2 Tim.", None),
    "tlg0031.tlg017": ("Titus", None),
    "tlg0031.tlg018": ("Phlm.", None),
    "tlg0031.tlg019": ("Heb.", None),
    "tlg0031.tlg020": ("Jas.", None),
    "tlg0031.tlg021": ("1 Pet.", None),
    "tlg0031.tlg025": ("3 John", None),
    "tlg0031.tlg026": ("Jude", None),
    "tlg0031.tlg027": ("Rev.", None),
}


def _strip_extension(file_name: str) -> str:
    for suffix in (".tb.xml", ".xml", ".conllu", ".conll"):
        if file_name.lower().endswith(suffix):
            return file_name[: -len(suffix)]
    return file_name


def _tlg_key(file_name: str) -> str | None:
    # The tlgAUTHOR.tlgWORK id from a filename, or None when it does not follow the TLG convention (custom URLs, arbitrary uploads).
    parts = _strip_extension(file_name).split(".")
    if len(parts) >= 2 and parts[0].startswith("tlg") and parts[1].startswith("tlg"):
        return f"{parts[0]}.{parts[1]}"
    return None


def _tlg_from_document_id(document_id: str | None) -> str | None:
    # The Gorman files are not TLG-named, but their CTS document_id identifies the work well enough to reuse the catalog; NaN (from a missing pandas category) is truthy, so check the type, since a float would reach re.search and raise.
    if not isinstance(document_id, str) or not document_id:
        return None
    # Newer CTS urn form: "...urn:cts:greekLit:tlg0540.tlg001.perseus-grc1".
    match = re.search(r"(tlg\d+\.tlg\d+)", document_id)
    if match:
        return match.group(1)
    # Older Gorman form: "0014-046" -> "tlg0014.tlg046".
    match = re.fullmatch(r"\s*(\d{1,4})-(\d{1,3})\s*", document_id)
    if match:
        return f"tlg{int(match.group(1)):04d}.tlg{int(match.group(2)):03d}"
    return None


def tlg_work_key(file_name: str, document_id: str | None = None) -> str | None:
    # The key the picker groups on: files sharing one are one work, so a work split across passage files collapses into a single entry.
    return _tlg_key(file_name) or _tlg_from_document_id(document_id)


# Boilerplate cluttering the XML <title> of texts not in the catalog; the title is cut at the first match.
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
    # Catalog entries win, then the cleaned XML author and title, then the filename; author may be None, which the caller buckets under "Unknown author", and there is no book/section suffix, so files of one work collapse into one entry.
    entry = WORK_CATALOG.get(tlg_work_key(file_name, document_id) or "")
    if entry:
        return entry
    author = (xml_author or "").strip() or None
    work = _clean_title(xml_title) or _strip_extension(file_name)
    return author, work


def _clean_subdoc(subdoc: str | None) -> str:
    # The usable passage reference, or "": subdoc is opaque (book.line, a Stephanus page, a bare section), so only whitespace and the empty / literal-"None" placeholders are rejected, and a non-string (None, or the NaN a missing pandas category yields) means no reference.
    if not isinstance(subdoc, str):
        return ""
    ref = subdoc.strip()
    return "" if ref.lower() == "none" else ref


def format_citation(
    file_name: str,
    document_id: str | None,
    subdoc: str | None,
) -> str:
    # A short citation: "Hdt. 1.1", "Hom. Il. 1.1-1.7", "Lys. 12.1"; a work with no siglum falls back to "Author, Work", with no passage reference the citation degrades to the work label, and "" only when even that is unknown.
    ref = _clean_subdoc(subdoc)
    entry = WORK_CITATION.get(tlg_work_key(file_name, document_id) or "")
    if entry:
        siglum, leading = entry
    else:
        author, work = resolve_author_work(file_name, None, None, document_id)
        siglum = f"{author}, {work}" if author else work
        leading = None
    reference = ".".join(part for part in (leading, ref) if part)
    return f"{siglum} {reference}".strip()
