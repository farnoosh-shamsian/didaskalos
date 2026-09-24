# About Didaskalos

Didaskalos is a corpus-driven system that builds an Ancient Greek textbook around the texts you actually want to read. Instead of relying on fixed curricula or generalized pedagogical sequences, Didaskalos builds each textbook directly from the texts a learner or instructor chooses. The aim is that the grammar you meet and the sentences you work on are the ones your own texts use. The explanations themselves are the same for every reader; what your corpus changes is which of them you get, and in what order.

At its core, Didaskalos operates by integrating three key resources: a corpus of selected Ancient Greek texts, linguistic treebanks, and a modular grammar framework. The process begins when a learner chooses a target corpus. The system then analyzes this corpus using treebank data, identifying vocabulary frequency and morphological patterns. These linguistic features are ranked by how often they occur in the selected texts.

Based on this analysis, Didaskalos constructs a frequency-driven learning path. Rather than following a predetermined order of topics, the system introduces vocabulary and grammar in descending order of how often each occurs in the chosen corpus, on the working assumption that this is the fastest route into those texts.

Each unit in the resulting textbook pairs a grammatical explanation, written to stand on its own wherever it can, with exercises drawn from the source material; where one lesson genuinely has to wait for another, the syllabus makes it wait. From the first frequency-driven lesson onward, every exercise sentence and every vocabulary word comes from the corpus you provided. Nothing is invented, reworded, or simplified, and every exercise sentence carries the citation of the text it was taken from. The seven fixed opening modules you are reading now run ahead of any corpus, so their illustrative examples are the standard ones of Greek grammar.

---

## How to Use This Book

**What a lesson looks like.** After these seven opening modules, every lesson has the same three parts: an explanation of one grammatical topic, ending with a section on how that topic behaves in other dialects and periods; a list of words to learn, drawn from your own corpus and ranked by how often they occur in it; and exercises built from real sentences, each with an answer key.

**The opening modules are long, and you do not have to master them before going on.** They are deliberately extensive. A reader working alone has no teacher on hand to say *that is normal, it happens all the time*, so these modules try to name the irregularities and oddities of Greek in advance rather than let them ambush you later. If that makes them feel overwhelming, read them through once for the shape of the thing and then go on to the lessons proper. Come back to them every so often, as you start meeting these phenomena in real sentences — they are written to be returned to, and they read very differently the second and third time.

**The vocabulary lists have no translations, and that is deliberate.** You look each word up yourself. Looking a word up and writing down what you find takes longer than reading a gloss, and the effort is the point: you will meet the word again, and the second meeting is what fixes it. This is a pedagogical choice rather than a proven one — glossed lists have their own evidence behind them — and it is made here partly because the corpus is yours, so no glossary written for someone else’s textbook would fit it. The sixth module teaches the skill in full — how a lexicon is organized, how to get from a form in a text back to the headword it belongs to, and how to record what you find. Every vocabulary entry in this book carries links to Logeion and to the Perseus word study tool, so looking up a listed word is a click. Words that turn up in the exercise sentences but not on the list are a harder job: there you have to get from the inflected form back to the headword yourself, which is exactly what that module is for.

**The sentences are real from the first lesson.** They have not been shortened or rewritten, which means that early on a sentence will contain words and constructions you have not met. This is normal and it is not a sign that you have fallen behind. Read for the thing the lesson is about: find the form under discussion, parse it, see what it is doing. Understanding the whole sentence is a bonus at first. It should get steadily more common as you go, because the frequency ordering means the words you learn are precisely the ones that keep coming back — though how fast that happens depends on the corpus you chose, and no one has yet measured it.

**Frequent is not the same as easy.** Because the syllabus follows frequency rather than a traditional teaching order, some genuinely difficult grammar arrives early — the μι-verbs ⟨mi-verbs⟩ and the participle are both awkward and both extremely common, so a Didaskalos book reaches them sooner than a conventional textbook would. Expect the difficulty curve to be lumpy rather than gentle. The compensation is that every topic earned its place by how often it occurs in *your* texts, so everything you learn matters for reading the texts you chose.

---

## The Greek This Book Teaches

Because the corpus is the learner's own, the Greek in a given textbook may span nine centuries and several dialects. The grammar modules therefore take **Attic** — the Greek of Classical Athens — as their reference point: it is the dialect around which grammars and dictionaries are organized, and the one Koine grew chiefly out of. Every paradigm is the Attic one unless the lesson says otherwise, and the grammar lessons close with a **Historical Development** section on how their forms behave in other dialects and periods. How much weight those sections carry depends entirely on the corpus chosen — background reading for an Attic prose selection, essential grammar for one built on Homer or Herodotus. The dialects module, the last of the fixed opening lessons, sets this out in full.

## A Note on Transliteration

In this opening group of modules — the alphabet, the dictionary module, the introductions to nouns, adjectives and verbs, and the dialects module — every Greek word and example is followed by its transliteration in Latin letters. This is a temporary scaffold, there to help you internalize the alphabet and check your reading while the letters are still new. It stops after these lessons: from the first frequency-driven lesson onward the Greek stands on its own, because by then you should be reading the Greek script directly rather than through a Latin echo of it.

The transliteration is written in **angle brackets**: ὕδωρ ⟨hýdōr⟩ "water". The brackets are there so that you never have to work out which of the two words on the page is the Greek — the Greek is the bare one, and the bracketed one is the scaffold that will be taken away. They also keep the notation honest, because this book marks two different things about a letter and they are easy to confuse. **Angle brackets are spelling**: ⟨h⟩ is how a rough breathing is written in Latin letters. **Slashes are sound**: /h/ is what you say. A table that gives both — as the alphabet table does — is telling you two separate facts about the same letter.

## About the Developer

Didaskalos was developed by Farnoosh Shamsian. For more information, please refer to the relevant project pages and publications, or check out <https://farnoosh-shamsian.github.io/>
