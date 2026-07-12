# Didaskalos

Didaskalos is a corpus-driven pipeline for generating personalized Ancient Greek textbooks based on the linguistic features of user-selected texts. The project departs from traditional fixed curricula by tailoring instructional content to the specific vocabulary, morphology, and syntactic structures attested in a target corpus.

A central principle of the project is the exclusive use of authentic texts. All instructional material, including examples and exercises, is derived directly from the selected corpus.

## Overview

Ancient Greek texts exhibit substantial variation across periods, genres, and authors. Standard pedagogical materials typically present a generalized sequence of grammatical topics that may not align with the needs of a given text.

Didaskalos addresses this limitation by allowing learners and instructors to select a corpus and automatically generating a corresponding textbook. The resulting material is organized according to the frequency and distribution of linguistic features within the selected texts.

## Methodology

The system integrates three main components:

1. Corpus Analysis  
   The pipeline processes annotated Ancient Greek treebanks to extract lexical frequencies, morphological patterns, and syntactic constructions.

2. Curriculum Generation  
   Based on frequency data, the system constructs a sequence of topics reflecting the distribution of linguistic features in the corpus.

3. Modular Textbook Assembly  
   The textbook is composed of self-contained modules covering grammatical and morphological categories, accompanied by examples and exercises drawn from the source texts.

## Exercises and Difficulty Scoring

Exercises are generated exclusively from authentic sentences in the corpus. To control progression and pedagogical suitability, each sentence is assigned a difficulty score.

The difficulty score is determined by two factors:

- Sentence length: shorter sentences receive lower difficulty scores
- Lexical frequency: sentences containing higher-frequency words (relative to the corpus at that stage) receive lower difficulty scores

These criteria allow the system to prioritize simpler, more accessible sentences in earlier stages, while gradually introducing more complex material.

## Grammar Module Generation

Initial versions of grammatical explanation modules were produced using a Retrieval-Augmented Generation (RAG) pipeline based on standard reference grammars (including Smyth and Crosby & Schaeffer). Current work focuses on refining these modules and preparing them for localization and translation.
The generation pipeline is language-independent. Scaling to additional languages primarily requires translation and adaptation of grammar modules.
This enables the production of textbooks in multiple target languages without modification to the underlying system. Current work will soon also include localization into Persian.

## Implementation

The project is implemented in Python and integrates:

- Treebank processing
- Frequency-based linguistic analysis
- Modular content generation and assembly

## Project Status

The core infrastructure is still being built! Ongoing work focuses on:

- Details of foe the frequency-based syllabus is generated
- Refinement of grammar modules
- Evaluation of exercices
- Localization and multilingual support (Persian already added)

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
