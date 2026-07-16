# The Writing Style of Grzegorz Chrupała

An explicit stylistic profile based on four scientific papers:

| Paper | Venue | Authorship |
| --- | --- | --- |
| *Correlating neural and symbolic representations of language* | ACL 2019 | with Alishahi |
| *Visually Grounded Models of Spoken Language: A Survey…* | JAIR 2022 | solo |
| *Learning English with Peppa Pig* | TACL 2022 | with Nikolaus & Alishahi |
| *Beyond Decodability: Reconstructing LM Representations with an Encoding Probe* | ACL submission | anonymized |

Two of the papers are co-authored, so some variance reflects collaborators; traits that recur in the solo-authored survey **and** the co-authored papers are treated here as the author's core style. Statistics were computed over the running prose only (references, tables, and equations stripped).

---

## 1. Quantitative fingerprint

Per-paper figures (rates per 1,000 words unless noted):

| Metric | ACL'19 | JAIR'22 (solo) | TACL'22 | ACL sub. |
| --- | --- | --- | --- | --- |
| Mean sentence length (words) | 23.9 | 25.0 | 24.8 | 24.4 |
| Median sentence length | 22 | 24 | 22 | 23 |
| 90th-percentile length | 40 | 44 | 46 | 39 |
| Longest sentence | 101 | 88 | 104 | 86 |
| Sentences > 40 words | 9% | 14% | 14% | 8% |
| Sentences < 10 words | 10% | 16% | 10% | 8% |
| Flesch–Kincaid grade | 14.7 | 15.1 | 14.3 | 15.0 |
| Mean word length (chars) | 5.2 | 5.2 | 5.1 | 5.2 |
| Type–token ratio (MATTR-500) | 0.45 | 0.50 | 0.48 | 0.45 |
| Commas | 41 | 63 | 53 | 46 |
| Semicolons | 2.5 | 2.7 | 6.9 | 2.7 |
| Colons | 7.2 | 7.0 | 10.5 | 5.4 |
| Parentheses (opening) | 18 | 23 | 17 | 18 |
| Dashes | 2.9 | 0.1 | 1.5 | 0.0 |
| Exclamation marks | 0 | 0 | 0 | 0 |
| Question marks | 0.0 | 0.9 | 0.0 | 0.2 |
| *we/our/us* | 18.6 | 3.4 | 23.1 | 18.3 |
| Hedges | 3.9 | 5.6 | 5.5 | 6.3 |
| Boosters | 1.4 | 1.3 | 2.2 | 2.5 |
| Passive constructions (approx.) | 10 | 13 | 8 | 9 |
| Nominalizations (-tion/-ity/-ment…) | — | — | — | 63 (corpus-wide) |

**Reading of the numbers.** Sentences are long by NLP-paper standards (mean ≈ 24–25 words; the field average is closer to 20–22) with a fat right tail — one sentence in ten runs past 40 words, and 85–105-word sentences occur, almost always because of stacked parenthetical citations or coordinated lists. Hedges consistently outnumber boosters two- or four-to-one. There is not a single exclamation mark in any body text (one lone "Thank you!" closes the survey's acknowledgments — the only place the register cracks). Graduate-level readability (FK grade 14–15) driven by Latinate vocabulary and heavy nominalization rather than by convoluted syntax.

---

## 2. Vocabulary and register

**Latinate, precise, high-density — with deliberate colloquial relief.** The base register is formal academic English rich in nominalizations (≈63 per 1,000 words: *decodability, tractability, generalization, segmentation, ecological validity, representational similarity*) and low-frequency precision words: *amenable, purview, endeavor, akin to, pervasive, displaced, disjoint, confounded, coincident, undoubtedly*. Against this backdrop the author regularly drops in a plain idiom, and the contrast is a signature:

> "when children **pick up** a language" · "harder to **zoom in on** the meaning-bearing aspects" · "grounding is **only part of the story**" · "Current approaches **work well enough** from an applied point of view" · "**But the point remains** that…" · "the scores **may keep going up**, but that does not mean…"

The effect is a learned voice that refuses stuffiness: technical where precision matters, conversational where it doesn't.

**Terminology discipline.** New concepts get explicit names, often typographically marked (small caps or all-caps in the originals): Encoding Probe / Decoding Probe, RSA_regress, the FIXED / JITTER / STATIC experimental conditions. Terms are defined at first use ("We refer to these as diagnostic models"), and the naming is then used with complete consistency. Abbreviations are expanded on first mention with the acronym in parentheses.

**Spelling and conventions.** American spelling (*modeling, behavior, analyze, endeavor*), yet enumerations use the British-flavored *Firstly, … Secondly, …*. Occasional *towards*. In less copy-edited venues an unpolished spelling slips through (*undoubtably* in the JAIR survey) — the prose is meticulous in structure but not fussily proofread.

**Field-anchoring vocabulary.** The conceptual home base shows in the recurring lexicon: *grounded, modality, ecological validity, naturalistic, acquisition, learner, cognitively motivated*. Even engineering-heavy papers are framed in the language of child language acquisition and cognitive science.

---

## 3. Syntax and sentence architecture

- **Long but load-bearing sentences.** Complexity comes from coordination and parenthetical insertions, not from deep center-embedding. A typical long sentence stacks a main claim, a subordinate qualification, and a parenthetical citation or example, in that order: "Although it is plausible that such non-semantic correlations can sometimes be useful to the learner in the general endeavor of making sense of the world, for the specific task of learning the semantics of linguistic units they are likely more often an obstacle, as they make it harder to zoom in on the meaning-bearing aspects of the audio signal."
- **Fronted adverbial and purpose clauses.** Sentences habitually open with a scene-setting adjunct: "In order to control for these factors, …", "To capture the similarities between these symbolic representations, …", "Following Nikolaus and Fourtassi (2021), …". The full "in order to" (21 corpus occurrences) is preferred over bare infinitival "to" for purposes.
- **"Here we…" as the pivot move.** The transition from problem to contribution is almost always "Here we present / introduce / address / take…" — locative *here*, present tense, active voice.
- **Agentive "we" in research papers** (18–23 per 1,000 words), including for opinions ("We suspect that…", "We foresee that…"). The solo survey, by contrast, goes impersonal (*we* drops to 3.4/1k), relying on constructions like "The current paper brings together…", "This survey zooms in on…".
- **Moderate passive** (≈8–13/1k), reserved for genuinely patient-focused statements ("The video is subsampled to 10 frames per second") — methods sections use it freely, argumentative prose stays active.
- **Relative clauses with *which*** are frequent (up to 5.9/1k), used for restrictive as well as non-restrictive relatives ("Analysis methods which enable us to better understand…") — no strict that/which discipline.
- **Occasional formal inversion** for flavor: "Should we wish to decode activation patterns into a structured target…, we would need to resort to…".
- **Enumerative scaffolding**: "Firstly … Secondly …", "two important limitations", "Our contributions are the following:", bulleted contribution lists, "(i) … (ii) …" inline enumeration.

---

## 4. Punctuation habits

- **The colon is the signature mark.** Well above field-average rates (5–10/1k), and — most distinctively — the colon frequently introduces a *complete capitalized clause* (65 corpus occurrences), a house style the author embraces: "We disregard portions of the video that are annotated as neither dialog nor narration: This means our data consists mostly of video clips that contain some speech", "Our heuristic to generate positive and negative examples is very simple: We consider an example positive if…". Colons announce definitions, explanations, and formulas.
- **Parentheses everywhere** (17–23/1k): citations, "(e.g., …)" exemplification, short clarifications ("(around 20 words and seven visual categories)"), and glosses. Parenthetical *e.g.* / *i.e.* are frequent (≈40 combined corpus-wide) and always in the Latin abbreviated form.
- **Dashes almost absent** (0–3/1k). Where another writer would use an em-dash aside, this author uses parentheses or a colon.
- **Semicolons sparing but purposeful**: mainly to chain closely related independent clauses in comparative statements ("Lower values of this parameter discount larger tree fragments…; the value 1 does not do any discounting") and inside multi-reference citation lists.
- **No exclamation marks, virtually no question marks.** The rare rhetorical question is a deliberate didactic device, answered immediately and tersely: "What can we conclude about the relative importance of these two features? Nothing definitive: the two numbers are not directly comparable…"

---

## 5. Discourse structure and rhetorical moves

1. **Problem–gap–solution engine.** Nearly every introduction follows the same gearbox: (a) describe the standard approach neutrally; (b) expose its limitation in one crisp sentence beginning "However," / "One important limitation…" / "A major unresolved issue…"; (c) pivot with "Here we…". The gap is always stated as a property of the *method or data*, never as a failing of named colleagues.
2. **Didactic worked micro-examples.** Abstract methodological points are made concrete through miniature thought experiments with invented numbers: "We may then recover speaker IDs with (e.g.) 95% accuracy and phone labels with 58% accuracy. What can we conclude…?". The reader is walked through the reasoning rather than told the conclusion.
3. **Relentless signposting.** "Our contributions are the following:", "In Section 5.1 we investigate…", "For the rest of this paper, we only report recall@10." The reader always knows where they are and why.
4. **Candid, itemized limitations.** Limitation discussions are unusually long, specific, and unprompted: "The probe remains observational rather than interventional…", "This approach is rather simplistic and does not match the real experience of language learners", "the reported ablation effects should be read as a case study, rather than as an exhaustive characterization". Negative and null results are reported flatly ("the effect of JITTER is only minor", "some runs fail to converge", "temporal information … may even have a detrimental effect").
5. **Simplicity presented as a virtue.** The author repeatedly emphasizes the deliberate plainness of their models: "We thus keep the components of our architecture simple", "a simple bi-modal architecture", "Our heuristic … is very simple". Novelty claims attach to *questions, data, and evaluation design*, not to architectural sophistication.
6. **Taxonomizing instinct.** Surveys and related-work sections organize the field into named families and timelines; experiments are organized into named ablation conditions with a summary table. Classification is the default mode of exposition.
7. **Interdisciplinary framing.** Engineering results are persistently connected to cognitive science: evaluation protocols "inspired by the intermodal preferential looking paradigm", data choices motivated by "ecological validity", speculation linked to "predictors of age of acquisition in the child language acquisition literature".

---

## 6. Stance, personality, and sentiment

**Sober, understated, and confident without heat.** The booster inventory is thin (1.3–2.5/1k) and almost entirely *scalar* — *substantially, strongly, clearly* — attached to measured effect sizes, never to the paper's own importance. There is no "remarkable", no "impressive", no "state-of-the-art" chest-beating. Enthusiasm is expressed through the mildest possible evaluatives: things are "interesting", "appealing", "worthwhile", "useful", at most "crucial".

**Hedged epistemics as the default.** Claims are graded with care: "Our results **suggest**…", "One **possible** explanation **could be**…", "We **suspect** that…", "it is **plausible** that…", "**arguably**", "**likely more often** an obstacle". Top hedges in the corpus: *may* (30), *often* (23), *rather* (23). Yet hedging never becomes mush — a hedged claim is immediately followed by the concrete evidence or the diagnostic that would settle it (e.g., the noun/verb clip-duration hypothesis is stated with "One possible explanation could be…" and then instantly tested with a Pearson correlation).

**Dry wit at the level of research design, deadpan at the level of prose.** The author who trains a model on *Peppa Pig* — and soberly discusses "when Daddy Pig explains that they need to clean up before Mummy Pig sees the mess that Peppa and George made" as an example of displaced language — is clearly having fun, but the sentence-level register never winks. The humor lives entirely in the choice of material and the incongruity of treating it with full scientific rigor.

**Intellectual honesty as a stylistic trait.** Confounds are volunteered before a reviewer could ask ("Note that there is one further confound we do not control for"), competing explanations are listed, and the writing repeatedly warns the reader off over-interpretation of its own results. The overall persona: a careful, slightly skeptical empiricist with a cognitive-science heart, allergic to hype, generous with caveats, and quietly playful in what they choose to study.

---

## 7. Citation and scholarly apparatus

- **Citation-dense**: ≈6 *et al.* per 1,000 words; long inline citation chains ("(e.g., Synnaeve et al., 2014; Harwath and Glass, …; Alishahi et al., 2017; …)") are a major cause of the longest sentences.
- Citations are woven into syntax as agents ("Kriegeskorte et al. (2008) present RSA as…", "Moschitti (2006) propose an efficient algorithm…") — plural verb agreement with *et al.* subjects.
- Prior work is characterized generously and precisely; criticism is systemic, not personal ("this technique has been carried over from text-based image-caption modeling" — the flaw belongs to the tradition).
- Practical reproducibility notes appear in the running text: exact URLs, "Our code is publicly available at…", hardware constraints ("not tractable on commodity GPU hardware"), even the note that episodes were "purchased … on DVD support".
- Footnotes are used liberally for such operational details and mild qualifications.

---

## 8. How to write like this author — a checklist

1. Aim for a **mean sentence length of ~24 words**; let one sentence in ten exceed 40 words, but only via coordination, citation chains, or parallel lists — never deep nesting.
2. Open the paper by **describing the mainstream approach fairly**, then expose one or two limitations in numbered form ("two important limitations"), then pivot with **"Here we…"**.
3. Use **active "we"** for everything the authors did or believe; keep passives for method mechanics. (Writing solo? Shift to "The current paper…" impersonals.)
4. Reach for a **colon** where others use a dash; let it introduce full capitalized clauses. Put asides and examples in **parentheses**, with *e.g.,* and *i.e.,* in Latin form. Never use an exclamation mark.
5. Hedge every interpretive claim (*suggest, may, likely, arguably, we suspect*) and then **immediately back it with a number, a test, or a planned analysis**. Use boosters only for effect sizes (*substantially, strongly*).
6. **Name your methods and conditions** (ideally in small caps), define every term at first use, and reuse names with total consistency.
7. Mix a **Latinate formal register** (*amenable, purview, tractability, ecological validity*) with occasional **plain idioms** (*pick up a language, zoom in on, part of the story*).
8. Signpost mercilessly: contribution bullets, "Firstly/Secondly", section forecasts, "For the rest of this paper we…".
9. Volunteer your **confounds and limitations** at length, unprompted, in concrete terms; report null and negative results in the same flat tone as positive ones.
10. Praise your own architecture only for being **simple**; locate the novelty in the question, the data, or the evaluation design.
11. Frame technical work in **cognitive terms** — acquisition, learners, ecological validity — and connect evaluation to experimental paradigms from psychology.
12. Keep sentiment at room temperature: the strongest allowed self-praise is "interesting" or "useful", and the only exclamation mark permitted is a "Thank you!" to your reviewers.
