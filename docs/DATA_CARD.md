# Data card: PromptShift Bluesky 2026 offensive-language set

400 English Bluesky posts written between 1 May and 15 November 2026, labelled OFF / NOT with OLID's level-A definition. They were built to test LLM annotators on posts written after the models were released (Llama 3.3: Dec 2024; Gemma 4 and Qwen 3.6: Apr 2026).

## 1. Collection: how to repeat it

| Step | Setting |
|---|---|
| API | Bluesky AT Protocol, `https://bsky.social/xrpc/app.bsky.feed.searchPosts`, session via `com.atproto.server.createSession` with an app password |
| Query | One search per OLID keyword available on Bluesky: `you are`, `she is`, `he is`, `gun control`, `conservatives`, `antifa`, `MAGA`, `liberals` (multi-word keywords in quotes). OLID's Twitter-only operators `to:BreitBartNews` and `-filter:safe` have no Bluesky equivalent |
| Parameters | `sort=latest`, `lang=en`, `limit=100`, `since`/`until` = one calendar month (May–Oct 2026, then 1–15 Nov 2026), up to 3 pages per keyword and month, 0.5 s between requests |
| Filters | created inside the month and not before 2026-05-01 · `en` in the post's languages · no quote posts · keyword present in the text · at least 3 content words |
| Cleaning | mentions → `@USER`, links → `URL` (as in OLID), whitespace collapsed |
| Duplicates | exact (normalised) and near duplicates (Jaccard ≥ 0.8 on word sets) removed, keeping the earliest |
| Sampling | seed `20261129`; equal share per keyword, spread over months, at most 2 posts per author; 470 posts split at random into 400 main, 40 pilot, 30 calibration |
| Date | sample drawn 25 November 2026 |
| Code | [`src/promptshift/collect_fresh.py`](../src/promptshift/collect_fresh.py) (`collect`, then `sample`) |
| Record | [`protocol/fresh_collection_fingerprint.json`](../protocol/fresh_collection_fingerprint.json) (counts and SHA-256 of every file) |

| Stage | Posts |
|---|---|
| Returned by the API | 15,561 |
| After filters (dropped: 2,006 quote posts, 1,180 keyword not in text, 714 too short, 31 outside the window, 225 already collected) | 11,405 |
| After duplicate removal | 10,926 |
| Balanced sample | 470 |
| Main set (labelled) | 400 |

Search results change over time (deleted posts, ranking), so a new run gives a similar but not identical pool. The released post IDs identify the exact posts.

**ID check (27 Nov 2026):** all 400 IDs were fetched from Bluesky's public API (`dataset_card.py verify-ids`). 396 returned exactly our stored text and creation date; 4 had been deleted or hidden by their authors; 0 differed ([`artifacts/fresh/id_verification.json`](../artifacts/fresh/id_verification.json)).

## 2. Composition (main set, final labels)

| Keyword | Posts | Offensive | Share |
|---|---|---|---|
| MAGA | 47 | 19 | 40.4% |
| he is | 50 | 17 | 34.0% |
| conservatives | 49 | 14 | 28.6% |
| antifa | 54 | 14 | 25.9% |
| she is | 46 | 10 | 21.7% |
| liberals | 49 | 10 | 20.4% |
| gun control | 52 | 7 | 13.5% |
| you are | 53 | 7 | 13.2% |
| **All** | **400** | **98** | **24.5%** |

- OLID test set for comparison: 240 of 860 offensive (27.9%).
- Posts per month: 50–62. Median length: 28 words (OLID: 21).
- Monthly tables: [`artifacts/fresh/dataset_description.md`](../artifacts/fresh/dataset_description.md).
- **Not a random sample of Bluesky.** Keyword sampling over-represents political and person-directed posts, so prevalence figures describe keyword-matched posts only.

## 3. Labelling

| | |
|---|---|
| Definition | OLID level A, verbatim. OFF: posts containing any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct. NOT: posts that do not contain offense or profanity |
| Guide | The definition plus 7 rules and 6 invented examples. The rules and examples were drafted with an LLM (Claude) and reviewed by the team. Full text: [`src/promptshift/annotate_fresh.py`](../src/promptshift/annotate_fresh.py) (`GUIDE`) |
| Procedure | 1. A and B label 30 calibration posts and discuss them. 2. A and B each label all 400 main posts plus 100 OLID test posts (anchors), in separate random orders. 3. C labels the posts where A and B disagree. 4. Final label = majority |
| Agreement | 365 / 400 (91.25%), Krippendorff's α = 0.78 |
| Disagreements | 35 (8.75%): 34 × A OFF / B NOT, 1 × A NOT / B OFF. 32 were marked "unsure" by at least one annotator. Most under *conservatives* (8) and *you are* (7). C: 26 NOT, 9 OFF (agreed with B 27 times, with A 8 times) |
| Annotator shares | A 123 OFF (30.8%), B 90 OFF (22.5%), final 98 (24.5%) |
| Against OLID's labels | 100 anchors: accuracy 0.78 (A), 0.77 (B); κ = 0.43 (A), 0.37 (B) |
| Note | After labelling, annotator A reported relying mainly on his own understanding of offensive language rather than the full rules and examples |
| Reasons for disagreement | Criticism vs insult 14 · veiled or implicit insult 8 · quoted or reported offensive words 4 · sarcasm or irony 4 · context missing 3 · profanity not aimed at anyone 2. Drafted by an LLM (Claude) from the posts and the annotators' comments, reviewed by annotator A ([`artifacts/fresh/disagreement_reasons.json`](../artifacts/fresh/disagreement_reasons.json)) |

No LLM produced any label.

## 4. Files and access

| File | Contents | Where |
|---|---|---|
| `bluesky2026_offensive_ids_labels.csv` | item_id, row_id, AT URI, bsky.app link, keyword, month, created_at, label_A, label_B, label_C, label_final | Course submission |
| `bluesky2026_offensive_CONTAINS_TWEET_TEXT.csv` | the above plus post text | Course submission, examiners only |
| `annotations_side_by_side_CONTAINS_TWEET_TEXT.xlsx` | all annotations side by side, disagreements highlighted, guide | Team and examiners |
| `artifacts/fresh/dataset_description.md` | counts only | Public repository |

All three release files are created by `python src/promptshift/dataset_card.py release` in `data/fresh/release/`, which git ignores.

## 5. Privacy

Posts are public. The public repository contains no post text and no post IDs; there, authors appear only as salted hashes. IDs and text go only to the examiners. The raw download is deleted after the course. Deleted posts can no longer be retrieved from their IDs.
