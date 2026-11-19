# Project map: before and after the midterm

**Research question:** if we change only the *wording* of a prompt, or add a *definition* of "offensive", do LLMs label the same social-media posts differently?

**Data:** TweetEval / OLID offensive-language test set (860 posts, human labels).
**Models:** Gemma 4 26B, Qwen 3.6 36B, Llama 3.3 70B (via the maKI API, temperature 0).

---

## 1. Before the midterm (old version, replaced)

```mermaid
flowchart LR
    A["Idea:<br/>does prompt wording<br/>change LLM labels?"] --> B["3 prompt wordings<br/>no definition given"]
    B --> C["Ran 3 models<br/>on offensive posts"]
    C --> D["Counted how often<br/>labels flipped"]
    D --> E["Midterm<br/>4.5 / 10"]
    E --> F["Main gaps:<br/>no hypotheses<br/>no noise baseline<br/>no definition in prompt<br/>contamination not checked<br/>unclear data splits<br/>slides too dense"]

    classDef old fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef bad fill:#FAECE7,stroke:#993C1D,color:#4A1B0C
    class A,B,C,D old
    class E,F bad
```

---

## 2. After the midterm (rebuilt from scratch, using the feedback)

```mermaid
flowchart TD
    Q["1. Clear question + 3 hypotheses<br/>H1 wording matters beyond noise<br/>H2 a definition changes quality<br/>H3 a definition reduces wording effect"]
    Q --> L["2. Literature review<br/>6 core papers"]
    L --> DS["3. Design: 8 prompt conditions<br/>3 wordings, 3 wordings + definition,<br/>2 exact repeats = noise baseline"]
    DS --> CODE["4. Code + 18 automated tests"]
    CODE --> P["5. Pilot on validation split<br/>960 / 960 valid answers"]
    P --> FR["6. Plan frozen in git<br/>before any test data<br/>commit 018b817"]
    FR --> RUN["7. Main run on test split<br/>20,640 calls, 0 invalid"]
    RUN --> PR["8. Contamination check<br/>1,200 calls"]
    PR --> AN["9. Statistics + figures<br/>commit fa704fd"]
    AN --> NEXT["Next: human check,<br/>presentation, report"]

    classDef done fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef todo fill:#FAEEDA,stroke:#854F0B,color:#412402
    class Q,L,DS,CODE,P,FR,RUN,PR,AN done
    class NEXT todo
```

### The 8 prompt conditions in one picture

```mermaid
flowchart LR
    subgraph NODEF["Without definition"]
        W0["W0"] --- W1["W1"] --- W2["W2"]
        R0["R0 = exact repeat of W0"]
    end
    subgraph DEF["With OLID definition"]
        D0["D0"] --- D1["D1"] --- D2["D2"]
        RD0["RD0 = exact repeat of D0"]
    end
    NODEF -. "same 3 wordings,<br/>definition added" .-> DEF
```

The repeats tell us how much a model changes its answer **by chance**. A wording effect only counts if it is bigger than that.

---

## 3. How each feedback point was fixed

| Midterm feedback | What we do now |
|---|---|
| No hypotheses | H1–H3 written down and tested with fixed statistical tests |
| No baseline / noise floor | Exact-repeat conditions (R0, RD0), plus majority and random baselines |
| No definition in the prompt | Definition conditions D0–D2 using the official OLID definition |
| Contamination not addressed | Memorisation probe (Golchin & Surdeanu 2024 method) |
| Unclear data splits | Pilot on the validation split only; final results on the test split only; dataset version pinned |
| Unclear analysis | Analysis plan frozen in git before the test data was collected |
| Presentation too dense | New talk: about 10 slides, one message per slide |

---

## 4. Results in plain words

| Finding | Number |
|---|---|
| Changing only the first sentence of the prompt flips labels on a small share of posts | about 2.5–3% per pair of wordings (repeats: about 0%) |
| Adding the definition helps 2 of 3 models | macro-F1 +0.09 (Gemma), +0.05 (Qwen), no change (Llama) |
| The definition pulls every model toward the human labelling standard | share labelled offensive moves toward the true 28% for all 3 models |
| The choice of model matters more than the wording | Gemma and Llama disagree on about 21% of posts |
| Contamination check | no signal for Gemma or Qwen; weak signal for Llama, no copied posts |

All 6 main tests ran as planned; the results are unchanged when duplicate posts are removed.

---

## 5. What is left

```mermaid
flowchart LR
    A["Human check<br/>2 annotators, 100 posts"] --> B["Final presentation<br/>9–10 Dec"] --> C["Report, ACL format<br/>8 Jan"]
    classDef todo fill:#FAEEDA,stroke:#854F0B,color:#412402
    class A,B,C todo
```

## 6. Questions we would like guidance on

1. Is a plan frozen in a git commit before data collection acceptable as our preregistration?
2. Is the human check (50 posts where labels flipped vs 50 where they did not, 2 annotators) enough?
3. Should the talk lead with the "model choice matters most" result or with the wording result?
4. How should we report the weak contamination signal for Llama?
5. Is an extra analysis using the models' confidence scores (log probabilities) worth adding, or should we keep the scope?
