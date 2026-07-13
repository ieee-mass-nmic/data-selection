# PCU-Select — Method Overview Figure: Drawing Spec

> Scope: a **paper method-overview figure** (Fig. 2, `\label{fig:architecture}`), not a
> software-implementation flowchart. This spec defines *what the figure must argue*, not just
> what boxes to draw. Cross-checked against `01_introduction.tex`, `03_problem_formulation.tex`,
> `04_method.tex`, `05_experiments.tex`, `08_appendix.tex`, and the current `figures/Architecture.png`.
>
> This round: **spec only**. Do not create `.drawio`, do not render.

---

## 1. The one-sentence core claim the figure must transmit

> **PCU-Select computes expensive PEFT-conditioned utility supervision *once* over a shared
> intervention-site space, then *reuses* it for cheap target-conditioned selection across an
> entire PEFT registry.**

Everything in the figure exists to support three sub-claims, which map to the three things the
user asked to foreground:

1. **Reusable PEFT-conditioned utility** — utility `u(x,p,t)` depends on the target PEFT `p`, not
   just the task; it is learned into a scorer `s_φ` that is trained once and reused.
2. **Shared intervention-site space** — samples, task sketches, and PEFT configurations are all
   projected onto the *same* 24 module-output sites `Ω`; that shared coordinate system is what
   makes cross-PEFT comparison possible and what makes supervision reusable.
3. **Low-cost target-conditioned selection** — online, a new target `(p*,t*)` pays only one
   scorer pass + clustering (0.18 GPU-h), no gradients, no new datastore.

If a reader takes away only "there is an offline part and an online part," the figure has failed.
The non-obvious message is the **shared site space + the reuse arrow**.

---

## 2. Entities, modules, and mathematical objects that MUST appear

### Offline supervision (left region)
- **Three input towers** (the three conditioning signals), each `raw source → encoder → code`:
  - Sample tower: `Candidate pool` → `Feature extract` → `z_x` (sample representation, ℝ^848)
  - PEFT tower: `PEFT registry` → `PEFT encoder` → `z_p` (PEFT code, ℝ^128)
  - Task tower: `Task sketches` → `Sketch encoder` → `z_t` (task representation, ℝ^848)
- **Intervention sites `Ω`** — the shared hub: `24 sites = 8 layers × 3` (attention / MLP /
  post-residual). Labeled caption underneath: `shared coordinates: samples / PEFTs / tasks`.
- **Low-fidelity site-weighted gradient proxy** `u^lo` (Eq. 4, cheap, broad coverage).
- **High-fidelity short-horizon labels** `u^hi` (10K triples; coverage / uncertainty / boundary).
- **Scorer training** `s_φ`: rank + regression + proxy distillation + heteroscedastic uncertainty
  (produces `(μ̂, σ̂)`). This is the reusable artifact.

### Online selection (right region)
- **Target** `(p*, t*)` in → **Encode** `z_{p*}, z_{t*}`.
- **Support-tier check**: `L0 near / L1 far / L2 new-family` (+ optional `calibration` residual head).
- **Score pool** → `(μ̂, σ̂)`.
- **Conservative score** `q = μ̂ − λσ̂` (λ = 0.2).
- **Coverage-aware allocation**: `cluster + quotas` → `top-b_k per cluster`.
- **Selected subset** `S` (the output).

### The bridge object (the point of the figure)
- **The trained scorer `s_φ` is the single object that crosses the offline→online boundary.**
  The **"reused"** arrow from `Train scorer` into `Score pool` is the amortization claim and must
  be visually the most important edge in the figure.

### Numbers to keep as small annotations (NOT as primary boxes)
- `z_x`: `848 = 768 + 16 + 64`
- `z_p`: `128 = 96 mask + 16 cap + 16 recipe`
- `z_t`: `848 + site grads`
- Amortization footnote: `offline cost amortizes after P* = 2.33 target configs`.

---

## 3. Edge semantics (what each arrow *means*, not just order)

Every edge carries a meaning; label the load-bearing ones. Ordering-only arrows stay unlabeled.

| # | From → To | Semantic meaning | Label on edge |
|---|-----------|------------------|---------------|
| E1 | `z_x` → `Ω` | sample contributes **per-site gradient alignment** signatures | *site gradients* (thin) |
| E2 | `z_p` → `Ω` | PEFT code selects **which sites are touched + operator + capacity weight** | *site mask / weights* |
| E3 | `z_t` → `Ω` | task sketch contributes **mean site gradients** (task direction) | *sketch mean grads* |
| E4 | `Ω` → `u^lo` | site-weighted cosine proxy over shared sites (Eq. 4) | — (structural) |
| E5 | `u^lo` → `u^hi` | proxy is **corrected near the boundary** by short-horizon labels | *proxy distillation* |
| E6 | `u^lo` + `u^hi` → `Train scorer` | **multi-fidelity** supervision trains the conditional scorer | *multi-fidelity labels* |
| **E7** | `Train scorer s_φ` → `Score pool` | **the trained scorer is reused per target — the amortization** | **reused** (heavy, accent) |
| E8 | `Target (p*,t*)` → `Encode` → `Support-tier` → `Score pool` | online conditioning path | — (flow) |
| E9 | `Support-tier` → `calibration` (dashed) | far / new-family targets **optionally** fit a residual head | *cal.* (dashed, optional) |
| E10 | `Score pool` → `Conservative score` | uncertainty penalty makes ranking risk-averse | — (flow) |
| E11 | `Conservative score` → `cluster+quotas` → `top-b_k` → `Selected subset` | coverage-aware budget allocation | — (flow) |

**Rule:** exactly one edge (E7, "reused") is drawn heavy and in the accent color. E1–E3 (the three
towers converging on `Ω`) are the second-most-important visual: they show the *shared* space.
E9 is the only dashed edge (optional/conditional). All others are plain flow arrows.

---

## 4. Offline / Online boundary

- **Vertical divider** separating a left **Offline supervision** panel from a right **Online
  selection** panel. Keep the two-panel framing from the current figure — it works.
- **Left panel banner:** `Offline supervision` — subtitle `once per backbone family`.
- **Right panel banner:** `Online selection` — subtitle `per target p*, t*`.
- The divider is crossed by **exactly one arrow: E7 ("reused")**. This is deliberate — the whole
  cost argument is that only the scorer, a cached artifact, crosses the line. Do not let any other
  edge cross the boundary; cached features are implied by the scorer crossing.
- Bottom full-width caption strip (keep from current figure, tighten):
  `Offline cost is cached and amortized (P* = 2.33); online selection is one scorer pass + clustering (0.18 GPU-h).`

---

## 5. Visual hierarchy — primary vs. auxiliary

**Primary layer (large nodes, bold labels, must survive thumbnail):**
1. The two panel banners + divider (Offline / Online).
2. `Ω` intervention-site hub (visually the anchor of the left panel).
3. The three input codes `z_x`, `z_p`, `z_t` (the conditioning triple).
4. `Train scorer s_φ` and the **reused** edge E7.
5. `Score pool → q → Selected subset` spine on the right.

**Secondary layer (medium, supporting the primary claim):**
- `u^lo` and `u^hi` label nodes.
- `Support-tier check (L0/L1/L2)`.
- `cluster + quotas`, `top-b_k`.

**Auxiliary layer (small text, annotations, de-emphasized — never boxes competing for attention):**
- Dimensionality breakdowns (`848 = …`, `128 = …`).
- `Feature extract`, `PEFT encoder`, `Sketch encoder` boxes (necessary but low-value — render small/neutral).
- `calibration` residual head (dashed, small).
- The `8 layers × 3` grid interior detail of `Ω` (show a compact glyph, not 24 individually labeled cells).

---

## 6. Final English labels (must match paper terminology EXACTLY)

Use these strings verbatim. Do not paraphrase; these are the terms a reader will grep for.

**Panels / banners**
- `Offline supervision` / `once per backbone family`
- `Online selection` / `per target $p^*, t^*$`

**Left towers (source → encoder → code)**
- `Candidate pool` → `Feature extract` → `$z_x$ sample` · annot `848 = 768 + 16 + 64`
- `PEFT registry` → `PEFT encoder` → `$z_p$ PEFT code` · annot `128 = 96 mask + 16 cap + 16 recipe`
- `Task sketches` → `Sketch encoder` → `$z_t$ task` · annot `848 + site grads`

**Hub**
- `24 sites` / `8 layers × 3`  (interior)
- caption: `shared coordinates: samples / PEFTs / tasks`

**Supervision**
- `Low-fidelity site-weighted proxy`
- `High-fidelity short-horizon labels`
- edge `multi-fidelity labels`
- `Train scorer` / `rank + reg + uncertainty`
- edge `reused`

**Online spine**
- `Target PEFT + task`
- `Encode $z_{p^*}, z_{t^*}$`
- `Support-tier check` / `near / far / new family`  (tiers L0 / L1 / L2)
- `cal.` (dashed side box)
- `Score pool` / `$(\hat\mu, \hat\sigma)$`
- `Conservative score` / `$q = \hat\mu - \lambda\hat\sigma$`
- `Cluster + quotas`
- `Top-$b_k$ per cluster`
- `Selected subset`

**Bottom caption**
- `Offline cost is cached and amortized ($P^\star = 2.33$); online selection is one scorer pass + clustering.`

> Terminology guards (from cross-check):
> - It is **"intervention sites"** / `Ω` / **24 sites**, never "hooks" or "layers" alone.
> - It is **"site-weighted"** low-fidelity proxy (not "gradient similarity").
> - **"short-horizon"** high-fidelity labels (horizons h∈{1,4}); do not say "full fine-tune."
> - **"conservative score"** for `q = μ̂ − λσ̂` (λ=0.2), not "final score."
> - Tiers are **near-support / far-support / unseen-family** = **L0 / L1 / L2**.
> - The scorer is **`s_φ`**, output **`(μ̂, σ̂)`** (mean + uncertainty).

---

## 7. Recommended horizontal layout and width ratios

Single-column-spanning `figure*` (full text width), aspect ≈ **2.05 : 1** (w:h), same envelope as
the current figure so it drops into `\includegraphics[width=\textwidth]`.

Left→right regions:

```
| Offline supervision  ~62% width          | ‖ | Online selection ~35% |
|                                          |div|                       |
| [towers 24%] → [Ω hub 12%] → [labels 14%]→[scorer 12%]  ‖  [spine]   |
```

- **Offline panel: ~62%** of width (3% divider gutter), **Online panel: ~35%**.
  Rationale: offline is where the intellectual contribution lives (three towers + shared hub +
  multi-fidelity supervision); online is a clean linear spine and should read as "cheap/simple."
- Within offline, sub-columns left→right:
  - **Input towers** (three stacked rows): ~24% — sample (top), PEFT (mid), task (bottom).
  - **`Ω` hub**: ~12%, vertically centered so all three towers converge on it (E1/E2/E3 fan in).
  - **Supervision labels** (`u^lo` top, `u^hi` mid): ~14%.
  - **`Train scorer`**: ~12%, positioned adjacent to the divider so E7 ("reused") has a short,
    prominent crossing.
- **Online spine**: single vertical column of ~7 nodes; keep it visually lighter/narrower than the
  offline panel to signal "low cost."
- Vertical center line of the whole figure aligns `Ω` ↔ `Train scorer` ↔ `Score pool` so the
  reuse arrow E7 is a near-horizontal, eye-level spine.

---

## 8. Color semantics, node types, arrow types, grouping

**Color = role, not decoration** (keep the current palette family; it already reads well):

| Color | Meaning | Applied to |
|-------|---------|-----------|
| Blue | **sample / data + offline structure** | sample tower, `Ω` hub, `u^lo`, offline panel border |
| Teal/green | **PEFT + online path** | PEFT tower, online panel border, online spine nodes |
| Orange | **labels / supervision + final outputs** | `u^hi`, `Conservative score`, `Selected subset` |
| Purple | **the reusable scorer** (the bridge) | `Train scorer` node **and the `reused` edge E7** |
| Neutral grey | **auxiliary encoders / plumbing** | `Feature extract`, `PEFT encoder`, `Sketch encoder`, `cal.` |

Make **purple the accent**: the scorer node + the E7 edge are the only purple elements, so the eye
lands on the amortization claim.

**Node types**
- Rounded rectangle: modules/stages (default).
- Rounded rectangle w/ colored fill = primary; white fill w/ colored border = secondary.
- Small grey rectangle = auxiliary encoder/plumbing.
- Compact glyph (stacked mini-cells) for `Ω` interior — suggest 24 sites without labeling each.
- Dashed rectangle = optional (`cal.` residual head only).

**Arrow types**
- Solid thin arrow = data/flow (default).
- **Solid heavy accent (purple) arrow = `reused` (E7)** — the one hero edge.
- Fan-in convergence (E1/E2/E3) into `Ω` = solid thin, all three same weight, to read as "shared."
- Dashed arrow = optional/conditional (E9 → `cal.`).

**Grouping rules**
- Two labeled container panels (Offline / Online) with a dashed vertical divider between.
- Inside offline, the three towers are visually a group (same left margin, equal row height) so
  "three conditioning signals" reads as one concept.
- `u^lo` + `u^hi` grouped as the "multi-fidelity supervision" cluster feeding the scorer.

---

## 9. Details to DELETE, MERGE, or SIMPLIFY (vs. current `Architecture.png`)

The current figure is closer to an implementation flowchart than a method overview. Trim toward the
argument:

- **DELETE** the fully drawn `24-site` ladder (8 numbered rows × 3 colored cells). It dominates
  the center visually but carries little argument. Replace with a compact `Ω` hub glyph + the text
  `24 sites (8 × 3: attn / MLP / post-residual)`. This frees the space that should show the
  three-towers-converge-on-shared-space idea.
- **MERGE** the online chain: current 7 stacked boxes (`Target` → `Encode` → `Support-tier` →
  `Score` → `Conservative` → `Cluster+quotas` → `Top-b_k` → `Subset`) is too long. Keep the spine
  but visually **group** it into 3 labeled phases: **(a) encode + tier**, **(b) score → q**,
  **(c) allocate → subset**. Fewer perceived steps ⇒ "cheap online" reads faster.
- **SIMPLIFY** dimensionality strings to small superscript-style annotations under each `z`, not
  multi-line inside the box (`z_x sample / 848 = 768+16+64` → `z_x` box + tiny `848` annotation).
- **DE-EMPHASIZE** the three encoder boxes (`Feature extract`, `PEFT encoder`, `Sketch encoder`) —
  shrink to small grey plumbing nodes; they are not contributions.
- **DROP** any per-loss enumeration inside `Train scorer` beyond `rank + reg + uncertainty`
  (FiLM / bilinear / stage A-B belong in text, not the overview).
- **KEEP** and **strengthen** the `reused` edge — currently a thin diagonal; make it the hero.
- **KEEP** the bottom amortization caption; add `P* = 2.33` so the cost claim is on the figure.

Net effect: reclaim center space from the site ladder and the long online chain, and spend it on
the two ideas that are actually novel — **shared site space** (towers converging on `Ω`) and
**reuse of the scorer** (hero edge crossing the boundary).

---

## 10. Reader comprehension checklist (5s / 20s / 60s)

**At 5 seconds (thumbnail / glance) the reader should grasp:**
- There is an **Offline** stage (left, larger) and an **Online** stage (right, smaller).
- One thing — the **scorer** — crosses from offline to online (the purple `reused` arrow).
- ⇒ "Expensive work is done once and reused." *(the cost thesis)*

**At 20 seconds (scan the labels) the reader should grasp:**
- Three inputs — **sample, PEFT, task** — all feed a **shared 24-site space `Ω`**.
- Offline produces **multi-fidelity** labels (cheap proxy + short-horizon high-fidelity) that
  **train the scorer**.
- Online: a new **target `(p*, t*)`** is scored, penalized by uncertainty (`q = μ̂ − λσ̂`), and
  turned into a **budgeted subset** via clustering.
- ⇒ "Utility is conditioned on the PEFT *and* the task, via a shared coordinate system."

**At 60 seconds (read annotations + edges) the reader should grasp:**
- **Why** the site space matters: PEFT code sets *which sites are touched + capacity weights*;
  sample and task contribute *per-site gradient signatures*; the low-fidelity proxy is a
  *site-weighted* cosine over exactly those sites (structural encodability).
- The **support-tier** mechanism (L0 near / L1 far / L2 new-family) with an optional calibration
  head handles unseen configurations.
- The **amortization** number: offline cost breaks even after **P* = 2.33** target configs; online
  is one scorer pass + clustering at 0.18 GPU-h.
- ⇒ "This is quality-preserving amortization: reusable PEFT-conditioned supervision replaces a
  per-target datastore."

---

## 11. One-line build constraints for the next round
- Full-text-width `figure*`, ≈ 2.05:1, offline ~62% / online ~35% / divider ~3%.
- Exactly one boundary-crossing edge (`reused`, purple, heavy).
- Purple used *only* for the scorer + reuse edge; dashed *only* for the optional `cal.` head.
- Labels verbatim from §6; no new terminology.
- Do **not** create `.drawio` or render this round — stop after the spec.
