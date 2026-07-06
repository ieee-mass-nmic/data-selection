# 3.5 Unverified empirical assumptions

**Track: A** (justification prose + Fig 1 relabel; no authorized reruns)

## Reviewer concern

1. 24 intervention sites are coarse (module-output / MLP-output / block-residual)
   and may drop module-/direction-level info; need justification that
   module-output gradient represents projection-level PEFT update.
2. Eq. (3) capacity weighting (`tanh(eta log(1+cap/dmodel))`, operator coeff
   `rho_op`, mask/capacity defs) is ad hoc; needs motivation + sensitivity.
3. Short-horizon sketch-loss-reduction supervision needs validation: a figure of
   short-horizon utility vs full-FT downstream marginal gain (Spearman/NDCG).
4. Figure 1 same-PEFT Top-5% overlap 0.31 called "high agreement" is a stretch;
   relabel and discuss label noise.

## Decisions

- (1)(2) -> **Track A** prose (justification + motivation). Optional
  capacity-weighting sensitivity sweep is NOT authorized -> present as motivated
  design + limitation, no new sweep.
- (3) correlation figure -> **NOT authorized** (needs per-sample full-FT gains,
  expensive). Handle by: justify the short-horizon proxy in prose and keep the
  proxy-bias limitation explicit (already partially at
  `paper/Sections/06_analysis.tex:34`). Claim no more than measured.
- (4) Fig 1 relabel -> **Track A**, do early (cheap, high value).

## Planned changes

1. **Site-mapping justification paragraph** in `paper/Sections/04_method.tex`:
   why module-output gradient is an adequate surrogate for projection-level
   updates; acknowledge the information lost as a scoped approximation.
2. **Eq. (3) motivation** rewrite: present capacity weighting as motivated
   (monotone in capacity, saturating) rather than bare feature engineering;
   move unproven specifics to a limitation.
3. **Fig 1 relabel:** "high agreement" -> "moderate but substantially higher than
   cross-PEFT overlap" + a label-noise sentence. Recompute 0.31 / ~0.05 from
   `result/data/motivation/values.parquet` before rewording to confirm numbers.
4. **Keep short-horizon-proxy limitation** explicit; add a sentence that
   downstream-correlation validation is future work (no figure claimed).

## Data sources / files

- Read: `result/data/motivation/values.parquet`,
  `src/pcu_select/proxy/*`, `peft_space/*`.
- Write: `paper/Sections/04_method.tex`, `06_analysis.tex`, Fig 1 caption.

## Reruns needed

No (only recompute overlap stats from existing parquet).

## Acceptance criteria

- Fig 1 wording matches measured overlap; label noise acknowledged.
- Site mapping and capacity weighting are motivated; unproven parts are scoped
  as limitations, not asserted.
