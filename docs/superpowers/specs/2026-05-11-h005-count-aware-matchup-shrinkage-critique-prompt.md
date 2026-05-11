# H005 External Critique Prompt

Use this prompt with another LLM reviewer before implementing the revised H005
design.

```text
Please critically review the document
`2026-05-11-h005-count-aware-matchup-shrinkage-design.md`.

Do not accept the design blindly. The design was revised after prior reviewers
objected that the original manual shrinkage formula was not supported by the
EBM evidence and that a global count denominator was position-biased.

Assess whether the revised H005 reliability-only design is ready for
implementation, needs further revision, or should be abandoned.

Focus on:

1. High-severity correctness issues.
2. Whether the Phase 0 mechanism audit is sufficient before feature work.
3. Whether position-normalized count reliability is actually supported by the
   EBM residual lead.
4. Whether the expected-count formula is cutoff-safe and statistically sound.
5. Whether keeping the raw matchup columns plus reliability columns makes the
   experiment too ambiguous.
6. Whether the decision statuses and acceptance gates are appropriate.
7. Whether a simpler or better alternative exists.
8. Implementation risks that could produce misleading results.

Return your assessment in this structure:

1. Critical Issues
2. Hidden Assumptions
3. Gaps And Missing Definitions
4. Statistical / Leakage Risks
5. Alternative Designs
6. Regression And Implementation Risks
7. Actionable Fixes, Prioritized
8. Final Verdict

For the final verdict, include:

- Clarity score from 1-10
- Correctness confidence from 1-10
- Production readiness from 1-10
- Risk level: Low, Medium, or High
- One of: approve, revise before implementation, abandon
```
