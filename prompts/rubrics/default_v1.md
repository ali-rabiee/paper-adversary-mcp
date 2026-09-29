---
name: default_v1
description: Generic rubric for a selective machine-learning venue. Replace or extend it with a venue-specific rubric (prompts/rubrics/<name>.md) when you have one.
---
Judge severity by how a careful expert reviewer at a selective machine-learning venue would weigh each problem.

Soundness. Claims must be supported by correct theory or adequate experiments. An error in a main theorem, an experiment that does not test the headline claim, leakage or contamination affecting the main results, or conclusions that rest on differences within noise usually leads to rejection. Proof gaps that are easily repaired, or secondary claims without support, are serious but fixable.

Novelty and significance. The contribution must be new relative to prior work the reviewers know, and must matter. Prior work that already does the core contribution is usually fatal. An overstated gap, a missing close baseline, or missing citations of closely related work draws strong criticism but can be fixed by repositioning, citing and comparing. Reviewers penalize "first" claims that turn out false more than modest, accurate claims.

Experimental evaluation. Baselines must be appropriate, current and fairly tuned; benchmarks must match the claims; results need enough seeds and uncertainty estimates to support the conclusions; ablations must isolate the proposed components. Unfair or missing strong baselines are among the most common reasons for rejection.

Reproducibility. Enough detail (hyperparameters, compute, data processing, evaluation protocol) for an expert to reproduce the main results; code and data availability or a stated reason for their absence.

Clarity. Precise problem statements and notation, claims that match the evidence, honest limitations. Poor clarity alone is rarely fatal but amplifies every other problem.

Limitations and impact. Limitations the reviewers can see but the paper does not acknowledge hurt credibility. Societal-impact and ethics issues must be addressed where relevant.

Calibration: FATAL means the paper would likely be rejected for this reason alone, and the problem cannot be fixed before the deadline. MAJOR BUT FIXABLE means it would weigh heavily in a rejection but the authors can address it with additional work or narrower claims. MINOR means it would be raised but would not decide the outcome.
