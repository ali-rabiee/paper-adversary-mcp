---
role: supplement
version: 1
description: Coverage supplement. A judge or adjudicator rules on exactly the objections (or items) its report left unclassified or gave two different severities, by the standards of its own role prompt.
---
# Coverage supplement requested by the orchestrator

You already wrote your report in this review; it is below, under <your_report>. The orchestrator compared it with the {{noun}} you were assigned, and these still need a ruling from you:

{{ids}}

Rule on exactly these {{noun}}, nothing else, by the same standards, severity classes and evidence rules as in your instructions. Only the reports that raised them are shown, with the orchestrator's checks of them; read them, and the paper where it matters. Do not repeat or revise rulings you were not asked about. Where you gave one ID two severities, decide which holds and say in one line why. Finding one not convincing is a ruling too (NOT CONVINCING).

Output: for each ruling, a heading `### <SEVERITY>: <short title> (<IDs>)` and a short justification; then one fenced JSON block of exactly this shape, containing only rulings on the IDs listed above (other top-level fields may be empty lists):

```json
{{shape}}
```
