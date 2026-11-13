# Paper 2 finalization: "Restoring Native CT Evidence at Fixed Proposals for Lung Nodule Ranking"

**Status: expanded working draft** (main text ends on page 9; the CVPR 2027 kit and guidelines set an 8-page limit, see Open items). Built on the official CVPR 2027 author kit.

**Scope set by the authors:** no change to the scientific content; only figures plus minor structural and paragraph changes, applying the finalization brief where applicable.

## One-sentence contribution (working note)
Keeping the crop, network and training fixed and varying only the visible native CT shows that a thin band around a 10 mm ball recovers about 70% of the wider-context CPM gain at fixed LUNA16 proposals, mostly by revealing more of large or off-centre lesions rather than through sparse access or sample geometry.

## Claim → evidence map
| Claim | Evidence |
|---|---|
| Band recovers ~70% of the ball-to-full gap | Tab. 1, Fig. 3a; +4.03 [1.85, 6.17] (adjusted), +4.21 (seed 2) |
| Stable across halves, uncapped, fixed thresholds | Fig. 4 |
| Dense coverage beats native sources in the same network | Fig. 5; +2.29 [0.94, 4.13] |
| Direction-aware attention adds ~nothing | Fig. 6a, Tab. 2 (seed-2 heads caveat) |
| Gains concentrate where the band reaches the lesion | Fig. 7, Tab. 3; +15.3/+16.7 vs +2.1/+1.1 at 1 FP/scan |
| Transfers to a single-stream 3D network | Tab. 1, Fig. 3a; +2.91 [0.69, 5.48] |

## Rounds (within the content-preserving scope)
1. **Figures.** Fig. 5 was reduced to its effects panel; its two-number bar panel now lives in the caption and text. Fig. 2 was compacted (about 0.3 in shorter, with the panel (b) note lifted clear of the divider). Every results caption now states its takeaway and what the intervals are computed over (2,000 patient-level bootstrap resamples of the 353 scans). Fig. 3b uses a compact legend instead of overlapping inline labels; the 5 mm scale bars are on a dark backing.
2. **Structure and paragraphs.** The abstract explains LUNA16 and the metric in plain words. The intro adds an explicit comparison with the closest approaches (multi-scale/nested context; occlusion of trained networks) and closes with "Our paper offers the following contributions." using short bold labels in connected sentences. Related work is now two connected paragraphs (no mini-headings). The discussion is connected paragraphs (no bold lead-ins). The conclusion now also names the main limitation and the next steps.
3. **Checks and narrow corrections from the final review.**
   - §5.6 now reports the seed-2 embedding contrast (−0.06) and states that the unpaired contrast is −0.31 for both seeds.
   - The morphology comparison in §5.6 is defined.
   - The three adjusted contrasts are named and mapped to Fig. 6b.
   - The confirmation/replication halves are defined in §4.
   - "Distant anatomy" → "surrounding anatomy within the crop", since the full field is only a ±16 mm crop.

Verified: every page rendered and inspected; no undefined references or overfull boxes; figures embed only Times-family fonts; the PDF Title/Author metadata is empty and review mode prints "Anonymous CVPR submission"; there are no identifying links in the PDFs; the BioRender attribution is in the Fig. 2 caption. "Exactly one matching proposal per never-reached lesion" was verified in the aggregate (377 pairs for 377 lesions).

## Independent final review (fresh agent, 0–10 internal rubric; approximate)
Originality/significance 5 · Soundness 7 · Evidence 6 · Clarity 7 · Presentation 8 · **Overall 5.5 (borderline), medium confidence.** It recomputed the reported arithmetic and found no inconsistencies.

Remaining objections that would change the content and are therefore left to the authors:
- **Baseline framing.** The 10 mm ball truncates nodules ≥10 mm, so part of the gain is undoing truncation. Say up front why the ball is the right baseline.
- **Scope of "wider context".** The full field is a 32.4 mm crop, so frame the conclusions as context within ±16 mm.
- **Two-scale field of view.** The central 16³ stream spans 19.2 mm, which is less than the 20 mm ball, so the band enters only through the pooled stream. Worth one sentence of discussion.
- **Decisive missing experiment.** Train a band condition with the contoured lesion voxels set to air ("band minus lesion"), and/or stratify by size within lesions ≥10 mm (70 reached vs. 35 not). This would turn the lesion-vs-surroundings conclusion from correlational into causal.
- **Coverage vs. continuity.** An equal-volume scattered control, plus extra seeds for the native-source and single-stream conditions.
- **Thick slices.** For d ≈ 2.7 mm the band's outer edge (≈16.9 mm) exceeds the crop half-width (16.2 mm). State how the 16,672-voxel count treats this.
- **To confirm.** Ball-only S₁ is 85.68% for both seeds (and the ball-only group sensitivities are identical). This is what the aggregate contains; confirm it against the raw score exports.
- **Earlier open items still stand:**
  - three seed-2 attention heads retained their first checkpoint (rerun);
  - verify the 14 added references;
  - check the LLM statement against the CVPR policy;
  - make the GitHub repo private;
  - decide on the 8-page limit.
