# SurgVU Cat-2 FINAL phase — scoring failure (draft email for the USER to send)

**To:** the SurgVU 2026 organizers (their preferred contact / GC forum)
**Subject:** Category 2 Final Phase — "PredictionProcessingError: scoring failed" on a submission whose algorithm ran successfully

Dear organizers,

Our team (AIMS, user `omarchoudhry`) submitted to the **Category 2 – Final Phase** on 26 Aug 2026 at 10:00.
The submission reports:

> `helpers.PredictionProcessingError: One or more errors occurred during prediction processing.`
> `Your algorithm ran successfully, but the scoring failed. If you would like more information, please contact the challenge organisers.`

Submission: `bf65a0e3-254f-4410-89db-9a8f66968fb4`
Algorithm: `aims-surgvu26-cat-2-deterministic-answerer`, active image `9b92047d-c495-4827-b211-de6a496f3448`

Three things suggest the issue is specific to Category 2 final-phase scoring rather than to our algorithm:

1. **The identical algorithm image succeeded on the Category 2 Prelim phase** on 21 Aug 2026 (status `Succeeded`).
2. Your own message states the algorithm ran successfully and that the failure occurred during scoring.
3. **Our Category 1 Final Phase submission, made the same morning, completed with status `Succeeded`** — so the
   final-phase pipeline as a whole is working; the failure appears isolated to Category 2 scoring.

We have also checked our answer generation against degenerate inputs (empty string, whitespace, punctuation-only,
very long strings, non-ASCII characters) and it returns a non-empty short declarative sentence in every case, so we
do not believe we are emitting an empty or malformed answer.

Could you let us know whether this is a known issue with the final-phase evaluation, or whether there is something we
should change in our output? We have a limited number of submissions for this phase and would rather not spend them
re-testing blindly.

Thank you,
Omar Choudhry, AIMS — University of Leeds
