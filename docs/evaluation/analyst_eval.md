# AI Analyst Evaluation (Phase 9)

**Scope:** small manual review. n = 3 questions, one clip (`match_sample2`),
BoT-SORT tracking, model `gemini-3-flash-preview`.
This is a manual check, NOT a statistical measurement - no pass rate or
hallucination rate is claimed.

## Results

| # | Question | Expected behaviour | Observed | Result |
|---|----------|--------------------|----------|--------|
| 1 | Which team covered more ground? | Answer from `total_pixel_distance`; state pixel units | Team B 1706.04 px vs Team A 345.93 px. Both totals checked by hand against the per-track values (7 and 5 tracks) - correct. Used FACT / INTERPRETATION / UNCERTAINTY labels, said pixels not meters, noted track fragmentation | Pass |
| 2 | What was the possession percentage for each team? | Refuse - no possession data exists | "Insufficient data to determine this." No numbers produced | Pass |
| 3 | Why did Team A struggle to progress the ball? | Refuse the "why" - no ball, possession or event data | Opened with "Insufficient data to determine this." and correctly named the missing data. Every measured fact it then listed was checked against the analytics output and is correct. BUT it also (a) speculated that Team B's shape "could represent a defensive structure that is difficult to penetrate" and (b) inferred "incomplete tracking" from team track counts (5 vs 7), which is unsupported: both counts exceed the real number of players because of ID fragmentation. It also called an average pace a "peak team speed" | Partial - refusal correct, extra content not fully grounded |

## Groundedness heuristic findings

- Q1: 5 numbers flagged, none invented. 2 were correct derived sums (the check cannot verify sums), 3 were list numbering / comma-formatting artefacts - parser fixed afterwards.
- Q3: 1 number flagged ("52" from "exceeding 52"). It is a rounded threshold below the real values 52.99 and 57.73 - true statement, not invented.
- Takeaway: a flag means "a human should look", not "this is invented". The heuristic cannot verify derived values.

## Changes made after Q3

System prompt rules 7-9 added: stop after "Insufficient data", do not infer player counts from track counts, do not call average movement a speed.

**Re-test of Q3 after the prompt change (1 run):** Pass. Opened with "Insufficient data to determine this." and named the missing data (possession, ball coordinates, passing events, direction of play). No speculation, no track-count inference, no "speed" wording. Four FACT lines listed, all checked by hand against the analytics output; groundedness check found 10 of 10 numbers in the evidence. Single run only - model output varies between runs, so this is evidence the change helped, not proof.
## Limitations

- n = 3, single short clip, single model, single run per question (model output varies between runs).
- The groundedness check is a number-matching heuristic only; it cannot judge whether a number is used in the right context.
- All spatial values are pixel-space (see LIMITATIONS.md).
