## Tracking Fragmentation (Phase 6 follow-up)

Phase 6 analytics on match_sample2 produced 11 distinct track IDs
across 25 sampled frames, despite only 4-5 real players ever being
visible in the clip. Frame-count-per-ID is heavily skewed (one ID
covers 14 frames, several others cover only 1-3), consistent with
the single ID-switch example documented in Phase 3's interview prep,
now observed at scale across the whole clip.

**What this means:** per-player distance/pace metrics in
analytics.json should be read as "movement under one temporary
identity," not a player's true total movement for the clip - a real
player's actual total is likely split across multiple track_id rows.

**What would fix it:** re-running tracking with BoT-SORT
(`--tracker botsort.yaml`, already supported, no code changes
needed) to test whether appearance-based re-identification reduces
fragmentation versus ByteTrack's motion-only approach - a concrete,
testable next step, not yet done.


**Tested fix:** re-ran tracking with BoT-SORT (`--tracker botsort.yaml`)
instead of ByteTrack on the same clip. Result: did NOT improve
fragmentation - BoT-SORT produced 12 track IDs after team
classification vs. ByteTrack's 11, for the same 4-5 real players.
Likely cause: BoT-SORT's appearance-based re-identification depends
on visually distinguishing players, but the specific occlusion case
in this clip involves two similarly-colored kits - exactly where
appearance matching has the least signal to work with. This is a
genuine negative result, not an unexplored option.