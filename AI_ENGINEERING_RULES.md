# FootballIQ — AI Engineering Rules

Working reference for how this project is engineered. Not required in the public README, but kept in-repo as the standing rules for development decisions.

## Core Principles

Always prioritize:

1. Working implementation over theoretical complexity.
2. Understanding over buzzwords.
3. Measurable results over unsupported claims.
4. Simplicity before complexity.
5. Modular architecture.
6. Reproducibility.
7. Interview explainability.
8. Production-quality engineering where practical.

## AI/ML Rules

Never fabricate:

* Model accuracy.
* Dataset statistics.
* Football statistics.
* Detection results.
* Tracking results.
* Benchmark results.

If something is estimated, label it as an estimate.
If something is experimental, label it as experimental.
If something cannot currently be calculated reliably, say so.

## LLM Rules

The LLM component must be grounded in structured match data.

If evidence is insufficient: "Insufficient data to determine this."

Never invent a statistic simply to produce a more convincing answer.

Clearly distinguish FACT from INTERPRETATION from UNCERTAINTY.

## Technology Selection

Before adding any technology, ask:

* Is it necessary?
* Is it maintained?
* Is it compatible with the rest of the project?
* Is it realistic for a student?
* Can I explain it in an interview?
* Does it materially improve the system?

## Project Scope

MVP priority:

1. Video processing
2. Player detection
3. Player tracking
4. Team classification
5. Pitch mapping
6. Basic football analytics
7. Dashboard
8. Grounded LLM analysis

Advanced features (RAG, player similarity, automated scouting, live analysis,
advanced event detection, multi-match analysis) only after the MVP works reliably.
