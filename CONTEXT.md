# Skipper AI Assistant — Project Context & Specifications

## 1. Project Overview

**Project name**: Skipper AI Assistant  
**Type**: Personal project / Proof of Concept (POC)  
**Domain**: Sailing performance analysis, explainable AI  
**Target users**: Skippers, performance engineers, sailing analysts  
**Execution mode**: Offline, local-only (no cloud dependency)

The Skipper AI Assistant is a post-navigation analysis tool designed to explain sailing performance losses using interpretable machine learning and natural language explanations.

The goal is NOT to predict race outcomes or optimize routing in real time, but to help understand *why* performance was lost during a navigation.

---

## 2. Core Philosophy

- Offline-first (designed for onboard or secure environments)
- Explainability over raw accuracy
- Decision support, not automation
- Realistic assumptions about limited and imperfect data
- Separation of concerns:
  - ML model → performance estimation
  - Explainability → SHAP
  - LLM → explanation and synthesis only

---

## 3. Target Boat Class

- Generic **Class40**
- No specific real-world boat
- Performance modeled using approximate or public polar data

---

## 4. Input Data

### 4.1 Accepted Formats
- CSV (primary)
- GPX / NMEA (converted to CSV upstream)

### 4.2 Minimum Required Fields

Each row represents a fixed time step (1–2 seconds).

| Column name | Type | Description |
|------------|------|-------------|
| timestamp | int | Unix timestamp |
| tws | float | True Wind Speed (knots) |
| twa | float | True Wind Angle (degrees) |
| boat_speed | float | Speed Over Water (knots) |
| heading | float | Boat heading (degrees) |
| heel | float | Heel angle (degrees, real or simulated) |
| sail_mode | categorical | Sail configuration (e.g. J1, J2, A3) |

### 4.3 Derived Features
- VMG
- Point of sail (upwind / reach / downwind)
- Expected speed from polar
- Performance ratio

---

## 5. Performance Reference Model

### 5.1 Polar-Based Expectation

Expected performance is computed using a simplified polar model:

```

expected_speed = polar_speed(tws, twa)

```

### 5.2 Performance Metric

```

performance_ratio = boat_speed / expected_speed

```

- < 1.0 → underperformance
- ≈ 1.0 → nominal
- > 1.0 → overperformance (possible, not an error)

This ratio is the **primary target variable**.

---

## 6. Machine Learning Model

- Type: Regression
- Model: XGBoost Regressor
- Target: performance_ratio
- Input: all numeric + encoded categorical features
- Training data:
  - Combination of public real sailing tracks
  - Synthetic data with injected performance losses

The model is not expected to generalize to all boats or conditions.
It is expected to support *relative explanations*.

---

## 7. Explainability

- Method: SHAP (TreeExplainer)
- Scope:
  - Local explanations (per maneuver / segment)
  - Aggregated explanations (entire run)

Explainability output must:
- Rank performance loss contributors
- Quantify relative impact
- Be robust to noisy data

---

## 8. Role of the LLM

### 8.1 Constraints

The LLM:
- Does NOT see raw sensor data
- Does NOT perform numerical computation
- Does NOT infer physics

### 8.2 Inputs to the LLM

- Top SHAP contributors
- Simple statistics (averages, deltas)
- Context metadata (run name, point of sail)

### 8.3 Outputs

- Natural language diagnostic
- Human-readable explanation
- Actionable recommendations (qualitative)

Example prompt intent:
> "Explain why performance was lost upwind during this run."

---

## 9. CLI Interface

The project exposes a command-line interface.

### Core Commands

```

skipper-ai ingest <file.csv>
skipper-ai analyze <run_id>
skipper-ai explain <run_id>
skipper-ai ask "<natural language question>"

```

The CLI must:
- Be deterministic
- Be scriptable
- Run entirely on macOS
- Not require internet access

---

## 10. Output Examples

### Performance Summary
- Total time under polar (%)
- Average performance ratio
- Main contributing factors

### Explanation Output
- Ranked causes (SHAP-based)
- Clear language
- No ML jargon in final user-facing text

---

## 11. Explicit Non-Goals

The project does NOT aim to:
- Perform real-time navigation
- Replace routing software
- Predict race results
- Provide certified performance metrics
- Compete with professional proprietary tools

---

## 12. Evaluation Criteria

The project is considered successful if:
- Injected synthetic performance losses are recovered by SHAP
- Explanations are understandable by non-technical users
- The full pipeline runs locally and reproducibly
- Limitations are clearly documented

---

## 13. Technology Stack

- Language: Python
- OS: macOS
- ML: XGBoost
- Explainability: SHAP
- LLM: Local model (e.g. Mistral / LLaMA)
- Interface: CLI

---

## 14. Project Positioning Statement

> The Skipper AI Assistant is a proof-of-concept offline tool for explainable sailing performance analysis, designed to bridge the gap between raw telemetry and actionable insight for skippers and performance engineers.
