# Abacum Outreach Scoring Dashboard

A dashboard prototype to help GTM teams prioritize prospect outreach. Scores companies based on industry fit, growth signals, tech stack overlap, and engagement—then suggests tailored talking points for each.

## What it does

- Scores prospects using weighted factors (adjustable via sliders)
- Shows why each prospect scored high or low
- Generates outreach angles based on industry (SaaS → ARR focus, Fintech → transaction drivers, etc.)
- Exports everything to CSV for sales sequences

## Data format

Your CSV needs these columns:

- `company_name`, `industry`, `stage`
- `industry_similarity`, `stage_score`, `growth_6mo_pct`
- `tech_overlap_score`, `engagement_score`
- `tech_stack` (optional - triggers integration angles)

Sample data included in `mock_prospects.csv`.

## Context

Built as a GTM engineering prototype for Abacum's RevOps team to show how I'd approach scoring and prioritizing outreach to support expansion efforts.