# Prompting Notes

This repo is being operated with a fast model that may need extra grounding.

## Prompt style rules

1. Give the model one bounded task at a time.
2. Name the exact files likely involved.
3. State what must not change.
4. Ask for a short plan before edits on non-trivial tasks.
5. Ask for validation commands and expected outcomes.
6. Ask it to update docs when changing contracts or behavior.

## Good prompt pattern

Use this structure:

Task:
- one concrete goal

Context:
- what behavior exists now
- what is wrong now
- which docs to read first

Constraints:
- what must not change
- what style to preserve
- what not to refactor

Files to inspect first:
- exact paths

Required output:
- short plan
- implementation
- validation
- changed files
- open risks

## Example constraints for this repo

- preserve Obsidian HUD style
- do not break MCI mode
- do not rename request fields casually
- do not claim hybrid RAG is live unless wired into live endpoints
- avoid changing training scripts unless task explicitly targets offline pipeline

## Escalation rule

Use the faster model for:
- bounded implementation
- doc updates
- small bug fixes
- UI polish
- route wiring

Use the stronger model for:
- cross-stack refactors
- retrieval redesign
- complex debugging
- architecture changes
- anything touching both demo reliability and data correctness
