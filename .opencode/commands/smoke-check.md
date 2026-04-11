Read `docs/ai/live-api-contracts.md` and `docs/ai/open-risks.md` first.

Task:
Run a repo smoke check focused on demo reliability.

Check for:
- frontend API URL assumptions
- Rust route expectations
- Python endpoint expectations
- known naming mismatches
- startup fragility
- obvious contract drift

Output:
1. critical blockers
2. likely breakpoints
3. quick fixes
4. validation commands to run next
