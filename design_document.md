# Design Document: French Decimal Comma Resolution

## Problem Summary

After datetime normalization, 27,275 rows (1.68%) remain malformed due to French decimal commas in numeric fields. These extra commas shift columns, making all downstream analysis incorrect.

---

## Algorithm

**Step 1 — Fix datetime commas** (existing, unchanged)

Apply the anchored regex `(DD/MM/YY HH:MM:SS),(NNNNNNNNN)` → replace comma with dot. This is safe because the pattern is fully anchored to datetime syntax.

**Step 2 — Count fields after datetime normalization**

Split the fixed line by `,`. Count the fields.

- **23 fields** → clean. Write as-is.
- **24 fields** → exactly one French decimal comma. Identify and merge.
- **25 fields** → exactly two French decimal commas. Merge both.
- **Any other count** → unknown corruption. Write as-is (let `on_bad_lines` handle it downstream).

**Step 3 — Identify split position**

For 24 fields, the extra field is at one of two candidate positions:

| Candidate | Split occurs between | Detection rule |
|-----------|---------------------|----------------|
| AMOUNT area | field[4] (integer part) and field[5] (decimal part) | `field[5]` matches `^\d{1,3}$` |
| FEES area | field[14] (integer part) and field[15] (decimal part) | `field[15]` matches `^\d{1,3}$` |

Check AMOUNT first. If no match, check FEES. If neither matches → leave as-is.

For 25 fields, apply both merges. After merging AMOUNT (deleting field[5]), indices shift: FEES is now at field[14] and field[15]. Merge FEES second.

**Step 4 — Merge and write**

Replace `field[N]` with `field[N].field[N+1]`, delete `field[N+1]`, rejoin with `,`, write.

---

## Why This Is Safe

**1. The datetime regex cannot produce false positives.** It requires the full `DD/MM/YY HH:MM:SS` pattern before the comma. No legitimate CSV delimiter would match this.

**2. Field count is the primary guard.** We only act when we have exactly 24 or 25 fields — a condition that can only arise from a French decimal comma in this dataset. A clean row (23 fields) is never touched.

**3. The numeric regex is narrow.** `^\d{1,3}$` matches only 1–3 unquoted digits. This prevents false positives on:
- Quoted strings (`"P"`, `"T"`, `""`)
- Empty fields (from `,,`)
- Long numeric strings (transaction IDs, phone numbers)
- Phone numbers with `TEL` prefix

**4. Candidate positions are fixed.** AMOUNT and FEES are the only fields that can contain French decimals in this dataset (per data documentation). We never touch other fields.

**5. The algorithm is deterministic.** For any given line, the output is always the same. No ambiguity, no randomness.

---

## Why It Cannot Corrupt Valid CSV Rows

| Scenario | What happens | Correct? |
|----------|-------------|----------|
| Clean row (23 fields) | Passed through untouched | Yes |
| Row with datetime comma only (23 fields after step 1) | Passed through untouched | Yes |
| Row with French decimal in AMOUNT (24 fields) | field[5] is `^\d{1,3}$` → merged | Yes |
| Row with French decimal in FEES (24 fields) | field[5] is not numeric → field[15] is `^\d{1,3}$` → merged | Yes |
| Row with both (25 fields) | Both merged | Yes |
| Row with other corruption (>25 fields) | Passed through untouched | Yes |
| Row with quoted field containing digits (`"P"`) | Not matched by `^\d{1,3}$` (quotes present) | Yes |
| Row with empty field (`,,`) | Not matched by `^\d{1,3}$` (empty string) | Yes |

---

## Computational Complexity

- **Time**: O(n) where n = 1,627,757 rows. Each row: regex sub (O(1)) + split (O(1)) + conditional merge (O(1)). Total: ~1.6M operations.
- **Space**: O(1) per row (streaming line-by-line). No row materialization.
- **I/O**: Two passes over the file (read raw, write normalized). Same as existing.

The normalized CSV is cached on disk (`DATASET_ESP-2026.normalized.csv`), so subsequent calls skip normalization entirely.

---

## Remaining Edge Cases

| Edge case | Handling |
|-----------|----------|
| AMOUNT = 0.5 (`0,5`) | field[4]="0", field[5]="5" → merged to `0.5` ✓ |
| FEES = 0.5 (`0,5`) | field[14]="0", field[15]="5" → merged to `0.5` ✓ |
| AMOUNT = 9900.99 (`9900,99`) | field[4]="9900", field[5]="99" → merged to `9900.99` ✓ |
| FEES = 495.05 (`495,05`) | field[14]="495", field[15]="05" → merged to `495.05` ✓ |
| 25-field row with `2,2` and `0,01` | Both merged: AMOUNT=2.2, FEES=0.01 ✓ |
| Row with 26+ fields | Unknown corruption → passed through untouched ✓ |
| Empty FEES (`""`) with extra comma | field[15] is not `^\d{1,3}$` → not merged by FEES rule; if field[5] is numeric, AMOUNT is merged instead ✓ |

---

## Files to Modify

| File | Change | Rationale |
|------|--------|-----------|
| `src/data/preprocessing.py` | Extend `_build_normalized_csv()` with step 2–4 above (lines ~226–249) | Core fix: single source of truth |
| `tests/test_preprocessing.py` | Add test cases for French decimal comma patterns (A, B, C) | Verify correctness of new logic |
| `notebooks/01_EDA.ipynb` | Replace raw `pd.read_csv()` with `preprocess()` call | Eliminate duplicate parsing; EDA uses same cleaned data as feature engineering |

---

## What We Will NOT Change

- The preprocessing API (`preprocess()`, `load_raw()`) — no signature changes
- `_parse_datetime_series()` — already correct
- `_parse_numerics()` — already strips commas from correctly-aligned columns
- `_validate_raw_csv()` — validation logic unchanged
- `feature_engineering.py` — no changes needed
- Tests for feature engineering — not affected
