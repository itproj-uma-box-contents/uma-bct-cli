# Catalogue and human review

[README](../README.md) · [CLI guide](usage.md)

This guide describes current output. Confluence holds the
[catalogue design](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/7209023)
and [requirements](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/2424836)
(access may be required). CSV and Excel use the same catalogue rules; all generated
output remains a draft for human review.

Both CSV and Excel keep this column order:

```text
Unit Type, Unit Number, Description, Dates, Earliest Date, Latest Date,
Transcription Confidence (%), Review Flags, Cropped Box Image Link
```

## Unit Type and Unit Number

`Unit Type` contains the model's `typed` transcription: all printed label text in
reading order except the unit-number phrase. Names, headings, logos, `Creator:`
labels, and identifiers remain. Line breaks and repeated whitespace become single
spaces. `Unit Number` is one unambiguous number padded to four digits. Recognized
labels are `Unit`, `Box`, `Box No.`, `Box Number`, and `No.`; the total in `Box 7 of
9` is excluded. Missing, invalid, or conflicting numbers stay blank and flagged,
including conflicts with handwritten identity evidence.

| Printed label text | Unit Type | Unit Number |
| --- | --- | --- |
| `Example Collection` / `2008.0036` / `Unit 7` | `Example Collection 2008.0036` | `0007` |
| `Example Collection` / `107/56` / `Box 7 of 9` | `Example Collection 107/56` | `0007` |

Only the unit phrase is removed from a shared line: `1975.0064 Unit 8` leaves
`1975.0064`. A label containing only `Unit 8` has a blank, flagged `Unit Type`.
Recognized identifiers include `YYYY.NNNN`, `YY/N`, and `UMA-(AC|SR)E?-\d{8,9}`; other
printed identifiers remain too. Malformed dotted text such as `1960.009` remains
visible with `Ambiguous identifier` and `Transcription mismatch`. Multiple valid
identifiers remain visible and receive `Multiple identifiers`; printed origin headings
are not extra identifiers.

`Description` joins handwriting and explicit content or condition annotations with
semicolons, excluding identity-only text. `Dates` comes from handwriting; `Earliest
Date` and `Latest Date` give its bounds. Local regex is the default; `--ai-dates` uses
the selected local or hosted model. When no dates are found, `Dates` is `No Dates`,
with `None` for the bounds. Failed extraction keeps the transcription, adds
`Date extraction failed`, and makes the run partial. Check dates against the crop.

Short years without a supported full-year anchor, such as `12 May 50` or `05/90`,
are omitted from the date columns and flagged `Ambiguous date`. Clear dates on the
same label are kept. A range such as `2021-23` retains its explicit century. When
no clear dates remain, `Dates` is `No Dates` and its bounds are `None`. Ambiguity is
a review flag; it does not by itself make processing partial. New readings use this
rule; completed catalogues and cached readings are preserved.

Rows sort case-insensitively by complete `Unit Type`, then numeric `Unit Number`;
incomplete rows stay stable at the end. Separate physical boxes retain separate rows,
even with matching identities; those matches receive `Potential Duplicate`.

## Missing-box detection

Missing-box suggestions are enabled by default. Only internal gaps of at most 100
missing units between valid numbers in the same complete `Unit Type` group (ignoring
case) receive rows. Different headings, identifiers, or other label text create
different groups; nothing is inferred before the lowest or after the highest observed
number. Suggestions are not proof that physical boxes are missing.

For example, `Example Collection 2008.0036 / 0001` and the same type with `0003`
produce a `0002` row. It retains the type, has `Missing Box`, and leaves description,
confidence, and crop link blank. Use `--disable-missing-box-rows` to disable this.

## Confidence, crops, and verification

`Low confidence` means strictly below 50% by default; exactly 50% is not low. `-c`
changes this review threshold without removing rows. Unavailable token confidence
exports as `0` and requires review. Scores use the least probable content token;
they are not measured accuracy or directly comparable between model families.
`Transcription mismatch` includes unavailable confidence and other `needs_review`
outcomes. Review flags alone do not make processing partial; unresolved work does.

CSV crop references are readable relative paths, such as `Box Images/Shelf - Box
01.png`, without hyperlink formulas. Excel crop filenames are clickable in desktop
Excel and LibreOffice and open with the system's associated image app. Keep the
catalogue and crop folder together when moving results. If links fail, open the named
file in `Box Images`; Apple Numbers and Google Sheets cannot open these private local
links. Existing completed catalogues stay unchanged; use `--reprocess` for a new batch.

Verification is off by default. `--verification always` adds a blind second reading of
the same images with the same selected model, local or hosted. `--verification ask`
asks after the first pass: **y** verifies this run; Enter finishes. The checking
prompt does not see the first answer. Only substantive differences receive a third
request; detection is not repeated.

A correction requires that third reading to corroborate an earlier candidate. Printed
text, identifiers, and unit number are reconciled together, including handwritten
identity claims. Unresolved differences, whole or partial content deletions, and
ambiguous box matches keep the first reading with review flags. Agreement never raises
confidence or proves accuracy. Run details preserve all readings, and managed runs
checkpoint them for resume. A failed model request keeps the first reading and marks
partial processing; a checkpoint failure stops the run. Only `--verification` enables
it for new runs; `UMA_BCT_VERIFICATION` and `.uma-bct-settings.json` do not.
Unfinished batches retain their recorded choice; older batches without a choice
continue without verification. See [CLI recovery](usage.md#photos-and-results).
JSON, redirected and unattended runs never prompt; use `always` to verify there.

Before import, check every source photo for missed boxes or false detections, then
every row against its crop. Check flags and confidence, preserve separate physical
boxes, and review unflagged rows too. Accuracy gains require reviewed references, not
agreement between model answers.

Current limitations: transcription and reading order depend on the selected model.
`Carton`, `Container`, and `Package` are not unit labels; `Unit 8A`, ranges, unclear,
and crossed-out unit numbers are not ordinary accepted numbers. Only malformed dotted
standard-looking identifiers are explicitly detected; malformed UMA or slash values
may pass as arbitrary identifiers. The parser cannot distinguish incidental `Research
Unit 8` from a unit claim, so it removes `Unit 8`. Printed logos, headings, and
`Creator:` labels remain under the all-printed-text rule.
