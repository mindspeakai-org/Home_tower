# Recorded evidence and local artifacts

Committed reports preserve the completed furniture audit, original strict
baseline, pending clean-evaluation status, CPU measurements, and reproducibility
scripts. The cached test predictions and inventory support recreating review
packs without repeating inference over the full dataset. They are unreviewed
source-derived metadata, not approved training labels.

Images, video, runtime logs, per-camera proposals, and individual annotation
review decisions remain local. Archived reports refer to these generated
artifacts; the published copy renders omitted artifact links as file paths.
The absence of those files from Git does not mean they were manually reviewed.

See `docs/DEVELOPMENT.md` for rebuilding local fixtures from the separately
provided dataset and existing model weights, and `docs/STATUS.md` for completed
and pending work. Source attribution is in `THIRD_PARTY_NOTICES.md`.
