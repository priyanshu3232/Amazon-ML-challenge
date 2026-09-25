#!/usr/bin/env bash
# Build <team_name>_submission.zip in the layout required by the challenge.
# Usage: ./make_submission_zip.sh <team_name>
set -euo pipefail
TEAM="${1:?team name required}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
STAGE="$ROOT/work/zip_stage"
rm -rf "$STAGE"; mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution/src"
cp "$ROOT/output/matching_results.tsv" "$ROOT/output/candidate_pairs.tsv" "$STAGE/output/"
cp "$ROOT/code/business_entity_resolution/src/"*.py "$STAGE/code/business_entity_resolution/src/"
cp "$ROOT/code/business_entity_resolution/README.md" "$ROOT/code/business_entity_resolution/requirements.txt" "$STAGE/code/business_entity_resolution/"
cp "$ROOT/Documentation_template.md" "$STAGE/"
( cd "$STAGE" && rm -f "$ROOT/${TEAM}_submission.zip" && zip -qr "$ROOT/${TEAM}_submission.zip" output code Documentation_template.md )
echo "wrote $ROOT/${TEAM}_submission.zip"; unzip -l "$ROOT/${TEAM}_submission.zip" | tail -n +1
