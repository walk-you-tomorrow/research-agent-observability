#!/usr/bin/env bash
# ObservabilityPowers/observable-research-agent → research-agent-observability 동기화

set -e

SRC="/Users/sung-a.generouspark/1Billion/ObservabilityPowers/observable-research-agent/"
DST="/Users/sung-a.generouspark/research-agent-observability/"

echo "▶ rsync 시작..."
rsync -av \
  --exclude='lightrag_storage/' \
  --exclude='logs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.venv/' \
  --exclude='venv*/' \
  --exclude='.env' \
  --exclude='lightrag.log' \
  --exclude='inputs/' \
  --exclude='docs/' \
  --exclude='CLAUDE.md' \
  --exclude='=0.11' \
  --exclude='=3.8' \
  --exclude='monitoring_schema_v2_backup.yaml' \
  --exclude='.claude/' \
  --exclude='.overstory/' \
  --exclude='.seeds/' \
  --exclude='.gitattributes' \
  --exclude='sync.sh' \
  "$SRC" "$DST"

cd "$DST"

if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
  echo "✓ 변경사항 없음, 스킵"
  exit 0
fi

DATE=$(date +%Y-%m-%d)
git add .
git commit -m "sync: update from ObservabilityPowers ($DATE)"
git push origin main
echo "✓ push 완료"
