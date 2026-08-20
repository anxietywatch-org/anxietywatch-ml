#!/usr/bin/env bash
# Script to apply branch protection rulesets to anxietywatch-ml repo
# Uses GitHub Rulesets API (preferred over classic branch protection)
# Idempotent: creates or updates rulesets by name
#
# Usage: chmod +x apply-branch-protection.sh && ./apply-branch-protection.sh
# Requires: gh CLI authenticated with admin:repo_hook + repo scope

set -euo pipefail

ORG="anxietywatch-org"
REPO="anxietywatch-ml"
RULESET_MAIN="Protect main branch"
RULESET_DEVELOP="Protect develop branch"

# ML CI job name (exact as it appears in GitHub Actions)
ML_CI_JOB="quality-and-test"

echo "Applying branch protection rulesets to $ORG/$REPO..."

# Helper: upsert ruleset
upsert_ruleset() {
  local name="$1"
  local branch="$2"
  local context="$3"

  echo "Processing ruleset: $name (refs/heads/$branch)..."

  local existing_id
  existing_id=$(gh api "/repos/$ORG/$REPO/rulesets" --jq ".[] | select(.name==\"$name\") | .id" 2>/dev/null || true)

  local payload
  payload=$(cat <<EOF
{
  "name": "$name",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["refs/heads/$branch"],
      "exclude": []
    }
  },
  "rules": [
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true,
        "allowed_merge_methods": ["merge", "squash", "rebase"]
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          { "context": "$context" }
        ]
      }
    },
    {
      "type": "non_fast_forward",
      "parameters": {}
    },
    {
      "type": "deletion",
      "parameters": {}
    }
  ]
}
EOF
)

  local temp_file
  temp_file=$(mktemp)
  echo "$payload" > "$temp_file"

  if [[ -n "$existing_id" ]]; then
    gh api --method PUT "/repos/$ORG/$REPO/rulesets/$existing_id" --input "$temp_file" >/dev/null
    echo "  ✓ Updated existing ruleset ($existing_id)"
  else
    gh api --method POST "/repos/$ORG/$REPO/rulesets" --input "$temp_file" >/dev/null
    echo "  ✓ Created new ruleset"
  fi

  rm -f "$temp_file"
}

# Apply protection to main
upsert_ruleset "$RULESET_MAIN" "main" "$ML_CI_JOB"

# Apply protection to develop
upsert_ruleset "$RULESET_DEVELOP" "develop" "$ML_CI_JOB"

echo "✓ Branch protection applied to main and develop"
echo "Required status check: $ML_CI_JOB"