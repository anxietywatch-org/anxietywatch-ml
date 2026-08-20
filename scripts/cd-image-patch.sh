#!/usr/bin/env bash
# Build a minimal image-only PATCH body for a Container App deploy (006-C-HOTFIX).
#
# Used by .github/workflows/publish-container.yml and exercised by
# scripts/test-cd-image-patch.sh against a sanitized fixture, so the CI path and
# the regression test share the exact transformation.
#
# Why template-only:
# The previous deploy GET the whole Container App, sanitized it and PATCHed the
# broad body back. The GET representation lists application secrets as bare
# metadata ({"name": "ml-api-key"}) WITHOUT a value or keyVaultUrl+identity, so
# round-tripping .properties.configuration made Azure reject the PATCH with
# ContainerAppSecretInvalid. The application secret itself was valid; the PATCH
# was the bug. We therefore emit ONLY .properties.template (the complete,
# unchanged revision template with exactly the image swapped) as a JSON Merge
# Patch; .properties.configuration - including secrets - is never sent and
# remains untouched in Azure.
set -euo pipefail

IMAGE_REF="${1:?usage: cd-image-patch.sh <image> <app-json> <out-patch-json>}"
APP_JSON="${2:?missing app-json path}"
OUT_PATCH="${3:?missing output patch path}"

jq '
  { properties: { template: .properties.template } } |
  .properties.template.containers[0].image = $img |
  del(.properties.template.revisionSuffix)
' --arg img "${IMAGE_REF}" "${APP_JSON}" > "${OUT_PATCH}"

# Defensive assertions: the PATCH must never carry configuration or secrets.
jq -e '.properties.template != null' "${OUT_PATCH}" >/dev/null || {
  echo "::error::PATCH body has no .properties.template" >&2
  exit 1
}
jq -e --arg img "${IMAGE_REF}" '.properties.template.containers[0].image == $img' "${OUT_PATCH}" >/dev/null || {
  echo "::error::PATCH body image does not match ${IMAGE_REF}" >&2
  exit 1
}
jq -e '.properties.configuration == null' "${OUT_PATCH}" >/dev/null || {
  echo "::error::PATCH body must NOT contain .properties.configuration (would round-trip secrets)" >&2
  exit 1
}
jq -e '.properties.template.containers[0].env[] | select(.name == "ANXIETYWATCH_API_KEY") | .secretRef == "ml-api-key"' "${OUT_PATCH}" >/dev/null || {
  echo "::error::ANXIETYWATCH_API_KEY must survive as secretRef ml-api-key (never an inline value)" >&2
  exit 1
}

echo "Validated template-only PATCH: image=${IMAGE_REF}, configuration excluded, secretRef preserved."