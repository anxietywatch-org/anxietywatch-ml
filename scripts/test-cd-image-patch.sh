#!/usr/bin/env bash
# Regression test for scripts/cd-image-patch.sh (006-C-HOTFIX).
#
# Proves an image-only deploy never round-trips Container App application secrets:
# the generated PATCH body keeps the full revision template (env incl.
# ANXIETYWATCH_API_KEY=secretref:ml-api-key, non-secret env, resources, probes,
# volume mounts, Azure Files volume, scale), contains the new immutable image,
# and contains NO .properties.configuration and NO application secret definition.
#
# Run: sh scripts/test-cd-image-patch.sh   (bash/jq not required on the host;
# busybox sh + jq in a container works, matching the ubuntu-latest runner).
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
FIXTURE="${SCRIPT_DIR}/fixtures/containerapp-with-secret.json"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

NEW_IMAGE="ghcr.io/anxietywatch-org/anxietywatch-ml-api:0123456789abcdef0123456789abcdef0123456789"
PATCH="${WORK}/patch.json"

sh "${SCRIPT_DIR}/cd-image-patch.sh" "${NEW_IMAGE}" "${FIXTURE}" "${PATCH}"

fail() { echo "FAIL: $*" >&2; exit 1; }

# 1. Full revision template preserved.
jq -e '.properties.template != null' "${PATCH}" >/dev/null || fail "no .properties.template"
# 2. New immutable image present.
jq -e --arg img "${NEW_IMAGE}" '.properties.template.containers[0].image == $img' "${PATCH}" >/dev/null \
  || fail "image not swapped to ${NEW_IMAGE}"

# Env: API-key secretRef + existing non-secret vars preserved.
jq -e '.properties.template.containers[0].env[] | select(.name == "ANXIETYWATCH_API_KEY") | .secretRef == "ml-api-key"' "${PATCH}" >/dev/null \
  || fail "ANXIETYWATCH_API_KEY secretRef missing in patch"
jq -e '.properties.template.containers[0].env[] | select(.name == "ANXIETYWATCH_MODEL_PATH") | .value == "/app/models/prototype_v0.1.0.pkl"' "${PATCH}" >/dev/null \
  || fail "ANXIETYWATCH_MODEL_PATH not preserved"
jq -e '.properties.template.containers[0].env[] | select(.name == "ANXIETYWATCH_REQUIRE_MODEL") | .value == "true"' "${PATCH}" >/dev/null \
  || fail "ANXIETYWATCH_REQUIRE_MODEL not preserved"
jq -e '.properties.template.containers[0].env[] | select(.name == "PORT") | .value == "8000"' "${PATCH}" >/dev/null \
  || fail "PORT not preserved"

# Resources / probes / volume mounts / Azure Files volume / scale preserved.
jq -e '.properties.template.containers[0].resources.cpu == 0.5' "${PATCH}" >/dev/null || fail "cpu not preserved"
jq -e '.properties.template.containers[0].resources.memory == "1Gi"' "${PATCH}" >/dev/null || fail "memory not preserved"
jq -e '.properties.template.containers[0].probes | length == 3' "${PATCH}" >/dev/null || fail "probes not preserved"
jq -e '.properties.template.containers[0].volumeMounts[] | select(.volumeName == "ml-model-volume") | .mountPath == "/app/models"' "${PATCH}" >/dev/null \
  || fail "volume mount not preserved"
jq -e '.properties.template.volumes[] | select(.name == "ml-model-volume" and .storageType == "AzureFile")' "${PATCH}" >/dev/null \
  || fail "Azure Files volume not preserved"
jq -e '.properties.template.scale.minReplicas == 0 and .properties.template.scale.maxReplicas == 1' "${PATCH}" >/dev/null \
  || fail "scale not preserved"

# 3. NO .properties.configuration and NO application secret definition.
if jq -e '.properties.configuration != null' "${PATCH}" >/dev/null 2>&1; then
  fail "PATCH body contains .properties.configuration"
fi
if jq -e '.properties.configuration.secrets' "${PATCH}" >/dev/null 2>&1; then
  fail "PATCH body contains application secret definitions"
fi

# revisionSuffix removed so Azure can create a fresh revision.
if jq -e '.properties.template.revisionSuffix != null' "${PATCH}" >/dev/null 2>&1; then
  fail "PATCH body retains revisionSuffix"
fi

echo "PASS: image-only PATCH preserves full template, swaps image, excludes configuration/secrets."