<# 
.SYNOPSIS
    Apply branch protection rulesets to anxietywatch-ml repo using GitHub Rulesets API.

.DESCRIPTION
    Idempotent: creates or updates rulesets by name for main and develop branches.
    Requires gh CLI authenticated with admin:repo_hook + repo scope.

.USAGE
    .\apply-branch-protection.ps1

.NOTES
    Follows anxietywatch-backend convention (script at repo root).
#>

param()

$ErrorActionPreference = "Stop"

$ORG = "anxietywatch-org"
$REPO = "anxietywatch-ml"
$RULESET_MAIN = "Protect main branch"
$RULESET_DEVELOP = "Protect develop branch"
$ML_CI_JOB = "quality-and-test"  # Exact GitHub Actions job name

function Upsert-Ruleset {
    param (
        [string]$Name,
        [string]$Branch,
        [string]$Context
    )

    Write-Host "Processing ruleset: $Name (refs/heads/$Branch)..." -ForegroundColor Cyan

    $existingId = gh api "/repos/$ORG/$REPO/rulesets" --jq ".[] | select(.name==\"$Name\") | .id" 2>$null

    $payload = @{
        name = $Name
        target = "branch"
        enforcement = "active"
        conditions = @{
            ref_name = @{
                include = @("refs/heads/$Branch")
                exclude = @()
            }
        }
        rules = @(
            @{
                type = "pull_request"
                parameters = @{
                    required_approving_review_count = 1
                    dismiss_stale_reviews_on_push = $true
                    require_code_owner_review = $false
                    require_last_push_approval = $false
                    required_review_thread_resolution = $true
                    allowed_merge_methods = @("merge", "squash", "rebase")
                }
            },
            @{
                type = "required_status_checks"
                parameters = @{
                    strict_required_status_checks_policy = $true
                    do_not_enforce_on_create = $false
                    required_status_checks = @(
                        @{ context = $Context }
                    )
                }
            },
            @{
                type = "non_fast_forward"
                parameters = @{}
            },
            @{
                type = "deletion"
                parameters = @{}
            }
        )
    }

    $json = $payload | ConvertTo-Json -Depth 8 -Compress
    $tempFile = [System.IO.Path]::GetTempFileName()
    [System.IO.File]::WriteAllText($tempFile, $json)

    try {
        if ($existingId) {
            gh api --method PUT "/repos/$ORG/$REPO/rulesets/$existingId" --input $tempFile | Out-Null
            Write-Host "  ✓ Updated existing ruleset ($existingId)" -ForegroundColor Green
        }
        else {
            gh api --method POST "/repos/$ORG/$REPO/rulesets" --input $tempFile | Out-Null
            Write-Host "  ✓ Created new ruleset" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "  ✗ Error: $($_.Exception.Message)" -ForegroundColor Red
        throw
    }
    finally {
        Remove-Item -Force $tempFile -ErrorAction SilentlyContinue
    }
}

Write-Host "Applying branch protection rulesets to $ORG/$REPO..." -ForegroundColor Cyan

Upsert-Ruleset $RULESET_MAIN "main" $ML_CI_JOB
Upsert-Ruleset $RULESET_DEVELOP "develop" $ML_CI_JOB

Write-Host "✓ Branch protection applied to main and develop" -ForegroundColor Green
Write-Host "Required status check: $ML_CI_JOB" -ForegroundColor Cyan