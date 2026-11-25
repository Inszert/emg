<#
.SYNOPSIS
  Push the current branch to EVERY configured git remote (e.g. two gits).

.EXAMPLE
  .\push_all.ps1                       # push current branch to all remotes
  .\push_all.ps1 -Setup gitlab https://gitlab.com/user/emg.git   # add 2nd remote once
  .\push_all.ps1 -Remotes origin,gitlab   # only these
  .\push_all.ps1 -Tags                 # also push tags
  .\push_all.ps1 -DryRun               # show what would happen

  Remotes listed in -Snapshot (default: gitlab) have a different history, so
  they get ONE new commit on top of their main that replaces only src/ and
  doc/ with the current tree. Their other root files (README.md, ...) are left alone.

  Never force-pushes. If one remote rejects (history differs), the others are
  still pushed and the script exits non-zero with a summary.
#>
param(
    [string]$Setup,            # name of a remote to add ...
    [string]$Url,              # ... with this URL (can also be 2nd positional)
    [string[]]$Remotes,        # default: all remotes
    [string[]]$Snapshot = @("gitlab"),   # remotes that get a src/ + doc/ snapshot
    [switch]$Tags,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot   # repo root (this script lives outside src/)

if ($Setup) {
    if (-not $Url -and $args.Count -gt 0) { $Url = $args[0] }
    if (-not $Url) { Write-Host "Usage: -Setup <name> <url>" -ForegroundColor Red; exit 2 }
    if (git remote | Select-String -SimpleMatch -Quiet -Pattern $Setup) {
        git remote set-url $Setup $Url
    } else {
        git remote add $Setup $Url
    }
    Write-Host "Remote '$Setup' -> $Url" -ForegroundColor Green
}

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -eq "HEAD") { Write-Host "Detached HEAD - check out a branch first." -ForegroundColor Red; exit 2 }

$all = @(git remote)
if (-not $Remotes) { $Remotes = $all }
if ($Remotes.Count -eq 0) { Write-Host "No remotes configured." -ForegroundColor Red; exit 2 }

$dirty = git status --porcelain
if ($dirty) { Write-Host "Note: uncommitted changes are NOT pushed (commit first)." -ForegroundColor Yellow }

$failed = @()
foreach ($r in $Remotes) {
    if ($all -notcontains $r) { Write-Host "[$r] no such remote" -ForegroundColor Red; $failed += $r; continue }
    if ($Snapshot -contains $r) {
        git fetch $r 2>&1 | Out-Null
        $tmp = Join-Path ([IO.Path]::GetTempPath()) ("push_all_" + [guid]::NewGuid().ToString("N"))
        $msg = (git log -1 --format=%B HEAD).Trim()
        git worktree add --detach $tmp "$r/main" 2>&1 | Out-Null
        Push-Location $tmp
        git rm -rq --ignore-unmatch src doc docs
        git checkout $branch -- src doc
        git add -A src doc
        if (git status --porcelain) {
            git commit -q -m $msg
            Write-Host "`n>>> snapshot of src/ + doc/ -> $r/main" -ForegroundColor Cyan
            $pc = @("push"); if ($DryRun) { $pc += "--dry-run" }; $pc += @($r, "HEAD:main")
            & git @pc
            if ($LASTEXITCODE -ne 0) { $failed += $r }
        } else {
            Write-Host "[$r] already up to date" -ForegroundColor Green
        }
        Pop-Location
        git worktree remove --force $tmp
        continue
    }
    $cmd = @("push")
    if ($DryRun) { $cmd += "--dry-run" }
    $cmd += @($r, $branch)
    Write-Host "`n>>> git $($cmd -join ' ')" -ForegroundColor Cyan
    & git @cmd
    $ok = ($LASTEXITCODE -eq 0)
    if ($ok -and $Tags) {
        $t = @("push") + $(if ($DryRun) { "--dry-run" }) + @($r, "--tags")
        & git @t
        $ok = ($LASTEXITCODE -eq 0)
    }
    if (-not $ok) { $failed += $r }
}

Write-Host ""
if ($failed.Count) {
    Write-Host "FAILED: $($failed -join ', ')  (others were pushed)" -ForegroundColor Red
    exit 1
}
Write-Host "OK: pushed '$branch' to $($Remotes -join ', ')" -ForegroundColor Green
