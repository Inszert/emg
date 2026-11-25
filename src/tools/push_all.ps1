<#
.SYNOPSIS
  Push the current branch to EVERY configured git remote (e.g. two gits).

.EXAMPLE
  .\tools\push_all.ps1                       # push current branch to all remotes
  .\tools\push_all.ps1 -Setup gitlab https://gitlab.com/user/emg.git   # add 2nd remote once
  .\tools\push_all.ps1 -Remotes origin,gitlab   # only these
  .\tools\push_all.ps1 -Tags                 # also push tags
  .\tools\push_all.ps1 -DryRun               # show what would happen

  Never force-pushes. If one remote rejects (history differs), the others are
  still pushed and the script exits non-zero with a summary.
#>
param(
    [string]$Setup,            # name of a remote to add ...
    [string]$Url,              # ... with this URL (can also be 2nd positional)
    [string[]]$Remotes,        # default: all remotes
    [switch]$Tags,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)   # project root

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
