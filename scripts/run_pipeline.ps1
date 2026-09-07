[CmdletBinding()]
param(
    [ValidateSet('setup', 'download', 'audit', 'preprocess', 'features', 'analysis', 'verify', 'all')]
    [string]$Stage = 'verify',
    [string]$WorkDir,
    [int]$MaxSubjects = 0,
    [switch]$AcceptDataResponsibility,
    [switch]$Force,
    [switch]$SkipLoo,
    [switch]$NoInstall
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $WorkDir) {
    $WorkDir = Join-Path $RepoRoot 'workdir'
}
$WorkDir = [System.IO.Path]::GetFullPath($WorkDir)
$PreprocessVenv = Join-Path $RepoRoot '.venv-preprocessing'
$AnalysisVenv = Join-Path $RepoRoot '.venv-analysis'
$PreprocessPython = Join-Path $PreprocessVenv 'Scripts\python.exe'
$AnalysisPython = Join-Path $AnalysisVenv 'Scripts\python.exe'

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host "`n> $Executable $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

function Initialize-Environments {
    if (-not (Test-Path -LiteralPath $PreprocessPython)) {
        Invoke-Checked -Executable 'py' -Arguments @('-3.12', '-m', 'venv', $PreprocessVenv)
    }
    if (-not (Test-Path -LiteralPath $AnalysisPython)) {
        Invoke-Checked -Executable 'py' -Arguments @('-3.12', '-m', 'venv', $AnalysisVenv)
    }
    Invoke-Checked -Executable $PreprocessPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--only-binary=:all:', '-r', (Join-Path $RepoRoot 'requirements\preprocessing-lock.txt'))
    Invoke-Checked -Executable $AnalysisPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--only-binary=:all:', '-r', (Join-Path $RepoRoot 'requirements\analysis-lock.txt'))
    Invoke-Checked -Executable $AnalysisPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', (Join-Path $RepoRoot 'requirements\dev.txt'))
}

function Assert-Environments {
    if (-not (Test-Path -LiteralPath $PreprocessPython)) {
        throw "Missing $PreprocessPython. Run -Stage setup first."
    }
    if (-not (Test-Path -LiteralPath $AnalysisPython)) {
        throw "Missing $AnalysisPython. Run -Stage setup first."
    }
}

function Invoke-Download {
    if (-not $AcceptDataResponsibility) {
        throw 'Read docs\DATA_ACCESS_AND_PRIVACY.md and rerun with -AcceptDataResponsibility.'
    }
    $parameters = @(
        (Join-Path $RepoRoot 'scripts\01_download_lemon.py'),
        '--project-root', $WorkDir,
        '--stage', 'all',
        '--accept-data-responsibility'
    )
    if ($MaxSubjects -gt 0) {
        $parameters += @('--max-subjects', $MaxSubjects.ToString())
    }
    Invoke-Checked -Executable $PreprocessPython -Arguments $parameters
}

function Invoke-Audit {
    Invoke-Checked -Executable $PreprocessPython -Arguments @(
        (Join-Path $RepoRoot 'scripts\02_audit_sources.py'),
        '--source-project', $WorkDir,
        '--output-root', $WorkDir
    )
}

function Invoke-Preprocessing {
    $parameters = @(
        (Join-Path $RepoRoot 'scripts\03_preprocess_eeg.py'),
        '--source-project', $WorkDir,
        '--output-root', $WorkDir,
        '--seed', '20260902'
    )
    if ($MaxSubjects -gt 0) {
        $parameters += @('--max-subjects', $MaxSubjects.ToString())
    }
    if ($Force) {
        $parameters += '--force'
    }
    Invoke-Checked -Executable $PreprocessPython -Arguments $parameters
}

function Invoke-Features {
    if ($MaxSubjects -gt 0) {
        throw '-MaxSubjects cannot be used with features because the frozen analysis requires all 139 participants.'
    }
    $psdParameters = @(
        (Join-Path $RepoRoot 'scripts\04_compute_psd_qc.py'),
        '--output-root', $WorkDir
    )
    if ($Force) {
        $psdParameters += '--force'
    }
    Invoke-Checked -Executable $PreprocessPython -Arguments $psdParameters
    Invoke-Checked -Executable $PreprocessPython -Arguments @(
        (Join-Path $RepoRoot 'scripts\05_freeze_qc.py'),
        '--project-root', $WorkDir
    )
    $inputParameters = @(
        (Join-Path $RepoRoot 'scripts\06_build_analysis_inputs.py'),
        '--project-root', $WorkDir
    )
    if ($Force) {
        $inputParameters += '--force-spectral-cache'
    }
    Invoke-Checked -Executable $AnalysisPython -Arguments $inputParameters
}

function Invoke-Analysis {
    $analysisOutput = Join-Path $WorkDir 'analysis\final'
    $parameters = @(
        (Join-Path $RepoRoot 'scripts\07_run_analysis.py'),
        '--project', $WorkDir,
        '--output', $analysisOutput
    )
    if ($SkipLoo) {
        $parameters += '--skip-loo'
    }
    Invoke-Checked -Executable $AnalysisPython -Arguments $parameters
}

function Invoke-RepositoryVerification {
    $python = if (Test-Path -LiteralPath $AnalysisPython) { $AnalysisPython } else { 'py' }
    $parameters = if ($python -eq 'py') {
        @('-3.12', (Join-Path $RepoRoot 'scripts\verify_repository.py'), '--repository', $RepoRoot)
    } else {
        @((Join-Path $RepoRoot 'scripts\verify_repository.py'), '--repository', $RepoRoot)
    }
    Invoke-Checked -Executable $python -Arguments $parameters
}

if ($Stage -eq 'setup') {
    Initialize-Environments
    exit 0
}
if ($Stage -eq 'verify') {
    Invoke-RepositoryVerification
    exit 0
}
if (-not $NoInstall) {
    Initialize-Environments
} else {
    Assert-Environments
}
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

switch ($Stage) {
    'download' { Invoke-Download }
    'audit' { Invoke-Audit }
    'preprocess' { Invoke-Preprocessing }
    'features' { Invoke-Features }
    'analysis' { Invoke-Analysis }
    'all' {
        if ($MaxSubjects -gt 0) {
            throw '-MaxSubjects is only for staged download/preprocessing, not -Stage all.'
        }
        Invoke-Download
        Invoke-Audit
        Invoke-Preprocessing
        Invoke-Features
        Invoke-Analysis
    }
}

Invoke-RepositoryVerification
Write-Host "`nCompleted stage '$Stage'. Work directory: $WorkDir" -ForegroundColor Green
