# Complete experiment, in the order used. Results: results\final (override with DRIFTIDS_RESULTS).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not $env:DRIFTIDS_RESULTS) { $env:DRIFTIDS_RESULTS = Join-Path (Get-Location) "results\final" }
$env:PYTHONWARNINGS = "ignore"
$J = if ($env:JOBS) { $env:JOBS } else { "22" }
function Run([string[]]$a) { & python @a; if ($LASTEXITCODE -ne 0) { throw "failed: $a" } }
Run @("-m", "unittest", "discover", "-s", "tests")
$env:OMP_NUM_THREADS = "1"; $env:DRIFTIDS_THREADS = $J
foreach ($ds in "unsw", "cic_s1a", "cic_s1b_ddos", "cic_s1b_infil", "ton_s1c") { Run @("phase0.py", "screen", "--config", "configs/phase0_$ds.json") }
foreach ($ds in "cic_s1a", "ton_s1c") { Run @("phase1.py", "--config", "configs/phase0_$ds.json", "--detector", "lightgbm", "--windows", "10", "--reps", "30", "--jobs", $J, "--interval", "posterior3") }
foreach ($ds in "unsw", "cic_s1a", "ton_s1c") { Run @("phase2a_attribution.py", "--config", "configs/phase0_$ds.json", "--detector", "lightgbm", "--null", "100", "--windows", "10", "--jobs", $J) }
foreach ($ds in "cic_s1a", "cic_s1b_ddos", "ton_s1c", "cic_s1b_infil") { Run @("phase2b_streams.py", "--config", "configs/phase0_$ds.json", "--detector", "lightgbm", "--reps", "10", "--jobs", $J) }
foreach ($p in @(@("unsw", "cic_s1a"), @("unsw", "ton_s1c"), @("cic_s1a", "ton_s1c"), @("ton_s1c", "cic_s1a"))) {
    Run @("phase2b_streams.py", "--config", "configs/phase0_$($p[0]).json", "--deploy-config", "configs/phase0_$($p[1]).json", "--detector", "lightgbm", "--reps", "10", "--jobs", $J) }
$env:OMP_NUM_THREADS = "2"; $env:DRIFTIDS_THREADS = "2"
foreach ($ds in "cic_s1a", "ton_s1c") { Run @("phase2b_policies.py", "--config", "configs/phase0_$ds.json", "--detector", "lightgbm", "--seeds", "10", "--jobs", "10") }
$main = $env:DRIFTIDS_RESULTS; $extra = Join-Path (Split-Path $main) "final_extra_seeds"
foreach ($ds in "cic_s1a", "ton_s1c") {
    $env:DRIFTIDS_POLICIES = "dad,periodic"; $env:DRIFTIDS_RESULTS = $extra
    Run @("phase2b_policies.py", "--config", "configs/phase0_$ds.json", "--detector", "lightgbm", "--seeds", "10", "--seed-start", "10", "--jobs", "10", "--no-post")
    Remove-Item Env:\DRIFTIDS_POLICIES; $env:DRIFTIDS_RESULTS = $main
    Run @("tools\merge_seeds.py", "$main\phase2b\policies_$ds", "$extra\phase2b\policies_$ds") }
Run @("analysis_hypotheses.py", "--s3", "cic_s1a", "ton_s1c", "--s1", "cic_s1a", "cic_s1b_ddos", "ton_s1c", "--s2", "unsw_to_cic_s1a", "unsw_to_ton_s1c", "cic_s1a_to_ton_s1c", "ton_s1c_to_cic_s1a", "--policy", "cic_s1a", "ton_s1c", "--s3-suffix", "_s3_posterior3_lightgbm", "--label", "final")
