# 나머지 감정 17개의 층별 탐색과 채점을 차례로 돌린다. 하나가 실패하면 멈춘다.
# 멈춰도 다시 실행하면 각 단계가 이어서 한다.
# 사용법: powershell -ExecutionPolicy Bypass -File scripts\layer_sweep_rest.ps1 [-Emotions random1,random2,random3,random4]
param([string]$Emotions = "rest")
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = "results\progress\layer_sweep_rest.log"
$py = ".venv\Scripts\python.exe"
$steps = @(
    @("scripts\run_layer_sweep.py", "--emotions", $Emotions),
    @("scripts\fluency_layer_sweep.py"),
    @("scripts\fluency_layer_sweep.py", "--generic"),
    @("scripts\score_layer_sweep.py")
)
function Note($text) { Add-Content $log "$(Get-Date -Format 'MM-dd HH:mm:ss') $text" -Encoding UTF8 }
foreach ($s in $steps) {
    Note "시작: $($s -join ' ')"
    & $py @s 2>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding UTF8
    if ($LASTEXITCODE -ne 0) {
        Note "실패(코드 $LASTEXITCODE): $($s -join ' ')"
        exit $LASTEXITCODE
    }
    Note "끝: $($s -join ' ')"
}
Note "모두 끝"
