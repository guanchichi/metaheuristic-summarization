$ErrorActionPreference = "Stop"
$Dataset = "data/processed/multi_news_test.jsonl"
$RunDir = "runs/tuning_experiments"

$BasePred = "runs/full_benchmark_result/base/predictions.jsonl"
$BertPred = "runs/full_benchmark_result/llm/predictions.jsonl"
$GraphPred = "runs/full_benchmark_result/graph/predictions.jsonl"

$UnionK20 = "$RunDir/union_k20.jsonl"
$UnionK40 = "$RunDir/union_k40.jsonl"
$UnionNoGraphK20 = "$RunDir/union_no_graph_k20.jsonl"
$UnionNoLmK20 = "$RunDir/union_no_lm_k20.jsonl"

Write-Host "=========================================="
Write-Host "Running Missing Ablation & Baseline Experiments"
Write-Host "=========================================="

# 1. Build unions if they don't exist
Write-Host "`n[1] Building Unions..." -ForegroundColor Cyan

if (-not (Test-Path $UnionK40)) {
    Write-Host "  > Building Union K=40..."
    python scripts/utils_fusion.py --input $Dataset --base_pred $BasePred --bert_pred $BertPred --graph_pred $GraphPred --out $UnionK40 --cap 120 --src_k 40
}

if (-not (Test-Path $UnionNoGraphK20)) {
    Write-Host "  > Building Union (No Graph) K=20..."
    python scripts/utils_fusion.py --input $Dataset --base_pred $BasePred --bert_pred $BertPred --out $UnionNoGraphK20 --cap 100 --src_k 20
}

if (-not (Test-Path $UnionNoLmK20)) {
    Write-Host "  > Building Union (No LM) K=20..."
    python scripts/utils_fusion.py --input $Dataset --base_pred $BasePred --graph_pred $GraphPred --out $UnionNoLmK20 --cap 100 --src_k 20
}

# 2. Define Tests
$Tests = @(
    @{ Name="Baseline_K20"; Config="configs/2_Fusion_Final.yaml"; Union=$UnionK20 },
    @{ Name="Baseline_K40"; Config="configs/2_Fusion_Final.yaml"; Union=$UnionK40 },
    @{ Name="Ablation_NoGraph_K20"; Config="configs/2_Fusion_Final.yaml"; Union=$UnionNoGraphK20 },
    @{ Name="Ablation_NoLM_K20"; Config="configs/2_Fusion_Final.yaml"; Union=$UnionNoLmK20 },
    @{ Name="Ablation_NoNSGA2_K20"; Config="configs/2_Fusion_NoNsga2.yaml"; Union=$UnionK20 }
)

Write-Host "`n[2] Running Inference and Evaluation..." -ForegroundColor Cyan

foreach ($t in $Tests) {
    $Name = $t.Name
    $Cfg = $t.Config
    $Union = $t.Union
    
    Write-Host "`n[Running] $Name..." -ForegroundColor Yellow
    
    $PredFile = "$RunDir/$Name/predictions.jsonl"
    if (-not (Test-Path $PredFile)) {
        python -m src.pipeline.select_sentences `
            --config $Cfg `
            --split test `
            --input $Union `
            --run_dir $RunDir `
            --stamp $Name
    } else {
        Write-Host "  > Skipping Prediction (Exists)"
    }

    if (Test-Path $PredFile) {
        Write-Host "  > Evaluating..."
        python -m src.pipeline.evaluate --pred $PredFile --out "$RunDir/$Name/metrics.csv"
            
        $Res = Get-Content "$RunDir/$Name/metrics.csv" | Select-Object -Skip 1 | Select-Object -First 3
        Write-Host "  > Result: $Res" -ForegroundColor Green
    } else {
        Write-Error "Prediction failed for $Name"
    }
}

Write-Host "`nAll missing experiments completed." -ForegroundColor Cyan
