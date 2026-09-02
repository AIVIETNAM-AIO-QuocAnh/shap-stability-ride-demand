from src.pipeline.build_pipeline import BuildPipeline

# 1) HPO on variant A (creates results/hpo/{model}/best_params.json) — run once, can be commented if already done
tuning = BuildPipeline(variant="A")
tuning.run_hpo()

# 2) Baseline (unfinetuned) on split hpo, reuse run_fold with tuned=False
tuning.run_fold(fold="hpo", tuned=False)

# 3) Core training + SHAP for all 3 variants x 5 folds, using frozen config
for variant in ["A", "B", "C"]:
    pipeline = BuildPipeline(variant=variant)
    for fold in ["fold1", "fold2", "fold3", "fold4", "final_test"]:
        pipeline.run_fold(fold, tuned=True)
        pipeline.run_shap(fold)
