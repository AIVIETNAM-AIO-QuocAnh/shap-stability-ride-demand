from src.pipeline.build_pipeline import BuildPipeline
### Run baseline to test the pipeline

# baseline_experiment = BuildPipeline(variant="A")
# baseline_experiment.run_baseline(fold = "fold1")

### Run HPO  -> finetuned models on variant_A
finetune = BuildPipeline(variant="A")
finetune.run_hpo()