from src.pipeline.build_pipeline import BuildPipeline

### Run HPO  -> finetuned models on variant_A
baseline_experiment = BuildPipeline(variant="A")
baseline_experiment.run(fold = "hpo")
baseline_experiment.run_shap(fold = "hpo")
### Run variant_A
# variant_A = BuildPipeline(variant="A")
# variant_A.run(fold = "fold1")
# variant_A.run(fold = "fold2")
# variant_A.run(fold = "fold3")
# variant_A.run(fold = "fold4")
# variant_A.run(fold = "final_test")

### Run variant_B
# Variant_B = BuildPipeline(variant="B")
# Variant_B.run(fold = "fold1")
# Variant_B.run(fold = "fold2")
# Variant_B.run(fold = "fold3")
# Variant_B.run(fold = "fold4")
# Variant_B.run(fold = "final_test")

### Run variant_C
# Variant_C = BuildPipeline(variant="C")
# Variant_C.run(fold = "fold1")
# Variant_C.run(fold = "fold2")
# Variant_C.run(fold = "fold3")
# Variant_C.run(fold = "fold4")
# Variant_C.run(fold = "final_test")
