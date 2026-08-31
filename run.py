

### Run HPO  -> finetuned models on variant_A
baseline_experiment = BuildPipeline(variant="A")
baseline_experiment.run(fold = "hpo")
### Run variant_A
variant_A = BuildPipeline(variant="A")
variant_A.run(fold = "1")
variant_A.run(fold = "2")
variant_A.run(fold = "3")
variant_A.run(fold = "4")
variant_A.run(fold = "final")

### Run variant_B
# Variant_B = BuildPipeline(variant="B")
# Variant_B.run(fold = "1")
# Variant_B.run(fold = "2")
# Variant_B.run(fold = "3")
# Variant_B.run(fold = "4")
# Variant_B.run(fold = "final")

### Run variant_C
# Variant_C = BuildPipeline(variant="C")
# Variant_C.run(fold = "1")
# Variant_C.run(fold = "2")
# Variant_C.run(fold = "3")
# Variant_C.run(fold = "4")
# Variant_C.run(fold = "final")