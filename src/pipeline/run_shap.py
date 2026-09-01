from src.utilities import load_config
import shap
cfg = load_config() 

class RunShap : 
    def __init__(self, variant: str, fold: str):
        self.variant = variant
        self.fold = fold

    def run(self, X_train, y_train, X_test, y_test):
        print(f"Running SHAP analysis for variant {self.variant} on fold {self.fold}")

        print(X_test)
        background = (
            X_test.groupby("PULocationID", group_keys=False, observed=False)
            .apply(
                lambda x: x.sample(n=cfg["shap"]["zone_sample_size"], random_state=cfg["seed"]),
                include_groups=True,
            )
            .reset_index(drop=True)
        )

        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(model, masker=masker, feature_names=X_test.columns.drop("PULocationID"))

        shap_values = explainer(X_test)

