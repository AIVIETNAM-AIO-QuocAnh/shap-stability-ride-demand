def save_shap_artifacts(self, shap_values, X_sample_model, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    shap_values_path = model_dir / "shap_values.pkl"
    with open(shap_values_path, "wb") as f:
        pickle.dump(shap_values, f)
    print(f"Saved SHAP values to: {shap_values_path}")

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(model_dir / "shap_bar.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved SHAP bar plot to: {model_dir / 'shap_bar.png'}")

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(model_dir / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved SHAP beeswarm plot to: {model_dir / 'shap_beeswarm.png'}")

def save_importance(self, shap_values, X_sample_model, model_dir: Path):
        importance = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": X_sample_model.columns,
            "importance": importance
        }).sort_values("importance", ascending=False)

        importance_path = model_dir / "shap_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"Saved SHAP importance to: {importance_path}")

        # Weekly-group aggregation based on variant
        variant_map_path = Path(__file__).resolve().parent.parent.parent / "data/processed/variant_feature_map.json"
        with open(variant_map_path) as f:
            variant_map = json.load(f)

        weekly_features = variant_map["variants"][self.variant]["weekly_features"]
        weekly_importance = importance_df[importance_df["feature"].isin(weekly_features)]["importance"].sum()

        weekly_group_path = model_dir / "shap_weekly_group.json"
        with open(weekly_group_path, "w") as f:
            json.dump({"weekly_group_importance": float(weekly_importance)}, f, indent=2)
        print(f"Saved SHAP weekly-group importance to: {weekly_group_path}")