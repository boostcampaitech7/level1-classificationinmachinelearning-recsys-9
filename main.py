import pandas as pd
from src.data.data_loader import load_external_files, load_data, merge_data
from src.features.feature_engineering import shift_feature, create_features
from src.models.lgb import evaluate_model, predict_lgb_model, train_lgb_model
from src.utils.plots import plot_barplots, plot_target_distribution


def main():
    data_path = "./data"

    # Load raw data
    train_df, test_df, submission_df = load_data(data_path)

    # Load external files
    external_data = load_external_files(data_path)

    # Merge external data
    merged_df = merge_data(train_df, external_data)

    # Feature engineering
    engineered_df = create_features(merged_df)
    conti_cols = ['buy_volume', 'sell_volume']  # Example continuous columns
    shift_features = shift_feature(engineered_df, conti_cols, [1, 2, 3])

    # Prepare for model
    x_train = engineered_df.drop("target", axis=1)
    y_train = engineered_df["target"]

    # Train model
    x_train_valid, x_valid = x_train.iloc[:1000], x_train.iloc[1000:]
    y_train_valid, y_valid = y_train.iloc[:1000], y_train.iloc[1000:]
    model = train_lgb_model(x_train_valid, y_train_valid, x_valid, y_valid)

    # Predict
    y_valid_pred, y_valid_pred_class = predict_lgb_model(model, x_valid)

    # Evaluate model
    accuracy, auroc = evaluate_model(y_valid, y_valid_pred, y_valid_pred_class)
    print(f"Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}")

    # Visualization
    plot_target_distribution(engineered_df)
    plot_barplots(engineered_df)

if __name__ == "__main__":
    main()
