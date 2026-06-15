from __future__ import annotations

import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(
    task_id: int,
    seed: int = 42,
    test_size: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder]:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")

    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), name="target", index=y.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=seed, stratify=y_enc
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), \
           y_train.reset_index(drop=True), y_test.reset_index(drop=True), le
