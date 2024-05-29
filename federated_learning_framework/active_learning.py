import numpy as np

def query_active_learning(model, unlabeled_data, num_samples):
    predictions = model.predict(unlabeled_data)
    uncertainty = predictions.max(axis=1)
    selected_indices = np.argsort(uncertainty)[:num_samples]
    return selected_indices
