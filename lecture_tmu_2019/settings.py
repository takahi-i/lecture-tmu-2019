SEARCH_PARAMETERS = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]
DATASET_BASE_PATH = "dataset"
POSITIVE_FILES = "{}/tokens/pos/*".format(DATASET_BASE_PATH)
NEGATIVE_FILES = "{}/tokens/neg/*".format(DATASET_BASE_PATH)
