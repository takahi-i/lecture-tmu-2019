from lecture_tmu_2019.utils import set_locale
from lecture_tmu_2019.ml import generate_feature_vectors, fit, load_data


set_locale()
feature_vectors_csr = generate_feature_vectors(load_data())
fit(feature_vectors_csr)
