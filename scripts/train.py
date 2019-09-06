from lecture_tmu_2019.utils import set_locale
from lecture_tmu_2019.ml import ReputationClassifier

set_locale()
classiier = ReputationClassifier()
feature_vectors_csr = classiier.generate_feature_vectors(classiier.load_data())
classiier.fit(feature_vectors_csr)
