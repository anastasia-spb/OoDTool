from tool.ood_score.experimental import ood_one_class_svm
from tool.ood_score.experimental import mahalanobis_ood


def one_class_svm_test(embeddings_pkl):
    ood_clf = ood_one_class_svm.OneClassOoD(embeddings_pkl)
    result = ood_clf()
    result.to_pickle('one_class_svm_test.ood.pkl')


def local_outlier_test(embeddings_pkl):
    ood_clf = ood_one_class_svm.LocalOutlierOoD(embeddings_pkl)
    result = ood_clf()
    result.to_pickle('local_outlier_test.ood.pkl')


def mahalanobis_ood_test(embeddings_pkl):
    ood_clf = mahalanobis_ood.MahalanobisOoD(embeddings_pkl)
    result = ood_clf()
    result.to_pickle('mahalanobis_ood_test.ood.pkl')


if __name__ == "__main__":
    input_data = '/home/vlasova/datasets/0metadata/CarLightsDVC/test_pipeline.emb.pkl'
    # one_class_svm_test(input_data)
    # local_outlier_test(input_data)
    mahalanobis_ood_test(input_data)
