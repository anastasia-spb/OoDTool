from oodtool.core.towhee_adapter.towhee_adapter import get_towheeresnet50_embedder_id, \
    get_swin_transformer_embedder_id
from oodtool.core.towhee_adapter import usecase

# OoD Methods names
OOD_ENTROPY = "OoD_Entropy"
OOD_ENTROPY_SWIN = "OoD_Entropy_Swin"
OOD_KNN_DIST = "OoD_KNN_Dist"
OOD_KNN_DIST_SWIN = "OoD_KNN_Dist_Swin"
OOD_CONFIDENT_LEARNING = "OoD_ConfidentLearning"

OOD_METHODS = [OOD_ENTROPY, OOD_ENTROPY_SWIN, OOD_KNN_DIST, OOD_KNN_DIST_SWIN, OOD_CONFIDENT_LEARNING]

OOD_METHOD_FEATURES = {
    OOD_ENTROPY: {usecase.OTHER: [get_towheeresnet50_embedder_id()]},
    OOD_ENTROPY_SWIN: {usecase.OTHER: [get_swin_transformer_embedder_id()]},
    OOD_KNN_DIST: {usecase.OTHER: [get_towheeresnet50_embedder_id()]},
    OOD_KNN_DIST_SWIN: {usecase.OTHER: [get_swin_transformer_embedder_id()]},
    OOD_CONFIDENT_LEARNING: {usecase.OTHER: []}
}
