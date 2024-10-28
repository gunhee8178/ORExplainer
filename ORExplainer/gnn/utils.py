import torch

def preprocess_features(features):
    """
    Preprocess the features using PyTorch and transforms them into a normalized representation.
    :param features: Original feature representation as a PyTorch tensor
    :return: Normalized feature tensor
    """
    # 각 노드의 특성 합 (각 행의 합)
    rowsum = features.sum(1)

    # 역수 계산
    r_inv = torch.pow(rowsum, -1)

    # 무한대(inf) 값이 있을 경우 0으로 처리
    r_inv[torch.isinf(r_inv)] = 0.

    # 역수 값을 각 행에 곱해주는 대각 행렬과의 곱셈 대신 broadcasting을 사용
    features = features * r_inv.unsqueeze(1)

    return features

def preprocess_model_keys(model_state_dict):
    renamed_state_dict = {}
    for key in model_state_dict['model_state_dict']:
        if key.startswith('conv') and key.endswith('weight') and ('lin' not in key):
            new_key = key[:5] + '.lin' + key[-7:]
            renamed_state_dict[new_key] = (model_state_dict['model_state_dict'][key]).T
            # renamed_state_dict[new_key] = (checkpoint['model_state_dict'][key]).T
        else:
            renamed_state_dict[key] = model_state_dict['model_state_dict'][key]

    return renamed_state_dict