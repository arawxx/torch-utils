from typing import Set, List

import torch.nn as nn


__all__ = ['param_groups_lrd']


def _get_layer_id_for_vit(
    name: str,
    num_layers: int,
) -> int:
    """
    Assigns a layer ID to a parameter based on its name. Used for ViT models.

    Args:
        `name` (str): The name of the parameter.
        `num_layers` (int): The total number of layers.

    Returns:
        int: The assigned layer ID for the parameter.

    Notes:
        This function follows the implementation in BEiT:
        https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33

    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def layerwise_lrd(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_weight_decay_list: Set[str] | List[str] = [],
    layer_decay=0.75,
) -> List[dict]:
    """
    Generate parameter groups for layer-wise learning rate decay.
    Currently supports only `timm` based ViTs, or implementations like `timm`'s.

    Args:
        `model` (nn.Module): The PyTorch model.
        `weight_decay` (float, optional): The weight decay factor. Defaults to 0.05.
        `no_weight_decay_list` (Set[str] | List[str], optional): A list of parameter names
            that should not be decayed. Defaults to an empty list.
        `layer_decay` (float, optional): The decay factor for each layer. Defaults to 0.75.

    Returns:
        List[dict]: A list of parameter groups, where each group is represented as a
        dictionary containing the following keys:
            "lr_scale" (float): The learning rate scale for the group.
            "weight_decay" (float): The weight decay factor for the group.
            "params" (List[str]): The list of parameter names in the group.

    Notes:
        This function generates parameter groups for layer-wise learning rate decay based on
            the BEiT implementation: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58.
        The layer-wise decay is controlled by the `layer_decay` parameter, which determines
            the decay factor for each layer.
        The `no_weight_decay_list` parameter allows specifying a list of parameter names that
            should not be decayed.

    Example 1:
        >>> model = MyModel()
        >>> param_groups = layerwise_lrd(
        >>>     model,
        >>>     weight_decay=0.01,
        >>>     no_weight_decay_list={'pos_embed', 'cls_token', 'dist_token'},
        >>>     layer_decay=0.9,
        >>> )
        >>> optimizer = torch.optim.AdamW(param_groups, lr=0.001)

    Example 2:
        >>> model = timm.create_model('vit_base_patch14_dinov2.lvd142m', num_classes=1000)
        >>> param_groups = layerwise_lrd(
        >>>     model,
        >>>     weight_decay=0.05,
        >>>     no_weight_decay_list=model.no_weight_decay(),  # timm models have such a method
        >>>     layer_decay=0.75,
        >>> )
        >>> optimizer = torch.optim.AdamW(param_groups, lr=0.001)
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if parameter.ndim == 1 or name in no_weight_decay_list:
            g_decay = 'no_decay'
            this_decay = 0.0
        else:
            g_decay = 'decay'
            this_decay = weight_decay
            
        layer_id = _get_layer_id_for_vit(name, num_layers)
        group_name = 'layer_%d_%s' % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                'lr_scale': this_scale,
                'weight_decay': this_decay,
                'params': [],
            }
            param_groups[group_name] = {
                'lr_scale': this_scale,
                'weight_decay': this_decay,
                'params': [],
            }

        param_group_names[group_name]['params'].append(name)
        param_groups[group_name]['params'].append(parameter)

    return list(param_groups.values())
