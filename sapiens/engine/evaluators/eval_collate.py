# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sapiens.registry import MODELS
from torch.utils.data import default_collate


@MODELS.register_module()
def eval_collate(batch: list):
    passthrough_keys = {"data_samples"}
    collated_data, passthrough_data = [], {key: [] for key in passthrough_keys}
    for item in batch:
        item_for_collation = {
            k: v for k, v in item.items() if k not in passthrough_keys
        }
        for key in passthrough_keys:
            passthrough_data[key].append(item[key])
        collated_data.append(item_for_collation)
    final_batch = default_collate(collated_data)
    final_batch.update(passthrough_data)
    return final_batch
