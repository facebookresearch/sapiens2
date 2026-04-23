# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .pointmap_base_dataset import PointmapBaseDataset
from .pointmap_render_people_dataset import PointmapRenderPeopleDataset


__all__ = ["PointmapBaseDataset", "PointmapRenderPeopleDataset"]
