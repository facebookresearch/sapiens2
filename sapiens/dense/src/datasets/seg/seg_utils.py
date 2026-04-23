# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ruff: noqa
# flake8: noqa
# fmt: off
from collections import OrderedDict

INTERNAL_CLASSES_32 = OrderedDict(
    [
        (0, {"name": "Background", "color": [0, 0, 0]}),
        (1, {"name": "Face_Neck", "color": [0, 0, 0]}),
        (2, {"name": "Nose", "color": [0, 0, 0]}),
        (3, {"name": "Left_Ear", "color": [0, 0, 0]}),
        (4, {"name": "Right_Ear", "color": [0, 0, 0]}),
        (5, {"name": "Left_Iris", "color": [0, 0, 0]}),
        (6, {"name": "Left_Sclera", "color": [0, 0, 0]}),
        (7, {"name": "Right_Iris", "color": [0, 0, 0]}),
        (8, {"name": "Right_Sclera", "color": [0, 0, 0]}),
        (9, {"name": "Left_Eyebrow", "color": [0, 0, 0]}),
        (10, {"name": "Right_Eyebrow", "color": [0, 0, 0]}),
        (11, {"name": "Lower_Lip", "color": [0, 0, 0]}),
        (12, {"name": "Upper_Lip", "color": [0, 0, 0]}),
        (13, {"name": "Lower_Teeth", "color": [0, 0, 0]}),
        (14, {"name": "Upper_Teeth", "color": [0, 0, 0]}),
        (15, {"name": "Tongue", "color": [0, 0, 0]}),
        (16, {"name": "Hair", "color": [0, 0, 0]}),
        (17, {"name": "Left_Arm", "color": [0, 0, 0]}),
        (18, {"name": "Left_Leg", "color": [0, 0, 0]}),
        (19, {"name": "Right_Arm", "color": [0, 0, 0]}),
        (20, {"name": "Right_Leg", "color": [0, 0, 0]}),
        (21, {"name": "Torso", "color": [0, 0, 0]}),
        (22, {"name": "Left_Hand", "color": [0, 0, 0]}),
        (23, {"name": "Right_Hand", "color": [0, 0, 0]}),
        (24, {"name": "Upper_Clothing", "color": [0, 0, 0]}),
        (25, {"name": "Lower_Clothing", "color": [0, 0, 0]}),
        (26, {"name": "Left_Shoe_Sock", "color": [0, 0, 0]}),
        (27, {"name": "Right_Shoe_Sock", "color": [0, 0, 0]}),
        (28, {"name": "Apparel", "color": [0, 0, 0]}),
        (29, {"name": "Glasses", "color": [0, 0, 0]}),
        (30, {"name": "Chair", "color": [0, 0, 0]}),
        (31, {"name": "Headset", "color": [0, 0, 0]}),
        (32, {"name": "Occluder", "color": [0, 0, 0]}),
    ]
)


DOME_CLASSES_34 = OrderedDict(
    [
        (0, {"name": "Background", "color": [50, 50, 50]}),
        (1, {"name": "Apparel", "color": [255, 218, 0]}),
        (2, {"name": "Chair", "color": [102, 204, 0]}),
        (3, {"name": "Eyeglass_Frame", "color": [14, 0, 204]}),
        (4, {"name": "Eyeglass_Lenses", "color": [0, 204, 160]}),
        (5, {"name": "Face_Neck", "color": [128, 200, 255]}),
        (6, {"name": "Hair", "color": [255, 0, 109]}),
        (7, {"name": "Headset", "color": [0, 255, 36]}),
        (8, {"name": "Left_Foot", "color": [189, 0, 204]}),
        (9, {"name": "Left_Hand", "color": [255, 0, 218]}),
        (10, {"name": "Left_Lower_Arm", "color": [0, 160, 204]}),
        (11, {"name": "Left_Lower_Leg", "color": [0, 255, 145]}),
        (12, {"name": "Left_Shoe", "color": [204, 0, 131]}),
        (13, {"name": "Left_Sock", "color": [182, 0, 255]}),
        (14, {"name": "Left_Upper_Arm", "color": [255, 109, 0]}),
        (15, {"name": "Left_Upper_Leg", "color": [0, 255, 255]}),
        (16, {"name": "Lower_Clothing", "color": [72, 0, 255]}),
        (17, {"name": "Lower_Spandex", "color": [204, 43, 0]}),
        (18, {"name": "Right_Foot", "color": [204, 131, 0]}),
        (19, {"name": "Right_Hand", "color": [255, 0, 0]}),
        (20, {"name": "Right_Lower_Arm", "color": [72, 255, 0]}),
        (21, {"name": "Right_Lower_Leg", "color": [189, 204, 0]}),
        (22, {"name": "Right_Shoe", "color": [182, 255, 0]}),
        (23, {"name": "Right_Sock", "color": [102, 0, 204]}),
        (24, {"name": "Right_Upper_Arm", "color": [32, 72, 204]}),
        (25, {"name": "Right_Upper_Leg", "color": [0, 145, 255]}),
        (26, {"name": "Torso", "color": [14, 204, 0]}),
        (27, {"name": "Upper_Clothing", "color": [0, 128, 72]}),
        (28, {"name": "Visible_Badge", "color": [204, 0, 43]}),
        (29, {"name": "Lower_Lip", "color": [235, 205, 119]}),
        (30, {"name": "Upper_Lip", "color": [115, 227, 112]}),
        (31, {"name": "Lower_Teeth", "color": [157, 113, 143]}),
        (32, {"name": "Upper_Teeth", "color": [132, 93, 50]}),
        (33, {"name": "Tongue", "color": [82, 21, 114]}),
    ]
)

DOME_CLASSES_29 = OrderedDict(
    [
        (0, {"name": "Background", "color": [50, 50, 50]}),
        (1, {"name": "Apparel", "color": [255, 218, 0]}),
        (2, {"name": "Eyeglass", "color": [14, 204, 182]}),
        (3, {"name": "Face_Neck", "color": [128, 200, 255]}),
        (4, {"name": "Hair", "color": [255, 0, 109]}),
        (5, {"name": "Left_Foot", "color": [189, 0, 204]}),
        (6, {"name": "Left_Hand", "color": [255, 0, 218]}),
        (7, {"name": "Left_Lower_Arm", "color": [0, 160, 204]}),
        (8, {"name": "Left_Lower_Leg", "color": [0, 255, 145]}),
        (9, {"name": "Left_Shoe", "color": [204, 0, 131]}),
        (10, {"name": "Left_Sock", "color": [182, 0, 255]}),
        (11, {"name": "Left_Upper_Arm", "color": [255, 109, 0]}),
        (12, {"name": "Left_Upper_Leg", "color": [0, 255, 255]}),
        (13, {"name": "Lower_Clothing", "color": [72, 0, 255]}),
        (14, {"name": "Right_Foot", "color": [204, 131, 0]}),
        (15, {"name": "Right_Hand", "color": [255, 0, 0]}),
        (16, {"name": "Right_Lower_Arm", "color": [72, 255, 0]}),
        (17, {"name": "Right_Lower_Leg", "color": [189, 204, 0]}),
        (18, {"name": "Right_Shoe", "color": [182, 255, 0]}),
        (19, {"name": "Right_Sock", "color": [102, 0, 204]}),
        (20, {"name": "Right_Upper_Arm", "color": [32, 72, 204]}),
        (21, {"name": "Right_Upper_Leg", "color": [0, 145, 255]}),
        (22, {"name": "Torso", "color": [14, 204, 0]}),
        (23, {"name": "Upper_Clothing", "color": [0, 128, 72]}),
        (24, {"name": "Lower_Lip", "color": [235, 205, 119]}),
        (25, {"name": "Upper_Lip", "color": [115, 227, 112]}),
        (26, {"name": "Lower_Teeth", "color": [157, 113, 143]}),
        (27, {"name": "Upper_Teeth", "color": [132, 93, 50]}),
        (28, {"name": "Tongue", "color": [82, 21, 114]}),
    ]
)

DOME_CLASSES_28 = OrderedDict(
    [
        (0, {"name": "Background", "color": [50, 50, 50]}),
        (1, {"name": "Apparel", "color": [255, 218, 0]}),
        (2, {"name": "Face_Neck", "color": [128, 200, 255]}),
        (3, {"name": "Hair", "color": [255, 0, 109]}),
        (4, {"name": "Left_Foot", "color": [189, 0, 204]}),
        (5, {"name": "Left_Hand", "color": [255, 0, 218]}),
        (6, {"name": "Left_Lower_Arm", "color": [0, 160, 204]}),
        (7, {"name": "Left_Lower_Leg", "color": [0, 255, 145]}),
        (8, {"name": "Left_Shoe", "color": [204, 0, 131]}),
        (9, {"name": "Left_Sock", "color": [182, 0, 255]}),
        (10, {"name": "Left_Upper_Arm", "color": [255, 109, 0]}),
        (11, {"name": "Left_Upper_Leg", "color": [0, 255, 255]}),
        (12, {"name": "Lower_Clothing", "color": [72, 0, 255]}),
        (13, {"name": "Right_Foot", "color": [204, 131, 0]}),
        (14, {"name": "Right_Hand", "color": [255, 0, 0]}),
        (15, {"name": "Right_Lower_Arm", "color": [72, 255, 0]}),
        (16, {"name": "Right_Lower_Leg", "color": [189, 204, 0]}),
        (17, {"name": "Right_Shoe", "color": [182, 255, 0]}),
        (18, {"name": "Right_Sock", "color": [102, 0, 204]}),
        (19, {"name": "Right_Upper_Arm", "color": [32, 72, 204]}),
        (20, {"name": "Right_Upper_Leg", "color": [0, 145, 255]}),
        (21, {"name": "Torso", "color": [14, 204, 0]}),
        (22, {"name": "Upper_Clothing", "color": [0, 128, 72]}),
        (23, {"name": "Lower_Lip", "color": [235, 205, 119]}),
        (24, {"name": "Upper_Lip", "color": [115, 227, 112]}),
        (25, {"name": "Lower_Teeth", "color": [157, 113, 143]}),
        (26, {"name": "Upper_Teeth", "color": [132, 93, 50]}),
        (27, {"name": "Tongue", "color": [82, 21, 114]}),
    ]
)

DOME_CLASSES_52 = OrderedDict(
    [
        (0, {"name": "Background", "color": [50, 50, 50]}),
        (1, {"name": "Hairnet", "color": [255, 105, 180]}),  # Hot pink
        (2, {"name": "Upper_Gum", "color": [220, 20, 60]}),  # Crimson
        (3, {"name": "Palate", "color": [255, 182, 193]}),  # Light pink
        (4, {"name": "Lower_Gum", "color": [178, 34, 34]}),  # Firebrick
        (5, {"name": "Floor_of_Mouth", "color": [255, 160, 122]}),  # Light salmon
        (6, {"name": "Cheeks_interior", "color": [219, 112, 147]}),  # Pale violet red
        (7, {"name": "Eyelash", "color": [0, 0, 0]}),  # Black
        (8, {"name": "Eyebrow", "color": [101, 67, 33]}),  # Brown
        (9, {"name": "Headset", "color": [0, 255, 36]}),  # Bright green
        (10, {"name": "Left_Foot", "color": [189, 0, 204]}),
        (11, {"name": "Left_Lower_Leg", "color": [0, 255, 145]}),
        (12, {"name": "Left_Shoe", "color": [204, 0, 131]}),
        (13, {"name": "Left_Sock", "color": [182, 0, 255]}),
        (14, {"name": "Left_Upper_Leg", "color": [0, 255, 255]}),
        (15, {"name": "Right_Foot", "color": [204, 131, 0]}),
        (16, {"name": "Right_Lower_Leg", "color": [189, 204, 0]}),
        (17, {"name": "Right_Shoe", "color": [182, 255, 0]}),
        (18, {"name": "Right_Sock", "color": [102, 0, 204]}),
        (19, {"name": "Right_Upper_Leg", "color": [0, 145, 255]}),
        (20, {"name": "Torso", "color": [14, 204, 0]}),
        (21, {"name": "Upper_Clothing", "color": [0, 128, 72]}),
        (22, {"name": "Lower_Clothing", "color": [72, 0, 255]}),
        (23, {"name": "Face_Neck_Skin", "color": [138, 190, 255]}),
        (24, {"name": "Tongue", "color": [82, 21, 114]}),
        (25, {"name": "Eyeglass_Frame", "color": [14, 0, 204]}),
        (26, {"name": "Eyeglass_Lenses", "color": [0, 204, 160]}),
        (27, {"name": "Badge", "color": [255, 165, 0]}),  # Orange
        (28, {"name": "Right_Hand", "color": [255, 0, 0]}),
        (29, {"name": "Right_Lower_Arm", "color": [72, 255, 0]}),
        (30, {"name": "Right_Upper_Arm", "color": [32, 72, 204]}),
        (31, {"name": "Left_Hand", "color": [255, 0, 218]}),
        (32, {"name": "Left_Lower_Arm", "color": [0, 160, 204]}),
        (33, {"name": "Left_Upper_Arm", "color": [255, 109, 0]}),
        (34, {"name": "Apparel", "color": [255, 218, 0]}),
        (35, {"name": "Upper_Teeth", "color": [132, 93, 50]}),
        (36, {"name": "Lower_Teeth", "color": [157, 113, 143]}),
        (37, {"name": "Lower_Lip", "color": [235, 205, 119]}),
        (38, {"name": "Upper_Lip", "color": [115, 227, 112]}),
        (39, {"name": "Hair", "color": [255, 0, 109]}),
        (40, {"name": "Chair", "color": [139, 69, 19]}),  # Saddle brown
        (41, {"name": "Right_thumbNail", "color": [255, 102, 102]}),
        (42, {"name": "Right_indexNail", "color": [255, 102, 102]}),
        (43, {"name": "Right_middleNail", "color": [255, 102, 102]}),
        (44, {"name": "Right_ringNail", "color": [255, 102, 102]}),
        (45, {"name": "Right_pinkyNail", "color": [255, 102, 102]}),
        (46, {"name": "Left_thumbNail", "color": [255, 102, 102]}),
        (47, {"name": "Left_indexNail", "color": [255, 102, 102]}),
        (48, {"name": "Left_middleNail", "color": [255, 102, 102]}),
        (49, {"name": "Left_ringNail", "color": [255, 102, 102]}),
        (50, {"name": "Left_pinkyNail", "color": [255, 102, 102]}),
        (51, {"name": "Occluder", "color": [255, 0, 255]}),
    ]
)

INTERNAL_MAPPING_32_to_29 = {
    0:  {"name": "Background",     "target_class_idx": 0,  "target_class_name": "Background"},
    1:  {"name": "Face_Neck",      "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    2:  {"name": "Nose",           "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    3:  {"name": "Left_Ear",       "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    4:  {"name": "Right_Ear",      "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    5:  {"name": "Left_Iris",      "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    6:  {"name": "Left_Sclera",    "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    7:  {"name": "Right_Iris",     "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    8:  {"name": "Right_Sclera",   "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    9:  {"name": "Left_Eyebrow",   "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    10: {"name": "Right_Eyebrow",  "target_class_idx": 3,  "target_class_name": "Face_Neck"},
    11: {"name": "Lower_Lip",      "target_class_idx": 24, "target_class_name": "Lower_Lip"},
    12: {"name": "Upper_Lip",      "target_class_idx": 25, "target_class_name": "Upper_Lip"},
    13: {"name": "Lower_Teeth",    "target_class_idx": 26, "target_class_name": "Lower_Teeth"},
    14: {"name": "Upper_Teeth",    "target_class_idx": 27, "target_class_name": "Upper_Teeth"},
    15: {"name": "Tongue",         "target_class_idx": 28, "target_class_name": "Tongue"},
    16: {"name": "Hair",           "target_class_idx": 4,  "target_class_name": "Hair"},
    17: {"name": "Left_Arm",       "target_class_idx": None, "target_class_name": None},   
    18: {"name": "Left_Leg",       "target_class_idx": None, "target_class_name": None}, 
    19: {"name": "Right_Arm",      "target_class_idx": None, "target_class_name": None},   
    20: {"name": "Right_Leg",      "target_class_idx": None, "target_class_name": None}, 
    21: {"name": "Torso",          "target_class_idx": 22, "target_class_name": "Torso"},
    22: {"name": "Left_Hand",      "target_class_idx": 6,  "target_class_name": "Left_Hand"},
    23: {"name": "Right_Hand",     "target_class_idx": 15, "target_class_name": "Right_Hand"},
    24: {"name": "Upper_Clothing", "target_class_idx": 23, "target_class_name": "Upper_Clothing"},
    25: {"name": "Lower_Clothing", "target_class_idx": 13, "target_class_name": "Lower_Clothing"},
    26: {"name": "Left_Shoe_Sock", "target_class_idx": None,  "target_class_name": None},       
    27: {"name": "Right_Shoe_Sock","target_class_idx": None, "target_class_name": None},       
    28: {"name": "Apparel",        "target_class_idx": 1,  "target_class_name": "Apparel"},
    29: {"name": "Glasses",        "target_class_idx": 2,  "target_class_name": "Eyeglass"},
    30: {"name": "Chair",          "target_class_idx": 0,  "target_class_name": "Background"},
    31: {"name": "Headset",        "target_class_idx": None, "target_class_name": None},             
    32: {"name": "Occluder",       "target_class_idx": 0,  "target_class_name": "Background"},
}


## source class idx: {name, target class idx, target class name}
DOME_MAPPING_52_to_29 = {
    0: {"name": "Background", "target_class_idx": 0, "target_class_name": "Background"},
    1: {"name": "Hairnet", "target_class_idx": 1, "target_class_name": "Apparel"},
    2: {"name": "Upper_Gum", "target_class_idx": 3, "target_class_name": "Face_Neck"},
    3: {"name": "Palate", "target_class_idx": 3, "target_class_name": "Face_Neck"},
    4: {"name": "Lower_Gum", "target_class_idx": 3, "target_class_name": "Face_Neck"},
    5: {
        "name": "Floor_of_Mouth",
        "target_class_idx": 3,
        "target_class_name": "Face_Neck",
    },
    6: {
        "name": "Cheeks_interior",
        "target_class_idx": 3,
        "target_class_name": "Face_Neck",
    },
    7: {"name": "Eyelash", "target_class_idx": 3, "target_class_name": "Face_Neck"},
    8: {"name": "Eyebrow", "target_class_idx": 3, "target_class_name": "Face_Neck"},
    9: {
        "name": "Headset",
        "target_class_idx": None,
        "target_class_name": None,
    },  # not mapped
    10: {"name": "Left_Foot", "target_class_idx": 5, "target_class_name": "Left_Foot"},
    11: {
        "name": "Left_Lower_Leg",
        "target_class_idx": 8,
        "target_class_name": "Left_Lower_Leg",
    },
    12: {"name": "Left_Shoe", "target_class_idx": 9, "target_class_name": "Left_Shoe"},
    13: {"name": "Left_Sock", "target_class_idx": 10, "target_class_name": "Left_Sock"},
    14: {
        "name": "Left_Upper_Leg",
        "target_class_idx": 12,
        "target_class_name": "Left_Upper_Leg",
    },
    15: {
        "name": "Right_Foot",
        "target_class_idx": 14,
        "target_class_name": "Right_Foot",
    },
    16: {
        "name": "Right_Lower_Leg",
        "target_class_idx": 17,
        "target_class_name": "Right_Lower_Leg",
    },
    17: {
        "name": "Right_Shoe",
        "target_class_idx": 18,
        "target_class_name": "Right_Shoe",
    },
    18: {
        "name": "Right_Sock",
        "target_class_idx": 19,
        "target_class_name": "Right_Sock",
    },
    19: {
        "name": "Right_Upper_Leg",
        "target_class_idx": 21,
        "target_class_name": "Right_Upper_Leg",
    },
    20: {"name": "Torso", "target_class_idx": 22, "target_class_name": "Torso"},
    21: {
        "name": "Upper_Clothing",
        "target_class_idx": 23,
        "target_class_name": "Upper_Clothing",
    },
    22: {
        "name": "Lower_Clothing",
        "target_class_idx": 13,
        "target_class_name": "Lower_Clothing",
    },
    23: {
        "name": "Face_Neck_Skin",
        "target_class_idx": 3,
        "target_class_name": "Face_Neck",
    },
    24: {"name": "Tongue", "target_class_idx": 28, "target_class_name": "Tongue"},
    25: {
        "name": "Eyeglass_Frame",
        "target_class_idx": 2,
        "target_class_name": "Eyeglass",
    },
    26: {
        "name": "Eyeglass_Lenses",
        "target_class_idx": 2,
        "target_class_name": "Eyeglass",
    },
    27: {
        "name": "Badge",
        "target_class_idx": None,
        "target_class_name": None,
    },  # don't care
    28: {
        "name": "Right_Hand",
        "target_class_idx": 15,
        "target_class_name": "Right_Hand",
    },
    29: {
        "name": "Right_Lower_Arm",
        "target_class_idx": 16,
        "target_class_name": "Right_Lower_Arm",
    },
    30: {
        "name": "Right_Upper_Arm",
        "target_class_idx": 20,
        "target_class_name": "Right_Upper_Arm",
    },
    31: {"name": "Left_Hand", "target_class_idx": 6, "target_class_name": "Left_Hand"},
    32: {
        "name": "Left_Lower_Arm",
        "target_class_idx": 7,
        "target_class_name": "Left_Lower_Arm",
    },
    33: {
        "name": "Left_Upper_Arm",
        "target_class_idx": 11,
        "target_class_name": "Left_Upper_Arm",
    },
    34: {"name": "Apparel", "target_class_idx": 1, "target_class_name": "Apparel"},
    35: {
        "name": "Upper_Teeth",
        "target_class_idx": 27,
        "target_class_name": "Upper_Teeth",
    },
    36: {
        "name": "Lower_Teeth",
        "target_class_idx": 26,
        "target_class_name": "Lower_Teeth",
    },
    37: {"name": "Lower_Lip", "target_class_idx": 24, "target_class_name": "Lower_Lip"},
    38: {"name": "Upper_Lip", "target_class_idx": 25, "target_class_name": "Upper_Lip"},
    39: {"name": "Hair", "target_class_idx": 4, "target_class_name": "Hair"},
    40: {"name": "Chair", "target_class_idx": 0, "target_class_name": "Background"},
    41: {
        "name": "Right_thumbNail",
        "target_class_idx": 15,
        "target_class_name": "Right_Hand",
    },
    42: {
        "name": "Right_indexNail",
        "target_class_idx": 15,
        "target_class_name": "Right_Hand",
    },
    43: {
        "name": "Right_middleNail",
        "target_class_idx": 15,
        "target_class_name": "Right_Hand",
    },
    44: {
        "name": "Right_ringNail",
        "target_class_idx": 15,
        "target_class_name": "Right_Hand",
    },
    45: {
        "name": "Right_pinkyNail",
        "target_class_idx": 15,
        "target_class_name": "Right_Hand",
    },
    46: {
        "name": "Left_thumbNail",
        "target_class_idx": 6,
        "target_class_name": "Left_Hand",
    },
    47: {
        "name": "Left_indexNail",
        "target_class_idx": 6,
        "target_class_name": "Left_Hand",
    },
    48: {
        "name": "Left_middleNail",
        "target_class_idx": 6,
        "target_class_name": "Left_Hand",
    },
    49: {
        "name": "Left_ringNail",
        "target_class_idx": 6,
        "target_class_name": "Left_Hand",
    },
    50: {
        "name": "Left_pinkyNail",
        "target_class_idx": 6,
        "target_class_name": "Left_Hand",
    },
    51: {"name": "Occluder", "target_class_idx": 0, "target_class_name": "Background"},
}

DOME_MAPPING_52_to_28 = {
    0: {"name": "Background", "target_class_idx": 0, "target_class_name": "Background"},
    1: {"name": "Hairnet", "target_class_idx": 1, "target_class_name": "Apparel"},
    2: {"name": "Upper_Gum", "target_class_idx": 2, "target_class_name": "Face_Neck"},
    3: {"name": "Palate", "target_class_idx": 2, "target_class_name": "Face_Neck"},
    4: {"name": "Lower_Gum", "target_class_idx": 2, "target_class_name": "Face_Neck"},
    5: {
        "name": "Floor_of_Mouth",
        "target_class_idx": 2,
        "target_class_name": "Face_Neck",
    },
    6: {
        "name": "Cheeks_interior",
        "target_class_idx": 2,
        "target_class_name": "Face_Neck",
    },
    7: {"name": "Eyelash", "target_class_idx": 2, "target_class_name": "Face_Neck"},
    8: {"name": "Eyebrow", "target_class_idx": 2, "target_class_name": "Face_Neck"},
    9: {"name": "Headset", "target_class_idx": None, "target_class_name": None},
    10: {"name": "Left_Foot", "target_class_idx": 4, "target_class_name": "Left_Foot"},
    11: {
        "name": "Left_Lower_Leg",
        "target_class_idx": 7,
        "target_class_name": "Left_Lower_Leg",
    },
    12: {"name": "Left_Shoe", "target_class_idx": 8, "target_class_name": "Left_Shoe"},
    13: {"name": "Left_Sock", "target_class_idx": 9, "target_class_name": "Left_Sock"},
    14: {
        "name": "Left_Upper_Leg",
        "target_class_idx": 11,
        "target_class_name": "Left_Upper_Leg",
    },
    15: {
        "name": "Right_Foot",
        "target_class_idx": 13,
        "target_class_name": "Right_Foot",
    },
    16: {
        "name": "Right_Lower_Leg",
        "target_class_idx": 16,
        "target_class_name": "Right_Lower_Leg",
    },
    17: {
        "name": "Right_Shoe",
        "target_class_idx": 17,
        "target_class_name": "Right_Shoe",
    },
    18: {
        "name": "Right_Sock",
        "target_class_idx": 18,
        "target_class_name": "Right_Sock",
    },
    19: {
        "name": "Right_Upper_Leg",
        "target_class_idx": 20,
        "target_class_name": "Right_Upper_Leg",
    },
    20: {"name": "Torso", "target_class_idx": 21, "target_class_name": "Torso"},
    21: {
        "name": "Upper_Clothing",
        "target_class_idx": 22,
        "target_class_name": "Upper_Clothing",
    },
    22: {
        "name": "Lower_Clothing",
        "target_class_idx": 12,
        "target_class_name": "Lower_Clothing",
    },
    23: {
        "name": "Face_Neck_Skin",
        "target_class_idx": 2,
        "target_class_name": "Face_Neck",
    },
    24: {"name": "Tongue", "target_class_idx": 27, "target_class_name": "Tongue"},
    25: {
        "name": "Eyeglass_Frame",
        "target_class_idx": 2,
        "target_class_name": "Face_Neck",
    },
    26: {
        "name": "Eyeglass_Lenses",
        "target_class_idx": 2,
        "target_class_name": "Face_Neck",
    },
    27: {
        "name": "Badge",
        "target_class_idx": 1,
        "target_class_name": "Apparel",
    },  # accessory → Apparel
    28: {
        "name": "Right_Hand",
        "target_class_idx": 14,
        "target_class_name": "Right_Hand",
    },
    29: {
        "name": "Right_Lower_Arm",
        "target_class_idx": 15,
        "target_class_name": "Right_Lower_Arm",
    },
    30: {
        "name": "Right_Upper_Arm",
        "target_class_idx": 19,
        "target_class_name": "Right_Upper_Arm",
    },
    31: {"name": "Left_Hand", "target_class_idx": 5, "target_class_name": "Left_Hand"},
    32: {
        "name": "Left_Lower_Arm",
        "target_class_idx": 6,
        "target_class_name": "Left_Lower_Arm",
    },
    33: {
        "name": "Left_Upper_Arm",
        "target_class_idx": 10,
        "target_class_name": "Left_Upper_Arm",
    },
    34: {"name": "Apparel", "target_class_idx": 1, "target_class_name": "Apparel"},
    35: {
        "name": "Upper_Teeth",
        "target_class_idx": 26,
        "target_class_name": "Upper_Teeth",
    },
    36: {
        "name": "Lower_Teeth",
        "target_class_idx": 25,
        "target_class_name": "Lower_Teeth",
    },
    37: {"name": "Lower_Lip", "target_class_idx": 23, "target_class_name": "Lower_Lip"},
    38: {"name": "Upper_Lip", "target_class_idx": 24, "target_class_name": "Upper_Lip"},
    39: {"name": "Hair", "target_class_idx": 3, "target_class_name": "Hair"},
    40: {
        "name": "Chair",
        "target_class_idx": 0,
        "target_class_name": "Background",
    },  # treated as background
    41: {
        "name": "Right_thumbNail",
        "target_class_idx": 14,
        "target_class_name": "Right_Hand",
    },
    42: {
        "name": "Right_indexNail",
        "target_class_idx": 14,
        "target_class_name": "Right_Hand",
    },
    43: {
        "name": "Right_middleNail",
        "target_class_idx": 14,
        "target_class_name": "Right_Hand",
    },
    44: {
        "name": "Right_ringNail",
        "target_class_idx": 14,
        "target_class_name": "Right_Hand",
    },
    45: {
        "name": "Right_pinkyNail",
        "target_class_idx": 14,
        "target_class_name": "Right_Hand",
    },
    46: {
        "name": "Left_thumbNail",
        "target_class_idx": 5,
        "target_class_name": "Left_Hand",
    },
    47: {
        "name": "Left_indexNail",
        "target_class_idx": 5,
        "target_class_name": "Left_Hand",
    },
    48: {
        "name": "Left_middleNail",
        "target_class_idx": 5,
        "target_class_name": "Left_Hand",
    },
    49: {
        "name": "Left_ringNail",
        "target_class_idx": 5,
        "target_class_name": "Left_Hand",
    },
    50: {
        "name": "Left_pinkyNail",
        "target_class_idx": 5,
        "target_class_name": "Left_Hand",
    },
    51: {"name": "Occluder", "target_class_idx": 0, "target_class_name": "Background"},
}

DOME_MAPPING_34_to_29 = {
    0: {"name": "Background", "target_class_idx": 0, "target_class_name": "Background"},
    1: {"name": "Apparel", "target_class_idx": 1, "target_class_name": "Apparel"},
    2: {"name": "Chair", "target_class_idx": 0, "target_class_name": "Background"},
    3: {
        "name": "Eyeglass_Frame",
        "target_class_idx": 2,
        "target_class_name": "Eyeglass",
    },
    4: {
        "name": "Eyeglass_Lenses",
        "target_class_idx": 2,
        "target_class_name": "Eyeglass",
    },
    5: {"name": "Face_Neck", "target_class_idx": 3, "target_class_name": "Face_Neck"},
    6: {"name": "Hair", "target_class_idx": 4, "target_class_name": "Hair"},
    7: {"name": "Headset", "target_class_idx": None, "target_class_name": None},
    8: {"name": "Left_Foot", "target_class_idx": 5, "target_class_name": "Left_Foot"},
    9: {"name": "Left_Hand", "target_class_idx": 6, "target_class_name": "Left_Hand"},
    10: {
        "name": "Left_Lower_Arm",
        "target_class_idx": 7,
        "target_class_name": "Left_Lower_Arm",
    },
    11: {
        "name": "Left_Lower_Leg",
        "target_class_idx": 8,
        "target_class_name": "Left_Lower_Leg",
    },
    12: {"name": "Left_Shoe", "target_class_idx": 9, "target_class_name": "Left_Shoe"},
    13: {"name": "Left_Sock", "target_class_idx": 10, "target_class_name": "Left_Sock"},
    14: {
        "name": "Left_Upper_Arm",
        "target_class_idx": 11,
        "target_class_name": "Left_Upper_Arm",
    },
    15: {
        "name": "Left_Upper_Leg",
        "target_class_idx": 12,
        "target_class_name": "Left_Upper_Leg",
    },
    16: {
        "name": "Lower_Clothing",
        "target_class_idx": 13,
        "target_class_name": "Lower_Clothing",
    },
    17: {
        "name": "Lower_Spandex",
        "target_class_idx": 13,
        "target_class_name": "Lower_Clothing",
    },
    18: {
        "name": "Right_Foot",
        "target_class_idx": 14,
        "target_class_name": "Right_Foot",
    },
    19: {
        "name": "Right_Hand",
        "target_class_idx": 15,
        "target_class_name": "Right_Hand",
    },
    20: {
        "name": "Right_Lower_Arm",
        "target_class_idx": 16,
        "target_class_name": "Right_Lower_Arm",
    },
    21: {
        "name": "Right_Lower_Leg",
        "target_class_idx": 17,
        "target_class_name": "Right_Lower_Leg",
    },
    22: {
        "name": "Right_Shoe",
        "target_class_idx": 18,
        "target_class_name": "Right_Shoe",
    },
    23: {
        "name": "Right_Sock",
        "target_class_idx": 19,
        "target_class_name": "Right_Sock",
    },
    24: {
        "name": "Right_Upper_Arm",
        "target_class_idx": 20,
        "target_class_name": "Right_Upper_Arm",
    },
    25: {
        "name": "Right_Upper_Leg",
        "target_class_idx": 21,
        "target_class_name": "Right_Upper_Leg",
    },
    26: {"name": "Torso", "target_class_idx": 22, "target_class_name": "Torso"},
    27: {
        "name": "Upper_Clothing",
        "target_class_idx": 23,
        "target_class_name": "Upper_Clothing",
    },
    28: {"name": "Visible_Badge", "target_class_idx": None, "target_class_name": None},
    29: {"name": "Lower_Lip", "target_class_idx": 24, "target_class_name": "Lower_Lip"},
    30: {"name": "Upper_Lip", "target_class_idx": 25, "target_class_name": "Upper_Lip"},
    31: {
        "name": "Lower_Teeth",
        "target_class_idx": 26,
        "target_class_name": "Lower_Teeth",
    },
    32: {
        "name": "Upper_Teeth",
        "target_class_idx": 27,
        "target_class_name": "Upper_Teeth",
    },
    33: {"name": "Tongue", "target_class_idx": 28, "target_class_name": "Tongue"},
}

# ## loss weights: 29 classes
# 0	Background	0.1
# 1	Apparel	10
# 2	Eyeglass	10
# 4	Face_Neck	3
# 5	Hair	2
# 6	Left_Foot	4
# 7	Left_Hand	4
# 8	Left_Lower_Arm	2
# 9	Left_Lower_Leg	2
# 10	Left_Shoe	6
# 11	Left_Sock	10
# 12	Left_Upper_Arm	3
# 13	Left_Upper_Leg	3
# 14	Lower_Clothing	1
# 15	Right_Foot	4
# 16	Right_Hand	4
# 17	Right_Lower_Arm	2
# 18	Right_Lower_Leg	2
# 19	Right_Shoe	6
# 20	Right_Sock	10
# 21	Right_Upper_Arm	3
# 22	Right_Upper_Leg	3
# 23	Torso	1
# 24	Upper_Clothing	1
# 25	Lower_Lip	10
# 26	Upper_Lip	10
# 27	Lower_Teeth	10
# 28	Upper_Teeth	10
# 29	Tongue	10

# ## loss weights: 34 classes
# 0	Background	0.1
# 1	Apparel	10
# 2	Chair	3
# 3	Eyeglass_Frame	10
# 4	Eyeglass_Lenses	10
# 5	Face_Neck	3
# 6	Hair	2
# 7	Headset	2
# 8	Left_Foot	4
# 9	Left_Hand	4
# 10	Left_Lower_Arm	2
# 11	Left_Lower_Leg	2
# 12	Left_Shoe	6
# 13	Left_Sock	10
# 14	Left_Upper_Arm	3
# 15	Left_Upper_Leg	3
# 16	Lower_Clothing	1
# 17	Lower_Spandex	0.5
# 18	Right_Foot	4
# 19	Right_Hand	4
# 20	Right_Lower_Arm	2
# 21	Right_Lower_Leg	2
# 22	Right_Shoe	6
# 23	Right_Sock	10
# 24	Right_Upper_Arm	3
# 25	Right_Upper_Leg	3
# 26	Torso	1
# 27	Upper_Clothing	1
# 28	Visible_Badge	10
# 29	Lower_Lip	10
# 30	Upper_Lip	10
# 31	Lower_Teeth	10
# 32	Upper_Teeth	10
# 33	Tongue	10
