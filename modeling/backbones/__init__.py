# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["build_backbone"]


def build_backbone(config, model_type):
    if model_type == "det":
        from .det_resnet import Resnet
        from .det_resnet_vd import ResNet_vd
        from .mvitv2 import mvit_v2_s
        support_dict = [
            "Resnet",
            "MViTv2_S",
            "ResNet_vd"
        ]
    module_name = config.pop("name")
    
    if module_name == "MViTv2_S":
        return mvit_v2_s(**config)
    
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(
            model_type, support_dict
        )
    )
    module_class = eval(module_name)(**config)
    return module_class
