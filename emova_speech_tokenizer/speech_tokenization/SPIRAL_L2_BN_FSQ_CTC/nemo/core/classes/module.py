# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import contextmanager

from torch.nn import Module

from nemo.core.classes.common import FileIO, Serialization, Typing

__all__ = ['NeuralModule']


class NeuralModule(Module, Typing, Serialization, FileIO):
    """
    Abstract class offering interface shared between all PyTorch Neural Modules.
    """

    @property
    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def input_example(self):
        """
        Override this method if random inputs won't work
        Returns:
            A tuple sample of valid input data.
        """

        return

    def freeze(self):
        r"""
        Freeze all params for inference.
        """
        requires_grad_states = []
        for param in self.parameters():
            requires_grad_states.append(param.requires_grad)
            param.requires_grad = False

        self.eval()
        return requires_grad_states

    def unfreeze(self, requires_grad_states) -> None:
        """
        Unfreeze all parameters for training.
        """
        for i, param in enumerate(self.parameters()):
            param.requires_grad = requires_grad_states[i]

        self.train()

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes a module, yields control and finally unfreezes the module.
        """
        requires_grad_states = self.freeze()

        try:
            yield
        finally:
            self.unfreeze(requires_grad_states)
