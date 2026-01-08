# Copyright 2025 Nous Research
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

import torch
from typing import Union


def pad_sequences(
    sequences: list[list[Union[int, float]]],
    max_len: int,
    pad_value: Union[int, float] = 0,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32 if isinstance(pad_value, float) else torch.long

    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_value, dtype=dtype)

    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=dtype)

    return padded
