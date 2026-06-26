# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2026 Baseten
# SPDX-License-Identifier: Apache-2.0

import os

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _get_trt_prepare_api_fn() -> Any:
    from spec_training.hidden_state_capture.trt_prepare_api import trt_prepare_api as trt_prepare_api_fn
    return trt_prepare_api_fn

_is_setup = False
def trt_prepare_api():
    global is_setup
    _is_setup = True
    return _get_trt_prepare_api_fn()()

def is_setup() -> bool:
    return _is_setup
