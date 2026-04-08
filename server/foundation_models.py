# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sentence_transformers import SentenceTransformer
import open_clip


def init_clip():
    # initialize CLIP
    print("loading clip model")
    (
        clip_model,
        _,
        clip_preprocess,
    ) = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k", device="cpu"
    )
    print("loaded clip model")
    print("loading clip tokenizer")
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    print("loaded clip tokenizer")
    return clip_model, clip_preprocess, clip_tokenizer

def init_sbert():
    # initialize sentence transformer
    print("loading sbert_model")
    sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    print("loaded sbert_model")
    return sbert_model



# --- Sequential mode support ---
# In sequential mode, models are loaded lazily and can be unloaded
try:
    from sequential_mode import sequential_enabled, SequentialManager
    _seq_available = True
except ImportError:
    _seq_available = False
    def sequential_enabled(): return False

clip_model = None
clip_preprocess = None
clip_tokenizer = None
sbert_model = None
_models_initialized = False


def _ensure_models_loaded():
    """Lazy initialization of models. In sequential mode, only loads when needed."""
    global clip_model, clip_preprocess, clip_tokenizer, sbert_model, _models_initialized
    if not _models_initialized:
        clip_model, clip_preprocess, clip_tokenizer = init_clip()
        sbert_model = init_sbert()
        _models_initialized = True
        # Register with sequential manager
        if _seq_available and sequential_enabled():
            mgr = SequentialManager.instance()
            mgr.register_stage("clip", loader=_load_clip_to_gpu, unloader=_unload_clip_from_gpu)


def _load_clip_to_gpu():
    """Move CLIP model to GPU."""
    global clip_model
    _ensure_models_loaded()
    if clip_model is not None:
        clip_model = clip_model.cuda()


def _unload_clip_from_gpu():
    """Move CLIP model back to CPU to free GPU memory."""
    global clip_model
    if clip_model is not None:
        clip_model = clip_model.cpu()


def get_clip_models():
    global clip_model, clip_preprocess, clip_tokenizer
    _ensure_models_loaded()
    return clip_model, clip_preprocess, clip_tokenizer


def get_sbert_model():
    global sbert_model
    _ensure_models_loaded()
    return sbert_model


# Eagerly initialize if NOT in sequential mode (preserves original behavior)
if not (_seq_available and sequential_enabled()):
    _ensure_models_loaded()