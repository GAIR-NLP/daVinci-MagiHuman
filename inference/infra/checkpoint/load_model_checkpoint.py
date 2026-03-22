# Copyright (c) 2026 SandAI. All Rights Reserved.
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

import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from inference.common import EngineConfig
from inference.utils import print_rank_0
from safetensors.torch import load as load_from_bytes
from safetensors.torch import load_file
from tqdm.auto import tqdm


def _load_shard(shard_path, param_names, num_threads=None):
    zstd_path = shard_path + ".zst"
    if os.path.exists(zstd_path):
        cmd = ["zstd", "-d"]
        if num_threads:
            cmd.extend(["-T", str(num_threads)])  # set parallelism

        process = subprocess.Popen(cmd + ["-c", zstd_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)

        decompressed_data = process.stdout.read()
        while True:
            new_data = process.stdout.read()
            if not new_data:
                break
            decompressed_data += new_data
        process.stdout.close()

        retcode = process.wait()
        if retcode != 0:
            raise RuntimeError(f"Decompression failed: {process.stderr.read().decode()}")

        buffer = io.BytesIO(decompressed_data)
        weights = load_from_bytes(buffer.getvalue())
        buffer.close()
    else:
        weights = load_file(shard_path)

    return {name: weights[name] for name in param_names}


def load_sharded_safetensors_parallel_with_progress(checkpoint_dir):
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        model_file_path = os.path.join(checkpoint_dir, "model.safetensors")
        state_dict = load_file(model_file_path)
        return state_dict

    with open(index_path, "r") as f:
        index = json.load(f)

    state_dict = {}
    shard_map = {}

    # Group parameters by shard file
    for param_name, shard_file in index["weight_map"].items():
        shard_path = os.path.join(checkpoint_dir, shard_file)
        if shard_path not in shard_map:
            shard_map[shard_path] = []
        shard_map[shard_path].append(param_name)

    # Load shards in parallel with a progress bar
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_load_shard, shard_path, param_names): shard_path for shard_path, param_names in shard_map.items()
        }
        pbar = tqdm(futures, desc="Loading shards", total=len(futures))
        for future in pbar:
            result = future.result()
            state_dict.update(result)

    return state_dict


def load_model_checkpoint(model, engine_config: EngineConfig):
    print_rank_0("Loading checkpoint with safetensors format from pretrained_folder")
    state_dict = load_sharded_safetensors_parallel_with_progress(engine_config.load)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print_rank_0(f"Load Weight Missing Keys: {missing_keys}")
    print_rank_0(f"Load Weight Unexpected Keys: {unexpected_keys}")
    print_rank_0("Load checkpoint successfully")
    return model
