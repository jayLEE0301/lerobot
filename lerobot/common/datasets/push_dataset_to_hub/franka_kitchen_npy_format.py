"""Process pickle files formatted like in: https://github.com/fyhMer/fowm"""

import pickle
import shutil
from pathlib import Path

import numpy as np
import einops
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def check_format(raw_dir):
    franka_kitchen_npy_files = set([file.name for file in list(raw_dir.glob("*.npy"))])
    
    required_files = {"observations_seq.npy", "actions_seq.npy", "existence_mask.npy"}

    assert len(franka_kitchen_npy_files) > 0
    assert all(k in franka_kitchen_npy_files for k in required_files)

    raw_dir = Path(raw_dir)
    observations = torch.from_numpy(
        np.load(raw_dir / "observations_seq.npy")
    )
    actions = torch.from_numpy(np.load(raw_dir / "actions_seq.npy"))
    masks = torch.from_numpy(np.load(raw_dir / "existence_mask.npy"))

    assert observations.shape[1] == 566
    assert actions.shape[1] == 566
    assert masks.shape[1] == 566

def load_from_raw(raw_dir, out_dir, fps, video, debug):
    raw_dir = Path(raw_dir)
    observations = torch.from_numpy(
        np.load(raw_dir / "observations_seq.npy")
    )
    actions = torch.from_numpy(np.load(raw_dir / "actions_seq.npy"))
    masks = torch.from_numpy(np.load(raw_dir / "existence_mask.npy"))

    # The current values are in shape T x N x Dim, move to N x T x Dim
    observations, actions, masks = transpose_batch_timestep(
        observations, actions, masks
    )
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0
    id_to = 0
    ep_idx = 0
    num_episodes = observations.shape[0]
    for i in tqdm.tqdm(range(num_episodes)):


        id_to += int(masks[i].sum())
        num_frames = id_to - id_from

        episode_observation = observations[i, :num_frames]
        episode_action = actions[i, :num_frames]
        episode_mask = masks[i, :num_frames]

        ep_dict = {}

        ep_dict["observation.state"] = episode_observation
        ep_dict["action"] = episode_action
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["next.done"] = torch.tensor([False] * num_frames, dtype=torch.int64)
        ep_dict["next.done"][-1] = True
        ep_dicts.append(ep_dict)

        episode_data_index["from"].append(id_from)
        episode_data_index["to"].append(id_from + num_frames)

        id_from = id_to
        ep_idx += 1

        # process first episode only
        if debug:
            break

    data_dict = concatenate_episodes(ep_dicts)
    return data_dict, episode_data_index


def to_hf_dataset(data_dict, video):
    features = {}

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)
    # TODO(rcadene): add success
    # features["next.success"] = Value(dtype='bool', id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=False, debug=False):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 10

    data_dict, episode_data_index = load_from_raw(raw_dir, out_dir, fps, video, debug)
    hf_dataset = to_hf_dataset(data_dict, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info

def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)