import json
import os
import random
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, TypedDict

import hydra
import prior
import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from utils import load_json

random.seed(42)

global_cfg = None
controllers = {}


class Vector3(TypedDict):
    x: float
    y: float
    z: float


class AgentPose(TypedDict):
    position: Vector3
    rotation: Vector3
    horizon: int
    standing: bool


def l2_distance(a: Vector3, b: Vector3, ignore_y: bool = True) -> float:
    """Return the L2 distance between two points."""
    if ignore_y:
        return sum((a[k] - b[k]) ** 2 for k in ["x", "z"]) ** 0.5
    return sum((a[k] - b[k]) ** 2 for k in ["x", "y", "z"]) ** 0.5


def get_target_objects_in_scene(args):
    global controllers

    house, house_ind, house_config = args

    pid = os.getpid()
    if pid not in controllers:
        gpu_index = pid % torch.cuda.device_count()
        controllers[pid] = Controller(
            branch="rkr",
            scene="Procedural",
            agentMode=global_cfg.agent.agentMode,
            fieldOfView=global_cfg.agent.fieldOfView,
            visibilityDistance=global_cfg.visibility_distance,
            platform=CloudRendering,
            x_display=f":0.{gpu_index}",
        )
    controller = controllers[pid]
    print(f"Processing house {house_ind}")

    controller.reset(scene=house, renderImage=False)

    pose = house["metadata"]["agent"].copy()
    if global_cfg.agent.agentMode == "locobot":
        del pose["standing"]
    event = controller.step(action="TeleportFull", **pose)

    # TODO: this is temporary because the stretch falls through the ground
    # upon teleportation.
    if global_cfg.agent.agentMode == "stretch":
        controller.step(
            action="TeleportFull",
            **house["metadata"]["agentPoses"][global_cfg.agent.agentMode],
            renderImage=False,
        )

    event = controller.step(
        action="GetReachablePositions", renderImage=False, raise_for_failure=True
    )

    reachable_positions = event.metadata["actionReturn"]

    target_object_configs = []
    for attribute, config in house_config.items():
        for scene_config in config:
            min_count = min(3, len(scene_config["positions"]))
            for positions in random.sample(scene_config["positions"], min_count):
                target_object_configs.append(
                    {
                        "attribute": attribute,
                        "attribute_metadata": scene_config["attribute_metadata"],
                        "positions": positions,
                    }
                )
    random.shuffle(target_object_configs)
    return target_object_configs, reachable_positions


def get_target_object_config_per_scenes(houses, house_inds, house_configs) -> dict:
    target_object_config_per_scene = dict()
    reachable_positions_per_scene = dict()
    with Pool(global_cfg.machine.processes) as p:
        house_by_ind = [houses[i] for i in house_inds]
        house_config_by_ind = [house_configs[str(i)] for i in house_inds]
        out = p.map(get_target_objects_in_scene, zip(house_by_ind, house_inds, house_config_by_ind))
        for i, (target_object_configs, reachable_positions) in enumerate(out):
            target_object_config_per_scene[house_inds[i]] = target_object_configs
            reachable_positions_per_scene[house_inds[i]] = reachable_positions
    return target_object_config_per_scene, reachable_positions_per_scene


@hydra.main(config_path="config", config_name="main")
def main(cfg) -> None:
    global global_cfg
    global_cfg = cfg
    print(cfg)


    if cfg.domain.name == "procthor-attr-onav":
        house_inds = range(global_cfg.domain.num_houses)
        houses = prior.load_dataset(
            cfg.dataset.name, entity=cfg.dataset.entity, revision=cfg.dataset.revision
        )[cfg.split]
    else:
        raise NotImplementedError()

    print("Dir: {}".format(os.getcwd()))
    house_config = load_json(cfg.dataset.scene_config)

    (
        target_object_configs_per_scene,
        reachable_positions_per_scene,
    ) = get_target_object_config_per_scenes(houses, house_inds, house_config)

    tasks = []
    for scene, target_object_configs in target_object_configs_per_scene.items():
        print(f"Processing house {scene} - {len(target_object_configs)} tasks.")

        # NOTE: So that it doesn't require computing GetShortestPath for all paths.
        print(reachable_positions_per_scene.keys())
        reachable_positions = reachable_positions_per_scene[scene]
        if len(reachable_positions) > cfg.max_reachable_positions:
            reachable_positions = random.sample(
                population=reachable_positions, k=cfg.max_reachable_positions
            )

        for target_object_config in target_object_configs:
            print(f"Processing task in scene {scene}")
            starting_position = random.choice(reachable_positions)

            target_object = random.choice(
                target_object_config["attribute_metadata"]
            )
            target_object_attribute = target_object[
                "shade" if target_object_config["attribute"] == "color" else target_object_config["attribute"]
            ]
            target_object_ids = [target_object["object_id"]]

            target_object_positions = target_object_config["positions"]

            min_length = l2_distance(target_object_positions[target_object["object_id"]]["position"], starting_position)

            standing = {}
            if cfg.agent.agentMode == "default":
                standing = {"standing": True}

            task = {
                "targetObjectConfig": target_object_config,
                "targetObjectIds": target_object_ids,
                "targetObjectColor": target_object["color_label"],
                "attributeType": target_object_attribute,
                "agentPose": AgentPose(
                    position=starting_position,
                    rotation=int(
                        random.choice(
                            np.arange(0, 360, cfg.rotate_step_degrees)
                        )
                    ),
                    horizon=30,
                    **standing,
                ),
                "shortestPathLength": min_length,
                "scene": scene,
            }
            tasks.append(task)
            print("adding task!")

    print("Saving {} tasks.... {}".format(len(tasks), os.listdir(".")))
    os.makedirs("tasks", exist_ok=True)
    with open(
        f"tasks/{cfg.domain.name}-{cfg.split}-{cfg.agent.agentMode}.json", "w"
    ) as f:
        json.dump(tasks, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
