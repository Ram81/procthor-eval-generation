#%%
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


def target_objects_in_scene(controller: Controller) -> Dict[str, List[str]]:
    """Return a map from the object type to the objectIds in the scene."""
    objects = controller.last_event.metadata["objects"]
    out = {}
    for obj in objects:
        if obj["objectType"] in global_cfg.domain.target_object_types:
            if obj["objectType"] not in out:
                out[obj["objectType"]] = []
            out[obj["objectType"]].append(obj["objectId"])
    return out


def get_nearest_positions(
    reachable_positions: List[Vector3], world_position: Vector3
) -> List[Vector3]:
    """Get the n reachable positions that are closest to the world_position."""
    reachable_positions.sort(
        key=lambda p: sum((p[k] - world_position[k]) ** 2 for k in ["x", "z"])
    )
    return reachable_positions[
        : min(len(reachable_positions), global_cfg.max_agent_positions)
    ]


def get_nearest_agent_height(y_coordinate: float) -> float:
    """Get the nearest valid agent height to a y_coordinate."""
    min_distance = float("inf")
    out = None
    for height in global_cfg.agent.valid_agent_heights:
        dist = abs(y_coordinate - height)
        if dist < min_distance:
            min_distance = dist
            out = height
    return out


def is_object_visible(
    controller: Controller, reachable_positions: List[Vector3], object_id: str
) -> bool:
    """Return True if object_id is visible without any interaction in the scene.
    This method makes an approximation based on checking if the object
    is hit with a raycast from nearby reachable positions.
    """

    obj = next(
        o
        for o in controller.last_event.metadata["objects"]
        if o["objectId"] == object_id
    )

    # NOTE: Get the nearest reachable agent positions to the target object.
    agent_positions = get_nearest_positions(
        reachable_positions=reachable_positions,
        world_position=obj["axisAlignedBoundingBox"]["center"],
    )
    visible_count = 0
    for agent_pos in agent_positions:
        # NOTE: Assumes the agent only has 1 height and 1 horizon
        for rotation in [0, 60, 120, 180, 240, 300]:
            event = controller.step(
                action="Teleport",
                rotation=rotation,
                position=agent_pos,
                horizon=0,
                renderImage=False,
            )
            if not event:
                continue
            obj_is_visible = next(
                o["visible"]
                for o in controller.last_event.metadata["objects"]
                if o["objectId"] == object_id
            )
            if obj_is_visible:
                visible_count += 1
                if visible_count >= global_cfg.min_visible_count:
                    return True
                break

    return False


def visible_target_objects_in_scene(
    controller: Controller, reachable_positions: List[Vector3]
) -> Dict[str, Dict[str, Any]]:
    """Return a map from the visible object type to the objectIds in the scene."""
    objects = controller.last_event.metadata["objects"]
    out = defaultdict(list)
    for obj in objects:
        if obj[
            "objectType"
        ] in global_cfg.domain.target_object_types and is_object_visible(
            controller=controller,
            reachable_positions=reachable_positions,
            object_id=obj["objectId"],
        ):
            out[obj["objectType"]].append(
                dict(
                    objectId=obj["objectId"],
                    position=obj["axisAlignedBoundingBox"]["center"],
                )
            )
    return out


def get_target_objects_in_scene(args):
    global controllers

    house, house_ind = args

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

    target_object_type_to_objs = visible_target_objects_in_scene(
        controller=controller, reachable_positions=reachable_positions
    )
    return target_object_type_to_objs, reachable_positions


def get_target_objects_per_scenes(houses, house_inds, num_scenes_with_target) -> dict:
    target_objects_per_scene = dict()
    reachable_positions_per_scene = dict()
    with Pool(global_cfg.machine.processes) as p:
        house_by_ind = [houses[i] for i in house_inds]
        out = p.map(get_target_objects_in_scene, zip(house_by_ind, house_inds))
        for i, (target_object_type_to_objs, reachable_positions) in enumerate(out):
            for obj_type in target_object_type_to_objs:
                num_scenes_with_target[obj_type] += 1
            target_objects_per_scene[house_inds[i]] = target_object_type_to_objs
            reachable_positions_per_scene[house_inds[i]] = reachable_positions
    return target_objects_per_scene, reachable_positions_per_scene


def get_num_target_trajectories_per_scene(
    target_objects_per_scene, num_scenes_with_target, scenes
):
    scenes_per_target_objects = defaultdict(list)
    for scene, target_objects in target_objects_per_scene.items():
        for obj_type in num_scenes_with_target:
            if obj_type in target_objects:
                scenes_per_target_objects[obj_type].append(scene)
    print("Scenes per object: {}".format(scenes_per_target_objects))
    for obj_type, scenes_with_type in scenes_per_target_objects.items():
        print(obj_type, len(scenes_with_type))
    for obj_type, scenes_with_type in scenes_per_target_objects.items():
        print("type: {} - {}".format(obj_type, len(scenes_with_type)))
        scenes_per_target_objects[
            obj_type
        ] *= global_cfg.tasks_per_target_object // len(scenes_with_type)
        rem = global_cfg.tasks_per_target_object % len(scenes_with_type)
        for _ in range(rem):
            scenes_per_target_objects[obj_type].append(random.choice(scenes_with_type))

    num_target_trajectories_per_scene = {
        scene: {
            obj_type: scenes_with_type.count(scene)
            for obj_type, scenes_with_type in scenes_per_target_objects.items()
            if obj_type in target_objects_per_scene[scene]
        }
        for scene in scenes
    }
    return num_target_trajectories_per_scene


@hydra.main(config_path="config", config_name="main")
def main(cfg) -> None:
    global global_cfg
    global_cfg = cfg

    if cfg.domain.name == "procthor":
        house_inds = range(global_cfg.domain.num_houses)
        houses = prior.load_dataset(
            cfg.dataset.name, entity=cfg.dataset.entity, revision=cfg.dataset.revision
        )[cfg.split]
    else:
        raise NotImplementedError()

    num_scenes_with_target = Counter()
    (
        target_objects_per_scene,
        reachable_positions_per_scene,
    ) = get_target_objects_per_scenes(houses, house_inds, num_scenes_with_target)
    num_target_trajectories_per_scene = get_num_target_trajectories_per_scene(
        target_objects_per_scene, num_scenes_with_target, scenes=house_inds
    )

    tasks = []
    for scene, target_object_counts in num_target_trajectories_per_scene.items():
        print(f"Processing house {scene}")

        # NOTE: So that it doesn't require computing GetShortestPath for all paths.
        print(reachable_positions_per_scene.keys())
        reachable_positions = reachable_positions_per_scene[scene]
        if len(reachable_positions) > cfg.max_reachable_positions:
            reachable_positions = random.sample(
                population=reachable_positions, k=cfg.max_reachable_positions
            )

        for target_obj_type, task_count in target_object_counts.items():
            print(f"Processing task in scene {scene}")
            target_objects = target_objects_per_scene[scene][target_obj_type]

            # NOTE: Get the length from each reachable position to each target
            # object.
            length_from_positions = []
            for _, agent_position in enumerate(reachable_positions):
                min_length_to_type = None
                for target_obj in target_objects:
                    path_length = l2_distance(target_obj["position"], agent_position)
                    if min_length_to_type is None or path_length < min_length_to_type:
                        min_length_to_type = path_length
                if min_length_to_type is not None:
                    length_from_positions.append((min_length_to_type, agent_position))

            length_from_positions.sort(key=lambda x: x[0])

            difficulties = [
                random.choice(["easy", "medium", "hard"]) for _ in range(task_count)
            ]

            easy_count = difficulties.count("easy")
            medium_count = difficulties.count("medium")
            hard_count = difficulties.count("hard")

            # NOTE: sort the reachable positions into easy, medium, and hard.
            agents_starting_data_map = {"easy": [], "medium": [], "hard": []}
            if easy_count > 0:
                upper_bound = int(len(length_from_positions) * cfg.medium_range[0])
                agents_starting_data_map["easy"] = random.choices(
                    length_from_positions[:upper_bound], k=easy_count
                )
            if medium_count > 0:
                lower_bound = int(len(length_from_positions) * cfg.medium_range[0])
                upper_bound = int(len(length_from_positions) * cfg.medium_range[1])
                agents_starting_data_map["medium"] = random.choices(
                    length_from_positions[lower_bound:upper_bound], k=medium_count
                )
            if hard_count > 0:
                lower_bound = int(len(length_from_positions) * cfg.medium_range[1])
                agents_starting_data_map["hard"] = random.choices(
                    length_from_positions[lower_bound:], k=hard_count
                )

            for difficulty, agents_starting_data in agents_starting_data_map.items():
                for min_length, starting_position in agents_starting_data:
                    standing = {}
                    if cfg.agent.agentMode == "default":
                        standing = {"standing": True}

                    task = {
                        "targetObjectType": target_obj_type,
                        "targetObjectIds": [obj["objectId"] for obj in target_objects],
                        "agentPose": AgentPose(
                            position=starting_position,
                            # rotation=houses[scene]["metadata"]["agentPoses"]["stretch"][
                            #     "rotation"
                            # ]["y"],
                            # TODO: this must be updated for the stretch agent.
                            rotation=int(
                                random.choice(
                                    np.arange(0, 360, cfg.rotate_step_degrees)
                                )
                            ),
                            horizon=30,
                            **standing,
                        ),
                        "difficulty": difficulty,
                        "shortestPathLength": min_length,
                        "scene": scene,
                    }
                    tasks.append(task)
                    print(task)
                    print("adding task!")

    print("Saving tasks.... {}".format(os.listdir(".")))
    os.makedirs("tasks", exist_ok=True)
    with open(
        f"tasks/{cfg.domain.name}-{cfg.split}-{cfg.agent.agentMode}.json", "w"
    ) as f:
        json.dump(tasks, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
