import json
import random
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
from ai2thor.controller import Controller

from main import (
    ROTATE_STEP_DEGREES,
    TARGET_OBJECT_TYPES,
    TARGET_OBJECT_TYPES_SET,
    AgentPose,
    l2_distance,
    visible_target_objects_in_scene,
)

AGENT = "locobot"
SPLIT = "test"

controller = Controller(agentMode=AGENT, branch="nanna")

scenes = {
    "validation": [
        f"FloorPlan{i + j}" for j in [0, 200, 300, 400] for i in range(21, 26)
    ],
    "test": [f"FloorPlan{i + j}" for j in [0, 200, 300, 400] for i in range(26, 31)],
}[SPLIT]

MEDIUM_RANGE = (0.2, 0.6)
TASKS_PER_HOUSE = 15
TASKS_PER_TARGET_OBJECT = 50
MAX_REACHABLE_POSITIONS = 50


counter = Counter()
target_objects_per_scene = dict()
for scene in scenes:
    event = controller.reset(scene=scene)
    target_objects = defaultdict(list)
    for obj in event.metadata["objects"]:
        if obj["objectType"] in TARGET_OBJECT_TYPES_SET:
            target_objects[obj["objectType"]].append(obj["objectId"])
    for obj in target_objects:
        counter[obj] += 1
    target_objects_per_scene[scene] = target_objects

scenes_per_target_objects = defaultdict(list)
for scene, target_objects in target_objects_per_scene.items():
    for obj in counter:
        if obj in target_objects:
            scenes_per_target_objects[obj].append(scene)
for obj_type, scenes_with_type in scenes_per_target_objects.items():
    scenes_per_target_objects[obj_type] *= TASKS_PER_TARGET_OBJECT // len(
        scenes_with_type
    )
    rem = TASKS_PER_TARGET_OBJECT % len(scenes_with_type)
    for i in range(rem):
        scenes_per_target_objects[obj_type].append(random.choice(scenes_with_type))

num_target_trajectories_per_scene = {
    scene: {
        obj_type: scenes_with_type.count(scene)
        for obj_type, scenes_with_type in scenes_per_target_objects.items()
        if obj_type in target_objects_per_scene[scene]
    }
    for scene in scenes
}

tasks = []
for scene, target_object_counts in num_target_trajectories_per_scene.items():
    print(f"Processing house {scene}")

    controller.reset(scene=scene)
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]

    target_objects = visible_target_objects_in_scene(
        controller=controller, reachable_positions=reachable_positions
    )

    object_counter = Counter({obj_type: 0 for obj_type in TARGET_OBJECT_TYPES})

    # NOTE: So that it doesn't require computing GetShortestPath for all paths.
    if len(reachable_positions) > MAX_REACHABLE_POSITIONS:
        reachable_positions = random.sample(
            population=reachable_positions, k=MAX_REACHABLE_POSITIONS
        )

    for obj_type, task_count in target_object_counts.items():
        object_ids = target_objects_per_scene[scene][obj_type]

        print(f"Processing task in scene {scene}")
        length_from_positions = []
        num_failures = 0
        for i, position in enumerate(reachable_positions):
            min_length_to_type = None
            for object_id in object_ids:
                obj = next(
                    obj
                    for obj in controller.last_event.metadata["objects"]
                    if obj["objectId"] == object_id
                )
                path_length = l2_distance(
                    obj["axisAlignedBoundingBox"]["center"], position
                )
                if min_length_to_type is None or path_length < min_length_to_type:
                    min_length_to_type = path_length
            if min_length_to_type is not None:
                length_from_positions.append((min_length_to_type, position))

        if len(length_from_positions) < 15:
            print(f"Shorest paths failed for {scene} to objectIds {object_ids}!")
            continue

        length_from_positions.sort(key=lambda x: x[0])
        difficulty = random.choice(["easy", "medium", "hard"])

        difficulties = [
            random.choice(["easy", "medium", "hard"]) for _ in range(task_count)
        ]

        easy_count = len([d for d in difficulties if d == "easy"])
        medium_count = len([d for d in difficulties if d == "medium"])
        hard_count = len([d for d in difficulties if d == "hard"])

        agents_starting_data_map = {"easy": [], "medium": [], "hard": []}

        if easy_count > 0:
            upper = int(len(length_from_positions) * MEDIUM_RANGE[0])
            agents_starting_data_map["easy"] = random.choices(
                length_from_positions[:upper], k=easy_count
            )
        if medium_count > 0:
            lower = int(len(length_from_positions) * MEDIUM_RANGE[0])
            upper = int(len(length_from_positions) * MEDIUM_RANGE[1])
            agents_starting_data_map["medium"] = random.choices(
                length_from_positions[lower:upper], k=medium_count
            )
        if hard_count > 0:
            lower = int(len(length_from_positions) * MEDIUM_RANGE[1])
            agents_starting_data_map["hard"] = random.choices(
                length_from_positions[lower:], k=hard_count
            )

        for difficulty, agents_starting_data in agents_starting_data_map.items():
            for min_length, starting_position in agents_starting_data:
                standing = {}
                if AGENT == "default":
                    standing = {"standing": True}

                task = {
                    "targetObjectType": obj_type,
                    "targetObjectIds": object_ids,
                    "agentPose": AgentPose(
                        position=starting_position,
                        rotation=int(
                            random.choice(np.arange(0, 360, ROTATE_STEP_DEGREES))
                        ),
                        horizon=30,
                        **standing,
                    ),
                    "difficulty": difficulty,
                    "shortestPathLength": min_length,
                    "scene": scene,
                }
                tasks.append(task)
                print("adding task!")

print(len(tasks))
with open(f"ithor-{AGENT}-{SPLIT}.json", "w") as f:
    for task in tasks:
        f.write(json.dumps(task) + "\n")

target_object_counts
