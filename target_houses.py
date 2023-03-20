import json
import random
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

AGENT = "default"
SPLIT = "validation"
MEDIUM_RANGE = (0.2, 0.6)

controller = Controller(agentMode=AGENT, branch="nanna")

scenes = [f"Target_House_0{i}" for i in range(1, 6)]
MAX_REACHABLE_POSITIONS: int = 50
TASKS_PER_OBJECT_TYPE_PER_SCENE: int = 8

failures = []
tasks = []
for scene in scenes:
    controller.reset(scene=scene)
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]

    target_objects = visible_target_objects_in_scene(
        controller=controller, reachable_positions=reachable_positions
    )

    # NOTE: So that it doesn't require computing GetShortestPath for all paths.
    if len(reachable_positions) > MAX_REACHABLE_POSITIONS:
        reachable_positions = random.sample(
            population=reachable_positions, k=MAX_REACHABLE_POSITIONS
        )

    for obj_type in TARGET_OBJECT_TYPES:
        object_ids = [
            obj["objectId"]
            for obj in controller.last_event.metadata["objects"]
            if obj["objectType"] == obj_type
        ]

        print(f"Processing {scene}|{obj_type}")

        length_from_positions = []
        num_failures = 0
        for i, position in enumerate(reachable_positions):
            min_length_to_type = None
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
            print(f"ERROR: Note enough shortest paths for {scene}|{obj_type}!")
            failures.append((scene, obj_type))
            continue

        length_from_positions.sort(key=lambda x: x[0])
        difficulties = [
            random.choice(["easy", "medium", "hard"])
            for _ in range(TASKS_PER_OBJECT_TYPE_PER_SCENE)
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

with open(f"datasets/target-{AGENT}-{SPLIT}.json", "w") as f:
    json.dump(tasks, f, indent=4, sort_keys=True)
