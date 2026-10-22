

def test_robots_launched_from_one_gate_share_the_gate_frame():
    # One-door seeds 922/923: B's frame is the gate frame (its brain starts at its pad), so it is placed in the house by
    # the pad at (0, 0, 0), not by B's own spawn. Control: without pads each robot starts at its own spawn.
    from brain.nav_io import start_pose
    gate = {"robots": [{"id": "robot_A", "spawn": {"at": [10.0, -1.2], "yaw": 0}, "launch_pad": [0, 0, 0]},
                       {"id": "robot_B", "spawn": {"at": [10.11, -2.37], "yaw": 2.06}, "launch_pad": [0, -1.2, 0]}]}
    assert list(start_pose(gate, "robot_B")) == list(start_pose(gate, "robot_A")) == [10.0, -1.2, 0.0]
    two = {"robots": [{"id": "robot_A", "spawn": {"at": [-1.2, 7.0], "yaw": 90}},
                      {"id": "robot_B", "spawn": {"at": [10.0, -1.2], "yaw": 0}}]}
    assert list(start_pose(two, "robot_B")[:2]) == [10.0, -1.2] and list(start_pose(two, "robot_A")[:2]) == [-1.2, 7.0]
