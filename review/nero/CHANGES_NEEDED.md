# Nero Cartesian support review

Historical review of official AgileX model files and the local Revobots Cartesian stack.
The subsequent implementation is documented in `../../NERO_IK.md`; native driver,
model wrapper, CLI and gamepad/VR presets are now present. No hardware was connected.

## Model and tool frames

Use the official seven-joint chain `base_link -> link7`. Keep joint names `joint1` through `joint7`, their order, origins, axes, and limits. `nero_flange_review.urdf` expands the official arm and flange include for this offline review; it retains package mesh references and is not a standalone visualization bundle.

The local SDK's Nero MDH FK matches `link7` (worst sampled discrepancy 0.0000571 mm and 0.0000103 degrees). Its API labels this the flange pose. The model's extra `flange_link` is a different frame, with link7-to-flange transform xyz `[0.031, 0, -0.0235]` m and RPY `[-1.5708, 0, -1.5708]` rad. Comparing it directly to SDK FK creates approximately 38.9 mm / 120 degree error. Keep SDK flange, accessory mounting flange, and tool TCP distinct.

For a measured tool offset `T_link7_tcp`, compute SDK flange commands as `T_base_link7 = T_base_tcp @ inverse(T_link7_tcp)`. Do not assume the TCP is the gripper base or a moving finger. Add a fixed TCP link with a measured translation and rotation. The gripper Xacro attaches its base 6 mm along the accessory flange's Z axis. The actual grasp point needs the installed tool geometry/calibration.

The current XML parser does not expand Xacro includes. Generate a plain URDF first. Mesh paths do not affect this numpy FK/IK, but must resolve for visualization or collision checking. The current solver does not check collisions.

## Driver contract

Expose all seven observed and commanded joints, in an explicit fixed order, with degrees at the LeRobot boundary and radians inside the SDK. Handle the gripper separately in its own units. Ensure FK(observation) describes the same configuration actually sent to the robot.

One local driver copy at `../aditya_lerobot_v0.4/src/lerobot/robots/agx_nero_follower/agx_nero_follower.py` exposes only joints 1, 2, 4, 5, 7 as five Koch-style names; its send path sets joints 3 and 6 to zero and applies leader offsets/sign changes. This is unsuitable as a native seven-joint Cartesian interface. Another nearby repository contains summaries of a newer seven-joint driver. Confirm the actual imported module path before changing the driver.

Move Koch-to-Nero retargeting into a separate leader action adapter. Native IK output must bypass that retargeting. Never rely on action dictionary iteration order. Require finite, complete joint commands and map reported/applied actions consistently.

## IK and command generation

Reuse `SerialChain` and `ArmKinematics`; they already support seven joints. Add `nero_ik.py` as a model/configuration wrapper and Cartesian move CLI. Use full orientation (`orientation_axes=None`), previous accepted solution warm starts, and measured-state seeding on start or clutch engage. Check both position and orientation residuals. The SO101 code's pitch/roll-only selector is inappropriate for Nero full-pose tracking.

Add a tertiary posture objective in the null space of the full pose task to control the redundant elbow. Use a preferred bent posture and joint-limit avoidance; keep its effect below position/orientation priorities. Add singularity-aware damping and a bounded solve budget, and test their effects rather than assuming more iterations improve continuity.

Use time-based joint velocity/acceleration limits, Cartesian translation/angular rate limits, and measured-vs-command tracking error checks. Preserve continuity when rejecting targets. Use the intersection of model and configured/controller joint limits; the SDK's rounded preset is slightly wider than this URDF. The URDF's uniform 5 rad/s velocity value is not a tuned teleoperation speed.

## Controls and presets

- `config.py`: retain the working leader preset; add Nero Cartesian/gamepad/VR presets with seven-joint model configuration and no Koch leader action mapping.
- `gamepad_teleop.py`: Nero uses `orientation_mode='frame'`. Reuse base/tool twists and full orientation; arm mode intentionally rotates the shoulder for yaw and can use direct wrist joints.
- `vr_teleop.py`: add full SE(3) clutch mode. Existing behavior tracks only controller pitch/roll and synthesizes arm pan orientation, so supplying a different URDF alone does not provide Nero orientation control.
- `so101_ik.py`: keep its five-joint wrapper separate; move reusable Cartesian trajectory execution into shared code for `nero_ik.py`.
- Keyboard Cartesian controls can use the full-pose jog path; direct joint controls need all seven joint keys.

## SDK and firmware

The current preset uses `v111`. Select the driver profile matching the actual arm firmware; do not infer that all versions >=1.11 use v111. Official documentation defines newer profiles.

`get_ik_joint_angles()` is feedback after controller Cartesian motion, requires firmware >=1.12, and is not an offline pose-to-joints solver. `move_p`/`move_l`/`move_c` use controller IK and can be considered a separate control backend; do not call them merely to obtain a dry-run solution. Choose and validate the SDK streaming mode for host-generated joint targets separately from one-shot trajectories.

## Offline verification

`check_nero.py` builds the arm + accessory flange review model and checks the actual shared solver without hardware. `results.json` records the outputs. Seed 42, 200 target configurations within the inner 70% of the URDF joint ranges, seeds perturbed by 10 degrees:

- 200/200 within 1 mm position and 1 degree orientation.
- Median position error 0.01995 mm; orientation error 0.000507 degrees.
- Typical solve time about 2.5 ms in the available Python environment.
- 239 subsequent samples on a small reachable continuous pose path: zero failures; largest joint step 1.94 degrees. These are trajectory samples, not velocity guarantees.
- Jacobian finite-difference maximum discrepancies: 3.05e-7 linear and 3.01e-10 angular.

This verifies model mathematics and nearby-seed IK, not collision safety, arbitrary workspace reachability, singularity robustness, transport timing, or physical smoothness. Before real Cartesian operation, validate zero/sign/order and TCP against measured poses, then test slow translation and orientation while recording tracking error.

Sources:
- https://github.com/agilexrobotics/agx_arm_urdf/tree/main/nero
- https://raw.githubusercontent.com/agilexrobotics/agx_arm_urdf/main/nero/urdf/nero_with_flange_description.xacro
- https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/api/constants.py
- https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/nero_api.md
- https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/firmware_reference.md
