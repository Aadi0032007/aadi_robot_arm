# Nero Cartesian IK

Use the shared numpy FK/Jacobian/task-priority IK through `src/nero_ik.py`.
It tracks XYZ and full RPY with all seven Nero arm joints. Like the SO101 tool,
omitted pose components keep the starting value and the move follows a Cartesian
line with orientation interpolation. The Nero version uses quintic timing and
preflights every point, including pose residuals, joint limits, speed and
acceleration, before enabling/sending targets.

## Commands

No hardware or LeRobot needed (numpy only):

```powershell
python -m src.nero_ik --self_test
python -m src.nero_ik --tip_link=gripper_tcp --self_test
python -m src.nero_ik --seed_deg=0,20,0,60,0,10,0 --z=0.50 --dry_run
```

Read the connected arm or solve a proposed move without enabling or sending
joint/gripper targets. These commands require pyAgxArm and the CAN adapter:

```powershell
python -m src.nero_ik --print_pose
python -m src.nero_ik --z=0.50 --dry_run
```

Execute after reviewing the target for your actual starting pose:

```powershell
python -m src.nero_ik --z=0.50 --move_time_s=5
```

XYZ values are metres in base_link; roll/pitch/yaw are degrees. The default
target point is `link7`, which agrees with the SDK's wrist/flange FK.
`--robot.interface`, `--robot.channel`, `--robot.firmware_version`,
`--robot.speed_percent` and `--fps` use the existing override syntax.
Firmware defaults to the current preset's v111; choose the profile matching
the installed arm firmware.

For the grasp point, select `--tip_link=gripper_tcp`. Its nominal XYZ is
`0,0,0.138` relative to gripper_base, with zero RPY. Calibrate this grasp reference
and supply `--tcp_xyz_m=x,y,z` and `--tcp_rpy_deg=r,p,y` as needed. These offsets
are relative to gripper_base, not link7. The nominal 138 mm reference is inferred
from finger joint origins and is not a measured fingertip centre.

`--max_pos_error_mm=1`, `--max_rot_error_deg=1`,
`--max_joint_speed_deg_s=30` and `--max_joint_accel_deg_s2=120` are move defaults.
If preflight reports a speed/acceleration violation, increase move_time_s.
An unreachable pose fails without sending targets. Motion uses SDK move_j;
controller tracking errors can still make the physical path differ from the plan.

## Gamepad and VR

Run with the existing LeRobot environment and pyAgxArm installed:

```powershell
python -m src.teleop --preset nero_gamepad
python -m src.teleop --preset nero_vr
```

For a gamepad on another computer, add `--port=5001` to the Nero gamepad command
and run `python -m src.gamepad_remote_client --ip=ROBOT_IP --port=5001` there.
Omit `--port` for local USB. See [GAMEPAD_REMOTE.md](GAMEPAD_REMOTE.md).

Both presets select the native seven-joint driver and `link7` tip. Gamepad uses
full base/tool-frame twists, with no RB requirement, and 0.04 m/s and 0.5 rad/s maximum
Cartesian input rates at scale 1. A opens the gripper and B closes it, just as on
SO101. The action has seven arm joints plus `gripper.pos` in 0..100% opening.
VR uses full controller rotation relative to
clutch engage, including yaw; its joint follower defaults to 30 deg/s and
120 deg/s². It interpolates both position and orientation when reducing an
unreachable target. SO101 remains in its existing arm orientation mode.

To use the nominal gripper reference in either preset:

```powershell
python -m src.teleop --preset nero_gamepad --teleop.tip_link=gripper_tcp
```

For a calibrated TCP in teleop, update gripper_tcp_joint in a copy of the generated
URDF and select it with `--teleop.urdf_path`; the CLI's offset flags apply to
`nero_ik` only.

## Native joint interface

`src/nero_robot.py` reports and accepts `joint1.pos` through `joint7.pos` in
degrees, explicitly ordered before conversion to SDK radians. It applies no
Koch leader offsets and does not zero redundant joints. Feedback is checked for
missing/nonfinite values and frozen timestamps. The driver rejects out-of-limit
targets and targets over 5 degrees from the measured state by default.

In `src.teleop`, the gamepad receives current joint feedback every tick. While
local/UDP input is missing it re-anchors its starting target to those measurements
and the loop sends no gamepad commands. If a live command differs from measured
joints by more than 5 degrees, gamepad control pauses and re-anchors. Release
sticks, triggers, D-pad rotation and A/B to re-arm, then move again. The driver
also checks immediately before transmission; a tracking rejection there is caught
for human gamepad control and pauses the session without sending the rejected
target. Its diagnostic identifies the joint, target, measurement and difference.
Other validation/feedback failures still stop the loop. The standalone move CLI
continues to abort on tracking faults.

The existing `nero` leader preset is retained. Native seven-joint recordings use
a separate dataset/schema; a five-joint leader policy is not directly compatible.
Disconnecting the native driver leaves torque enabled by default, avoiding an
automatic drop of the raised arm. `disable_on_disconnect` is configurable.

The `nero_gamepad` preset enables the gripper. It uses `gripper_mode=auto`, which
retains the SDK's reported width/angle mode and exposes a normalized 0..100
command: 100 means 0.1 m in width mode or 100 degrees in angle mode. A/B change
this value at 60 units/s. Startup seeds the command from measured opening without
switching gripper modes or moving to a preset opening. Width and motor angle are
not mechanically converted into each other.

Gripper commands remain off in the standalone Cartesian and VR presets, preserving
opening during their arm moves. Optional explicit `gripper_mode=width` uses metres;
`gripper_mode=angle` uses degrees, and each must match reported feedback. If enabling
width control in VR use `gripper_max=0.1`. Keep these units consistent.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m src.nero_ik --self_test
python -m src.so101_ik --self_test
```

The tests use numerical models and a fake SDK, never real CAN or servos. The
Nero self-test covers Jacobian finite differences, 300 nearby-seed pose round
trips, and a continuous Cartesian path. Hardware motion, tool calibration,
collision avoidance and singularity coverage require separate validation.
