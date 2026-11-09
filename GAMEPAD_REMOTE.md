# Local or UDP gamepad control

Leave out `--port` for the current local USB gamepad behavior. Add `--port=5001`
to use a UDP listener on `0.0.0.0:5001` instead. The robot does not need pygame
for UDP input. It still needs its normal robot/LeRobot/IK environment.

Nero, on the robot computer:

```powershell
python -m src.teleop --preset nero_gamepad --port=5001
```

The `nero_gamepad` preset inverts X, Y, Z and roll by default; no
`--teleop.invert` argument is needed. An explicit argument overrides this default.

SO101, on the robot computer:

```powershell
python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --teleop.type=gamepad_ee --port=5001
```

On the computer with the physical gamepad, use your robot computer's IP:

```powershell
python -m src.gamepad_remote_client --ip=192.168.1.20 --port=5001
```

The client now sends Porter/RevoPilot **protocol v2** (named `lx/ly/rx/ry`,
`dx/dy`, `L1/R1`, â€¦). The robot still accepts the older `gamepad_state` v1
packets. Use `--protocol=v1` only if you need the original schema.

Or run the file directly:

```powershell
python src/gamepad_remote_client.py --ip=192.168.1.20 --port=5001
```

Client defaults: `--ip=127.0.0.1`, `--port=5001`, `--fps=60`, `--protocol=v2`. Thus
`python -m src.gamepad_remote_client` works when sender and robot run on the
same computer. Install pygame on the gamepad computer; it uses the existing
`GamepadReader` and identical SDL/W3C mapping. No robot-specific configuration,
LeRobot or robot SDK is needed by the client. Configure IK and robot hardware on
the receiving computer as usual. The client is the same for every robot using
`gamepad_ee`, including custom URDF/motor mappings.

Local controls and UDP controls share stick/button processing, precision and
reference-frame toggles, and RB/LB gripper open/close. D-pad up/down controls
pitch; D-pad left/right controls roll. Set speed with `--teleop.initial_speed_level`,
`--teleop.max_linear_speed` and `--teleop.max_angular_speed`; X toggles precision.
A/B and RT are unused. LT is the optional deadman when `--teleop.require_deadman=true`
(disabled by default). All robot-specific
settings remain on the receiver. The receiver also supports
`--teleop.udp_port=5001` and `--teleop.udp_timeout_s=0.3`. `--port` is a shortcut
for gamepad input, separate from robot serial/CAN settings and remote leader ports.
The same override works in `src.record`.

To compare exactly what is sent and received, enable packet debug on both computers:

```powershell
# Robot computer (Nero example; --debug also works with the SO101 command)
python -m src.teleop --preset nero_gamepad --robot.channel=can1 --port=5001 --debug=true
# Gamepad computer
python -m src.gamepad_remote_client --ip=ROBOT_IP --port=5001 --debug=true
```

Both default to `--debug=false`. A bare `--debug` also enables it. The robot
accepts `--teleop.debug=true` as well. `GAMEPAD TX` lines show the destination and
the full JSON packet; `GAMEPAD RX` lines show the sender, accepted/rejected/ignored
status and packet contents. Compare `session` and `seq` across the two terminals.
The payload includes connection state, all four axes and all seventeen button/
trigger values. The receiver reports timeouts and duplicate/out-of-order frames.
The robot's existing debug output also shows active inputs after mapping. Packet
debug prints every packet, so use it temporarily when diagnosing input/network
behavior; console output can slow the control loop.

SO101 arm-mode translation now prefers the current tilt/roll but relaxes this
orientation hold when it blocks Cartesian progress. This helps reach positions
that need the wrist to change angle, as it naturally does during leader control.
Use `--teleop.translation_orientation=hold` for the previous behavior or `free`
for position-only translation. Explicit rotation inputs still retain their targets.
The model joint limits, 1-degree margin and 2 cm tool-height floor remain in force.
Nero full-frame orientation control is unchanged.

The client streams unchanged idle states as heartbeats. Unplugging the gamepad
or exiting sends a disconnected packet. Missing updates for 0.3 seconds return
no input. In `src.teleop`, ordinary stick commands stop during that wait; stored-pose moves continue. For stick control,
the integrated target is re-anchored to measured joints before input resumes.
Nero's 5-degree tracking guard pauses gamepad control instead of terminating
teleop. Release sticks, D-pad and RB/LB to re-arm from the
measured pose. Malformed input and duplicate
or reordered packets do not refresh the watchdog. The reader drains queued
packets to use the newest state; one active sender/session owns control until
it disconnects or times out. Sending resumes after client/gamepad reconnect.

Default packet schema is Porter/RevoPilot v2 (UTF-8 JSON):

```json
{"seq":0,"t":0,"brain":"HI","protocol":"v2","brake":0,"A":0,"B":0,"X":0,"Y":0,"L1":0,"R1":0,"L3":0,"R3":0,"Options":0,"Menu":0,"Home":0,"L4":0,"R4":0,"M1":0,"M2":0,"lx":0,"ly":0,"rx":0,"ry":0,"dx":0,"dy":0,"l2":0,"r2":0}
```

`lx/ly/rx/ry` are sticks in [-1,1] (W3C: up/left negative). `dx/dy` are the
hat (`+1` right/up, `-1` left/down). `L1`/`R1` are the bumpers. `brake=1`
(unplug or Ctrl-C) makes controller input unavailable; ordinary stick control holds,
but an active stored-pose move continues. Sequence numbers increase throughout a
client session.

The older v1 schema is still accepted (`--protocol=v1`):

```json
{"type":"gamepad_state","version":1,"session":"client-session-id","seq":0,"connected":true,"axes":[0,0,0,0],"buttons":[]}
```

Nero gamepad mapping (same as the working reference robot):

    Left stick up/down     X
    Left stick left/right  Y
    Right stick up/down    Z
    Right stick left/right yaw (Cartesian Z; often felt as turning)
    Hat up/down            pitch
    Hat left/right         roll
    L1 / R1                gripper open / close

Tests use loopback sockets and fake controllers; they never access robot CAN
or servos. Networked use between two physical computers remains to be tested.


Nero gamepad preset updates: SocketCAN `can1`, firmware profile `v121` (no firmware flashing),
startup rest pose, X = pick, Y = place, B = look. Pose JSON files override the configured
fallback angles. Stored-pose moves continue if controller input disconnects; ordinary
stick control holds. LB/L1 opens the gripper and RB/R1 closes it in this preset.
