# Nero + AGX gripper model

`source/` contains the official arm URDF and its flange/gripper Xacro includes,
downloaded from https://github.com/agilexrobotics/agx_arm_urdf/tree/main/nero/urdf.
`nero_with_gripper.urdf` expands those three files into plain XML for the numpy
kinematics parser. Rebuild from the workspace root:

```powershell
python scripts/build_nero_urdf.py
```

Frames:

- `base_link`: Cartesian reference frame.
- `link7`: matches the SDK Nero MDH endpoint; default IK tip.
- `flange_link`: accessory mounting frame with the official fixed transform.
- `gripper_base`: 6 mm along flange_link Z.
- `gripper_tcp`: a nominal midpoint at the finger joint reference plane, 138 mm
  along gripper_base Z. This is not a measured fingertip/contact calibration.
  `NeroKinematics` can override its translation and rotation without editing XML.

The arm chain to any of these tips contains only joint1 through joint7.
The opening/finger mimic joints remain separate branches. Mesh references retain
the upstream `package://agx_arm_description/...` paths; install the upstream
description package for visualization. Meshes are not needed by numpy FK/IK.

Joint geometry, limits and inertia are preserved from upstream. Neither the
URDF parser nor the IK solver performs collision checking.
