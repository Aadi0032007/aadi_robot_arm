"""Offline Nero URDF/Jacobian/IK review. Does not connect to hardware."""
from pathlib import Path
import ast
import copy
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.kinematics import SerialChain, make_transform, matrix_to_rotvec, solve_ik

HERE = Path(__file__).resolve().parent
robot = ET.parse(HERE / 'nero_description.urdf').getroot()
flange = ET.parse(HERE / 'nero_with_flange_description.xacro').getroot()
for element in flange:
    if element.tag in ('link', 'joint'):
        robot.append(copy.deepcopy(element))
ET.indent(robot)
ET.ElementTree(robot).write(HERE / 'nero_flange_review.urdf', encoding='utf-8', xml_declaration=True)
chain = SerialChain.from_urdf(HERE / 'nero_flange_review.urdf', 'flange_link', 'base_link')
sdk_chain = SerialChain.from_urdf(HERE / 'nero_description.urdf', 'link7', 'base_link')
assert chain.n == 7
constants = ast.parse(Path(r'C:\Users\Aadi\aditya\pyAgxArm\pyAgxArm\api\constants.py').read_text(encoding='utf-8'))
mdh = next(ast.literal_eval(n.value)['nero'] for n in constants.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROBOT_MDH_PRESET' for t in n.targets))
def sdk_fk(q):
    T = np.eye(4)
    for qi, (d, a, alpha, offset) in zip(q, mdh):
        T = T @ make_transform([a, 0, 0], [alpha, 0, 0]) @ make_transform([0, 0, d], [0, 0, qi+offset])
    return T

rng = np.random.default_rng(42)
mid, half = .5*(chain.lower+chain.upper), .5*(chain.upper-chain.lower)
worst_jv = worst_jw = worst_sdk_p = worst_sdk_r = 0.
worst_link7_p = worst_link7_r = 0.
hits = 0
pos, rot, times = [], [], []
for i in range(200):
    q = mid + .7*half*rng.uniform(-1,1,7)
    T, J = chain.fk_jacobian(q)
    S = sdk_fk(q)
    U = sdk_chain.fk(q)
    worst_link7_p = max(worst_link7_p, float(np.linalg.norm(U[:3,3]-S[:3,3])))
    worst_link7_r = max(worst_link7_r, float(np.linalg.norm(matrix_to_rotvec(U[:3,:3] @ S[:3,:3].T))))
    worst_sdk_p = max(worst_sdk_p, float(np.linalg.norm(T[:3,3]-S[:3,3])))
    worst_sdk_r = max(worst_sdk_r, float(np.linalg.norm(matrix_to_rotvec(T[:3,:3] @ S[:3,:3].T))))
    if i < 20:
        for k in range(7):
            qp = q.copy()
            qp[k] += 1e-6
            Tp = chain.fk(qp)
            worst_jv = max(worst_jv, float(np.linalg.norm((Tp[:3,3]-T[:3,3])/1e-6-J[:3,k])))
            worst_jw = max(worst_jw, float(np.linalg.norm(matrix_to_rotvec(Tp[:3,:3] @ T[:3,:3].T)/1e-6-J[3:,k])))
    seed = np.clip(q+rng.normal(0, math.radians(10), 7), chain.lower, chain.upper)
    started = time.perf_counter()
    result = solve_ik(chain, T, seed, orientation_axes=None, max_iters=100)
    times.append(1000*(time.perf_counter()-started))
    pos.append(1000*result.pos_err)
    rot.append(math.degrees(result.rot_err))
    hits += result.pos_err < .001 and result.rot_err < math.radians(1)

q = mid.copy()
path_failures = 0
max_jump = 0.
for t in np.linspace(0, 2*math.pi, 240):
    truth = mid + .2*half*np.sin(t+np.arange(7)*.3)
    target = chain.fk(truth)
    result = solve_ik(chain, target, q, max_iters=30)
    if t > 0:
        max_jump = max(max_jump, math.degrees(float(np.max(np.abs(result.q-q)))))
        path_failures += result.pos_err > .001 or result.rot_err > math.radians(1)
    q = result.q
report = dict(joints=chain.joint_names, tip=chain.tip_link,
    limits_deg=np.degrees(np.stack([chain.lower,chain.upper],axis=1)).tolist(),
    jacobian_error_linear=worst_jv, jacobian_error_angular=worst_jw,
    sdk_fk_worst_position_mm=1000*worst_sdk_p, sdk_fk_worst_orientation_deg=math.degrees(worst_sdk_r),
    sdk_link7_worst_position_mm=1000*worst_link7_p, sdk_link7_worst_orientation_deg=math.degrees(worst_link7_r),
    round_trip_success=hits, round_trip_trials=200, median_position_error_mm=float(np.median(pos)),
    median_orientation_error_deg=float(np.median(rot)), median_solve_ms=float(np.median(times)),
    trajectory_failures=path_failures, trajectory_steps=239, max_trajectory_joint_step_deg=max_jump)
(HERE / 'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
