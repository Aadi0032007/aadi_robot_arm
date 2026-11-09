from pathlib import Path
import json,hashlib,ast
root=Path.cwd()
p=root/'GAMEPAD_REMOTE.md'
t=p.read_text()
t=t.replace('In `src.teleop`, no gamepad commands are transmitted during that wait;', 'In `src.teleop`, ordinary stick commands stop during that wait; stored-pose moves continue. For stick control,')
t=t.replace('`brake=1`\n(unplug or Ctrl-C) holds the arm.', '`brake=1`\n(unplug or Ctrl-C) makes controller input unavailable; ordinary stick control holds,\nbut an active stored-pose move continues.')
p.write_text(t,encoding='utf-8')
manifest=json.loads((root/'work/before-nero-merge-20261031-191451/manifest.json').read_text())
changed=[]
for rel,digest in manifest.items():
    p=root/rel
    assert p.exists(),rel
    if hashlib.sha256(p.read_bytes()).hexdigest()!=digest: changed.append(rel.replace('\\','/'))
expected={'GAMEPAD_REMOTE.md','src/config.py','src/gamepad_remote_client.py','src/gamepad_teleop.py','src/gamepad_udp.py','src/nero_robot.py','src/teleop.py','tests/test_gamepad_udp.py','tests/test_nero_ik.py'}
assert set(changed)==expected,changed
for p in (root/'src').glob('*.py'): ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
from src.config import load
cfg=load('nero_gamepad')
assert cfg.robot['channel']=='can1' and cfg.robot['firmware_version']=='v121'
assert all(cfg.teleop[k] for k in ('pick_on_x','place_on_y','look_on_b','invert_gripper'))
assert hasattr(cfg,'stream') and hasattr(cfg.ai,'max_repeated_frames')
print('Verified: only 9 intended files changed; all other existing files preserved; source syntax valid; merged presets and streaming configuration present.')
