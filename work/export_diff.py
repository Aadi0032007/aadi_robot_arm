from pathlib import Path
import difflib
root=Path.cwd()
backup=root/'work/before-nero-merge-20261031-191451'
files=['GAMEPAD_REMOTE.md','src/config.py','src/gamepad_remote_client.py','src/gamepad_teleop.py','src/gamepad_udp.py','src/nero_robot.py','src/teleop.py','tests/test_gamepad_udp.py','tests/test_nero_ik.py']
output=root/'outputs'
output.mkdir(exist_ok=True)
parts=[]
for rel in files:
    before=(backup/rel).read_text(encoding='utf-8').splitlines(keepends=True)
    after=(root/rel).read_text(encoding='utf-8').splitlines(keepends=True)
    parts.append('diff --git a/'+rel+' b/'+rel+'\n')
    parts.extend(difflib.unified_diff(before,after,fromfile='a/'+rel,tofile='b/'+rel))
path=output/'nero-gamepad-changes.diff'
path.write_text(''.join(parts),encoding='utf-8')
print(path)
print(''.join(parts).count('\n'),'diff lines')
