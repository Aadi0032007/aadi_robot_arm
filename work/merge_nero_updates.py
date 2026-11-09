from pathlib import Path
import shutil, hashlib, json, datetime
root=Path(r'C:\Users\Aadi\aditya\revobots_arm')
source=Path(r'C:\Users\Aadi\Desktop\updated\aadi_robot_arm')
files=['src/gamepad_remote_client.py','src/gamepad_udp.py','src/gamepad_teleop.py','src/nero_robot.py','tests/test_gamepad_udp.py','tests/test_nero_ik.py','GAMEPAD_REMOTE.md','src/config.py','src/teleop.py']
backup=root/'work'/('before-nero-merge-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and 'work' not in p.relative_to(root).parts and '__pycache__' not in p.parts}
for rel in files:
    dest=backup/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/rel,dest)
(backup/'manifest.json').write_text(json.dumps(manifest,indent=2))
for rel in files[:7]:
    shutil.copy2(source/rel,root/rel)
p=root/'src/config.py'
text=p.read_text(encoding='utf-8')
new=(source/'src/config.py').read_text(encoding='utf-8')
a,b=text.index('def nero()'),text.index('def koch()')
c,d=new.index('def nero()'),new.index('def koch()')
text=text[:a]+new[c:d]+text[b:]
text=text.replace('# Same button behavior as SO101: sticks are live, RB opens, LB closes.', '# Nero controls: sticks are live, LB opens, RB closes; X/Y/B select poses.')
p.write_text(text,encoding='utf-8')
p=root/'src/teleop.py'
text=p.read_text(encoding='utf-8')
new=(source/'src/teleop.py').read_text(encoding='utf-8')
a=text.index('                except NeroTrackingError as e:')
b=text.index('\n            obs_processed = None',a)
c=new.index('                except NeroTrackingError as e:')
d=new.index('\n            obs_processed = None',c)
p.write_text(text[:a]+new[c:d]+text[b:],encoding='utf-8')
p=root/'GAMEPAD_REMOTE.md'
text=p.read_text(encoding='utf-8').replace('    L1 / R1                gripper close / open','    L1 / R1                gripper open / close')
text += '\nNero gamepad preset updates: SocketCAN `can1`, firmware profile `v121` (no firmware flashing),\nstartup rest pose, X = pick, Y = place, B = look. Pose JSON files override the configured\nfallback angles. Stored-pose moves continue if controller input disconnects; ordinary\nstick control holds. LB/L1 opens the gripper and RB/R1 closes it in this preset.\n'
p.write_text(text,encoding='utf-8')
print('Backup:',backup)
changed=[rel for rel,old in manifest.items() if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=old]
print('Changed files:',json.dumps(changed,indent=2))
assert {Path(p).as_posix() for p in changed}==set(files)

