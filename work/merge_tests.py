from pathlib import Path
import ast
root=Path.cwd()
backup=root/'work/before-nero-merge-20261031-191451'
p=root/'tests/test_nero_ik.py'
src=p.read_text()
text=(backup/'tests/test_nero_ik.py').read_text()
tree=ast.parse(src)
lines=src.splitlines(keepends=True)
methods=[]
for node in ast.walk(tree):
    if isinstance(node,ast.FunctionDef) and node.name in ('test_b_glides_to_look_pose','test_pose_glide_stays_within_tracking_limit_on_slow_tick','test_pose_glide_holds_on_tracking_reject_instead_of_pausing'):
        methods.append(''.join(lines[node.lineno-1:node.end_lineno])+'\n\n')
text=text.replace('    def test_optional_deadman_uses_lt_without_opening_gripper(self):',''.join(methods)+'    def test_optional_deadman_uses_lt_without_opening_gripper(self):')
text=text.replace('from src.gamepad import GamepadState, BTN_A, BTN_B, BTN_RT, BTN_LT','from src.gamepad import GamepadState, BTN_A, BTN_RT, BTN_LT').replace('for button in (BTN_A, BTN_B, BTN_RT, BTN_LT):','for button in (BTN_A, BTN_RT, BTN_LT):')
old="""        for value in (float('nan'),200.):
            with self.assertRaises(ValueError):
                robot.send_action({'joint1.pos':value})
        self.assertEqual(len(self.sent),1)
"""
new="""        with self.assertRaises(ValueError):
            robot.send_action({'joint1.pos':float('nan')})
        self.assertEqual(len(self.sent),1)
        robot.send_action({'joint1.pos':200.})
        self.assertEqual(len(self.sent),2)
        self.assertAlmostEqual(self.sent[-1][0], np.radians(robot._upper[0]))
"""
assert old in text
text=text.replace(old,new).replace('test_native_order_no_leader_offsets_and_limit_rejection','test_native_order_no_leader_offsets_and_limit_clipping')
# Existing Nero gripper tests use RB as the opening input; swap their inputs to match the new preset.
for name in ('test_optional_deadman_uses_lt_without_opening_gripper','test_tracking_error_pauses_until_neutral_and_then_resumes','test_nero_sticks_and_bumpers_send_seven_joints_and_gripper_without_deadman'):
    tree=ast.parse(text)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=text.splitlines(keepends=True)
    segment=''.join(lines[node.lineno-1:node.end_lineno])
    segment=segment.replace('BTN_RB','BTN_SWAP').replace('BTN_LB','BTN_RB').replace('BTN_SWAP','BTN_LB')
    text=''.join(lines[:node.lineno-1])+segment+'\n'+''.join(lines[node.end_lineno:])
p.write_text(text,encoding='utf-8')
print('Preserved existing Nero tests, merged three pose tests, updated changed behavior expectations.')
