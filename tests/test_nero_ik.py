"""No CAN, servo enable, gamepad or headset access: numerical/fake SDK checks."""
import ast
from dataclasses import dataclass
import importlib.machinery
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load
from src.devices import make_robot
from src.kinematics import IKResult, matrix_to_rotvec, rotvec_to_matrix
from src.nero_ik import nero_kinematics, plan_cartesian, reachable
from src.nero_robot import NeroJointRobot, NeroRobotConfig, NeroTrackingError, NERO_JOINTS


class NeroIKTests(unittest.TestCase):
    def test_chain_and_motor_round_trip(self):
        kin = nero_kinematics()
        self.assertEqual(kin.chain.joint_names, NERO_JOINTS)
        values = {f'{n}.pos': float(i) for i,n in enumerate(NERO_JOINTS)}
        np.testing.assert_allclose(list(kin.q_to_motor(kin.motor_to_q(values)).values()), list(values.values()))

    def test_tcp_fixed_transform_and_no_fingers_in_arm_chain(self):
        wrist = nero_kinematics()
        tcp = nero_kinematics(tip_link='gripper_tcp',tcp_xyz_m=[0,0,.15],tcp_rpy_deg=[0,0,0])
        self.assertEqual(tcp.chain.n,7)
        q = np.radians([0,20,0,60,0,10,0])
        transform = np.linalg.inv(wrist.fk(q)) @ tcp.fk(q)
        q[0] += .4
        np.testing.assert_allclose(np.linalg.inv(wrist.fk(q)) @ tcp.fk(q), transform, atol=1e-12)
        self.assertGreater(np.linalg.norm(transform[:3,3]), .1)

    def test_continuous_path_and_infeasible_target(self):
        kin = nero_kinematics()
        q = np.radians([0,20,0,60,0,10,0])
        T = kin.fk(q)
        T[2,3] += .02
        path,dt = plan_cartesian(kin,q,T)
        self.assertLess(np.linalg.norm(kin.fk(path[-1])[:3,3]-T[:3,3]),.001)
        velocity = np.diff(np.vstack([q,path]),axis=0)/dt
        self.assertLess(np.max(np.abs(np.degrees(velocity))),30)
        T[0,3] = 10.
        with self.assertRaises(ValueError):
            plan_cartesian(kin,q,T)

    def test_orientation_failure_rejected(self):
        self.assertFalse(reachable(IKResult(np.zeros(7),0.,.5,1),.001,.02))
        self.assertTrue(reachable(IKResult(np.zeros(7),0.,.5,1),.001,.02,0.))

    def test_invalid_parameters_and_start_limits(self):
        kin = nero_kinematics()
        q = np.zeros(7)
        for kwargs in ({'duration_s':0},{'fps':float('nan')},{'orientation_weight':-1}):
            with self.assertRaises(ValueError):
                plan_cartesian(kin,q,kin.fk(q),**kwargs)
        q[3] = 3.
        with self.assertRaises(ValueError):
            plan_cartesian(kin,q,kin.fk(q))

    def test_presets_preserve_leader_and_construct_native(self):
        self.assertEqual(load('nero').robot_type,'agx_nero_follower')
        for name in ('nero_cartesian','nero_gamepad','nero_vr'):
            cfg = load(name)
            self.assertEqual(cfg.robot_type,'agx_nero_native')
            self.assertNotIn('teleop_action_joints',cfg.robot)
            self.assertEqual(len(cfg.teleop['motors']),7)
            self.assertEqual(cfg.teleop['orientation_mode'],'frame')
            self.assertIsInstance(make_robot(cfg),NeroJointRobot)

    def native(self, with_gripper=False):
        robot = NeroJointRobot(NeroRobotConfig(with_gripper=with_gripper,max_relative_target=None))
        robot._connected = robot._enabled = True
        robot._arm = SimpleNamespace(get_joint_angles=lambda:SimpleNamespace(msg=[0.]*7,timestamp=time.monotonic()),
            move_j=lambda q: self.sent.append(q),disconnect=lambda:None)
        self.sent = []
        return robot

    def test_native_order_no_leader_offsets_and_limit_rejection(self):
        robot = self.native()
        action = {f'joint{i}.pos':float(i) for i in reversed(range(1,8))}
        robot.send_action(action)
        np.testing.assert_allclose(self.sent[0],np.radians(np.arange(1,8)))
        for value in (float('nan'),200.):
            with self.assertRaises(ValueError):
                robot.send_action({'joint1.pos':value})
        self.assertEqual(len(self.sent),1)
        robot.config.max_relative_target = 1.
        with self.assertRaises(ValueError):
            robot.send_action({'joint1.pos':2.})

    def test_stale_feedback_rejected(self):
        robot = self.native()
        robot._arm.get_joint_angles = lambda:SimpleNamespace(msg=[0.]*7,timestamp=1)
        robot.get_observation()
        robot._feedback_changed_at -= 1.
        with self.assertRaises(TimeoutError):
            robot.send_action({'joint1.pos':.1})
        self.assertFalse(self.sent)

    def test_tracking_fault_identifies_joint_and_sends_nothing(self):
        robot=self.native()
        robot.config.max_relative_target=5.
        with self.assertRaises(NeroTrackingError) as caught:
            robot.send_action({'joint3.pos':7.})
        self.assertIn('joint3 target 7.00 deg, measured 0.00 deg',str(caught.exception))
        self.assertEqual(caught.exception.measured_action['joint3.pos'],0.)
        self.assertEqual(self.sent,[])

    def test_gripper_width_is_not_degrees(self):
        robot = self.native(with_gripper=True)
        commands = []
        robot._gripper = SimpleNamespace(get_gripper_status=lambda:SimpleNamespace(msg=SimpleNamespace(value=.04,mode='width')),
            move_gripper_m=lambda **kw:commands.append(kw))
        robot.send_action({'gripper.pos':.03})
        self.assertEqual(commands[0]['value'],.03)
        with self.assertRaises(ValueError):
            robot.send_action({'gripper.pos':30})

    def test_read_only_sdk_connection_never_enables_or_moves(self):
        calls = []
        sdk = ModuleType('pyAgxArm')
        arm = SimpleNamespace(connect=lambda:calls.append('connect'),disconnect=lambda:calls.append('disconnect'),
            get_joint_angles=lambda:SimpleNamespace(msg=[0.]*7,timestamp=time.monotonic()))
        sdk.create_agx_arm_config = lambda **kw:kw
        sdk.AgxArmFactory = SimpleNamespace(create_arm=lambda cfg:arm)
        with patch.dict(sys.modules,{'pyAgxArm':sdk}):
            robot = NeroJointRobot(NeroRobotConfig(enable_on_connect=False))
            robot.connect()
            self.assertFalse(robot._enabled)
            with self.assertRaises(RuntimeError):
                robot.send_action({'joint1.pos':0.})
            robot.disconnect()
        self.assertEqual(calls,['connect','disconnect'])


class TeleopIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only the base API is stubbed. Actual config/goal/IK methods execute.
        @dataclass
        class Config:
            id: str | None = None
        class Teleoperator:
            def __init__(self,config):
                self.config=config
        lr = ModuleType('lerobot')
        lr.__path__=[]
        lr.__spec__=importlib.machinery.ModuleSpec('lerobot',loader=None,is_package=True)
        teleoperators = ModuleType('lerobot.teleoperators')
        teleoperators.Teleoperator,teleoperators.TeleoperatorConfig=Teleoperator,Config
        cls.stub = patch.dict(sys.modules,{'lerobot':lr,'lerobot.teleoperators':teleoperators})
        cls.stub.start()

    @classmethod
    def tearDownClass(cls):
        cls.stub.stop()

    def test_gamepad_frame_jog(self):
        from src.gamepad_teleop import GamepadEETeleop,GamepadEETeleopConfig
        spec=dict(load('nero_gamepad').teleop)
        spec.pop('type')
        teleop=GamepadEETeleop(GamepadEETeleopConfig(**spec))
        q=np.radians([0,20,0,60,0,10,0])
        teleop._q=q.copy()
        p=teleop.kin.fk(q)[:3,3]
        teleop._step_frame(np.array([.001,0,0]),np.zeros(3))
        self.assertLess(np.linalg.norm(teleop.kin.fk(teleop._q)[:3,3]-p-[.001,0,0]),.0005)

    def nero_gamepad_control(self):
        from src.gamepad_teleop import GamepadEETeleop,GamepadEETeleopConfig
        spec=dict(load('nero_gamepad').teleop)
        spec.pop('type')
        return GamepadEETeleop(GamepadEETeleopConfig(**spec))

    def test_no_udp_input_does_not_enable_commands_and_reanchors_drift(self):
        from src.gamepad import GamepadState
        control=self.nero_gamepad_control()
        obs={k:0. for k in control.kin.keys}
        obs['gripper.pos']=40.
        control.seed_from_observation(obs)
        control._reader=SimpleNamespace(read=lambda:None)
        control.get_action()
        self.assertFalse(control.command_ready)
        # Arm settles while waiting; old seed would now violate the 5-degree guard.
        obs['joint2.pos']=12.
        obs['gripper.pos']=45.
        control.update_observation(obs)
        control.get_action()
        self.assertFalse(control.command_ready)
        self.assertAlmostEqual(np.degrees(control._q[1]),12.)
        control._reader=SimpleNamespace(read=lambda:GamepadState())
        first=control.get_action()
        self.assertTrue(control.command_ready)
        self.assertAlmostEqual(first['joint2.pos'],12.)
        self.assertAlmostEqual(first['gripper.pos'],45.)

    def test_tracking_error_pauses_until_neutral_and_then_resumes(self):
        from src.gamepad import GamepadState,BTN_A
        control=self.nero_gamepad_control()
        q=np.array([0.,20.,0.,60.,0.,10.,0.])
        obs=dict(zip(control.kin.keys,q))
        obs['gripper.pos']=40.
        control.seed_from_observation(obs)
        control._input_available=True
        control._q[0]+=np.radians(7.)
        control.update_observation(obs)
        held=GamepadState(axes=[0.,-.5,0.,0.])
        held.buttons[BTN_A]={'pressed':True,'value':1.}
        control._reader=SimpleNamespace(read=lambda:held)
        paused=control.get_action()
        self.assertFalse(control.command_ready)
        np.testing.assert_allclose([paused[k] for k in control.kin.keys],q)
        self.assertEqual(paused['gripper.pos'],40.)
        control._reader=SimpleNamespace(read=lambda:GamepadState())
        control.get_action()
        self.assertTrue(control.command_ready)
        control._reader=SimpleNamespace(read=lambda:held)
        control._last_t=99.95
        with patch('src.gamepad_teleop.time.perf_counter',return_value=100.):
            resumed=control.get_action()
        self.assertGreater(resumed['gripper.pos'],40.)
        self.assertGreater(np.max(np.abs(control._q-np.radians(q))),1e-6)

    def test_driver_rejection_between_reads_reanchors_and_latches_pause(self):
        from src.gamepad import GamepadState
        control=self.nero_gamepad_control()
        initial={k:0. for k in control.kin.keys}
        initial['gripper.pos']=40.
        control.seed_from_observation(initial)
        control._input_available=True
        newer=dict(initial)
        newer['joint1.pos']=8.
        control.on_target_rejected(newer)
        self.assertFalse(control.command_ready)
        self.assertAlmostEqual(np.degrees(control._q[0]),8.)
        control._reader=SimpleNamespace(read=lambda:GamepadState())
        resumed=control.get_action()
        self.assertTrue(control.command_ready)
        self.assertAlmostEqual(resumed['joint1.pos'],8.)

    def test_so101_translation_relaxes_orientation_when_it_blocks_reach(self):
        from src.gamepad_teleop import GamepadEETeleop,GamepadEETeleopConfig
        seed=np.radians([99.01688,-34.45797,-78.73527,-61.57115,-151.31603])
        dp=np.array([-.0000594,.0007911,.0018359])
        errors={}
        for mode in ('hold','adaptive','free'):
            teleop=GamepadEETeleop(GamepadEETeleopConfig(translation_orientation=mode))
            teleop._q=seed.copy()
            target=teleop.kin.fk(seed)[:3,3]+dp
            teleop._step_arm(dp,0.,0.,0.)
            errors[mode]=np.linalg.norm(teleop.kin.fk(teleop._q)[:3,3]-target)
            self.assertTrue(np.all(teleop._q>=teleop.kin.chain.lower+teleop.kin.limit_margin-1e-10))
            self.assertTrue(np.all(teleop._q<=teleop.kin.chain.upper-teleop.kin.limit_margin+1e-10))
            self.assertLessEqual(np.max(np.abs(teleop._q-seed)),np.radians(5)+1e-10)
        self.assertGreater(errors['hold'],.0005)
        self.assertLess(errors['adaptive'],.0001)
        self.assertLess(errors['free'],.0001)

    def test_so101_easy_translation_preserves_original_orientation_solution(self):
        from src.gamepad_teleop import GamepadEETeleop,GamepadEETeleopConfig
        outputs=[]
        seed=np.radians([0,20,-30,10,0])
        for mode in ('hold','adaptive'):
            teleop=GamepadEETeleop(GamepadEETeleopConfig(translation_orientation=mode))
            teleop._q=seed.copy()
            teleop._step_arm(np.array([.001,0,0]),0.,0.,0.)
            outputs.append(teleop._q)
        np.testing.assert_allclose(*outputs,atol=1e-12)

    def test_so101_deliberate_rotation_is_not_relaxed(self):
        from src.gamepad_teleop import GamepadEETeleop,GamepadEETeleopConfig
        teleop=GamepadEETeleop(GamepadEETeleopConfig())
        teleop._q=np.radians([99.01688,-34.45797,-78.73527,-61.57115,-151.31603])
        with patch.object(teleop.kin,'solve',wraps=teleop.kin.solve) as solve:
            teleop._step_arm(np.array([-.0000594,.0007911,.0018359]),0.,.005,0.)
        self.assertEqual(solve.call_count,1)
        self.assertEqual(solve.call_args.kwargs['orientation_weight'],1.)

    def test_so101_floor_stays_enabled(self):
        from src.gamepad_teleop import GamepadEETeleop,GamepadEETeleopConfig
        teleop=GamepadEETeleop(GamepadEETeleopConfig())
        np.testing.assert_allclose(teleop._floor(np.array([.2,0,.02]),np.array([.201,0,.019])),[.201,0,.02])

    def test_nero_sticks_and_ab_without_rb_send_seven_joints_and_gripper(self):
        from src.gamepad import GamepadState, BTN_A, BTN_B, BTN_RB
        from src.gamepad_teleop import GamepadEETeleop,GamepadEETeleopConfig
        for mode in ('width','angle'):
            with self.subTest(feedback_mode=mode):
                cfg=load('nero_gamepad')
                robot=make_robot(cfg)
                self.assertEqual(len(robot.action_features),8)
                self.assertFalse(cfg.teleop['require_deadman'])
                arm_targets,gripper_targets=[],[]
                q0=np.radians([0,20,0,60,0,10,0])
                robot._connected=robot._enabled=True
                robot._arm=SimpleNamespace(
                    get_joint_angles=lambda:SimpleNamespace(msg=q0.tolist(),timestamp=time.monotonic()),
                    move_j=lambda q:arm_targets.append(q))
                def capture_mode(received_mode):
                    return lambda **kw:gripper_targets.append((received_mode,kw['value']))
                robot._gripper=SimpleNamespace(
                    get_gripper_status=lambda:SimpleNamespace(msg=SimpleNamespace(
                        value=.04 if mode=='width' else 40.,mode=mode)),
                    move_gripper_m=capture_mode('width'),
                    move_gripper_deg=capture_mode('angle'))
                spec=dict(cfg.teleop)
                spec.pop('type')
                teleop=GamepadEETeleop(GamepadEETeleopConfig(**spec))
                teleop.seed_from_observation(robot.get_observation())
                self.assertAlmostEqual(teleop._gripper,40.)
                state=GamepadState()
                state.axes[1]=-.5  # translation plus gripper input, without RB
                state.buttons[BTN_A]={'pressed':True,'value':1.}
                self.assertFalse(state.pressed(BTN_RB))
                teleop._reader=SimpleNamespace(read=lambda:state)
                teleop._last_t=99.95
                with patch('src.gamepad_teleop.time.perf_counter',return_value=100.):
                    opened=teleop.get_action()
                self.assertEqual(set(opened),set(robot.action_features))
                self.assertGreater(opened['gripper.pos'],40.)
                self.assertGreater(np.max(np.abs(teleop._q-q0)),1e-6)
                robot.send_action(opened)
                self.assertEqual(len(arm_targets[-1]),7)
                self.assertEqual(gripper_targets[-1][0],mode)
                scale=1000. if mode=='width' else 1.
                self.assertAlmostEqual(gripper_targets[-1][1],opened['gripper.pos']/scale)
                state.buttons[BTN_A]={'pressed':False,'value':0.}
                state.buttons[BTN_B]={'pressed':True,'value':1.}
                teleop._last_t=99.95
                with patch('src.gamepad_teleop.time.perf_counter',return_value=100.):
                    closed=teleop.get_action()
                self.assertLess(closed['gripper.pos'],opened['gripper.pos'])
                robot.send_action(closed)
                self.assertLess(gripper_targets[-1][1],gripper_targets[-2][1])

    def test_vr_full_orientation_keeps_position(self):
        from src.vr_teleop import VREETeleop,VREETeleopConfig,_Sample
        spec=dict(load('nero_vr').teleop)
        spec.pop('type')
        teleop=VREETeleop(VREETeleopConfig(**spec))
        q=np.radians([0,20,0,60,0,10,0])
        T=teleop.kin.fk(q)
        teleop._q_ik=q.copy()
        teleop._p0,teleop._R0=T[:3,3],T[:3,:3]
        teleop._align=teleop._H0=np.eye(3)
        teleop._c0=np.zeros(3)
        teleop._z_floor=None
        teleop._precision=False
        H=rotvec_to_matrix([0,0,.02])
        result=teleop._goal(_Sample(np.zeros(3),H,1.,0.))
        reached=teleop.kin.fk(result)
        self.assertLess(np.linalg.norm(reached[:3,3]-T[:3,3]),.001)
        self.assertLess(np.linalg.norm(matrix_to_rotvec((H@T[:3,:3])@reached[:3,:3].T)),np.radians(1.5))

    def test_udp_and_local_gamepad_produce_same_robot_commands(self):
        import select
        import socket
        from dataclasses import replace
        from src.gamepad import GamepadState,BTN_A,BTN_X,BTN_RT
        from src.gamepad_teleop import GamepadEETeleop,GamepadEETeleopConfig
        from src.gamepad_udp import UDPGamepadReader,encode_state
        for preset,q_deg in [('so101_gamepad',[0,20,-30,10,0]),('nero_gamepad',[0,20,0,60,0,10,0])]:
            with self.subTest(robot=preset):
                spec=dict(load(preset).teleop)
                spec.pop('type')
                cfg=GamepadEETeleopConfig(**spec)
                local=GamepadEETeleop(cfg)
                reader=UDPGamepadReader(0,host='127.0.0.1')
                remote=GamepadEETeleop(replace(cfg,udp_port=reader.address[1]))
                sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                state=GamepadState(axes=[.2,-.3,.1,-.1])
                state.buttons[BTN_A]={'pressed':True,'value':1.}
                state.buttons[BTN_X]={'pressed':True,'value':1.}
                state.buttons[BTN_RT]={'pressed':False,'value':.2}
                fake_local=SimpleNamespace(read=lambda:state,device_name='local test pad',close=lambda:None)
                try:
                    # Each branch opens only the selected input backend.
                    with patch('src.gamepad_udp.UDPGamepadReader',side_effect=AssertionError('Local mode opened UDP')), \
                         patch('src.gamepad.GamepadReader',return_value=fake_local):
                        local.connect()
                    with patch('src.gamepad_udp.UDPGamepadReader',return_value=reader), \
                         patch('src.gamepad.GamepadReader',side_effect=AssertionError('UDP mode opened local gamepad')):
                        remote.connect()
                    obs={k:float(v) for k,v in zip(local.kin.keys,q_deg)}
                    obs['gripper.pos']=40.
                    local.seed_from_observation(obs)
                    remote.seed_from_observation(obs)
                    sender.sendto(encode_state(state,'comparison',1),reader.address)
                    self.assertTrue(select.select([reader._sock],[],[],.5)[0])
                    local._last_t=remote._last_t=99.95
                    with patch('src.gamepad_teleop.time.perf_counter',return_value=100.):
                        expected=local.get_action()
                        actual=remote.get_action()
                    self.assertEqual(actual,expected)
                    self.assertEqual(remote._precision,local._precision)
                    reader._received_at-=1.
                    with patch('src.gamepad_teleop.time.perf_counter',return_value=100.05):
                        held=remote.get_action()
                    self.assertEqual(held,actual)  # no continued stick/A movement after dropout
                finally:
                    sender.close()
                    local.disconnect()
                    remote.disconnect()
                    reader.close()


if __name__ == '__main__':
    unittest.main()
