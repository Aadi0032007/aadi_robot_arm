"""UDP transport checks on loopback; no pygame, robot SDK or hardware."""
import json
from pathlib import Path
import select
import socket
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load
from src.devices import apply_overrides, parse_overrides
from src.gamepad import GamepadState, BTN_A, BTN_RT
from src.gamepad_udp import UDPGamepadReader, decode_state, encode_state
from src.gamepad_remote_client import run_client, main as client_main


class UDPGamepadTests(unittest.TestCase):
    def setUp(self):
        self.reader=UDPGamepadReader(0, host='127.0.0.1')
        self.sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    def tearDown(self):
        self.sender.close()
        self.reader.close()

    def send(self,state,seq=0,session='client',sender=None):
        (sender or self.sender).sendto(encode_state(state,session,seq),self.reader.address)
        self.assertTrue(select.select([self.reader._sock],[],[],.5)[0])
        return self.reader.read()

    def test_normalized_sticks_buttons_and_triggers(self):
        state=GamepadState(axes=[-.5,.3,1.,-1.])
        state.buttons[BTN_A]={'pressed':True,'value':1.}
        state.buttons[BTN_RT]={'pressed':False,'value':.43}
        received=self.send(state)
        self.assertEqual(received,state)
        self.assertIsNot(received,state)
        received.axes[0]=0.
        self.assertEqual(self.reader.read().axes[0],-.5)

    def test_watchdog_disconnect_and_reordered_packets(self):
        self.assertIsNone(self.reader.read())
        state=GamepadState(axes=[1.,0.,0.,0.])
        self.assertIsNotNone(self.send(state,seq=10))
        self.reader._received_at-=1.
        self.assertIsNone(self.reader.read())
        self.assertIsNone(self.send(state,seq=9))
        self.assertIsNone(self.send(state,seq=10))
        self.assertIsNotNone(self.send(state,seq=11))
        self.assertIsNone(self.send(None,seq=12))
        self.assertIsNone(self.send(state,seq=11))
        self.assertIsNotNone(self.send(state,seq=0,session='restarted'))

    def test_only_active_sender_then_takeover_after_timeout(self):
        self.send(GamepadState(),seq=5)
        other=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            moving=GamepadState(axes=[1.,0.,0.,0.])
            self.assertEqual(self.send(moving,sender=other).axes[0],0.)
            self.reader._received_at-=1.
            self.assertEqual(self.send(moving,sender=other).axes[0],1.)
        finally:
            other.close()

    def test_newest_frame_used_without_playing_queue(self):
        for seq in range(8):
            state=GamepadState(axes=[seq/10.,0.,0.,0.])
            self.sender.sendto(encode_state(state,'client',seq),self.reader.address)
        self.assertTrue(select.select([self.reader._sock],[],[],.5)[0])
        self.assertAlmostEqual(self.reader.read().axes[0],.7)

    def test_bad_packets_do_not_refresh_watchdog(self):
        base=json.loads(encode_state(GamepadState(),'client',1))
        variants=[b'not json',b'[]',b'0'*9000]
        for field,value in [('axes',[0,0,0]),('axes',[float('nan'),0,0,0]),('axes',[2,0,0,0]),
                            ('seq',True),('connected',1),('session','')]:
            msg=dict(base)
            msg[field]=value
            variants.append(json.dumps(msg).encode())
        for packet in variants:
            with self.subTest(packet=packet[:70]):
                with self.assertRaises(ValueError):
                    decode_state(packet)
                self.sender.sendto(packet,self.reader.address)
                self.assertTrue(select.select([self.reader._sock],[],[],.5)[0])
                self.assertIsNone(self.reader.read())

    def test_port_alias_any_gamepad_robot_and_argument_order(self):
        for preset in ('so101_gamepad','nero_gamepad'):
            cfg=load(preset)
            apply_overrides(cfg,parse_overrides(['--port','5001']))
            self.assertEqual(cfg.teleop['udp_port'],5001)
        cfg=load('koch')
        apply_overrides(cfg,parse_overrides(['--port=5001','--teleop.type=gamepad_ee']))
        self.assertEqual(cfg.teleop['udp_port'],5001)
        self.assertNotIn('udp_port',load('so101_gamepad').teleop)
        with self.assertRaises(SystemExit):
            apply_overrides(load('nero'),{'port':'5001'})

    def test_debug_boolean_on_robot_and_client_cli(self):
        for text,expected in [('true',True),('false',False)]:
            cfg=load('nero_gamepad')
            apply_overrides(cfg,parse_overrides([f'--debug={text}']))
            self.assertIs(cfg.teleop['debug'],expected)
            with patch('sys.argv',['gamepad_remote_client',f'--debug={text}']), \
                 patch('src.gamepad_remote_client.run_client') as run:
                client_main()
                self.assertIs(run.call_args.kwargs['debug'],expected)
        with patch('sys.argv',['gamepad_remote_client','--debug']), \
             patch('src.gamepad_remote_client.run_client') as run:
            client_main()
            self.assertIs(run.call_args.kwargs['debug'],True)

    def test_debug_shows_exact_sent_and_received_payloads_and_timeout(self):
        state=GamepadState(axes=[-.5,.25,0.,1.])
        state.buttons[BTN_A]={'pressed':True,'value':1.}
        reader=SimpleNamespace(read=lambda:state,device_name='debug test pad',close=lambda:None)
        self.reader.debug=True
        with patch('src.gamepad_remote_client.GamepadReader',return_value=reader), \
             patch('src.gamepad_remote_client.time.sleep',side_effect=KeyboardInterrupt), \
             self.assertLogs('src.gamepad_remote_client',level='INFO') as tx:
            run_client('127.0.0.1',self.reader.address[1],debug=True)
        self.assertTrue(select.select([self.reader._sock],[],[],.5)[0])
        with self.assertLogs('src.gamepad_udp',level='INFO') as rx:
            self.reader.read()
        tx_packets=[json.loads(record.args[-1]) for record in tx.records if record.msg.startswith('GAMEPAD TX')]
        rx_packets=[json.loads(record.args[-1]) for record in rx.records if record.msg.startswith('GAMEPAD RX from=')]
        self.assertEqual(tx_packets,rx_packets)
        self.assertEqual(len(tx_packets),2)  # state and exit/disconnect
        self.assertEqual(tx_packets[0]['axes'],state.axes)
        self.assertEqual(tx_packets[0]['buttons'],state.buttons)
        self.assertFalse(tx_packets[1]['connected'])
        with self.assertLogs('src.gamepad_udp',level='INFO') as logs:
            self.send(state,seq=2,session=tx_packets[0]['session'])
            self.send(state,seq=1,session=tx_packets[0]['session'])
            self.reader._received_at-=1.
            self.assertIsNone(self.reader.read())
        self.assertTrue(any('duplicate/out-of-order' in line for line in logs.output))
        self.assertTrue(any('timeout' in line for line in logs.output))

    def test_packet_debug_is_off_by_default(self):
        with self.assertNoLogs('src.gamepad_udp',level='INFO'):
            self.send(GamepadState())
        packets=[]
        state=GamepadState()
        reader=SimpleNamespace(read=lambda:state,device_name='fake pad',close=lambda:None)
        sock=SimpleNamespace(sendto=lambda *args:packets.append(args),close=lambda:None)
        with patch('src.gamepad_remote_client.GamepadReader',return_value=reader), \
             patch('src.gamepad_remote_client.socket.socket',return_value=sock), \
             patch('src.gamepad_remote_client.socket.gethostbyname',return_value='127.0.0.1'), \
             patch('src.gamepad_remote_client.time.sleep',side_effect=KeyboardInterrupt), \
             self.assertLogs('src.gamepad_remote_client',level='INFO') as logs:
            run_client()
        self.assertFalse(any('GAMEPAD TX' in line for line in logs.output))

    def test_client_reuses_reader_sends_state_and_disconnect(self):
        packets=[]
        state=GamepadState()
        state.buttons[BTN_A]={'pressed':True,'value':1.}
        reader=SimpleNamespace(read=lambda:state,device_name='fake pad',close=lambda:packets.append('reader closed'))
        sock=SimpleNamespace(sendto=lambda data,target:packets.append((data,target)),close=lambda:packets.append('socket closed'))
        with patch('src.gamepad_remote_client.GamepadReader',return_value=reader), \
             patch('src.gamepad_remote_client.socket.socket',return_value=sock), \
             patch('src.gamepad_remote_client.socket.gethostbyname',return_value='127.0.0.1'), \
             patch('src.gamepad_remote_client.time.sleep',side_effect=KeyboardInterrupt):
            run_client()
        first,disconnect=packets[:2]
        session,seq,received=decode_state(first[0])
        end_session,end_seq,empty=decode_state(disconnect[0])
        self.assertEqual(first[1],('127.0.0.1',5001))
        self.assertEqual(received,state)
        self.assertEqual(session,end_session)
        self.assertGreater(end_seq,seq)
        self.assertIsNone(empty)
        self.assertEqual(packets[-2:],['socket closed','reader closed'])


if __name__=='__main__':
    unittest.main()
