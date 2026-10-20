"""Stream a local USB gamepad's normalized controls to any robot's gamepad_ee.

    python -m src.gamepad_remote_client --ip=192.168.1.20 --port=5001
    python src/gamepad_remote_client.py --ip=127.0.0.1 --port=5001

Uses the same GamepadReader/SDL mapping as local control. Requires pygame only;
LeRobot, robot SDKs and IK are needed only on the robot computer.
"""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'src'

import argparse
import logging
import math
import socket
import time
import uuid

from .gamepad import GamepadReader
from .gamepad_udp import encode_state

logger = logging.getLogger(__name__)


def run_client(ip='127.0.0.1', port=5001, fps=60.):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Port must be an integer in [1, 65535]')
    if not math.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError('FPS must be in [1, 240]')
    # Resolve before opening the controller; use one socket/source port per run.
    target = (socket.gethostbyname(ip), port)
    reader = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    session, seq, connected = uuid.uuid4().hex, 0, None
    try:
        reader = GamepadReader()
        logger.info('Gamepad -> %s:%s at %g Hz (Ctrl-C to stop)', *target, fps)
        while True:
            started = time.perf_counter()
            state = reader.read()
            present = state is not None
            if present != connected:
                logger.info('Gamepad connected: %s' if present else 'No gamepad; sending disconnected state',
                            *([reader.device_name] if present else []))
                connected = present
            # Send idle states too: this heartbeat distinguishes rest from dropout.
            sock.sendto(encode_state(state, session, seq), target)
            seq += 1
            time.sleep(max(0., 1./fps-(time.perf_counter()-started)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.sendto(encode_state(None, session, seq), target)
        except OSError:
            pass
        sock.close()
        if reader is not None:
            reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ip', default='127.0.0.1', help='Robot computer IP (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5001, help='Robot UDP port (default: 5001)')
    parser.add_argument('--fps', '--hz', type=float, default=60., help='State/heartbeat rate (default: 60 Hz)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        run_client(args.ip, args.port, args.fps)
    except (ValueError, OSError, ImportError) as e:
        parser.exit(1, f'{e}\n')


if __name__ == '__main__':
    main()
