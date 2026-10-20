"""Expand the bundled official Nero includes without a ROS/Xacro dependency."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / 'urdf' / 'nero'


def build(destination: Path = ROOT / 'nero_with_gripper.urdf') -> Path:
    robot = ET.Element('robot', name='nero_with_agx_gripper')
    for name in ('nero_description.urdf', 'nero_with_flange_description.xacro', 'nero_with_gripper_description.xacro'):
        source = ET.parse(ROOT / 'source' / name).getroot()
        for item in source:
            if item.tag in ('link', 'joint'):
                robot.append(copy.deepcopy(item))
    # Nominal reference at the midpoint of the finger joint frames. This is
    # configurable in NeroKinematics; it is not a measured fingertip calibration.
    ET.SubElement(robot, 'link', name='gripper_tcp')
    joint = ET.SubElement(robot, 'joint', name='gripper_tcp_joint', type='fixed')
    ET.SubElement(joint, 'parent', link='gripper_base')
    ET.SubElement(joint, 'child', link='gripper_tcp')
    ET.SubElement(joint, 'origin', xyz='0 0 0.138', rpy='0 0 0')
    ET.indent(robot)
    ET.ElementTree(robot).write(destination, encoding='utf-8', xml_declaration=True)
    return destination


if __name__ == '__main__':
    print(build())
