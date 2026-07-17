"""
Build a multi-humanoid MJCF by duplicating the single <body name="torso"> subtree
N times, renaming every joint/body/geom/site/camera under each copy with a
"_i" suffix, and duplicating the matching actuator/tendon/contact entries.

Each copy keeps its own <freejoint>, so each robot is fully independent
(they only interact through contacts, which we allow by default so you can
disable/enable proximity effects by adjusting `spacing`).
"""

import copy
import os
import xml.etree.ElementTree as ET

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_XML = os.path.join(THIS_DIR, "Humanoid.xml")


def _suffix_names(elem, i, joint_names):
    for tag in ("body", "geom", "site", "camera", "joint", "freejoint"):
        for e in elem.iter(tag):
            name = e.get("name")
            if name is not None:
                new_name = f"{name}_{i}"
                if e.tag == "joint":
                    joint_names.append(name)
                e.set("name", new_name)


def build_multi_humanoid_xml(n_agents, spacing=3.0, out_path=None):
    tree = ET.parse(BASE_XML)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    actuator = root.find("actuator")
    tendon = root.find("tendon")
    contact = root.find("contact")

    torso_template = None
    for body in worldbody.findall("body"):
        if body.get("name") == "torso":
            torso_template = body
            break
    if torso_template is None:
        raise ValueError("Could not find <body name='torso'> in base XML")

    worldbody.remove(torso_template)

    actuator_templates = list(actuator)
    for a in actuator_templates:
        actuator.remove(a)

    tendon_templates = list(tendon) if tendon is not None else []
    if tendon is not None:
        for t in tendon_templates:
            tendon.remove(t)

    contact_templates = list(contact) if contact is not None else []
    if contact is not None:
        for c in contact_templates:
            contact.remove(c)

    for light in worldbody.findall("light"):
        if light.get("target") == "torso":
            light.set("target", "torso_0")

    grid = int(np.ceil(np.sqrt(n_agents)))
    for i in range(n_agents):
        gx, gy = i % grid, i // grid
        x, y = gx * spacing, gy * spacing

        body = copy.deepcopy(torso_template)
        base_pos = [float(v) for v in body.get("pos", "0 0 0").split()]
        body.set("pos", f"{base_pos[0] + x} {base_pos[1] + y} {base_pos[2]}")

        joint_names = []
        _suffix_names(body, i, joint_names)
        body.set("name", f"torso_{i}")
        worldbody.append(body)

        for a in actuator_templates:
            a2 = copy.deepcopy(a)
            j = a.get("joint")
            a2.set("joint", f"{j}_{i}")
            a2.set("name", f"{a.get('name')}_{i}")
            actuator.append(a2)

        if tendon is not None:
            for t in tendon_templates:
                t2 = copy.deepcopy(t)
                t2.set("name", f"{t.get('name')}_{i}")
                for j in t2.findall("joint"):
                    j.set("joint", f"{j.get('joint')}_{i}")
                tendon.append(t2)

        if contact is not None:
            for c in contact_templates:
                c2 = copy.deepcopy(c)
                c2.set("body1", f"{c.get('body1')}_{i}")
                c2.set("body2", f"{c.get('body2')}_{i}")
                contact.append(c2)

    xml_str = ET.tostring(root, encoding="unicode")

    if out_path:
        with open(out_path, "w") as f:
            f.write(xml_str)

    return xml_str


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = os.path.join(THIS_DIR, f"Humanoid_multi_{n}.xml")
    build_multi_humanoid_xml(n, spacing=3.0, out_path=out)
    print(f"Wrote {out}")