import os
import random
import math
import argparse
import xml.etree.ElementTree as ET

CELL = 200
LANES = 2
SPEED = 13.89
EXT_LEN = 100

TRAFFIC_LEVELS = {
    "low": 300,
    "medium": 700,
    "high": 1200,
}

# ─────────────────────────────────────────────
# NET GENERATOR (DIRECT .net.xml → FAST & SAFE)
# ─────────────────────────────────────────────

def generate_net(size: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    net_file = os.path.join(out_dir, "net.net.xml")

    net = ET.Element("net", {
        "version": "1.20",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/net_file.xsd",
    })

    total = (size - 1) * CELL

    ET.SubElement(net, "location", {
        "netOffset": "0,0",
        "convBoundary": f"-{EXT_LEN},-{EXT_LEN},{total+EXT_LEN},{total+EXT_LEN}",
        "origBoundary": "-1e10,-1e10,1e10,1e10",
        "projParameter": "!"
    })

    nodes = {}
    node_coords = {}

    # ── INTERSECTION NODES ──
    for y in range(size):
        for x in range(size):
            nid = f"n_{x}_{y}"
            nodes[(x, y)] = nid

            ET.SubElement(net, "junction", {
                "id": nid,
                "type": "traffic_light",
                "x": str(x * CELL),
                "y": str(y * CELL),
            })
            node_coords[nid] = (x * CELL, y * CELL)

    # ── EXTERNAL NODES ──
    ext_nodes = {}

    dirs = [(-1,0,"W"),(1,0,"E"),(0,-1,"S"),(0,1,"N")]

    for y in range(size):
        for x in range(size):
            for dx, dy, tag in dirs:
                if 0 <= x+dx < size and 0 <= y+dy < size:
                    continue

                nid = f"ext_{x}_{y}_{tag}"
                ext_nodes[(x,y,tag)] = nid

                ET.SubElement(net, "junction", {
                    "id": nid,
                    "type": "dead_end",
                    "x": str(x * CELL + dx * EXT_LEN),
                    "y": str(y * CELL + dy * EXT_LEN),
                })
                node_coords[nid] = (x * CELL + dx * EXT_LEN, y * CELL + dy * EXT_LEN)

    # ── EDGES (IMPORTANT: NO SHAPE PROBLEM) ──
    def edge(eid, frm, to):
        e = ET.SubElement(net, "edge", {
            "id": eid,
            "from": frm,
            "to": to,
            "numLanes": str(LANES),
            "speed": str(SPEED),
        })

        # compute coordinates for shape and accurate length
        if frm in node_coords and to in node_coords:
            fx, fy = node_coords[frm]
            tx, ty = node_coords[to]
        else:
            fx = fy = tx = ty = 0

        dist = math.hypot(tx - fx, ty - fy) or CELL

        shape_str = f"{fx},{fy} {tx},{ty}"

        for i in range(LANES):
            ET.SubElement(e, "lane", {
                "id": f"{eid}_{i}",
                "index": str(i),
                "speed": str(SPEED),
                "length": str(dist),
                "shape": shape_str
            })

    # internal edges
    for y in range(size):
        for x in range(size):
            if x + 1 < size:
                edge(f"e_{x}_{y}_E", nodes[(x,y)], nodes[(x+1,y)])
                edge(f"e_{x}_{y}_W", nodes[(x+1,y)], nodes[(x,y)])

            if y + 1 < size:
                edge(f"e_{x}_{y}_N", nodes[(x,y)], nodes[(x,y+1)])
                edge(f"e_{x}_{y}_S", nodes[(x,y+1)], nodes[(x,y)])

    # external edges
    for (x,y,tag), ext in ext_nodes.items():
        edge(f"in_{ext}", ext, nodes[(x,y)])
        edge(f"out_{ext}", nodes[(x,y)], ext)

    # traffic lights (fixed 2-phase)
    for y in range(size):
        for x in range(size):
            tl = ET.SubElement(net, "tlLogic", {
                "id": nodes[(x,y)],
                "type": "static",
                "programID": "0",
                "offset": "0"
            })

            ET.SubElement(tl, "phase", {"duration": "30", "state": "GGrrGGrr"})
            ET.SubElement(tl, "phase", {"duration": "5",  "state": "yyrryyrr"})
            ET.SubElement(tl, "phase", {"duration": "30", "state": "rrGGrrGG"})
            ET.SubElement(tl, "phase", {"duration": "5",  "state": "rryyrryy"})

    ET.indent(net, space="  ")
    ET.ElementTree(net).write(net_file, encoding="utf-8", xml_declaration=True)

    print(f"[OK] {size}x{size} → {net_file}")
    return net_file


# ─────────────────────────────────────────────
# ROUTES (FAST RL VERSION)
# ─────────────────────────────────────────────

def generate_routes(net_file, out_dir, level, duration=3600, seed=42):
    rng = random.Random(seed)
    vph = TRAFFIC_LEVELS[level]

    mean = 3600 / vph
    t = 0
    vid = 0

    file = os.path.join(out_dir, f"routes_{level}.rou.xml")

    with open(file, "w") as f:
        f.write('<routes>\n')
        f.write('<vType id="car" accel="2.6" decel="4.5" length="5" maxSpeed="13.89"/>\n')

        while t < duration:
            t += rng.expovariate(1.0 / mean)
            if t >= duration:
                break

            f.write(f'<vehicle id="v{vid}" type="car" depart="{t:.2f}"/>\n')
            vid += 1

        f.write('</routes>\n')

    print(f"[OK] routes {level} → {vid} veh")
    return file


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[2,3,4])
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "networks")

    for size in args.sizes:
        out = os.path.join(base, f"{size}x{size}")

        print(f"\n[GRID {size}x{size}]")

        net = generate_net(size, out)

        for lvl in TRAFFIC_LEVELS:
            generate_routes(net, out, lvl)

    print("\nOK: redes listas para RL (fast mode)")


if __name__ == "__main__":
    main()