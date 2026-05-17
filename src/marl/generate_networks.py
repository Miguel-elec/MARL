"""
generate_networks.py
--------------------
Genera redes de cuadrícula y archivos de demanda para SUMO.

Uso:
    python generate_networks.py          # genera 2x2, 3x3 y 4x4
    python generate_networks.py --size 2 # solo la 2x2

Las redes se guardan en:
    networks/
        2x2/net.net.xml
        2x2/routes_low.rou.xml
        2x2/routes_medium.rou.xml
        2x2/routes_high.rou.xml
        3x3/...
        4x4/...
"""

import argparse
import os
import random
import subprocess
import xml.etree.ElementTree as ET

SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
NETGENERATE = os.path.join(SUMO_HOME, "bin", "netgenerate")


# ─────────────────────────────────────────────────────────────────────────────
# Red
# ─────────────────────────────────────────────────────────────────────────────

def generate_grid_network(size: int, out_dir: str):
    """Genera una cuadrícula size×size con netgenerate."""
    os.makedirs(out_dir, exist_ok=True)
    net_file = os.path.join(out_dir, "net.net.xml")

    cmd = [
        NETGENERATE,
        "--grid",
        f"--grid.x-number={size}",
        f"--grid.y-number={size}",
        "--grid.x-length=200",
        "--grid.y-length=200",
        "--default.lanenumber=2",
        "--default.speed=13.89",     # 50 km/h
        "--tls.guess=true",
        "--tls.default-type=static",
        "-o", net_file,
        "--no-warnings",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] netgenerate falló:\n{result.stderr}")
        return None
    print(f"[OK] Red {size}×{size} → {net_file}")
    return net_file


# ─────────────────────────────────────────────────────────────────────────────
# Demanda (archivos .rou.xml)
# ─────────────────────────────────────────────────────────────────────────────

TRAFFIC_LEVELS = {
    "low":    300,    # vehículos/hora (tráfico ligero)
    "medium": 700,    # tráfico moderado
    "high":   1200,   # tráfico saturado
}

VEHICLE_TYPE = """
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5"
           length="5" maxSpeed="13.89" color="0.8,0.8,0"/>
"""

def _get_edge_ids(net_file: str) -> list[str]:
    """Extrae los IDs de los bordes externos de la red."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    edges = []
    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        # Los bordes externos no tienen ':'
        if not eid.startswith(":") and edge.get("function") != "internal":
            edges.append(eid)
    return edges


def generate_routes(net_file: str, out_dir: str, level: str,
                    duration: int = 3600, seed: int = 42):
    """Genera un archivo de rutas con distribución Poisson para el nivel dado."""
    rng = random.Random(seed)
    vph = TRAFFIC_LEVELS[level]
    edges = _get_edge_ids(net_file)
    if not edges:
        print(f"[WARN] No se encontraron bordes en {net_file}")
        return

    rou_file = os.path.join(out_dir, f"routes_{level}.rou.xml")
    vehicles = []
    t = 0.0
    v_id = 0

    # Distribución de Poisson: inter-llegadas exponenciales
    mean_interval = 3600.0 / vph
    while t < duration:
        interval = rng.expovariate(1.0 / mean_interval)
        t += interval
        if t >= duration:
            break
        origin = rng.choice(edges)
        dest   = rng.choice([e for e in edges if e != origin])
        vehicles.append((round(t, 2), v_id, origin, dest))
        v_id += 1

    with open(rou_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
        f.write(' xsi:noNamespaceSchemaLocation=')
        f.write('"http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write(VEHICLE_TYPE + "\n")
        for depart, vid, orig, dest in vehicles:
            f.write(
                f'    <vehicle id="v{vid}" type="car" depart="{depart}">\n'
                f'        <route edges="{orig} {dest}"/>\n'
                f'    </vehicle>\n'
            )
        f.write("</routes>\n")

    print(f"[OK] Rutas {level} ({len(vehicles)} veh) → {rou_file}")
    return rou_file


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4],
                        help="Tamaños de cuadrícula a generar (ej: 2 3 4)")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Duración de cada escenario en segundos")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "networks")

    for size in args.sizes:
        out_dir  = os.path.join(base_dir, f"{size}x{size}")
        net_file = generate_grid_network(size, out_dir)
        if net_file:
            for level in TRAFFIC_LEVELS:
                generate_routes(net_file, out_dir, level, args.duration)

    print("\n✓ Redes generadas. Próximo paso: python main.py")


if __name__ == "__main__":
    main()
