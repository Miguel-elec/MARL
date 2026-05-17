"""
main.py
-------
Punto de entrada del proyecto MARL para control de semáforos.

Flujo:
    1. Crea el entorno SUMO multi-agente
    2. Instancia IQL (un DQN por semáforo)
    3. Entrena durante N episodios
    4. Guarda checkpoints y métricas

Primer uso:
    # 1. Genera las redes (solo la primera vez)
    python generate_networks.py --sizes 2

    # 2. Entrena IQL en la cuadrícula 2×2 con tráfico moderado
    python main.py
"""

import os
import sys

# ─── Asegura que SUMO esté en el PATH ────────────────────────────────────────
SUMO_HOME = os.environ.get("SUMO_HOME")
if SUMO_HOME is None:
    # Intento automático en rutas comunes
    for path in ["/usr/share/sumo", "/opt/homebrew/share/sumo"]:
        if os.path.isdir(path):
            SUMO_HOME = path
            os.environ["SUMO_HOME"] = path
            break

if SUMO_HOME:
    sys.path.append(os.path.join(SUMO_HOME, "tools"))
else:
    print("[ERROR] SUMO_HOME no está definido.")
    print("  Instala SUMO y ejecuta: export SUMO_HOME=/ruta/a/sumo")
    sys.exit(1)

# ─── Imports del proyecto ────────────────────────────────────────────────────
from env.sumo_env        import SumoMultiAgentEnv
from algorithms.iql      import IQL
from utils.logger        import MetricsLogger

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Entorno
    "net_file"   : "networks/2x2/net.net.xml",
    "route_file" : "networks/2x2/routes_medium.rou.xml",
    "num_seconds": 3600,          # 1 hora simulada por episodio
    "delta_time" : 5,             # segundos por paso del entorno
    "min_green"  : 5,
    "max_green"  : 50,
    "use_gui"    : False,         # pon True para ver la simulación

    # Entrenamiento
    "episodes"   : 100,
    "eval_every" : 10,            # evaluar (greedy) cada N episodios
    "save_every" : 20,

    # IQL
    "lr"             : 1e-3,
    "gamma"          : 0.99,
    "epsilon_start"  : 1.0,
    "epsilon_end"    : 0.05,
    "epsilon_decay"  : 0.9995,
    "buffer_size"    : 50_000,
    "batch_size"     : 64,
    "target_update"  : 200,

    # Logging
    "log_dir"  : "results",
    "run_name" : "iql_2x2_medium",
}


# ─────────────────────────────────────────────────────────────────────────────
# Bucle de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def train():
    env = SumoMultiAgentEnv(
        net_file    = CONFIG["net_file"],
        route_file  = CONFIG["route_file"],
        num_seconds = CONFIG["num_seconds"],
        delta_time  = CONFIG["delta_time"],
        min_green   = CONFIG["min_green"],
        max_green   = CONFIG["max_green"],
        use_gui     = CONFIG["use_gui"],
        seed        = 42,
    )

    # Primera reset para conocer las dimensiones de los espacios
    obs = env.reset()

    iql_kwargs = {k: CONFIG[k] for k in (
        "lr", "gamma", "epsilon_start", "epsilon_end",
        "epsilon_decay", "buffer_size", "batch_size", "target_update"
    )}
    iql = IQL(env.obs_dims, env.action_dims, **iql_kwargs)

    logger = MetricsLogger(CONFIG["log_dir"], CONFIG["run_name"])

    print(f"\n{'='*55}")
    print(f"  Agentes: {env.agents}")
    print(f"  Obs dims: { {k: v for k, v in env.obs_dims.items()} }")
    print(f"  Action dims: { {k: v for k, v in env.action_dims.items()} }")
    print(f"{'='*55}\n")

    for episode in range(1, CONFIG["episodes"] + 1):
        greedy = (episode % CONFIG["eval_every"] == 0)
        obs    = env.reset()
        done   = False

        while not done:
            actions          = iql.act(obs, greedy=greedy)
            next_obs, rewards, done, info = env.step(actions)

            if not greedy:
                iql.store(obs, actions, rewards, next_obs, done)
                losses = iql.update()
            else:
                losses = {}

            logger.log_step(info, losses, rewards)
            obs = next_obs

        logger.log_episode(episode, extra={
            "epsilon": round(iql.mean_epsilon, 4),
            "mode"   : "eval" if greedy else "train",
        })

        if episode % CONFIG["save_every"] == 0:
            ckpt_dir = os.path.join(CONFIG["log_dir"], "checkpoints",
                                    CONFIG["run_name"], f"ep{episode:04d}")
            iql.save(ckpt_dir)
            print(f"  → Checkpoint guardado en {ckpt_dir}")

    env.close()
    logger.close()
    print("\n✓ Entrenamiento completado.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
