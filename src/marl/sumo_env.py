"""
sumo_env.py
-----------
Entorno multi-agente para control de semáforos con SUMO.

API compatible con Gymnasium (sin heredar la clase para evitar dependencias):
    obs  = env.reset()
    obs, rewards, done, info = env.step(actions)

Cada semáforo (TrafficSignal) actúa como un agente independiente.
El protocolo de paso es:

    t=0          t=YELLOW_TIME    t=delta_time
    |──amarillo──|────verde────────|
"""

import os
import sys
import traci
import numpy as np

from env.traffic_signal import TrafficSignal


class SumoMultiAgentEnv:
    """
    Parámetros
    ----------
    net_file    : ruta al archivo .net.xml de SUMO
    route_file  : ruta al archivo .rou.xml de SUMO
    num_seconds : duración total de cada episodio en segundos de simulación
    delta_time  : segundos de simulación por paso del entorno (>= YELLOW_TIME)
    min_green   : verde mínimo antes de permitir cambio de fase
    max_green   : verde máximo (para normalizar la observación)
    use_gui     : arranca sumo-gui en lugar de sumo (solo para depuración)
    seed        : semilla de aleatoriedad para SUMO
    """

    YELLOW_TIME = TrafficSignal.YELLOW_TIME

    def __init__(
        self,
        net_file: str,
        route_file: str,
        num_seconds: int = 3600,
        delta_time: int = 5,
        min_green: int = 5,
        max_green: int = 50,
        use_gui: bool = False,
        seed: int = 42,
    ):
        assert delta_time >= self.YELLOW_TIME, (
            f"delta_time ({delta_time}) debe ser >= YELLOW_TIME ({self.YELLOW_TIME})"
        )
        self.net_file    = net_file
        self.route_file  = route_file
        self.num_seconds = num_seconds
        self.delta_time  = delta_time
        self.min_green   = min_green
        self.max_green   = max_green
        self.use_gui     = use_gui
        self.seed        = seed

        self._sim_step        = 0
        self.ts_ids: list[str]                        = []
        self.traffic_signals: dict[str, TrafficSignal] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> dict[str, np.ndarray]:
        """Reinicia la simulación y devuelve las observaciones iniciales."""
        self._close_sumo()
        self._start_sumo()
        self._sim_step = 0

        self.ts_ids = list(traci.trafficlight.getIDList())
        self.traffic_signals = {
            ts_id: TrafficSignal(
                ts_id, self.delta_time, self.min_green, self.max_green
            )
            for ts_id in self.ts_ids
        }

        # Primer paso para que SUMO genere los primeros vehículos
        traci.simulationStep()
        self._sim_step += 1

        return {ts: sig.get_obs() for ts, sig in self.traffic_signals.items()}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict, dict, bool, dict]:
        """
        Ejecuta un paso del entorno.

        Parámetros
        ----------
        actions : {ts_id: acción (índice de fase verde)}

        Devuelve
        --------
        obs     : {ts_id: np.ndarray}
        rewards : {ts_id: float}
        done    : bool
        info    : dict con métricas globales
        """
        # 1. Solicitar fases → activa amarillo si hay cambio
        for ts_id, action in actions.items():
            self.traffic_signals[ts_id].request_phase(action)

        # 2. Simular YELLOW_TIME segundos (transición amarilla)
        for _ in range(self.YELLOW_TIME):
            traci.simulationStep()
        self._sim_step += self.YELLOW_TIME

        # 3. Confirmar nuevas fases verdes
        for ts in self.traffic_signals.values():
            ts.commit_phase()

        # 4. Simular el resto del delta_time
        remaining = self.delta_time - self.YELLOW_TIME
        for _ in range(remaining):
            traci.simulationStep()
        self._sim_step += remaining

        # 5. Recopilar observaciones y recompensas
        obs     = {ts: sig.get_obs()    for ts, sig in self.traffic_signals.items()}
        rewards = {ts: sig.get_reward() for ts, sig in self.traffic_signals.items()}
        done    = self._sim_step >= self.num_seconds

        info = self._collect_metrics()
        return obs, rewards, done, info

    def close(self):
        self._close_sumo()

    # ──────────────────────────────────────────────────────────────────────────
    # Propiedades de los espacios
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def agents(self) -> list[str]:
        return self.ts_ids

    @property
    def obs_dims(self) -> dict[str, int]:
        """Dimensión del vector de observación por agente."""
        return {ts: sig.obs_dim for ts, sig in self.traffic_signals.items()}

    @property
    def action_dims(self) -> dict[str, int]:
        """Número de acciones (fases verdes) por agente."""
        return {ts: sig.num_phases for ts, sig in self.traffic_signals.items()}

    # ──────────────────────────────────────────────────────────────────────────
    # Métricas
    # ──────────────────────────────────────────────────────────────────────────

    def _collect_metrics(self) -> dict:
        total_waiting  = 0.0
        total_halted   = 0
        total_vehicles = traci.vehicle.getIDCount()

        for sig in self.traffic_signals.values():
            for lane in sig.lanes:
                total_waiting += traci.lane.getWaitingTime(lane)
                total_halted  += traci.lane.getLastStepHaltingNumber(lane)

        return {
            "sim_step"      : self._sim_step,
            "total_waiting" : total_waiting,
            "total_halted"  : total_halted,
            "total_vehicles": total_vehicles,
            "mean_waiting"  : total_waiting / max(total_vehicles, 1),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # SUMO lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _start_sumo(self):
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-warnings",
            "--no-step-log",
            "--time-to-teleport", "-1",   # deshabilitar teleports
            "--seed", str(self.seed),
        ]
        traci.start(cmd)

    def _close_sumo(self):
        try:
            traci.close()
        except Exception:
            pass
