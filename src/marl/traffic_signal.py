"""
traffic_signal.py
-----------------
Encapsula un semáforo como agente MARL.

Observación:
    [cola_carril_0..N, densidad_carril_0..N, fase_onehot, tiempo_en_fase]

Recompensa:
    Diferencia de tiempo de espera acumulado entre pasos (negativa = mejora).

Acción:
    Índice de la siguiente fase verde a activar.
"""

import numpy as np
import traci


class TrafficSignal:
    YELLOW_TIME = 3   # segundos de transición amarilla fija

    def __init__(self, ts_id: str, delta_time: int,
                 min_green: int = 5, max_green: int = 50):
        self.id         = ts_id
        self.delta_time = delta_time
        self.min_green  = min_green
        self.max_green  = max_green

        # Fases verdes del programa por defecto
        logic = traci.trafficlight.getAllProgramLogics(self.id)[0]
        self.green_phases = [
            (i, p.state)
            for i, p in enumerate(logic.phases)
            if "G" in p.state          # solo fases con verde principal
        ]
        self.num_phases = len(self.green_phases)
        assert self.num_phases > 0, f"El semáforo {ts_id} no tiene fases verdes"

        # Carriles controlados (sin duplicados)
        self.lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(self.id)
        ))
        self.num_lanes = len(self.lanes)

        # Estado interno
        self.current_phase  = 0       # índice en self.green_phases
        self.time_on_phase  = 0       # segundos acumulados en la fase actual
        self._last_waiting  = 0.0     # para calcular la recompensa diferencial
        self._pending_phase = None    # fase pendiente tras el amarillo

        # Activar la primera fase verde
        traci.trafficlight.setPhase(self.id, self.green_phases[0][0])

    # ──────────────────────────────────────────────────────────────────────────
    # Espacios
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        """Dimensión del vector de observación."""
        return self.num_lanes * 2 + self.num_phases + 1

    # ──────────────────────────────────────────────────────────────────────────
    # Observación
    # ──────────────────────────────────────────────────────────────────────────

    def get_obs(self) -> np.ndarray:
        """
        Devuelve el vector de observación normalizado:
            - Cola por carril (vehículos detenidos / 15)
            - Densidad por carril (vehículos / longitud × 10)
            - Fase actual en one-hot
            - Tiempo en fase normalizado por max_green
        """
        queue = np.array(
            [traci.lane.getLastStepHaltingNumber(l) / 15.0 for l in self.lanes],
            dtype=np.float32,
        )
        density = np.array(
            [
                traci.lane.getLastStepVehicleNumber(l)
                / max(traci.lane.getLength(l), 1.0)
                * 10.0
                for l in self.lanes
            ],
            dtype=np.float32,
        )
        phase_oh = np.zeros(self.num_phases, dtype=np.float32)
        phase_oh[self.current_phase] = 1.0

        time_norm = np.array(
            [min(self.time_on_phase / self.max_green, 1.0)], dtype=np.float32
        )
        return np.concatenate([queue, density, phase_oh, time_norm])

    # ──────────────────────────────────────────────────────────────────────────
    # Recompensa
    # ──────────────────────────────────────────────────────────────────────────

    def get_reward(self) -> float:
        """
        Recompensa diferencial basada en tiempo de espera acumulado.
        Positiva si la espera bajó, negativa si subió.
        """
        waiting = sum(traci.lane.getWaitingTime(l) for l in self.lanes)
        reward  = self._last_waiting - waiting
        self._last_waiting = waiting
        return reward

    # ──────────────────────────────────────────────────────────────────────────
    # Acción
    # ──────────────────────────────────────────────────────────────────────────

    def can_change(self) -> bool:
        """Respeta el verde mínimo antes de permitir cambio."""
        return self.time_on_phase >= self.min_green

    def request_phase(self, action: int) -> bool:
        """
        Registra una acción.
        Si es un cambio válido: activa el amarillo y guarda la fase pendiente.
        Devuelve True si se inició una transición amarilla.
        """
        if action == self.current_phase or not self.can_change():
            self.time_on_phase += self.delta_time
            self._pending_phase = None
            return False

        # Construir estado amarillo reemplazando G/g → y
        current_state = self.green_phases[self.current_phase][1]
        yellow_state  = current_state.replace("G", "y").replace("g", "y")
        traci.trafficlight.setRedYellowGreenState(self.id, yellow_state)
        self._pending_phase = action
        return True

    def commit_phase(self):
        """
        Llama a esto después de simular YELLOW_TIME pasos.
        Activa la fase verde pendiente si la hay.
        """
        if self._pending_phase is not None:
            new_sumo_idx = self.green_phases[self._pending_phase][0]
            traci.trafficlight.setPhase(self.id, new_sumo_idx)
            self.current_phase  = self._pending_phase
            self.time_on_phase  = 0
            self._pending_phase = None
