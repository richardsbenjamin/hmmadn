from __future__ import annotations

from numpy import repeat

from hmmadn._typing import TYPE_CHECKING
from hmmadn.utils import ProbVec

if TYPE_CHECKING:
    from hmmadn._typing import Callable, List, Observation, State, Tuple, ndarray


class SemiGen:
    
    def __init__(
            self,
            states: List[State],
            semi_trans_mat: ndarray,
            obs_law: Callable,
            duration_law: Callable,
            mus: List[float] | None = None,
        ) -> None:
        self.states = states
        self.n = len(states)
        self._set_trans_vectors(semi_trans_mat)
        self._set_duration_law(duration_law)
        self._set_mus(mus)
        self.obs_law = obs_law

    def _set_trans_vectors(self, semi_trans_mat: ndarray) -> None:
        self.trans_vectors = {
            self.states[i]: ProbVec(semi_trans_mat[i], self.states)
            for i in range(self.n)
        }

    def _set_duration_law(self, duration_law: Callable) -> None:
        self._duration_law = duration_law

    def _set_mus(self, mus: List[float]) -> None:
        if mus is None:
            # Des probabilités égales par défaut.
            self.mus = ProbVec((repeat(1, self.n) / self.n), self.states)
        else:
            if len(mus) != self.n:
                raise ValueError
            self.mus = ProbVec(mus, self.states)
        self.state = self.mus.gen_value()

    def _next_state(self) -> float:
        self.state = self.trans_vectors[self.state].gen_value()
    
    def get_duration(self):
        return self._duration_law()
    
    def gen_semi_hmm(self, num_states: int) -> dict:
        segmented_obs = []
        states_only = []
        states_only2 = []
        duration_only = []
        obs_only = []
        states_and_durations = []

        for _ in range(num_states):
            duration = self.get_duration()
            obs_sequence = []
            for _ in range(duration):
                curr_obs = self.obs_law(self.state)
                obs_sequence.append(curr_obs)
                obs_only.append(curr_obs)
                states_only2.append(self.state)

            states_only.append(self.state)
            states_and_durations.append((self.state, duration))
            segmented_obs.append(obs_sequence)
            duration_only.append(duration)
            self._next_state()

        return SemiGenRes(**{
            "segmented_obs": segmented_obs,
            "states_only": states_only,
            "states_only2": states_only2,
            "duration_only": duration_only,
            "obs_only": obs_only,
            "states_and_durations": states_and_durations,
            "n_obs": len(obs_only),
        })


class SemiGenRes:

    def __init__(
            self,
            segmented_obs: List[ndarray[Observation]],
            states_only: List[State],
            states_only2: List[State],
            duration_only: List[int],
            obs_only: List[Observation],
            states_and_durations: List[Tuple[int, State]],
            n_obs: int,
        ) -> None:
        self.segmented_obs = segmented_obs
        self.states_only = states_only
        self.states_only2 = states_only2
        self.duration_only = duration_only
        self.obs_only = obs_only
        self.states_and_durations = states_and_durations
        self.n_obs = n_obs
