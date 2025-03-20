from __future__ import annotations

from numpy import array, isnan, log, unravel_index

from hmmadn._typing import TYPE_CHECKING
from hmmadn.utils import sum_delta_arrays
from .semigen import SemiGenRes

if TYPE_CHECKING:
    from hmmadn._typing import Callable, List, ndarray


class SemiViterbi:

    def __init__(
            self,
            n_states: int,
            d_max: int, 
            gen_res: SemiGenRes,
            mus: List[float],
            pd: Callable,
            b_Sj_Ot_function: Callable,
            trans_mat: ndarray[float],
        ) -> None:
        self.n_obs = gen_res.n_obs
        self.obs_list = gen_res.obs_only
        self.mus = mus
        self.pd = pd
        self.n = n_states
        self._max_d = d_max
        self.num_obs_states = len(gen_res.segmented_obs)
        self.b_Sj_Ot_function = b_Sj_Ot_function
        self.trans_mat = trans_mat

    def run_viterbi(self) -> None:
        self.set_deltas_and_phis()
        self.set_optimal_sequence()

    def set_deltas_and_phis(self) -> list:
        self.deltas = []
        self.phis = []
        for t in range(self.n_obs):
            if t == 0:
                self.deltas.append(
                    self.get_init_delta_array(t, t+1)
                )
                self.phis.append(
                    array([
                        [(float('nan'), float('nan')) for _ in range(self.n)]
                        for _ in range(1, self._max_d+1)
                    ])
                )
            else:
                self.deltas.append(self.get_delta(t))
                self.phis.append(
                    array([
                        [self._get_phi_tj(t, j, d) for j in range(self.n)]
                        for d in range(1, self._max_d+1) 
                    ])  
                )

    def get_init_delta_array(self, t: int, init_d: int) -> ndarray:
        return array([
            [
                log(self.mus[j])
                + log(self.pd(d))
                + log(self._b_Sj_Ot(j, self.obs_list[(t+1-d):t+1]))
                if d == init_d else -float('inf')
                for j in range(self.n)
            ]
            for d in range(1, self._max_d+1)
        ])

    def get_delta(self, t: int):
        delta_array = self.get_delta_array(t)
        if t < self._max_d:
            return sum_delta_arrays(
                self.get_init_delta_array(t, t+1),
                delta_array
            )
        return delta_array

    def get_delta_array(self, t: int) -> ndarray:
        return array([
            [self._get_dtj(t, j, d) for j in range(self.n)]
            for d in range(1, self._max_d+1)
        ])

    def set_optimal_sequence(self) -> None:
        self.states_only = []
        self.states_and_durations = []
        self.durations_only = []

        a = self.deltas[-1]
        index_ = unravel_index(a.argmax(), a.shape)
        d = int(index_[0] + 1)
        j = int(index_[1])

        self.states_only.append(j)
        self.states_and_durations.append((j, d))
        self.durations_only.append(d)

        t = self.n_obs

        while t > 0:
            j_star, d_star = self.phis[t-1][d-1][j]
            t -= d
            if isnan(j_star) or isnan(d_star):
                if t != 0:
                    raise Exception
                else:
                    continue
            
            j, d = int(j_star), int(d_star)
            self.states_only.append(j)
            self.states_and_durations.append((j, d))
            self.durations_only.append(d)

        self.states_only = self.states_only[::-1]
        self.states_and_durations = self.states_and_durations[::-1]
        self.durations_only = self.durations_only[::-1]

    def _b_Sj_Ot(self, j: int, obs_segment: List[ndarray]) -> float:
        return self.b_Sj_Ot_function(j, obs_segment)
        
    def _get_dtj(self, t: int, j: int, d: int) -> float:
        # d needs to be 1-indexed
        obs_segment = self.obs_list[(t+1-d):t+1]
        bsjot = self._b_Sj_Ot(j, obs_segment)
        max_ = -float('inf')
        for i in range(self.n):
            if i != j:
                for d_ in range(self._max_d):
                    pd = self.pd(d)
                    if (
                        self.trans_mat[i][j] == 0
                        or t < d
                        or bsjot == 0
                        or pd == 0
                    ):
                        temp = -float('inf')
                    else:
                        temp = (
                            log(self.trans_mat[i][j])
                            + log(pd)
                            + self.deltas[t-d][d_][i]
                            + log(bsjot)
                        )
                    if temp > max_:
                        max_ = temp
        return max_
    
    def _get_phi_tj(self, t: int, j: int, d: int) -> float:
        # d needs to be 1-indexed
        obs_segment = self.obs_list[(t+1-d):t+1]
        bsjot = self._b_Sj_Ot(j, obs_segment)
        max_ = -float('inf')
        state_duration_pair = (float('nan'), float('nan'))
        
        for i in range(self.n):
            if i != j:
                for d_ in range(self._max_d):
                    pd = self.pd(d)
                    if (
                        self.trans_mat[i][j] == 0
                        or t < d
                        or bsjot == 0
                        or pd == 0.0
                    ):
                        temp = -float('inf')
                    else:
                        temp = (
                            log(self.trans_mat[i][j])
                            + log(pd)
                            + self.deltas[t-d][d_][i]
                            + log(bsjot)
                        )
                    if temp > max_:
                        max_ = temp
                        state_duration_pair = (i, d_+1)
        return state_duration_pair