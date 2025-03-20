from __future__ import annotations

from math import log

from hmmadn._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hmmadn._typing import Callable, List, Observation, State, ndarray


class Viterbi:

    def __init__(
            self,
            states: List[State],
            obs_list: List[Observation],
            trans_mat: ndarray[float],
            mus: ndarray[float],
            b_Sj_Ot_function: Callable
        ) -> None:
        self.states = states
        self.n = len(states)
        self.deltas = []
        self.phis = []
        self.obs_list = obs_list
        self.n_obs = len(obs_list)
        self.trans_mat = trans_mat
        self.mus = mus
        self.b_Sj_Ot_function = b_Sj_Ot_function

    def run_viterbi(self) -> None:
        self._get_deltas_and_phis()
        phi_T = self.deltas[-1].index(max(self.deltas[-1]))
        states_star = [phi_T]
        for phis in list(reversed(self.phis[1:])):
            states_star.append(
                phis[states_star[-1]]
            )
        return list(reversed(states_star))

    def _get_deltas_and_phis(self):
        for t in range(self.n_obs):
            if t == 0:
                self.deltas.append([
                    log(self.mus[j]) + log(self._b_Sj_Ot(j, t))
                    for j in range(self.n)
                ])
                self.phis.append(
                    [0 for _ in range(self.n)]
                )
            else:
                self.deltas.append(
                    [self._get_dtj(t, j) for j in range(self.n)]
                )
                self.phis.append(
                    [self._get_phi_tj(t, j) for j in range(self.n)]
                )

    def _b_Sj_Ot(self, j: int, t: int) -> float:
        return self.b_Sj_Ot_function(j, self.obs_list[t])
    
    def _get_dtj(self, t: int, j: int) -> float:
        max_ = -float('inf')
        bsjot = self._b_Sj_Ot(j, t)
        for i in range(self.n):
            try:
                temp = (
                    log(self.trans_mat[i][j])
                    + self.deltas[t-1][i]
                    + log(bsjot)
                )
            except ValueError:
                # Si la probabilité est 0, le log
                # donnera une erreur
                temp = -float('inf')
            if temp > max_:
                max_ = temp
        return max_

    def _get_phi_tj(self, t: int, j: int) -> float:
        max_value = -float('inf')
        max_S = None
        for i in range(self.n):
            try:
                temp = (
                    log(self.trans_mat[i][j])
                    + self.deltas[t-1][i]
                )
            except ValueError:
                temp = -float('inf')
            if temp > max_value:
                max_value = temp
                max_S = i
        return max_S

