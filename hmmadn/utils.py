from __future__ import annotations

from abc import ABC, abstractmethod

from numpy import append, array, cumsum, where, ndarray
from numpy.random import Generator, PCG64

from hmmadn._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hmmadn._typing import List, Observation, State


def sum_delta_arrays(delta1: ndarray, delta2: ndarray) -> ndarray:
    delta1[delta1 == -float('inf')] = 0
    delta2[delta2 == -float('inf')] = 0
    res = delta1 + delta2
    res[res == 0] = -float('inf')
    return res


class ProbVec:
    """Une classe pour tirer des valeurs d'un vecteur selon une loi donnée.

    Attributs
    ---------
    n : int
        La longeur du vecteur. 
    cum_prob_vec : ndarray
        Le vecteur des probabilités en manière cumulative à partir de zéro.
    value_vec : list
        Une liste des valeurs qui seront sorties.
    _gen : Generator
        Objet de Generator responsable pour générer des nombres aléatoire.
    
    Méthodes
    --------
    gen_value : any
        Tirer une valeur du value_vec selon les probabilités en prob_vec.

    """

    def __init__(self, prob_vec: ndarray, value_vec: List[State]) -> None:
        """Initialiser l'objet de ProbVec.

        Paramètres
        ----------
        prob_vec : ndarray
            Un array des probabilités en forme non cumulative. Chaque
            élément corréspond à la probabilité à tirer la valeur dans
            l'array de value_vec au respect de l'ordre.
        value_vec : List[State]
            La liste des valeurs à tirer.

        Raises
        ------
        ValueError
            Si les longeurs de prob_vec et de value_vec ne sont pas
            pareils.
            Si la somme des probabilitées en prob_vec n'égale pas à 1.

        """
        if len(prob_vec) != len(value_vec):
            raise ValueError
        self.n = len(value_vec)
        if prob_vec.sum() != 1:
            raise ValueError
        self.prob_vec = prob_vec
        self.cum_prob_vec = append(0., cumsum(prob_vec))
        self.value_vec = value_vec
        self._gen = Generator(PCG64())

    def gen_value(self) -> State:
        """Tirer une valeur du value_vec selon les probabilités en prob_vec.

        Pour tirer une valeur, un nombre aléatoire est généré entre 0 et 1,
        puis on trouve la position en prob_vec qui inclut ce nomnre. La
        position trouvée est utilisé pour sortir la valeur finale.

        Sortie
        -------
        State
            La valeur tirée.

        """
        rand = self._gen.uniform()
        for i in range(self.n):
            if self.cum_prob_vec[i] < rand <= self.cum_prob_vec[i+1]:
                return self.value_vec[i]
            
    def __call__(self) -> State:
        return self.gen_value()
            
    def __getitem__(self, index: int) -> float:
        return self.prob_vec[index]
            

class ObservationLaw(ABC):
    """Classes abstraite pour générer des observations d'un HMM.

    L'idée est que la classe de HMMGen peut passer l'état actuel
    au objet d'ObservationLaw pour générer une observation. Ainsi,
    l'implementation est définie par l'utilisateur.

    Méthodes
    --------
    gen_obs(state)
        Méthode abstraite pour générer une observation basée un état
        actuel.

    """

    @abstractmethod
    def gen_obs(self, state: State) -> Observation:
        raise NotImplementedError
    
    def __call__(self, state: State) -> Observation:
        return self.gen_obs(state)

def get_error(
        obs_states: List[State],
        viterbi_states: List[State],
        n: int,
    ) -> float:
    return (abs(array(viterbi_states) - array(obs_states)).sum()) / n

def get_duration_error(
        obs_durations: List[int],
        viterbi_durations: List[int],
    ) -> float:
    h = where(
        (array(obs_durations) - array(viterbi_durations)) != 0,
        1,
        0
    )
    return h.sum() / h.size


class DurationLaw(ABC):

    @abstractmethod
    def gen_value(self) -> State:
        raise NotImplementedError
    
    @abstractmethod
    def get_prob(self, state: State) -> float:
        raise NotImplementedError
    

class DurationProbVec(ProbVec, DurationLaw):

    def __init__(self, prob_vec: ndarray, value_vec: List[State]) -> None:
        ProbVec.__init__(self, prob_vec, value_vec)
        self.prob_dict = dict(zip(value_vec, prob_vec))

    def get_prob(self, state: State) -> float:
        return self.prob_dict[state]

    def __call__(self, state: State | None = None) -> State:
        if state is not None:
            return self.get_prob(state)
        return self.gen_value()