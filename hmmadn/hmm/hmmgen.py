from __future__ import annotations

from numpy import ndarray, repeat

from hmmadn._typing import TYPE_CHECKING
from hmmadn.utils import ObservationLaw, ProbVec

if TYPE_CHECKING:
    from hmmadn._typing import List, Observation, State
    

class HMMGen:
    """Classe responsable pour générer une séquence des états et des observations.

    Attributes
    ----------
    states : List[State]
        La liste des états du modèle.
    n : int
        La longeur de la liste des états.
    trans_matrix : ndarray
        ...
    trans_vectors : dict[any, ProbVec]
        ...
    obs_laws : ObservationLaw
        ...
    state : any
        ...

    Méthodes
    --------
    
    
    """

    def __init__(
            self,
            states: List[State],
            trans_matrix: ndarray,
            obs_laws: ObservationLaw,
            mus: List[float] | None = None
        ):
        """Initialiser l'objet de HMMGen.
        
        Paramètres
        ----------
        states : list[any]
            La liste des états du modèle. En effet, le type de données des valeurs
            ne sont pas pertinentes.
        trans_matrix : ndarray
            Une matrice de probabilités de transition, où l'élément a_ij donne la
            probabilité de transitionner de l'état i à l'état j. La matrice doit
            être carré, et la longeur doit égale celle de la liste des états.
        obs_laws : ObservationLaw
            Un objet qui implémente l'interface d'ObservationLaw.
        mus: list[float] | None; optionel, défaut = None
            Une liste de probabilités de commencer dans les états. Si une valeur n'est 
            pas passée, chaque valeur aura une probabilité égale d'être le premier dans
            la séquence.

        """
        self.states = states
        self.n = len(states)
        self._set_trans_matrix(trans_matrix)
        self._set_trans_vectors()
        self._set_obs_laws(obs_laws)
        self._set_mus(mus)

    def _set_trans_matrix(self, trans_matrix: ndarray) -> None:
        """Valider le trans_matrix et le déclarer.

        Parameters
        ----------
        trans_matrix : ndarray
            Une matrice de probabilités de transition.

        Raises
        ------
        ValueError
            Si le trans_matrix n'est pas de type de ndarray. 
            Si il n'est pas d'une forme carrée.
            Si sa longeur n'égale pas à celle des états.
            Si la somme de chaque ligne n'égale pas à 1.

        """
        if not isinstance(trans_matrix, ndarray):
            raise ValueError
        if trans_matrix.shape[0] != trans_matrix.shape[1]:
            raise ValueError
        if trans_matrix.shape[0] != self.n:
            raise ValueError
        for i in range(self.n):
            if sum(trans_matrix[i]) != 1.:
                raise ValueError
        self.trans_matrix = trans_matrix

    def _set_trans_vectors(self) -> None:
        """Déclarer la variable de classe de trans_vectors.

        En effet, pour bien utiliser le trans_matrix pour
        tirer des états, il faut qu'on utilise une structure
        de ProbVec. Ainsi, pour chaque état actuel, on peut
        générer le prochain selon les probabilités.

        """
        self.trans_vectors = {
            self.states[i]: ProbVec(self.trans_matrix[i], self.states)
            for i in range(self.n)
        }

    def _set_obs_laws(self, obs_laws: ObservationLaw) -> None:
        """Valider et déclarer la variable d'obs_laws.

        Parameters
        ----------
        obs_laws : ObservationLaw
            Un objet qui implémente l'interface d'ObservationLaw.

        Raises
        ------
        ValueError
            Si la variable d'obs_laws n'est pas d'ObservationLaw.

        """
        if not isinstance(obs_laws, ObservationLaw):
            raise ValueError
        self.obs_laws = obs_laws

    def _set_mus(self, mus: List[float]) -> None:
        """_summary_

        Parameters
        ----------
        mus : list[float]
            Une liste de probabilités de commen dans l'état donné.
            L'élément m_i corréspond à la probabilité de commencer
            la séquence dans l'état s_i, selon les emplacements dans
            la liste des états.

        Raises
        ------
        ValueError
            Si la longeur de mus n'égale pas à celle de states.

        """
        if mus is None:
            # Des probabilités égales par défaut.
            self.mus = ProbVec((repeat(1, self.n) / self.n), self.states)
        else:
            if len(mus) != self.n:
                raise ValueError
            self.mus = ProbVec(mus, self.states)
        self.state = self.mus.gen_value()

    def gen_obs(self, m: int, states: bool = False) -> List[Observation | tuple]:
        """Générer une séquence des observations selon le modéle.

        Parameters
        ----------
        m : int
            La longeur de la séquence générée.
        states : bool, optionel, défaut = False
            Si True, la séquence inclut les états associés aux
            observations.

        Returns
        -------
        list[Observation | tuple]
            La sortie pourrait être une liste de :
                - l'observation
                - un tuple de l'observation et l'état
            Le type de donnée de l'observation dépend d'objet
            d'ObservationLaw qui l'a générée. Généralement, elle
            va être un float, ou bien un ndarray des floats (si
            la sortie est multi-dimensionnelle)

        """
        res = []
        states_res = []
        for _ in range(m):
            curr_obs = self.obs_laws(self.state)
            if states:
                states_res.append(self.state)
            res.append(curr_obs)
            self.next_state()
        if states:
            return res, states_res
        return res

    def next_state(self) -> None:
        """Déclarer le prochain état selon l'état actuel."""
        self.state = self.trans_vectors[self.state].gen_value()

