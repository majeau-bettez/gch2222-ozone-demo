"""Solveur d'équilibre du mécanisme de Chapman — GCH2222.

MIROITE LE CORRIGÉ, NE LE REMPLACE PAS. Les fonctions ci-dessous
reproduisent volontairement l'équilibre_(catalyse)/vitesses/balayage
présentés comme exercices à compléter dans le TD étudiant
(`td_ozone_etudiant.py`, sept fonctions marquées « à remplir ») et fournis
dans le corrigé (`td_ozone_enseignant.py`). Elles NE sont PAS importées par
ces notebooks : le TD reste un exercice où la personne étudiante écrit ces
fonctions elle-même, et ce module ne doit jamais devenir une façon de
contourner cet exercice.

Cette duplication déclarée existe pour un usage distinct : une application
interactive (`cours/04_ozone/webapp/`) a besoin d'un solveur importable,
sans effets de bord d'exécution de notebook. Si le corrigé change, ce
module doit être mis à jour en conséquence — c'est le prix assumé de ne
pas avoir une unique source de vérité ici.

CONVENTION D'UNITÉS : système international (voir parametres_ozone).
"""

import numpy as np
from scipy.optimize import fsolve


def vitesses(C_O, C_O3, C_O2, C_M, k1, k2, k3, k4):
    """Vitesses des quatre réactions de Chapman, mol m-3 s-1.

    Retourne le quadruplet (r1, r2, r3, r4).
    """
    r1 = k1 * C_O2
    r2 = k2 * C_O * C_O2 * C_M
    r3 = k3 * C_O3
    r4 = k4 * C_O * C_O3
    return r1, r2, r3, r4


def vitesse_catalytique(C_X, kC1, kC2, kC3, C_O, C_O3, fraction_jour):
    """Destruction d'oxygène impair par le cycle catalytique NOx, moyennée
    sur 24 h, mol m-3 s-1.

    C_X : concentration molaire de la famille NOx = NO + NO2, mol m-3.
    fraction_jour : durée du jour rapportée à 24 h, sans dimension.
    """
    return fraction_jour * (C_X * kC1 * kC2 * C_O3 * C_O
                             / (kC1 * C_O3 + kC2 * C_O + kC3))


def equilibre(k1, k2, k3, k4, C_M, C_tot, estimation,
              C_X=0.0, kC1=0.0, kC2=0.0, kC3=1.0, fraction_jour=0.0):
    """Équilibre photochimique local (C_O, C_O3, C_O2), mol m-3.

    Résout les deux conditions d'équilibre du cycle rapide (R2=R3, à un
    terme de puits catalytique près) plus le bilan atomique
    C_O + 2 C_O2 + 3 C_O3 = C_tot. Le puits catalytique est optionnel :
    C_X = 0 (valeur par défaut) redonne l'équilibre de Chapman pur.

    MISE À L'ÉCHELLE. Les conditions d'équilibre portent sur des vitesses
    de l'ordre de 1e-11 mol m-3 s-1, le bilan atomique sur des
    concentrations de l'ordre de 1e-1 mol m-3 : on divise les deux
    premiers résidus par r3 pour que le critère d'arrêt absolu de fsolve
    ne soit pas saturé par le seul bilan atomique.
    """

    def residus(inconnues):
        C_O, C_O3, C_O2 = inconnues
        r1, r2, r3, r4 = vitesses(C_O, C_O3, C_O2, C_M, k1, k2, k3, k4)
        r_cat = vitesse_catalytique(C_X, kC1, kC2, kC3, C_O, C_O3,
                                     fraction_jour)
        residu_O = (2.0 * r1 - r2 + r3 - r4 - r_cat) / r3
        residu_O3 = (r2 - r3 - r4 - r_cat) / r3
        residu_atomes = C_O + 2.0 * C_O2 + 3.0 * C_O3 - C_tot
        return [residu_O, residu_O3, residu_atomes]

    solution, _, code, message = fsolve(residus, np.array(estimation),
                                         full_output=True)
    if code != 1:
        raise RuntimeError(f"non-convergence : {message}")
    if np.any(solution <= 0.0):
        raise RuntimeError("concentration négative : racine sans sens "
                            "physique, changez l'estimation de départ")
    return solution


def balayage(z, dans_domaine, k1, k2, k3, k4, C_M, C_tot,
             C_X=None, kC1=None, kC2=None, kC3=1.0, fraction_jour=0.0):
    """Résout l'équilibre à chaque altitude du domaine, par continuation.

    Descend depuis le sommet du domaine ; chaque résolution est amorcée
    avec la solution de l'altitude précédente (les concentrations varient
    de plusieurs ordres de grandeur sur le profil).

    C_X, kC1, kC2 : tableaux (taille de z) ou None pour l'équilibre de
    Chapman pur, sans puits catalytique.

    Retourne (C_O, C_O3), tableaux de la taille de z, NaN hors du domaine.
    """
    C_O = np.full(len(z), np.nan)
    C_O3 = np.full(len(z), np.nan)
    avec_catalyse = C_X is not None

    indices = [i for i in range(len(z)) if dans_domaine[i]]
    indices.reverse()  # du haut vers le bas

    i_depart = indices[0]
    estimation = (1e-9, 1e-8, C_tot[i_depart] / 2)

    for i in indices:
        kwargs = {}
        if avec_catalyse:
            kwargs = dict(C_X=C_X[i], kC1=kC1[i], kC2=kC2[i], kC3=kC3,
                           fraction_jour=fraction_jour)
        solution = equilibre(k1[i], k2[i], k3[i], k4[i], C_M[i], C_tot[i],
                              estimation, **kwargs)
        C_O[i] = solution[0]
        C_O3[i] = solution[1]
        estimation = solution

    return C_O, C_O3
