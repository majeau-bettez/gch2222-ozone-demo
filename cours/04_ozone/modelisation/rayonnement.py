"""Atmosphère de référence et constantes de photolyse — GCH2222.

Ce module regroupe tout ce que le travail dirigé n'a PAS à construire :
l'atmosphère de référence et le calcul des constantes de photolyse k1 et k3.
Les personnes étudiantes l'importent et l'utilisent comme une boîte noire
documentée.

Le contenu n'est pas difficile, il est simplement long : écrantage par deux
absorbants, moyennage sur la géométrie solaire d'une journée, et résolution
d'une rétroaction. Le détail est traité en cours, pas au TD.

    from rayonnement import atmosphere, photolyse_simple, photolyse_complete

CONVENTION D'UNITÉS : système international (voir parametres_ozone).
"""

import numpy as np

import parametres_ozone as P


# ===========================================================================
# 1. Atmosphère de référence
# ===========================================================================
def atmosphere(z):
    """(T, C_M, C_O2) à l'altitude z (m).

    Température interpolée dans la table AFGL 45 N juillet [AFGL86], puis
    pression par intégration de l'équilibre hydrostatique
    dp/dz = -M_air g p / (R T). La densité ainsi obtenue reproduit celle que
    la table donne directement à 4 % près en moyenne.

    Retourne des tableaux en K et mol m-3.
    """
    z = np.atleast_1d(np.asarray(z, float))
    T = np.interp(z, P.z_ref, P.T_ref)
    T = np.where(z < P.z_ref[0], P.T_ref[0] + 6.5e-3 * (P.z_ref[0] - z), T)
    T_moy = 0.5 * (T[1:] + T[:-1])
    ln_p = np.concatenate([
        [np.log(101325.0)],
        np.log(101325.0) + np.cumsum(-P.M_air * P.g / (P.R * T_moy) * np.diff(z))])
    C_M = np.exp(ln_p) / (P.R * T)
    return T, C_M, P.y_O2 * C_M


def colonne_au_dessus(z, C):
    """Colonne intégrée au-dessus de chaque altitude, mol m-2."""
    N = np.zeros(len(z))
    # On remonte depuis le sommet, où la colonne au-dessus est nulle.
    for i in range(len(z) - 2, -1, -1):
        epaisseur = z[i + 1] - z[i]
        N[i] = N[i + 1] + 0.5 * (C[i] + C[i + 1]) * epaisseur
    return N


# ===========================================================================
# 1bis. Version naïve : aucun écrantage
#
# Ajout pour l'application interactive de démonstration (webapp/), niveau
# d'entrée avant tout écrantage. N'apparaît pas dans le TD.
# ===========================================================================
def photolyse_naive(z):
    """(k1, k3) en s-1, constantes avec l'altitude.

    Aucun écrantage ultraviolet : k1 = k1_infini et k3 = k3_ref partout,
    comme si l'atmosphère sus-jacente était parfaitement transparente.
    Sert à montrer qu'aucune couche réaliste n'émerge tant que k1 ne
    dépend pas de la colonne d'air traversée --- la dépendance en
    altitude de l'écrantage, pas la valeur des constantes, est ce qui
    manque.
    """
    return np.full_like(z, P.k1_infini), np.full_like(z, P.k3_ref)


# ===========================================================================
# 2. Version simple : soleil au zénith, écrantage par O2 seul
# ===========================================================================
def photolyse_simple(z):
    """(k1, k3) en s-1, version naïve.

    Deux simplifications, toutes deux fausses mais instructives :
      - SOLEIL AU ZÉNITH, en permanence;
      - écrantage par l'OXYGÈNE SEUL, l'ozone étant ignoré comme absorbant.

    k1 suit une loi de Beer-Lambert, k1 = k1_inf exp(-sigma_eff N_O2), dont
    les deux paramètres sont calibrés sur les valeurs publiées de [Nicolet86]
    à 20 km et à la stratopause. k3 est prise constante.
    """
    _, C_M, C_O2 = atmosphere(z)
    N_O2 = colonne_au_dessus(z, C_O2)
    N20, N50 = np.interp(20e3, z, N_O2), np.interp(50e3, z, N_O2)
    sigma_eff = np.log(100.0) / (N20 - N50)
    k1_inf = P.k1_infini * np.exp(sigma_eff * N50)
    return k1_inf * np.exp(-sigma_eff * N_O2), np.full_like(z, P.k3_ref)


# ===========================================================================
# 3. Version complète : deux absorbants, moyenne diurne, auto-écrantage
# ===========================================================================
def _geometrie_solaire(n_pas=1441):
    """(facteur de masse d'air, masque de jour) sur une journée complète.

    Le trajet oblique allonge le chemin optique d'un facteur 1/cos(theta),
    plafonné à 40 pour éviter la divergence à l'horizon (sans conséquence :
    l'exponentielle écrase la contribution des grands angles).
    """
    h = np.linspace(-np.pi, np.pi, n_pas)
    mu = (np.sin(P.latitude) * np.sin(P.declinaison)
          + np.cos(P.latitude) * np.cos(P.declinaison) * np.cos(h))
    jour = mu > 0.0
    return np.where(jour, 1.0 / np.maximum(mu, 1.0 / 40.0), 0.0), jour


def _k1_moyen(tau_vertical, k1_inf, m_air, jour):
    """Moyenne sur 24 h de k1_inf exp(-tau m), s-1.

    On moyenne l'EXPONENTIELLE, pas l'exposant : la moyenne d'une
    exponentielle n'est pas l'exponentielle de la moyenne, et c'est toute
    la différence entre les deux versions de ce module.

    Une boucle sur les altitudes, vectorisée sur les instants de la journée.
    """
    k1 = np.zeros(len(tau_vertical))
    for i in range(len(tau_vertical)):
        valeurs_du_jour = k1_inf * np.exp(-tau_vertical[i] * m_air)
        k1[i] = np.where(jour, valeurs_du_jour, 0.0).mean()
    return k1


def photolyse_complete(z, n_iterations=100, tolerance=1e-7, historique=False):
    """(k1, k3) en s-1, version raffinée. Trois corrections :

    1. DEUX ABSORBANTS. L'ultraviolet qui dissocie O2 (200-240 nm) est
       absorbé par l'O2 et par l'OZONE, dont la bande de Hartley présente
       un minimum local vers 210 nm. Deux des trois paramètres viennent de
       la littérature (k1_infini, sigma_O2); seule sigma_O3 est calibrée,
       sur l'ancrage de transmittance à 20 km [Nicolet86]. Elle sort à
       3,4e-19 cm2 par molécule, contre ~3e-19 pour le minimum de Hartley :
       contrôle de cohérence réussi, non imposé par la calibration.

    2. MOYENNE DIURNE. k1 traverse une grande épaisseur optique : allonger
       le trajet l'écrase exponentiellement. k3 opère dans des bandes peu
       opaques : seule compte la fraction de jour. Comme C_O3 varie en
       sqrt(k1/k3), cette asymétrie réduit l'ozone calculé.

    3. AUTO-ÉCRANTAGE. k1 dépend de la colonne d'ozone, qui est justement
       ce que le modèle calcule : rétroaction NÉGATIVE, résolue par
       itération de point fixe. Plus d'ozone -> plus d'écrantage -> moins de
       production. C'est un correcteur d'erreur.

    RÉSERVE. Rien ne garantit l'unicité du point fixe; la sensibilité à
    l'amorce n'a pas été explorée.

    historique : si True, retourne un troisième élément -- la liste des
    estimations de C_O3 (mol m-3, l'approximation analytique utilisée en
    interne pour la rétroaction, PAS la solution fsolve de l'équilibre
    complet) à chaque itération, amorce comprise. Sert à visualiser la
    convergence; n'affecte pas (k1, k3).
    """
    T, C_M, C_O2 = atmosphere(z)
    N_O2 = colonne_au_dessus(z, C_O2)
    C_O3_ref = np.interp(z, P.z_ref, P.C_O3_ref, left=0.0, right=0.0)
    N_O3_ref = colonne_au_dessus(z, C_O3_ref)

    # Unique paramètre libre, calibré une seule fois sur la colonne de
    # référence : une section efficace est une propriété de la molécule,
    # pas un état de l'atmosphère.
    N2, N3 = np.interp(20e3, z, N_O2), np.interp(20e3, z, N_O3_ref)
    sigma_O3 = (-np.log(P.transmittance_20km) - P.sigma_O2 * N2) / N3

    m_air, jour = _geometrie_solaire()
    k3 = np.full_like(z, P.k3_ref * jour.mean())
    k2, k4 = P.k2_de_T(T), P.k4_de_T(T)
    strato = (z >= 12e3) & (z <= 55e3)

    def ozone(N_O3):
        k1 = _k1_moyen(P.sigma_O2 * N_O2 + sigma_O3 * N_O3,
                       P.k1_infini, m_air, jour)
        return k1, C_O2 * np.sqrt(k1 * k2 * C_M / (k3 * k4))

    k1, C_O3 = ozone(N_O3_ref)
    trace = [np.where(strato, C_O3, np.nan)] if historique else None
    for _ in range(n_iterations):
        k1, suivant = ozone(colonne_au_dessus(z, np.where(strato, C_O3, 0.0)))
        ecart = np.nanmax(np.abs(suivant / np.maximum(C_O3, 1e-30) - 1.0))
        C_O3 = 0.5 * C_O3 + 0.5 * suivant
        if historique:
            trace.append(np.where(strato, C_O3, np.nan))
        if ecart < tolerance:
            break
    if historique:
        return k1, k3, trace
    return k1, k3
