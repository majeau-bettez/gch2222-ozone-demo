"""Paramètres du module Ozone stratosphérique — GCH2222, automne 2026.

Source unique de vérité pour les constantes du module. Chaque valeur porte son
nom complet, son unité et sa référence.

CONVENTION D'UNITÉS
-------------------
Tout est en unités du système international :
    concentrations molaires   C_i    mol m-3
    vitesses de réaction      r_j    mol m-3 s-1
    constantes du 1er ordre   k      s-1
    constantes du 2e ordre    k      m3 mol-1 s-1
    constantes du 3e ordre    k      m6 mol-2 s-1

La littérature en chimie atmosphérique utilise des densités numériques en
molécules cm-3 et note « j » les constantes de photolyse. Les facteurs de
conversion sont fournis plus bas.

Clés de références (voir gch2222.bib) :
    [Chapman30]   chapman1930
    [JPL19]       burkholder2019   -- https://jpldataeval.jpl.nasa.gov/
    [Solomon99]   solomon1999
    [Jacob99]     jacob1999
    [COESA76]     coesaUSStandardAtmosphere1976
    [AFGL86]      anderson1986
    [CODATA]      CODATA 2018 / révision du SI de 2019
"""

import numpy as np

# ===========================================================================
# 1. Constantes physiques
# ===========================================================================
R = 8.314462618
"""Constante des gaz parfaits, J mol-1 K-1. Exacte depuis 2019. [CODATA]"""

N_A = 6.02214076e23
"""Constante d'Avogadro, mol-1. Exacte depuis 2019. [CODATA]"""

g = 9.80665
"""Accélération gravitationnelle normale, m s-2. Exacte par convention.
[COESA76]"""

M_air = 28.9647e-3
"""Masse molaire de l'air sec, kg mol-1. [COESA76]"""

# ===========================================================================
# 2. Facteurs de conversion vers les unités de la chimie atmosphérique
# ===========================================================================
CM3_PAR_M3 = 1.0e6
"""Nombre de cm3 dans 1 m3."""

MOLEC_CM3_PAR_MOL_M3 = N_A / CM3_PAR_M3
"""Multiplier une concentration en mol m-3 par ce facteur donne la densité
numérique en molécules cm-3. Vaut 6,022e17."""

CONV_K2 = 1.0e-12 * N_A**2
"""Multiplier une constante termoléculaire en cm6 molécule-2 s-1 par ce
facteur donne des m6 mol-2 s-1. Vaut 3,626e35."""

CONV_K3 = 1.0e-6 * N_A
"""Multiplier une constante bimoléculaire en cm3 molécule-1 s-1 par ce facteur
donne des m3 mol-1 s-1. Vaut 6,022e17."""

DOBSON = 2.687e20
"""Unité Dobson exprimée en molécules m-2. 1 DU = 2,687e16 molécules cm-2,
soit une couche de 10 micromètres d'ozone pur aux conditions normales.
La colonne totale moyenne du globe vaut environ 300 DU. [Jacob99]"""

# ===========================================================================
# 3. Composition
# ===========================================================================
y_O2 = 0.2095
"""Fraction molaire de O2 dans l'air sec, sans dimension. Homogène dans toute
l'homosphère (jusqu'à ~100 km). [COESA76]"""

# ===========================================================================
# 4. Constantes cinétiques du mécanisme de Chapman
#
# Les expressions ci-dessous sont des FONCTIONS de la température, données
# dans les unités de la littérature puis converties. Elles proviennent de
# l'évaluation JPL n° 19. [JPL19]
# ===========================================================================
def k2_de_T(T):
    """Constante de R2 : O + O2 + M -> O3 + M, en m6 mol-2 s-1.

    Paramètre
    ---------
    T : température, K

    Forme JPL : k = 6,0e-34 (T/300)^-2,4 cm6 molécule-2 s-1.
    NOTER LE SIGNE : l'exposant est négatif, donc k2 AUGMENTE quand T diminue.
    La recombinaison termoléculaire est plus rapide dans le froid --- contre-
    exemple utile à l'intuition d'Arrhenius. [JPL19]
    """
    return 6.0e-34 * (T / 300.0) ** (-2.4) * CONV_K2


def k4_de_T(T):
    """Constante de R4 : O + O3 -> 2 O2, en m3 mol-1 s-1.

    Paramètre
    ---------
    T : température, K

    Forme JPL : k = 8,0e-12 exp(-2060/T) cm3 molécule-1 s-1. [JPL19]
    """
    return 8.0e-12 * np.exp(-2060.0 / T) * CONV_K3


# ===========================================================================
# 5. Constantes cinétiques du cycle catalytique NOx
#     C1 : NO  + O3 -> NO2 + O2
#     C2 : NO2 + O  -> NO  + O2
# ===========================================================================
def kC1_de_T(T):
    """Constante de C1 : NO + O3 -> NO2 + O2, en m3 mol-1 s-1.
    Forme JPL : k = 3,0e-12 exp(-1500/T) cm3 molécule-1 s-1. [JPL19]"""
    return 3.0e-12 * np.exp(-1500.0 / T) * CONV_K3


def kC2_de_T(T):
    """Constante de C2 : NO2 + O -> NO + O2, en m3 mol-1 s-1.
    Forme JPL : k = 5,1e-12 exp(210/T) cm3 molécule-1 s-1. [JPL19]"""
    return 5.1e-12 * np.exp(210.0 / T) * CONV_K3


k_C3 = 1.0e-2
"""Constante de photolyse de NO2 (NO2 + hv -> NO + O), s-1, soleil au zénith.

CETTE RÉACTION EST ESSENTIELLE et facile à oublier : elle rend l'atome O et
ferme un CYCLE NUL (NO + O3 -> NO2 + O2, puis NO2 + hv -> NO + O, puis
O + O2 + M -> O3 + M : bilan net nul). Comme k_C3 est très supérieure à
k_C2 * C_O, l'essentiel du NO2 emprunte cette voie et NE détruit PAS d'ozone.
L'omettre surestime l'efficacité du cycle NOx d'un ordre de grandeur.
Valeur pour un soleil au zénith. NO2 photolyse dans le proche ultraviolet
et le visible, où l'atmosphère est transparente : k_C3 varie peu avec
l'altitude, ce qui justifie de la prendre constante. Correspond à une durée
de vie de NO2 d'environ 100 s. [JPL19, brasseur2005]"""

z_NOx = np.array([15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]) * 1e3
"""Altitudes du profil de NOx, m."""

y_NOx = np.array([0.45, 1.65, 4.43, 8.61, 14.4, 15.2, 12.9, 10.7]) * 1e-9
"""Fraction molaire DIURNE de la famille NOx = NO + NO2, sans dimension.

Somme des profils tabulés de NO et de NO2 de [AFGL86], table 2b. Ces deux
espèces n'y sont définies que par un profil unique, de type « U.S.
Standard », et non par modèle climatique --- leur variabilité réelle est
donc plus grande que celle de l'ozone.

Les auteurs précisent avoir retenu les estimations DE JOUR pour les
espèces à cycle diurne, dont NO et NO2. C'est bien ce qu'il nous faut.

POURQUOI UN PROFIL ET NON UNE CONSTANTE. Vers le bas, l'azote réactif est
progressivement SÉQUESTRÉ dans des espèces réservoirs (HNO3, N2O5, ClONO2)
qui ne détruisent pas l'ozone : la fraction de NOx actif chute d'un facteur
30 entre 40 et 15 km. Nous ne modélisons pas cette séquestration --- elle
exigerait la famille HOx, la séparation des canaux de photolyse de
l'ozone, un profil de vapeur d'eau et une intégration diurne pour N2O5.
Nous en PRESCRIVONS le résultat. Fermeture déclarée, du même type que
celle sur k1(z)."""



# ===========================================================================
# 6. Constantes de photolyse --- FERMETURE CALIBRÉE
#
# Ces valeurs sont pour un SOLEIL AU ZÉNITH (angle zénithal solaire nul).
# Elles ne valent donc pas pour une moyenne diurne ni pour les hautes
# latitudes; c'est une hypothèse déclarée du carnet.
# ===========================================================================
k1_infini = 5.0e-10
"""Constante de photolyse de O2 (R1) non atténuée, s-1, soleil au zénith.

[Nicolet86] établit que J(O2) ne peut excéder environ 5e-10 s-1 au niveau de
la stratopause pour l'intervalle 200-240 nm (continuum de Herzberg), source
dominante d'oxygène impair dans la stratosphère. À cette altitude
l'atténuation est faible : la borne sert donc de valeur non atténuée.

Les bandes de Schumann-Runge (175-205 nm) ajoutent une contribution au-dessus
de ~50 km; elles sont négligées ici."""

sigma_O2 = 1.0e-23 * 1e-4 * N_A
"""Section efficace effective d'absorption de O2 dans le continuum de
Herzberg, m2 mol-1 (valeur saisie en cm2 par molécule puis convertie).

VALEUR DE LABORATOIRE, non ajustée. Elle a une histoire : révisée à la
baisse dans les années 1980, avec des conséquences directes sur l'ozone
stratosphérique calculé. Connue aujourd'hui à 10-20 % près. [Nicolet86]"""

transmittance_20km = 1.0e-2
"""Transmittance verticale à 20 km dans l'intervalle 200-240 nm, sans
dimension.

[Nicolet86] : à 20 km et en dessous, la transmittance tombe sous 1e-2. Cet
ancrage sert à calibrer le SEUL paramètre libre de la fermeture, la section
efficace effective de l'ozone dans la fenêtre à 210 nm (section 4b du
carnet).

CONTRÔLE DE COHÉRENCE. La calibration donne environ 3,4e-19 cm2 par
molécule, à comparer au minimum local de la bande de Hartley dans cette
fenêtre, de l'ordre de 3e-19 cm2. Rien ne l'imposait."""

k3_ref = 1.0e-3
"""Constante de photolyse de O3 (R3), s-1, tous canaux confondus (bandes de
Hartley, Huggins et Chappuis), soleil au zénith.

PROVENANCE. Correspond à une durée de vie photolytique de l'ozone de 1000 s,
soit 17 minutes --- l'ordre de grandeur admis dans la stratosphère moyenne,
vers 30 km. [brasseur2005]

CE QUE CETTE VALEUR N'EST PAS. Une moyenne. La constante de photolyse de
l'ozone CROÎT d'environ un ordre de grandeur entre 20 et 45 km : [jacob1999]
retient 1e-2 s-1 pour la haute stratosphère. La traiter comme constante avec
l'altitude est une simplification déclarée, calibrée sur ~30 km.

Contrairement à k1, k3 n'est PAS écrantée dans ce modèle : l'ozone photolyse
dans des bandes où l'atmosphère sus-jacente est peu opaque au-dessus de
20 km. Le moyennage diurne ne lui applique donc que le rapport cyclique
jour/nuit, sans facteur de masse d'air --- asymétrie essentielle, puisque
C_O3 varie comme sqrt(k1/k3)."""

# ===========================================================================
# 7. ATMOSPHÈRE ET PROFIL D'OZONE DE RÉFÉRENCE
#
# Modèle 2 « Midlatitude Summer » (45 N, juillet) de [AFGL86].
# Rapport en accès libre : https://apps.dtic.mil/sti/tr/pdf/ADA175173.pdf
#
# POURQUOI CELUI-LÀ. Polytechnique est à 45,5 N : la latitude de référence
# du modèle est la nôtre. Le rapport précise en outre que, pour les espèces
# à cycle diurne (O3, NO, NO2), ce sont les valeurs DE JOUR qui sont
# tabulées --- ce qui est cohérent avec notre hypothèse de photolyse
# diurne.
#
# CE QUE CE PROFIL EST. Une climatologie composite : ozone troposphérique
# de LOWTRAN, stratosphérique des satellites SBUV et LIMS (Nimbus 7),
# mésosphérique de SME. Lissée et interpolée. Précision annoncée par les
# auteurs : 10 à 30 % dans la troposphère et la stratosphère.
#
# CE QU'IL N'EST PAS. Une mesure. Les auteurs avertissent que les
# gradients horizontaux, saisonniers et latitudinaux dépassent souvent un
# facteur 2 à 10 dans la stratosphère, et qu'un profil représentatif ne
# reproduit aucune situation observée en particulier.
# ===========================================================================
_AFGL_MLS = np.array([
    # z (km),   T (K),   n (cm-3),  y_O3 (ppmv)
    [12.0, 222.3, 6.814e18, 2.23e-01],
    [13.0, 215.8, 6.012e18, 3.00e-01],
    [14.0, 215.7, 5.141e18, 4.40e-01],
    [15.0, 215.7, 4.368e18, 5.00e-01],
    [16.0, 215.7, 3.730e18, 6.00e-01],
    [17.0, 215.7, 3.192e18, 7.00e-01],
    [18.0, 216.8, 2.715e18, 1.00e+00],
    [19.0, 217.9, 2.312e18, 1.50e+00],
    [20.0, 219.2, 1.967e18, 2.00e+00],
    [21.0, 220.4, 1.677e18, 2.40e+00],
    [22.0, 221.6, 1.429e18, 2.90e+00],
    [23.0, 222.8, 1.223e18, 3.40e+00],
    [24.0, 223.9, 1.042e18, 4.00e+00],
    [25.0, 225.1, 8.919e17, 4.80e+00],
    [27.5, 228.5, 6.050e17, 6.00e+00],
    [30.0, 233.7, 4.094e17, 7.00e+00],
    [32.5, 239.0, 2.820e17, 8.10e+00],
    [35.0, 245.2, 1.927e17, 8.90e+00],
    [37.5, 251.3, 1.338e17, 8.70e+00],
    [40.0, 257.5, 9.373e16, 7.55e+00],
    [42.5, 263.7, 6.624e16, 5.90e+00],
    [45.0, 269.9, 4.726e16, 4.50e+00],
    [47.5, 275.2, 3.398e16, 3.50e+00],
    [50.0, 275.7, 2.500e16, 2.80e+00],
    [55.0, 269.3, 1.386e16, 1.80e+00],
    [60.0, 257.1, 7.668e15, 1.30e+00],
])
"""Table de référence AFGL, modèle 2 (45 N, juillet). [AFGL86]

Colonnes : altitude (km), température (K), densité numérique totale
(molécules cm-3), fraction molaire d'ozone (ppmv).

La pression tabulée du rapport n'est pas reprise : quelques valeurs sont
corrompues par la numérisation du document de 1986. Elle se recalcule
exactement par p = n k T, ce que fait `atmosphere_reference` ci-dessous."""

z_ref = _AFGL_MLS[:, 0] * 1e3
"""Altitudes de la table de référence, m."""

T_ref = _AFGL_MLS[:, 1]
"""Températures de la table de référence, K."""

C_M_ref = _AFGL_MLS[:, 2] * CM3_PAR_M3 / N_A
"""Concentration molaire totale de l'air, mol m-3, convertie depuis la
densité numérique tabulée."""

y_O3_ref = _AFGL_MLS[:, 3] * 1e-6
"""Fraction molaire d'ozone de référence, sans dimension."""

C_O3_ref = y_O3_ref * C_M_ref
"""Concentration molaire d'ozone de référence, mol m-3.

Maximum d'environ 4,3e12 molécules cm-3 vers 24-25 km, et maximum de
fraction molaire (8,9 ppm) vers 35 km : les deux ne coïncident pas, la
densité de l'air décroissant plus vite que l'ozone."""

# ===========================================================================
# 7bis. Géométrie solaire du cas de référence
#
# La table AFGL retenue vaut pour 45 N en juillet, valeurs de jour. Nous
# calquons la géométrie solaire sur ces conditions.
# ===========================================================================
latitude = np.deg2rad(45.5)
"""Latitude du cas de référence, rad. Montréal : 45,5 N --- à un demi-degré
de la latitude nominale du modèle AFGL. [AFGL86]"""

declinaison = np.deg2rad(21.5)
"""Déclinaison solaire à la mi-juillet, rad.

Valeur au 15 juillet, cohérente avec le « July » du modèle AFGL. Elle varie
de 23,4 (solstice) à 20,4 degrés sur le mois."""

# ===========================================================================
# 8. Énergétique de l'absorption ultraviolette
# ===========================================================================
h_Planck = 6.62607015e-34
"""Constante de Planck, J s. Exacte depuis 2019. [CODATA]"""

c_lumiere = 2.99792458e8
"""Vitesse de la lumière dans le vide, m s-1. Exacte par définition. [CODATA]"""

lambda_Hartley = 255e-9
"""Longueur d'onde du maximum de la bande de Hartley, m.

L'ozone y absorbe avec une section efficace d'environ 1,15e-17 cm2 par
molécule. La transition est ÉLECTRONIQUE et mène à un état DISSOCIATIF :
le rendement quantique de dissociation vaut 0,90 à 0,95. [JPL19]

Ne pas confondre avec la bande de vibration à 9,6 micromètres, dans
l'infrarouge, qui contribue à l'effet de serre et non à l'écran
ultraviolet."""

D_O3 = 101.0e3
"""Énergie de dissociation O3 -> O2 + O, J mol-1.

Faible devant l'énergie d'un photon de la bande de Hartley : l'excédent
part en translation des fragments, en vibration de l'O2 et en excitation
électronique (O(1D), O2(1Delta_g)), puis se thermalise par collisions."""

cp_air = 29.1
"""Capacité thermique molaire de l'air sec à pression constante,
J mol-1 K-1. Faiblement dépendante de T dans la stratosphère. [COESA76]"""

# ===========================================================================
# 9. Transport stratosphérique
# ===========================================================================
SENSIBILITE = """SENSIBILITÉ DU RÉSULTAT AUX CONSTANTES DE PHOTOLYSE.

La solution d'équilibre varie comme C_O3 ~ sqrt(k1 / k3). Une erreur d'un
facteur 4 sur k1 déplace donc l'ozone d'un facteur 2, soit l'ampleur même de
l'effet que le carnet met en évidence.

k1 est ici ancrée sur deux valeurs publiées (Nicolet et Kennes 1986) et la
section efficace effective qui en résulte concorde avec la valeur de
laboratoire du continuum de Herzberg. k3 et k_C3 restent des valeurs
typiques, non ancrées sur une mesure précise.

Trois hypothèses limitent la portée quantitative des chiffres :
  1. soleil au zénith --- pas de moyenne diurne ni de dépendance en latitude;
  2. bandes de Schumann-Runge négligées --- sans effet sous 50 km;
  3. écrantage par l'ozone non résolu, absorbé dans la section efficace
     effective; le couplage rayonnement-composition est donc découplé de
     force, et le modèle ne peut pas décrire l'auto-écrantage.

Pour raffiner : le calculateur Quick TUV du NCAR est gratuit et en ligne,
https://www.acom.ucar.edu/Models/TUV/Interactive_TUV/
Régler l'altitude de mesure, l'angle zénithal à 0 et la colonne d'ozone,
puis lire J(O2), J(O3) et J(NO2) dans la sortie « photolysis frequencies ».
"""


z_transition_litt = 25.0e3
"""Altitude de transition entre contrôle photochimique et contrôle dynamique
d'après la littérature, m.

L'évaluation WMO de 1985 (chapitre 8, d'après le modèle bidimensionnel de
Garcia et Solomon 1985) énonce que sous environ 25 km, la durée de vie
photochimique de l'oxygène impair est longue devant le temps caractéristique
du transport méridien et vertical. La distribution d'ozone y est donc
gouvernée par la dynamique. [wmo1985]"""

tau_Ox_transition = (1.0e7, 1.0e8)
"""Encadrement de la durée de vie photochimique de l'oxygène impair au
voisinage de la transition, s --- soit environ 4 mois à 3 ans.

Lu sur les isocontours de la figure 8-2 de [wmo1985] (log10 tau = 7 et 8),
qui borde la zone de transition. Ordre de grandeur, pas une mesure ponctuelle."""

D_t = 0.5
"""Diffusivité turbulente verticale dans la stratosphère, m2 s-1.
Varie de 0,1 à 1 selon l'altitude et la source : ordre de grandeur seulement.
À comparer à 10-100 m2 s-1 dans la couche limite atmosphérique.
[Solomon99, brasseur2005]"""
