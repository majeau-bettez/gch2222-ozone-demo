"""Application interactive — la couche d'ozone, modèle par modèle.

GCH2222, Système Terre et enjeux environnementaux. Conçue pour une journée
portes ouvertes : montre, niveau par niveau, comment le mécanisme de
Chapman se rapproche du profil d'ozone observé à mesure qu'on y ajoute des
ingrédients physiques -- jamais en calant des paramètres sur l'observation.

Réutilise telles quelles les briques déjà validées du TD :
    parametres_ozone.py  -- constantes, cinétique, atmosphère de référence
    rayonnement.py       -- fermetures radiatives (photolyse_naive à
                             photolyse_complete)
    chapman.py            -- solveur d'équilibre (miroir déclaré du corrigé
                             du TD ; voir son en-tête)

Lancement : streamlit run app.py
(environnement conda gch2222-prep, voir programming/environment-prep.yml)
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ICI, "..", "modelisation"))

# Logo officiel, vendoré dans cours/commun/img/ -- source unique déjà
# utilisée ailleurs dans le cours (slides Beamer). Résolu relativement à ce
# fichier, pas au cwd (même convention que explose_img.py, voir CLAUDE.md).
LOGO_POLY = os.path.join(_ICI, "..", "..", "commun", "img",
                          "Logo_Alternatif_Polytechnique_Montreal_noir_rgb.png")

import parametres_ozone as P  # noqa: E402
import rayonnement as R  # noqa: E402
import chapman as C  # noqa: E402


st.set_page_config(
    page_title="La couche d'ozone, modèle par modèle",
    layout="wide",
)

COULEUR_MODELE = "#1f77b4"
COULEUR_REF = "crimson"
COULEUR_PRECEDENT = "0.6"


# ===========================================================================
# Atmosphère de référence -- indépendante des contrôles, calculée une fois
# ===========================================================================
@st.cache_resource
def preparer_atmosphere():
    z = np.linspace(0.0, 60e3, 241)
    T, C_M, C_O2 = R.atmosphere(z)
    return dict(
        z=z, T=T, C_M=C_M, C_O2=C_O2,
        k2=P.k2_de_T(T), k4=P.k4_de_T(T),
        kC1=P.kC1_de_T(T), kC2=P.kC2_de_T(T),
        dans_domaine=(z >= 12e3) & (z <= 55e3),
        C_tot=2.0 * C_O2,
        # Zéro hors du domaine tabulé : correct pour l'intégrale de colonne
        # (metriques), mais PAS pour l'affichage -- voir graphique().
        C_O3_ref_colonne=np.interp(z, P.z_ref, P.C_O3_ref, left=0.0, right=0.0),
        # Pas de bornes explicites : prolonge par la valeur au bord (comme
        # le TD), pour éviter un NOx artificiellement nul sous 15 km.
        y_NOx_ref=np.interp(z, P.z_NOx, P.y_NOx),
    )


ATM = preparer_atmosphere()
z = ATM["z"]

# Borne droite fixe du graphique du niveau 0, mol m-3. Le profil naïf peut
# dépasser d'ordres de grandeur (surtout aux valeurs extrêmes des sliders) ;
# sans borne fixe, l'axe s'étire et écrase la référence contre l'axe des y.
# Voir graphique(xlim_max=...).
#
# ATTENTION (piège vécu) : ne JAMAIS laisser une chaîne nue (style docstring
# de parametres_ozone.py) sur sa propre ligne dans ce fichier -- Streamlit
# interprète toute expression "nue" qui n'est pas un appel de fonction comme
# une commande magique et l'affiche sur la page. Utiliser des `#` ici.
X_MAX_NAIF = 3.0 * np.nanmax(P.C_O3_ref)


def xlim_raisonnable(*courbes, marge=1.15):
    """Borne droite adaptée au(x) profil(s) qu'on veut garder pleinement
    lisible(s) -- typiquement le modèle du niveau courant et la
    référence. Une courbe de comparaison délibérément énorme (p. ex. le
    niveau naïf affiché en pâle) n'est PAS incluse ici : elle peut sortir
    du cadre sans que l'axe s'étire pour l'accommoder."""
    return marge * max(np.nanmax(c) for c in courbes)


def note_hors_cadre(C_modele, xlim_max):
    """Affiche une légende si C_modele dépasse la borne d'affichage."""
    pic = np.nanmax(C_modele)
    if pic > xlim_max:
        st.caption(
            f"Le profil sort du cadre à droite (pic à {pic:.1e} mol m⁻³, "
            f"soit {pic / np.nanmax(P.C_O3_ref):.0f}× le pic de référence) "
            ": l'échelle est volontairement bornée pour garder la courbe "
            "rouge lisible."
        )


@st.cache_data(show_spinner=False)
def profil_chapman(fermeture, k1_expose=None, k3_expose=None, facteur_NOx=0.0):
    """Profil vertical de C_O3 (mol m-3) pour une fermeture radiative donnée.

    fermeture : "naive" | "simple" | "complete"
    k1_expose, k3_expose : log10(k1), log10(k3), utilisés seulement si
        fermeture == "naive" (contrôle du niveau 0). None -> valeurs par
        défaut de parametres_ozone.
    facteur_NOx : multiplicateur du profil NOx de référence AFGL, 0 = pas
        de cycle catalytique.
    """
    if fermeture == "naive":
        k1_inf = 10.0 ** k1_expose if k1_expose is not None else P.k1_infini
        k3_val = 10.0 ** k3_expose if k3_expose is not None else P.k3_ref
        k1, k3 = np.full_like(z, k1_inf), np.full_like(z, k3_val)
    elif fermeture == "simple":
        k1, k3 = R.photolyse_simple(z)
    elif fermeture == "complete":
        k1, k3 = R.photolyse_complete(z)
    else:
        raise ValueError(fermeture)

    if facteur_NOx > 0.0:
        C_NOx = facteur_NOx * ATM["y_NOx_ref"] * ATM["C_M"]
        fraction_jour = k3[0] / P.k3_ref if fermeture == "complete" else 1.0
        _, C_O3 = C.balayage(
            z, ATM["dans_domaine"], k1, ATM["k2"], k3, ATM["k4"],
            ATM["C_M"], ATM["C_tot"],
            C_X=C_NOx, kC1=ATM["kC1"], kC2=ATM["kC2"], kC3=P.k_C3,
            fraction_jour=fraction_jour,
        )
    else:
        _, C_O3 = C.balayage(
            z, ATM["dans_domaine"], k1, ATM["k2"], k3, ATM["k4"],
            ATM["C_M"], ATM["C_tot"],
        )
    return C_O3


@st.cache_data(show_spinner=False)
def trace_iterations():
    """Estimation de C_O3 (mol m-3) à chaque passe du point fixe de
    l'auto-écrantage (niveau 2), amorce comprise -- pour l'animation de
    convergence. Voir rayonnement.photolyse_complete(historique=True)."""
    _, _, trace = R.photolyse_complete(z, historique=True)
    return trace


def colonne_DU(C_O3):
    """Colonne stratosphérique (12-55 km) en unités Dobson."""
    C_rempli = np.nan_to_num(C_O3, nan=0.0)
    colonne_mol_m2 = R.colonne_au_dessus(z, C_rempli)[0]
    return colonne_mol_m2 * P.N_A / P.DOBSON


def graphique(C_modele, etiquette_modele, C_precedent=None,
              etiquette_precedent=None, xlim_max=None):
    """xlim_max : borne droite fixe (mol m-3), mol m-3. None -> échelle
    automatique. Un profil qui la dépasse sort simplement du cadre --
    utile au niveau 0, où le profil naïf peut être des ordres de grandeur
    trop grand et écraserait sinon la référence contre l'axe des y."""
    fig, ax = plt.subplots(figsize=(5.0, 5.6))
    if C_precedent is not None:
        ax.plot(C_precedent, z / 1e3, color=COULEUR_PRECEDENT, lw=1.8,
                ls="--", label=etiquette_precedent)
    ax.plot(C_modele, z / 1e3, color=COULEUR_MODELE, lw=2.4,
            label=etiquette_modele)
    # Tableau natif AFGL (commence à 12 km) : pas de valeur fabriquée
    # sous la tropopause, contrairement à une interpolation zéro-remplie.
    ax.plot(P.C_O3_ref, P.z_ref / 1e3, "o--", color=COULEUR_REF, ms=3,
            lw=1.4, label="référence AFGL, 45 N juillet")
    ax.set_xlabel(r"concentration molaire d'ozone $C_{O_3}$ (mol m$^{-3}$)")
    ax.set_ylabel("altitude $z$ (km)")
    ax.set_ylim(10, 55)
    # Statement if/else, pas une expression ternaire nue : cette dernière
    # (pas un simple appel) se ferait attraper par la commande magique de
    # Streamlit et son retour (le tuple xlim) s'afficherait sur la page --
    # piège vécu, voir la remarque équivalente près de X_MAX_NAIF plus haut.
    if xlim_max is not None:
        ax.set_xlim(0, xlim_max)
    else:
        ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    plt.close(fig)
    return fig


def graphique_iteration(trace, i, C_precedent=None, etiquette_precedent=None):
    """Une image de l'animation de convergence : la passe `i` en trait
    plein, les passes 0..i-1 en pâle (plus une passe est ancienne, plus
    elle est pâle). L'axe des concentrations est fixé sur l'ensemble de
    la trace pour que l'échelle ne saute pas d'une image à l'autre."""
    fig, ax = plt.subplots(figsize=(5.0, 5.6))
    if C_precedent is not None:
        ax.plot(C_precedent, z / 1e3, color=COULEUR_PRECEDENT, lw=1.5,
                ls=":", label=etiquette_precedent)
    n = len(trace)
    for j in range(i):
        alpha = 0.08 + 0.32 * (j + 1) / max(i, 1)
        ax.plot(trace[j], z / 1e3, color=COULEUR_MODELE, lw=1.1, alpha=alpha)
    ax.plot(trace[i], z / 1e3, color=COULEUR_MODELE, lw=2.4,
            label=f"itération {i} / {n - 1}")
    ax.plot(P.C_O3_ref, P.z_ref / 1e3, "o--", color=COULEUR_REF, ms=3,
            lw=1.4, label="référence AFGL, 45 N juillet")
    ax.set_xlabel(r"concentration molaire d'ozone $C_{O_3}$ (mol m$^{-3}$)")
    ax.set_ylabel("altitude $z$ (km)")
    ax.set_ylim(10, 55)
    borne_x = 1.05 * max(np.nanmax(c) for c in trace)
    ax.set_xlim(0, borne_x)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    plt.close(fig)
    return fig


def afficher_metriques(C_modele):
    ref = ATM["C_O3_ref_colonne"]
    i_pic_m = int(np.nanargmax(C_modele))
    i_pic_r = int(np.nanargmax(ref))
    rapport_pic = C_modele[i_pic_m] / ref[i_pic_r]
    ecart_altitude = (z[i_pic_m] - z[i_pic_r]) / 1e3
    du_modele = colonne_DU(C_modele)
    du_ref = colonne_DU(ref)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rapport au pic (modèle / référence)", f"{rapport_pic:.2f}×")
    c2.metric("Écart d'altitude du pic", f"{ecart_altitude:+.1f} km")
    c3.metric("Colonne strato. (modèle / réf.)",
              f"{du_modele:.0f} / {du_ref:.0f} DU")


def glossaire():
    with st.expander("Glossaire des symboles"):
        st.markdown(r"""
| Symbole | Nom complet | Dimension | Unité d'exemple |
|---|---|---|---|
| $C_{O_3}$ | concentration molaire d'ozone | mol / volume | mol m$^{-3}$ |
| $C_{O_2}$, $C_M$ | concentration molaire de O$_2$, du mélange total | mol / volume | mol m$^{-3}$ |
| $k_1$ | constante de photolyse de O$_2$ (R1) | 1 / temps | s$^{-1}$ |
| $k_3$ | constante de photolyse de O$_3$ (R3) | 1 / temps | s$^{-1}$ |
| $N_{O_2}$, $N_{O_3}$ | colonne d'air (resp. d'ozone) au-dessus de $z$ | quantité / surface | mol m$^{-2}$ |
| $\sigma$ | section efficace effective d'absorption | surface / quantité | m$^2$ mol$^{-1}$ |
| $z$ | altitude | longueur | km (affiché), m (calcul) |
| $y_{NO_x}$ | fraction molaire de la famille NO + NO$_2$ | sans dimension | ppb |
| $DU$ | unité Dobson (colonne d'ozone) | quantité / surface | 1 DU $\approx$ 2,687$\times10^{20}$ molécules m$^{-2}$ |
""")


# ===========================================================================
# En-tête (fixe en haut) et navigation (panneau latéral)
# ===========================================================================
# Le titre reste visible même en défilant : utile en salle, quand on
# discute avec une personne qui a fait défiler la page pour lire une
# équation. `position: sticky` seule sur .titre-fixe ne suffit PAS ici :
# vérifié dans un vrai Chrome (Playwright), le conteneur défilant de
# Streamlit est section[data-testid=stMain], et c'est son ENFANT DIRECT
# (le div[data-testid=stElementContainer] généré par Streamlit autour de
# ce st.markdown) qui doit porter position:sticky pour que ça s'accroche
# réellement -- la sticky posée sur notre propre div interne, plus bas
# dans l'arbre, n'avait aucun effet visible (testé, confirmé cassé).
# top:60px (pas 0) car la barre d'outils Streamlit (data-testid=stHeader,
# z-index très supérieur) occupe déjà les 60 premiers pixels et
# recouvrirait un titre collé à top:0. Couleur de fond blanche en dur :
# ce Streamlit n'expose aucune variable CSS de thème (vérifié, --*
# absentes de :root) ; à revoir si le thème sombre de Streamlit est activé.
st.markdown(
    """
    <style>
    div[data-testid="stElementContainer"]:has(.titre-fixe) {
        position: sticky;
        top: 60px;
        z-index: 998;
        background-color: #ffffff;
    }
    .titre-fixe {
        padding: 0.6rem 0 0.5rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    }
    /* Plus gros que les sous-titres de section (st.subheader, 28px dans le
       contenu principal) pour garder une hiérarchie visuelle claire. */
    .titre-fixe h1 { margin: 0; font-size: 2.6rem; }
    </style>
    <div class="titre-fixe">
        <h1>On modélise la couche d'ozone <u>ensemble</u> en classe</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Libellés volontairement courts pour tenir sur les boutons du sélecteur
# de niveau sans troncature -- la description complète de chaque niveau
# réapparaît en sous-titre dans le contenu principal une fois sélectionné.
niveaux = [
    "0 — Chapman naïf",
    "1 — écran solaire",
    "2 — auto-écrantage",
    "3 — destruction NOx",
]
with st.sidebar:
    # Sigle et titre du cours, en haut du panneau -- déplacés hors du
    # bandeau principal pour lui laisser toute la place au titre. Logo
    # officiel plutôt que le nom en texte de l'établissement.
    st.markdown(
        '<div style="font-size:1.7rem; font-weight:700; '
        'margin-bottom:0.2rem;">GCH2222</div>',
        unsafe_allow_html=True,
    )
    st.caption("Système Terre et enjeux environnementaux")
    st.image(LOGO_POLY, width=180)
    st.divider()
    st.subheader("Niveau de raffinement")
    # st.segmented_control reste en interne une rangée horizontale
    # (role="radiogroup", flex-wrap: wrap) qui replie chaque option sur sa
    # propre ligne faute de largeur dans le panneau latéral -- mais les
    # coins arrondis restent calculés comme pour une seule barre continue
    # (premier bouton arrondi à gauche seulement, boutons du milieu carrés
    # des deux côtés, dernier arrondi à droite seulement), d'où l'effet de
    # troncature visuelle une fois replié en colonne. Vérifié dans un vrai
    # Chrome (Playwright) : border-radius vaut "8px 0 0 8px" / "0" / "0" /
    # "0 8px 8px 0" sur les quatre boutons. On force un arrondi complet et
    # uniforme sur chacun, ainsi qu'une largeur pleine (donc égale) --
    # ciblé via data-testid/data-variant (attributs exposés par Streamlit)
    # plutôt que les classes .st-emotion-cache-* générées, mais reste
    # dépendant de la structure interne du widget et à revalider à chaque
    # montée de version de Streamlit.
    #
    # Bordure gauche : dans la barre continue d'origine, Streamlit rend
    # transparente la bordure gauche du bouton qui suit immédiatement le
    # bouton sélectionné (pour ne pas doubler visuellement la bordure
    # accent du bouton actif). Une fois les boutons empilés, ça laisse une
    # encoche sur ce bouton-là -- vérifié : c'est systématiquement le
    # voisin direct du sélectionné, jamais un bouton fixe par position.
    # Couleur de bordure (rgba(49,51,63,0.2)) copiée de l'état normal des
    # autres boutons ; codée en dur faute de variable de thème exposée
    # (même limite déjà notée pour le fond blanc du titre fixe plus haut)
    # -- à revoir si le thème sombre de Streamlit est activé.
    st.markdown(
        """
        <style>
        div[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
            border-radius: 8px !important;
            margin: 0 !important;
            width: 100% !important;
        }
        div[data-testid="stButtonGroup"]
            button[data-variant="segmented_control"][aria-checked="true"]
            + button[data-variant="segmented_control"] {
            border-left-color: rgba(49, 51, 63, 0.2) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    choix = st.segmented_control("Niveau de raffinement", niveaux,
                                  default=niveaux[0],
                                  label_visibility="collapsed")
# Un deuxième clic sur le segment actif le désélectionne (choix devient
# None) -- contrairement à st.radio, où un choix reste toujours actif. On
# retombe alors sur le dernier niveau valide plutôt que de planter.
if choix is None:
    niveau = st.session_state.get("dernier_niveau", 0)
else:
    niveau = niveaux.index(choix)

# Détection d'une arrivée sur le niveau -- pas seulement la première fois
# de la session, mais à chaque re-sélection depuis un autre niveau. Sert
# à l'animation automatique du niveau 2 (voir plus bas).
arrivee_niveau = niveau != st.session_state.get("dernier_niveau")
st.session_state["dernier_niveau"] = niveau

col_science, col_resultat = st.columns([2, 3])


# ===========================================================================
# Niveau 0 — Chapman naïf
# ===========================================================================
if niveau == 0:
    with col_science:
        st.subheader("Niveau 0 — Chapman naïf")
        st.markdown(
            "Le mécanisme complet, quatre réactions élémentaires :"
        )
        st.latex(r"""
        \begin{array}{lll}
        R_1 & O_2 + h\nu \rightarrow 2\,O & k_1 \\
        R_2 & O + O_2 + M \rightarrow O_3 + M & k_2 \\
        R_3 & O_3 + h\nu \rightarrow O_2 + O & k_3 \\
        R_4 & O + O_3 \rightarrow 2\,O_2 & k_4
        \end{array}
        """)
        st.markdown(
            "**Hypothèse naïve testée ici :** $k_1$ et $k_3$ sont "
            "**constantes avec l'altitude**, comme si l'atmosphère "
            "au-dessus de chaque point était parfaitement transparente à "
            "l'ultraviolet."
        )
        st.latex(r"k_1(z) = k_1^{\infty} \qquad k_3(z) = k_{3,\text{réf}}")
        st.markdown("**Contrôle : ajustez les deux constantes vous-même.**")
        k1_exp = st.slider(r"log₁₀ k₁ (s⁻¹)", -11.0, -7.0,
                            float(np.log10(P.k1_infini)), 0.1)
        k3_exp = st.slider(r"log₁₀ k₃ (s⁻¹)", -5.0, -1.0,
                            float(np.log10(P.k3_ref)), 0.1)
        st.caption(
            "Essayez de trouver des valeurs qui reproduisent la courbe "
            "rouge en pointillé. Vous n'y arriverez pas : sans "
            "dépendance en altitude, $C_{O_3}$ suit la densité de l'air "
            "et **décroît de façon monotone** — aucun jeu de constantes "
            "ne crée un maximum. C'est un échec de **structure**, pas de "
            "calibration."
        )

    with col_resultat:
        C_O3 = profil_chapman("naive", k1_exp, k3_exp)
        st.pyplot(graphique(C_O3, "Chapman, $k_1,k_3$ constantes",
                             xlim_max=X_MAX_NAIF))
        note_hors_cadre(C_O3, X_MAX_NAIF)
        afficher_metriques(C_O3)

# ===========================================================================
# Niveau 1 — écran solaire (soleil au zénith)
# ===========================================================================
elif niveau == 1:
    with col_science:
        st.subheader("Niveau 1 — écran solaire (soleil au zénith)")
        st.markdown(
            "$k_1$ dépend maintenant de la colonne d'air traversée par "
            "le rayonnement, via une loi de Beer-Lambert :"
        )
        st.latex(
            r"k_1(z) = k_1^{\infty}\, e^{-\sigma_{\text{eff}}\, N_{O_2}(z)}"
        )
        st.markdown(
            r"où $N_{O_2}(z)$ est la colonne de O$_2$ **au-dessus** de "
            r"$z$ et $\sigma_{\text{eff}}$ est calibrée sur deux "
            r"transmittances publiées (Nicolet et Kennes, 1986). "
            r"$k_3$ reste constante : l'ozone photolyse dans des bandes "
            r"peu opaques."
        )
        st.markdown("**Contrôle : activez ou retirez l'écran.**")
        actif = st.toggle("Écrantage par O₂ actif (loi de Beer-Lambert)",
                           value=True)
        st.caption(
            "Une couche apparaît, à peu près à la bonne altitude — déjà "
            "remarquable pour quatre réactions élémentaires. Mais la "
            "concentration reste trop forte en bas de la couche."
        )

    with col_resultat:
        C_O3 = profil_chapman("simple") if actif else profil_chapman("naive")
        C_precedent = profil_chapman("naive") if actif else None
        # Avec écran : le profil naïf n'est qu'une ombre de comparaison,
        # on borne sur la courbe courante. Sans écran : c'est le profil
        # naïf lui-même qui est affiché, même traitement qu'au niveau 0.
        xlim_max = (xlim_raisonnable(C_O3, P.C_O3_ref) if actif
                    else X_MAX_NAIF)
        st.pyplot(graphique(
            C_O3, "avec écran O₂" if actif else "sans écran (niveau 0)",
            C_precedent, "sans écran (niveau 0)" if actif else None,
            xlim_max=xlim_max,
        ))
        note_hors_cadre(C_O3, xlim_max)
        afficher_metriques(C_O3)

# ===========================================================================
# Niveau 2 — auto-écrantage itératif
# ===========================================================================
elif niveau == 2:
    with col_science:
        st.subheader("Niveau 2 — auto-écrantage itératif")
        st.markdown("Trois corrections combinées :")
        st.markdown(
            "1. **Deux absorbants** : O$_2$ *et* O$_3$ atténuent "
            "l'ultraviolet qui photolyse O$_2$.\n"
            "2. **Moyenne sur 24 h**, avec l'asymétrie jour/nuit propre "
            "à chaque réaction.\n"
            "3. **Auto-écrantage** : $k_1$ dépend de la colonne "
            "d'ozone, que le modèle est justement en train de calculer "
            "— une rétroaction négative, résolue par itération."
        )
        st.latex(
            r"k_1(z) = \overline{k_1^{\infty}\, "
            r"e^{-(\sigma_{O_2} N_{O_2} + \sigma_{O_3} N_{O_3}(z))\,m(\theta)}}"
            r"^{\;\text{jour}}"
        )
        st.markdown(
            r"$N_{O_3}(z)$ dépend de $C_{O_3}(z)$, la solution elle-même "
            r"— d'où l'itération à point fixe."
        )
        st.markdown("**Contrôle : activez l'auto-écrantage.**")
        actif = st.toggle(
            "Auto-écrantage itératif actif (sinon : écran O₂ seul, "
            "niveau 1)", value=True)
        st.caption(
            "Le facteur d'erreur passe d'environ 14× à environ 2×, sans "
            "qu'aucun paramètre n'ait été calé sur l'observation : le "
            "gain vient de la restitution de couplages physiques."
        )
        animer = False
        if actif:
            animer = st.button(
                "Rejouer la convergence, itération par itération")
            st.caption(
                "Chaque passe recalcule l'écrantage à partir de la "
                "colonne d'ozone de la passe précédente. Les traits "
                "pâles sont les passes intermédiaires."
            )

    with col_resultat:
        if actif and (animer or arrivee_niveau):
            trace = trace_iterations()
            C_precedent = profil_chapman("simple")
            espace = st.empty()
            for i in range(len(trace)):
                espace.pyplot(graphique_iteration(
                    trace, i, C_precedent, "écran O₂ seul (niveau 1)"))
                time.sleep(0.12)
            afficher_metriques(profil_chapman("complete"))
        else:
            C_O3 = profil_chapman("complete") if actif else profil_chapman("simple")
            C_precedent = profil_chapman("simple") if actif else None
            st.pyplot(graphique(
                C_O3,
                "auto-écrantage itératif" if actif else "écran O₂ seul (niveau 1)",
                C_precedent, "écran O₂ seul (niveau 1)" if actif else None,
            ))
            afficher_metriques(C_O3)

# ===========================================================================
# Niveau 3 — destruction catalytique NOx
# ===========================================================================
else:
    with col_science:
        st.subheader("Niveau 3 — destruction catalytique (NOx)")
        st.markdown("Un cycle catalytique à trois réactions :")
        st.latex(r"""
        \begin{array}{lll}
        C_1 & NO + O_3 \rightarrow NO_2 + O_2 & k_{C1} \\
        C_2 & NO_2 + O \rightarrow NO + O_2 & k_{C2} \\
        C_3 & NO_2 + h\nu \rightarrow NO + O & k_{C3} \;(\text{cycle nul})
        \end{array}
        """)
        st.markdown(
            r"$C_1$+$C_2$ détruit un O et un O$_3$ ; $C_3$ referme un "
            r"cycle nul qui **ne détruit rien**. La vitesse nette, "
            r"moyennée sur 24 h :"
        )
        st.latex(
            r"r_{\text{cat}} = f_{\text{jour}}\, C_{NO_x}\,"
            r"\frac{k_{C1} k_{C2}\, C_{O_3} C_O}"
            r"{k_{C1} C_{O_3} + k_{C2} C_O + k_{C3}}"
        )
        st.markdown(
            r"$C_{NO_x}$ suit un profil prescrit (climatologie AFGL). "
            r"**Contrôle : faites varier son intensité.**"
        )
        facteur = st.slider(
            r"Facteur d'échelle du profil de NOx (1,0 = climatologie AFGL)",
            0.0, 8.0, 1.0, 0.1)
        st.caption(
            "Quelques parties par milliard de NOx suffisent à dominer "
            "la destruction directe (R4) : une espèce ultra-minoritaire "
            "contrôle le budget d'ozone parce qu'elle n'est pas "
            "consommée, seulement recyclée par C₃."
        )

    with col_resultat:
        C_O3 = profil_chapman("complete", facteur_NOx=facteur)
        C_precedent = profil_chapman("complete", facteur_NOx=0.0)
        st.pyplot(graphique(
            C_O3, f"avec NOx, facteur {facteur:.1f}×",
            C_precedent, "sans NOx (niveau 2)",
        ))
        afficher_metriques(C_O3)

st.divider()
glossaire()

with st.expander("À propos de ce modèle"):
    st.markdown(
        "Quatre versions du même mécanisme chimique (Chapman, 1930), "
        "comparées au même profil d'ozone observé (modèle AFGL, "
        "45° N, juillet — la latitude de Montréal).\n\n"
        "**Ce qui ne change jamais d'un niveau à l'autre :** les quatre "
        "réactions élémentaires, l'atmosphère de référence, le soleil au "
        "zénith d'un point de vue géométrique de base. **Ce qui change :** "
        "uniquement la façon de calculer $k_1$ et $k_3$, et éventuellement "
        "un puits catalytique."
    )
    st.caption("Limites déclarées du modèle")
    st.text(P.SENSIBILITE)
