"""
Calendrier lunaire : jour lunaire à partir d'une photo de la lune.

Utilisation :
    python lune.py images/ma_photo.jpg               # lune croissante (par défaut)
    python lune.py images/ma_photo.jpg decroissante  # lune décroissante

Ou modifie CHEMIN et CROISSANTE ci-dessous et lance simplement : python lune.py

Étapes :
    1. Filtre        : niveaux de gris + filtre médian
    2. Segmentation  : seuil d'Otsu, plus grand objet, trous bouchés
    3. Disque + âge  : ajustement d'un cercle sur le bord extérieur, fraction éclairée, âge lunaire, affichage
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from scipy.spatial.distance import cdist
from skimage import color, filters, measure, morphology, segmentation

# ----------------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------------
CHEMIN = "images/moon.7041.jpg"
CROISSANTE = True       # True : lune croissante (avant la pleine lune) | False : décroissante
PERIODE = 29.53         # durée du mois lunaire (jours)


# ----------------------------------------------------------------------------
# Étape 1 : chargement et filtre
# ----------------------------------------------------------------------------
def charger(chemin):
    I = plt.imread(chemin)
    if I.dtype == np.uint8:                  # jpg : entiers 0..255 -> flottants 0..1
        I = I.astype(float) / 255
    if I.ndim == 3:                          # couleur -> niveaux de gris
        I = color.rgb2gray(I[:, :, :3])
    k = max(1, max(I.shape) // 600)          # grosse photo : on garde 1 pixel sur k
    return I[::k, ::k]


def filtrer(I):
    # Filtre médian 3x3 : enlève le bruit et les étoiles sans rendre les bords flous
    # (3x3 et pas plus : un noyau plus grand efface les croissants très fins)
    return signal.medfilt2d(I, kernel_size=3)


# ----------------------------------------------------------------------------
# Étape 2 : segmentation de la lune
# ----------------------------------------------------------------------------
def segmenter(J):
    masque = J > filters.threshold_otsu(J)                  # seuil d'Otsu : ciel sombre / lune claire
    L, n = measure.label(masque, return_num=True)           # objets connexes
    if n == 0:
        raise ValueError("Aucun objet lumineux détecté dans l'image")
    aires = [p.area for p in measure.regionprops(L)]
    masque = (L == np.argmax(aires) + 1)                    # on garde le plus grand objet
    return ndimage.binary_fill_holes(masque)                # remplit les zones sombres internes


# ----------------------------------------------------------------------------
# Étape 3 : disque complet, fraction éclairée, âge lunaire
# ----------------------------------------------------------------------------
def ajuster_cercle(x, y):
    # Moindres carrés (méthode de Kåsa) : x² + y² = 2·a·x + 2·b·y + c  -> équation linéaire en (a, b, c)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    second_membre = x**2 + y**2
    (a, b, c), *_ = np.linalg.lstsq(A, second_membre, rcond=None)
    return a, b, np.sqrt(c + a**2 + b**2)                   # centre (a, b) et rayon


def estimer_disque(masque):
    # Enveloppe convexe de la zone éclairée (pour un croissant : le limbe + la corde entre les 2 pointes)
    enveloppe = morphology.convex_hull_image(masque)
    bord_env = segmentation.find_boundaries(enveloppe, mode='inner')

    # Estimation 1 : cercle dont le diamètre est la plus grande corde (= distance entre les 2 pointes)
    pts = np.argwhere(bord_env)[:, ::-1].astype(float)      # coordonnées (x, y)
    D = cdist(pts, pts)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    centre, R = (pts[i] + pts[j]) / 2, D[i, j] / 2
    if masque.sum() / (np.pi * R**2) >= 0.5:                # gibbeuse ou pleine : l'estimation 1 suffit
        return centre[0], centre[1], R

    # Estimation 2 (croissant) : cercle ajusté sur le limbe = points du contour qui touchent l'enveloppe
    contour = segmentation.find_boundaries(masque, mode='inner')
    proche = ndimage.binary_dilation(bord_env, iterations=2)
    y, x = np.nonzero(contour & proche)
    x, y = x.astype(float), y.astype(float)
    cx, cy, R = ajuster_cercle(x, y)
    for _ in range(3):                                      # on écarte les points loin du cercle, puis on refait
        ecart = np.abs(np.hypot(x - cx, y - cy) - R)
        garde = ecart < max(2.0, 2 * np.median(ecart))
        if garde.sum() < 10:
            break
        cx, cy, R = ajuster_cercle(x[garde], y[garde])
    return cx, cy, R


def age_lunaire(f, croissante):
    # f = (1 - cos(phi)) / 2  avec  phi = 2*pi * age / PERIODE
    phi = np.arccos(1 - 2 * f)
    age = phi * PERIODE / (2 * np.pi)
    return age if croissante else PERIODE - age


def afficher(J, masque, cx, cy, R, f, age):
    jour = int(age) + 1                                     # jour 1 = nouvelle lune
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(J, cmap='gray', vmin=0, vmax=1);   ax[0].set_title("Image filtrée")
    ax[1].imshow(masque, cmap='gray');              ax[1].set_title("Partie éclairée extraite")
    ax[2].imshow(J, cmap='gray', vmin=0, vmax=1)
    ax[2].add_patch(plt.Circle((cx, cy), R, fill=False, color='r', lw=2))
    ax[2].set_title("Disque complet estimé (rouge)")
    for a in ax:
        a.axis('off')
    fig.suptitle(f"Jour lunaire : {jour}   (âge ≈ {age:.1f} j, {100 * f:.1f} % du disque éclairé)", fontsize=14)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main(chemin, croissante):
    I = charger(chemin)                                     # étape 1
    J = filtrer(I)

    masque = segmenter(J)                                   # étape 2

    cx, cy, R = estimer_disque(masque)                      # étape 3
    f = min(1.0, masque.sum() / (np.pi * R**2))
    age = age_lunaire(f, croissante)

    print(f"Fraction éclairée : {100 * f:.1f} %")
    print(f"Âge de la lune    : {age:.1f} jours")
    print(f"Jour lunaire      : {int(age) + 1}")
    afficher(J, masque, cx, cy, R, f, age)


if __name__ == "__main__":
    chemin = sys.argv[1] if len(sys.argv) > 1 else CHEMIN
    croissante = (sys.argv[2].lower() != "decroissante") if len(sys.argv) > 2 else CROISSANTE
    main(chemin, croissante)