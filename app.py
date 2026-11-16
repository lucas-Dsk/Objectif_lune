import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal, ndimage
from scipy.spatial.distance import cdist
from skimage import color, filters, measure, morphology, segmentation
import math

# --- Fonctions de traitement de l'image (issues de lune.py) ---

PERIODE = 29.53  # durée du mois lunaire (jours)

@st.cache_data
def charger_image_pil(image_file):
    '''Charge une image via PIL depuis un uploader Streamlit'''
    img = Image.open(image_file)
    I = np.array(img)
    
    if I.dtype == np.uint8:
        I = I.astype(float) / 255
    if I.ndim == 3:
        I = color.rgb2gray(I[:, :, :3])
    
    # Redimensionnement pour performances
    k = max(1, max(I.shape) // 600)
    return I[::k, ::k]

def evaluer_nettete_fft(image):
    '''Évaluation de la netteté via la transformée de Fourier rapide'''
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    h, w = magnitude_spectrum.shape
    haute_freq = magnitude_spectrum[0:h//4, :] 
    
    return np.mean(haute_freq)

def filtrer(I):
    '''Filtre médian pour lisser le bruit tout en préservant les contours'''
    return signal.medfilt2d(I, kernel_size=3)

def segmenter(J):
    '''Segmentation par seuillage d'Otsu et opérations morphologiques'''
    seuil = filters.threshold_otsu(J)
    masque = J > seuil
    L, n = measure.label(masque, return_num=True)
    if n == 0:
        return None, "Aucun objet lumineux détecté dans l'image"
    
    aires = [p.area for p in measure.regionprops(L)]
    masque = (L == np.argmax(aires) + 1)
    return ndimage.binary_fill_holes(masque), None

def ajuster_cercle(x, y):
    '''Ajustement géométrique par la méthode des moindres carrés de Kåsa'''
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    second_membre = x**2 + y**2
    result = np.linalg.lstsq(A, second_membre, rcond=None)
    a, b, c = result[0]
    return a, b, np.sqrt(c + a**2 + b**2)

def estimer_disque(masque):
    '''Reconstruction du disque lunaire complet à partir de la portion visible'''
    enveloppe = morphology.convex_hull_image(masque)
    bord_env = segmentation.find_boundaries(enveloppe, mode='inner')

    pts = np.argwhere(bord_env)[:, ::-1].astype(float)
    D = cdist(pts, pts)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    centre, R = (pts[i] + pts[j]) / 2, D[i, j] / 2
    
    # Si la lune est gibbeuse ou pleine, l'estimation simple suffit
    if masque.sum() / (np.pi * R**2) >= 0.5:
        return centre[0], centre[1], R

    contour = segmentation.find_boundaries(masque, mode='inner')
    proche = ndimage.binary_dilation(bord_env, iterations=2)
    y, x = np.nonzero(contour & proche)
    x, y = x.astype(float), y.astype(float)
    
    if len(x) < 3: 
        return centre[0], centre[1], R
        
    cx, cy, R = ajuster_cercle(x, y)
    
    # Élimination itérative des points aberrants
    for _ in range(3):
        ecart = np.abs(np.hypot(x - cx, y - cy) - R)
        garde = ecart < max(2.0, 2 * np.median(ecart))
        if garde.sum() < 10:
            break
        cx, cy, R = ajuster_cercle(x[garde], y[garde])
    return cx, cy, R

def cote_eclaire(masque, cx):
    '''Déduction automatique de la phase (croissante/décroissante)'''
    _, x = np.nonzero(masque)
    return x.mean() > cx

def age_lunaire(f, croissante):
    '''Calcul mathématique de l'âge lunaire à partir de la fraction éclairée'''
    f_clamped = max(0.0, min(1.0, f))
    phi = np.arccos(1 - 2 * f_clamped)
    age = phi * PERIODE / (2 * np.pi)
    return age if croissante else PERIODE - age

def generer_figure_resultat(J, masque, cx, cy, R):
    '''Génère le graphique Matplotlib à afficher dans Streamlit'''
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(J, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title("Image filtrée", color='white')
    
    ax[1].imshow(masque, cmap='gray')
    ax[1].set_title("Partie éclairée", color='white')
    
    ax[2].imshow(J, cmap='gray', vmin=0, vmax=1)
    cercle = plt.Circle((cx, cy), R, fill=False, color='red', lw=2)
    ax[2].add_patch(cercle)
    ax[2].set_title("Reconstruction géométrique", color='white')
    
    for a in ax:
        a.axis('off')
        
    fig.patch.set_facecolor('#0e1117') 
    fig.tight_layout()
    return fig


# --- Interface Utilisateur Streamlit ---

st.set_page_config(page_title="Objectif Lune", layout="wide")

# Masquer le menu Streamlit, le bouton Deploy et le footer pour un aspect logiciel sur mesure
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Projet : Objectif Lune")
st.markdown("""
**Estimation de la phase et de l'âge de la Lune à partir d'une photographie.**  
*Méthodes appliquées : Traitement du signal (FFT, Filtres) et analyse d'images (Seuillage d'Otsu, méthode de Kåsa).*
""")

# Barre latérale pour les paramètres
with st.sidebar:
    st.header("Configuration")
    mode_croissance = st.radio(
        "Détection du sens lunaire :",
        ["Automatique", "Forcer : Croissante", "Forcer : Décroissante"]
    )
    st.markdown("---")
    st.markdown("*Note : Le programme analyse la courbure du limbe pour reconstruire virtuellement le disque lunaire complet et en déduire l'angle de phase.*")

# Zone de chargement d'image
uploaded_file = st.file_uploader("Importer une photographie de la Lune (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Analyse en cours..."):
        I = charger_image_pil(uploaded_file)
        
        # 2. Qualité de l'image (FFT)
        score_nettete = evaluer_nettete_fft(I)
        
        if score_nettete < 10.0:
            st.warning(f"Analyse FFT : Score de netteté faible ({score_nettete:.1f}). L'image est potentiellement floue.")
        else:
            st.info(f"Analyse FFT : Image nette (Score = {score_nettete:.1f})")
            
        # 3. Traitement d'image
        J = filtrer(I)
        masque, erreur = segmenter(J)
        
        if erreur:
            st.error(f"Erreur de segmentation : {erreur}")
        else:
            # 4. Mathématiques
            cx, cy, R = estimer_disque(masque)
            f = min(1.0, masque.sum() / (np.pi * R**2))
            
            if mode_croissance == "Automatique":
                croissante = cote_eclaire(masque, cx)
            elif mode_croissance == "Forcer : Croissante":
                croissante = True
            else:
                croissante = False
                
            age = age_lunaire(f, croissante)
            jour = int(age) + 1
            
            st.markdown("---")
            st.markdown("### Résultats de l'analyse")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Jour Lunaire", f"{jour}")
            col2.metric("Âge Estimé", f"{age:.1f} jours")
            col3.metric("Illumination", f"{f * 100:.1f} %")
            col4.metric("Phase", "Croissante" if croissante else "Décroissante")

            st.markdown("### Reconstruction géométrique")
            fig = generer_figure_resultat(J, masque, cx, cy, R)
            st.pyplot(fig)