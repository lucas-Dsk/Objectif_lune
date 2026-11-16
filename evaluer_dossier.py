import os
import glob
import numpy as np
from lune import charger, evaluer_nettete_fft, filtrer, segmenter, estimer_disque, cote_eclaire, age_lunaire

def evaluer_dossier(dossier="images"):
    fichiers = glob.glob(os.path.join(dossier, "*.*"))

    print(f"{'Fichier':<20} | {'Netteté':<8} | {'Sens':<12} | {'Éclairé (%)':<11} | {'Âge (j)'}")
    print("-" * 70)

    for chemin in fichiers:
        nom = os.path.basename(chemin)
        try:
            I = charger(chemin)
            score_nettete = evaluer_nettete_fft(I)

            J = filtrer(I)
            masque = segmenter(J)

            cx, cy, R = estimer_disque(masque)
            f = min(1.0, masque.sum() / (np.pi * R**2))
            croissante = cote_eclaire(masque, cx)
            age = age_lunaire(f, croissante)

            sens_str = "Croissant" if croissante else "Décroissant"
            print(f"{nom:<20} | {score_nettete:<8.1f} | {sens_str:<12} | {100*f:<11.1f} | {age:.1f}")

        except Exception as e:
            print(f"{nom:<20} | ERREUR : {e}")

if __name__ == "__main__":
    evaluer_dossier()