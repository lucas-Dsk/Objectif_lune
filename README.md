# Objectif_lune
# Projet : Estimation de la phase et de l'âge de la Lune à partir d'une photographie
## Présentation du projet

Ce projet a pour objectif de développer un programme Python capable
d'analyser une photographie de la Lune afin d'estimer sa phase et son
âge lunaire.

Le programme s'appuie principalement sur des techniques classiques de
traitement d'image et de traitement du signal : filtrage, segmentation,
opérations morphologiques, détection de contours, ajustement
géométrique et analyse fréquentielle (FFT).

L'idée est de détecter la partie éclairée de la Lune sur une image, de
reconstruire virtuellement le disque lunaire complet, puis d'utiliser la
surface éclairée et la forme du terminateur pour estimer la phase. Le
programme doit également distinguer une Lune croissante d'une Lune
décroissante, car une même fraction éclairée peut correspondre aux deux
situations.

## Objectifs

Le projet est organisé autour des étapes suivantes :

1.  **Prétraitement de l'image**
    -   Conversion en niveaux de gris ou utilisation du canal V de
        l'espace HSV.
    -   Évaluation de la netteté de l'image à l'aide de l'analyse fréquentielle (FFT) pour prévenir les erreurs liées au flou.
    -   Réduction du bruit avec un filtre gaussien ou médian.
    -   Estimation et correction du fond du ciel.
    -   Prise en compte de la surexposition et du halo autour de la
        Lune.
        
2.  **Segmentation de la partie éclairée**
    -   Utilisation du seuillage d'Otsu comme méthode de base.
    -   Comparaison éventuelle avec un seuillage adaptatif.
    -   Nettoyage du masque avec des opérations morphologiques.
    -   Conservation de la composante correspondant à la Lune.
      
3.  **Reconstruction du disque lunaire**
    -   Détection du bord extérieur de la Lune.
    -   Ajustement d'un cercle à partir du limbe extérieur.
    -   Distinction entre le limbe extérieur et le terminateur, qui
        sépare la partie éclairée de la partie sombre.
        
4.  **Estimation de la phase**
    -   Calcul de la fraction illuminée : aire éclairée divisée par aire
        du disque lunaire complet.
    -   Analyse de la forme du terminateur afin d'affiner l'estimation
        de la phase.
        
5.  **Distinction entre Lune croissante et décroissante**
    -   Analyse du côté éclairé de la Lune.
    -   Prise en compte de l'orientation de la photographie et de
        l'hémisphère.
    -   Utilisation éventuelle des métadonnées EXIF comme référence de
        validation, et non comme méthode principale de prédiction.
        
6.  **Estimation de l'âge lunaire**
    -   Conversion de la phase estimée en âge exprimé en jours.
    -   Utilisation d'un mois synodique moyen d'environ 29,53 jours.
    -   Possibilité de commencer par une classification en huit phases
        lunaires avant de chercher une estimation plus précise.

## Lancement du programme

### Prérequis
-   Python 3 installé sur la machine.
-   Les bibliothèques Python nécessaires au projet installées (`numpy`, `scipy`, `matplotlib`, `scikit-image`, `pandas`, `streamlit`, `Pillow`).
-   Une photographie de la Lune accessible depuis le terminal.

Si le projet contient un fichier `requirements.txt`, installez les
dépendances avec :

``` bash
python -m pip install -r requirements.txt
```

Si aucun fichier `requirements.txt` n'est fourni, installez les
bibliothèques utilisées directement par le script, selon ses imports.


### Exécution en ligne de commande (Base)



Placez-vous dans le dossier qui contient le fichier `lune.py`, puis

exécutez la commande suivante :



``` bash

python lune.py images/nomDeLimage

```



Remplacez `images/nomDeLimage` par le chemin vers la photographie à

analyser.



**Exemple :**



``` bash

python lune.py images/lune.jpg

```



Le chemin peut être relatif au dossier depuis lequel la commande est

exécutée. Sous Windows, vous pouvez également utiliser un chemin absolu,

par exemple :



``` bash

python lune.py "C:/Users/Utilisateur/Images/lune.jpg"

```



### Exécution via l'Interface Web (Démonstration)



Pour une utilisation ergonomique avec interface graphique, lancez l'application Streamlit intégrée au projet :



```bash

streamlit run app.py

```

Cela ouvrira une page web interactive dans votre navigateur pour uploader et analyser visuellement les photos.



### Exécution du traitement par lots (Validation)



Pour analyser automatiquement tout un dossier d'images et exporter les résultats :



```bash

python evaluer_dossier.py

```

Ce script génère un tableau récapitulatif, un fichier d'export `resultats_lune.csv` et un graphique global `evolution_lune.png`.



### Vérifier que Python est installé



Vous pouvez vérifier l'installation de Python avec :



``` bash

python --version

```



Sur certaines installations Windows, la commande suivante peut être

nécessaire :



``` bash

py --version

```



Dans ce cas, lancez le programme avec :



``` bash

py lune.py images/nomDeLimage

```



## Méthode de calcul de l'âge lunaire



Le mois synodique moyen dure environ 29,53 jours. La fraction illuminée

`f` peut être reliée à l'angle de phase `φ` par la relation :



``` text

f = (1 - cos(φ)) / 2

```



On peut donc estimer l'angle de phase avec :



``` text

φ = arccos(1 - 2f)

```

Puis calculer un âge lunaire approximatif :

``` text
âge = φ × 29,53 / (2π)
```
Cette dernière formule donne une valeur correspondant à la moitié
croissante du cycle. Pour une Lune décroissante, l'âge estimé est :
``` text
âge = 29,53 - âge_croissant
```
La fraction illuminée ne permet donc pas, à elle seule, de savoir si la
Lune est croissante ou décroissante. Il faut également déterminer le
côté éclairé ou disposer d'une autre information permettant de lever
cette ambiguïté.

## Validation des résultats

Pour évaluer la qualité du programme, il est nécessaire de constituer un
ensemble de photographies dont la date et l'heure de prise de vue sont
connues. (Un script automatisé `evaluer_dossier.py` a été développé à cet effet pour faciliter la validation à grande échelle).

Les estimations du programme pourront ensuite être comparées à des
données astronomiques de référence. L'erreur moyenne, exprimée en jours,
permettra d'évaluer la précision de la méthode.

Il sera également intéressant d'identifier les situations dans
lesquelles le programme fonctionne moins bien :

-   Croissant lunaire très fin.
-   Lune surexposée ou entourée d'un halo important.
-   Présence de nuages ou de pollution lumineuse.
-   Image floue ou bruitée (partiellement géré par l'analyse FFT).
-   Orientation inconnue de la photographie.

## Améliorations possibles

Plusieurs évolutions pourront être envisagées après la réalisation d'une
première version fonctionnelle :

-   Détecter la Lune en présence de nuages ou de sources lumineuses
    parasites.

-   Améliorer la reconstruction du disque lunaire lorsque seule une
    petite partie est éclairée.

-   Comparer les méthodes classiques de traitement d'image avec un
    modèle d'apprentissage automatique, par exemple un petit réseau de
    neurones convolutif (CNN).
    
-   Prendre en compte la libration et l'inclinaison apparente de l'axe
    lunaire.


## Remarque


La méthode recherchée repose avant tout sur l'analyse de la
photographie. Les métadonnées de date et d'heure peuvent servir à
vérifier les résultats, mais ne doivent pas remplacer l'analyse d'image
si l'objectif est bien d'estimer la phase lunaire à partir de la
photographie.
