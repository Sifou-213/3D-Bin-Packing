# Colis · plan de chargement en 3D

![Colis](static/img/favicon.svg)

Choisissez un camion, ajoutez vos colis, et Colis vous montre comment tout ranger, colis par colis, en 3D.

## Ce que l'on peut faire

- **Un camion** : remplir un seul camion au mieux.
- **Plusieurs camions** : répartir une grosse commande sur le moins de camions possible.
- **Une palette** : empiler des colis sur une palette (Europe, standard, US ou sur mesure).

Pour chaque colis on indique la quantité, la taille (en cm) et le poids, et on peut cocher **Toujours debout** ou **Fragile : rien dessus**. Les produits peuvent aussi être choisis dans la liste (`static/Data/Items data.json`), et les camions viennent de `static/Data/camions.json`.

Le plan s'affiche en 3D. On peut le tourner (glisser) et zoomer (molette ou pincer), et survoler un colis pour voir son nom. On peut aussi le revoir se construire avec **↻ Rejouer**, couche par couche, du fond du camion vers la porte. Après le calcul, les colis sont poussés les uns contre les autres pour qu'il ne reste pas de vide entre eux (`compaction.py`).

**Colis par colis** : la barre sous le plan (⏮ ◀ ▶ Lecture ▶ ⏭ et le curseur) ou le clavier (`←` `→`, `Début`, `Fin`, `Espace`) permettent d'avancer et de reculer d'un colis à la fois. Le colis en cours est entouré de blanc, et une phrase dit ce que c'est et où le poser. L'ordre respecte l'appui : un colis vient toujours après ceux sur lesquels il repose.

**↓ Rapport Excel** crée un rapport dans le même thème sombre (`report.py`), avec trois onglets :
- **Plan** : l'image de chaque camion et le tableau des colis ;
- **Ordre de chargement** : chaque colis dans l'ordre où il faut le poser, avec sa position ;
- **Paramètres**.

Une copie est aussi gardée dans `shipment_results_archive/<date>/Rapports/`.

## Lancer l'application

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Puis ouvrir http://localhost:5000.

Pour mettre le projet sur GitHub et obtenir un lien public, voir [DEPLOIEMENT.md](DEPLOIEMENT.md).

## Organisation du code

```
app.py                      Serveur Flask : pages, calcul, export Excel
hga.py                      Algorithme génétique hybride (chargement par couches)
multi_container_solver.py   Plusieurs camions : remplit un camion après l'autre
compaction.py               Pousse les colis vers le fond pour supprimer les vides
report.py                   Rapport Excel (thème sombre)
pallet_loading.py           Ancien algorithme de palette (gardé pour l'API /solve_pallet_loading)
templates/                  landing.html, workspace.html, _logo.html
static/css/app.css          Styles (thème sombre du plan de chargement)
static/js/app.js            Interface
static/js/viewer.js         Vue 3D interactive (Three.js)
static/js/plan.js           Ordre de chargement (partagé)
static/js/landing.js        Plan de démonstration de la page d'accueil
tools/make_demo_plan.py     Régénère static/Data/demo_plan.json
render.yaml                 Mise en ligne sur Render
static/Data/                Camions, produits, exemples
```
