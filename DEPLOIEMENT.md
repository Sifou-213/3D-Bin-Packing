# Mettre Colis sur GitHub et en ligne

## 1. Envoyer le projet sur GitHub

Dépôt : https://github.com/Sifou-213/3D-Bin-Packing (branche `main`). Les commandes ci-dessous remplacent l'ancienne version par celle-ci, en gardant l'historique.

**Installer Git** (une seule fois), puis fermer et rouvrir le terminal :

```powershell
winget install --id Git.Git -e
```

**Se présenter à Git** (une seule fois) :

```powershell
git config --global user.name "Sifou-213"
git config --global user.email "benabidasifeddine@gmail.com"
```

**Dans le dossier du projet** :

```powershell
cd "D:\Users\BONJOUR\Downloads\3d bin packing solver"
git init -b main
git remote add origin https://github.com/Sifou-213/3D-Bin-Packing.git
git fetch origin
git reset origin/main
git add -A
git commit -m "Nouvelle version : Colis, plan de chargement en 3D"
git push -u origin main
```

`git reset origin/main` met le projet par-dessus l'ancienne version sans toucher à vos fichiers, et `git add -A` enregistre aussi la suppression des anciens fichiers (`__pycache__`, anciens scripts). À la première commande `git push`, une fenêtre GitHub s'ouvre pour vous connecter.

Pour les mises à jour suivantes :

```powershell
git add -A
git commit -m "Ce qui a changé"
git push
```

Sans Git, il est aussi possible de passer par le site GitHub : **Add file → Upload files**, glisser le contenu du dossier, puis supprimer à la main les anciens fichiers qui ne servent plus (dont `__pycache__`). Ne pas envoyer `shipment_results_archive` ni `.claude`.

## 2. Obtenir un lien pour tout le monde (Render)

GitHub Pages ne sait pas faire tourner du Python, il faut donc un hébergeur. Render est gratuit et se met à jour tout seul à chaque `git push`.

1. Créer un compte sur https://render.com avec **Sign in with GitHub**.
2. **New → Blueprint**, choisir le dépôt `3D-Bin-Packing`. Render lit le fichier `render.yaml` du projet.
3. Cliquer **Apply** et attendre la fin de l'installation (quelques minutes).
4. Le lien apparaît en haut de la page du service, par exemple `https://colis.onrender.com` (le nom exact peut varier). C'est ce lien que vous partagez.

À savoir avec l'offre gratuite :
- le site s'endort après 15 minutes sans visite, et la première visite suivante prend environ une minute ;
- la machine est petite, donc le calcul est plus lent qu'en local. Choisir « Rapide » aide ; l'offre payante « Starter » est nettement plus rapide ;
- les rapports Excel gardés sur le serveur sont effacés à chaque redémarrage. Ce n'est pas grave : chaque utilisateur télécharge le sien.

## 3. Travailler en local

```powershell
pip install -r requirements.txt
python app.py
```

Puis ouvrir http://localhost:5000.
