# Rapport de projet : habitudes de consommation (Online Retail, UCI)

**Cours :** Apprentissage machine appliqué. Modules 3 (Sélection des variables), 4 (Introduction au clustering) et 5 (Techniques avancées de clustering)
**Fichiers livrés :**
- `projet_online_retail.py` : le code complet, découpé en cellules `# %%` (on peut l'exécuter dans VS Code ou Jupyter)
- `projet_online_retail.ipynb` : le même code sous forme de notebook Jupyter
- `figures/` : tous les graphiques (`01_…png` à `12_…png`) et les tableaux de résultats (`.csv`)

---

## 1. Objectif du projet

Le fichier « objectif du projet » demande d'analyser les habitudes de consommation des clients d'un site de vente en ligne britannique. Il y a trois tâches :

| Tâche demandée | Ce qui est fait dans le projet | Module |
|---|---|---|
| Clustering des clients (segmentation RFM) | Partie A : K-Means, hiérarchique, DBSCAN, HDBSCAN, Spectral | 4 et 5 |
| Classification de la probabilité de réachat | Partie B : prédire si un client rachètera dans les 3 prochains mois | 3 |
| Ensembles pour mieux prédire les comportements futurs | Random Forest et Gradient Boosting, optimisés par validation croisée imbriquée | 3 |

Les cinq étapes attendues se retrouvent dans le code : (1) nettoyage, (2) comparaison de modèles de classification, (3) clustering, (4) méthodes d'ensembles, (5) visualisations et rapport.

---

## 2. Respect du cadre pédagogique

Le code n'utilise que les notions, bibliothèques et fonctions présentées dans les modules 3, 4 et 5 :

| Notion du cours | Fonction utilisée | Section du code |
|---|---|---|
| Normalisation / standardisation (M4) | `StandardScaler` | §4, §15, §17 |
| Suppression du bruit / valeurs aberrantes (M4) | transformation `np.log1p` | §4 |
| K-Means + méthode du coude (M4) | `KMeans`, `inertia_` | §5 |
| Silhouette, Davies-Bouldin (M4) | `silhouette_score`, `davies_bouldin_score` | §5, §10 |
| Hiérarchique agglomératif + dendrogramme (M4) | `AgglomerativeClustering`, `linkage`, `dendrogram` | §6 |
| Hiérarchique divisif (M4) | `BisectingKMeans` | §6 |
| DBSCAN, choix de eps par la courbe k-distance, minPts ≈ 2×dimension (M4, M5) | `DBSCAN`, `NearestNeighbors` | §7 |
| HDBSCAN (M5) | `HDBSCAN` | §8 |
| Clustering spectral (M5) | `SpectralClustering` | §9 |
| Calinski-Harabasz, Dunn, Stabilité (M5) | `calinski_harabasz_score`, formule de Dunn, `adjusted_rand_score` | §10 |
| Visualisation PCA et t-SNE (M5) | `PCA`, `TSNE` | §11 |
| Filtres : VarianceThreshold, corrélation, ANOVA, information mutuelle (M3) | `VarianceThreshold`, `.corr()`, `f_classif`, `mutual_info_classif` | §14 |
| Enveloppantes : RFE, RFECV (M3) | `RFE`, `RFECV` | §15 |
| Intégrées : Lasso, importances d'arbres (M3) | `Lasso`, `feature_importances_` | §16 |
| Pipeline scaling → modèle (M3) | `Pipeline` | §17 |
| Validation croisée KFold / Stratified / Repeated (M3) | `StratifiedKFold`, `RepeatedStratifiedKFold` | §17 |
| Moyenne ± écart-type, boxplots (M3) | `cross_val_score`, `plt.boxplot` | §17 |
| Test statistique de Wilcoxon (M3) | `scipy.stats.wilcoxon` | §18 |
| CV imbriquée avec GridSearchCV (M3) | `GridSearchCV` + `cross_val_score` | §19 |
| Précision, rappel, F1 (M3) | `classification_report`, `confusion_matrix` | §20 |

Le style du code reste simple : une suite d'étapes numérotées, des listes Python et des tableaux pandas. Il n'y a que deux petites fonctions (indice de Dunn et stabilité), qui traduisent les formules du Module 5 parce que scikit-learn ne les fournit pas. Les seuls dictionnaires sont les grilles d'hyperparamètres que `GridSearchCV` exige.

---

## 3. Chargement et nettoyage des données (§1 et §2)

Le jeu brut compte **541 909 lignes**, une par produit acheté dans une facture. Il couvre la période du 01/12/2010 au 09/12/2011.

| Étape de nettoyage | Pourquoi |
|---|---|
| Supprimer les lignes sans `CustomerID` (135 080 lignes) | On ne peut pas segmenter un client inconnu. |
| Supprimer les doublons exacts | Ils fausseraient les montants et les fréquences. |
| Mettre de côté les factures commençant par « C » | Ce sont des annulations. On les réutilise comme variable (`NbAnnulations`) dans la Partie B. |
| Garder seulement `Quantity > 0` et `UnitPrice > 0` | Les valeurs nulles ou négatives sont des erreurs ou des ajustements. |
| Créer `TotalPrice = Quantity × UnitPrice` | C'est le montant de chaque ligne. |

**Résultat :** 392 692 lignes propres et **4 338 clients**.

---

## 4. Partie A : segmentation RFM par clustering (Modules 4 et 5)

### 4.1 Variables RFM (§3)
Pour chaque client, on calcule trois variables :
- **Récence** : nombre de jours depuis le dernier achat. La date de référence est le lendemain de la dernière facture.
- **Fréquence** : nombre de factures différentes.
- **Montant** : somme totale dépensée.

Les distributions sont très asymétriques (figure `01_distribution_rfm.png`). Le montant médian est de 669 £, mais le maximum atteint 280 206 £.

### 4.2 Prétraitement (§4, Module 4 « Prétraitement des données »)
1. **Logarithme (`log1p`)** : il réduit le poids des très gros clients. Le cours rappelle que K-Means est sensible aux valeurs aberrantes.
2. **Standardisation** : chaque variable a une moyenne de 0 et un écart-type de 1. Sans cette étape, le montant (en milliers) dominerait la récence (en jours) dans la distance euclidienne.

### 4.3 Choix de k pour K-Means (§5, figure `02_choix_k_kmeans.png`)

| k | Inertie | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|
| 2 | 6484 | **0,433** | **0,892** | **4367** |
| 3 | 4869 | 0,337 | 1,048 | 3625 |
| **4** | **3939** | **0,338** | **1,009** | 3328 |
| 5 | 3297 | 0,316 | 0,988 | 3193 |
| 6–8 | ↓ | ≈ 0,30–0,31 | ≈ 0,98–1,02 | ↓ |

**Choix : k = 4.** Voici pourquoi :
- k = 2 obtient les meilleurs scores, mais il sépare seulement les « bons » et les « mauvais » clients, ce qui ne sert pas à grand-chose en marketing.
- La courbe d'inertie forme un coude vers k = 3–4.
- Entre 3 et 8, **k = 4 a la meilleure silhouette** et un Davies-Bouldin meilleur que k = 3.
- Le cours rappelle que l'inertie diminue toujours quand k augmente. Il faut donc la croiser avec les autres métriques et avec l'interprétation métier.

### 4.4 Méthodes comparées (§6 à §9)
- **Hiérarchique agglomératif (Ward)** : le dendrogramme est dans `03_dendrogramme.png`. On le coupe en 4 groupes.
- **Hiérarchique divisif (Bisecting K-Means)** : c'est l'approche top-down présentée au Module 4.
- **DBSCAN** : minPts = 2 × 3 dimensions = 6. eps = 0,35 est lu au coude de la courbe k-distance (`04_courbe_k_distance.png`).
- **HDBSCAN** : la version hiérarchique, moins sensible à eps (`min_cluster_size = 50`).
- **Spectral** : graphe des 15 plus proches voisins, puis K-Means sur les vecteurs propres du Laplacien.

### 4.5 Comparaison avec les métriques avancées (§10)

| Méthode | Nb clusters | Bruit | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-H. ↑ | Dunn ↑ | Stabilité (ARI) ↑ |
|---|---|---|---|---|---|---|---|
| **K-Means** | 4 | 0 | **0,338** | **1,009** | **3328** | 0,002 | 0,911 |
| Hiérarchique (Ward) | 4 | 0 | 0,280 | 1,099 | 2740 | 0,007 | 0,376 |
| Bisecting K-Means | 4 | 0 | 0,282 | 1,175 | 3095 | 0,003 | 0,884 |
| DBSCAN | 3 | 175 | 0,175 | 1,689 | 2078 | **0,069** | **0,970** |
| HDBSCAN | 6 | 701 | 0,101 | 2,695 | 907 | 0,030 | 0,922 |
| Spectral | 4 | 0 | 0,178 | 1,466 | 1634 | 0,003 | 0,921 |

*(Les métriques de DBSCAN et HDBSCAN sont calculées sans les points de bruit. La stabilité est la moyenne de l'ARI sur 10 échantillons bootstrap.)*

**Lecture des résultats :**
- **K-Means est le meilleur sur les trois métriques principales** (silhouette, Davies-Bouldin, Calinski-Harabasz). Sa stabilité est aussi très bonne (0,91) : les segments se retrouvent d'un rééchantillonnage à l'autre.
- Le **hiérarchique Ward** est **instable** (ARI 0,38). Sur un nuage de points continu, une petite modification des données change l'ordre des fusions.
- **DBSCAN et HDBSCAN** ont un indice de Dunn et une stabilité élevés. Ils écartent cependant beaucoup de clients comme « bruit » (jusqu'à 701 pour HDBSCAN), et leurs silhouettes sont faibles. Le cours le prévoit : ces méthodes **fonctionnent mal quand la densité est variable**, ce qui est le cas ici.
- L'**indice de Dunn est très faible pour toutes les méthodes**. Les clients forment un **continuum** et non des « îlots » bien séparés, donc la distance minimale entre deux clusters voisins est presque nulle. La segmentation découpe ce continuum en zones utiles.
- La projection PCA (`05_comparaison_pca.png`) le confirme : K-Means découpe le nuage en 4 zones nettes, alors que DBSCAN et HDBSCAN produisent des groupes qui se chevauchent dans la projection. La figure `06_kmeans_pca_tsne.png` compare la PCA et le t-SNE pour K-Means.

**Méthode retenue : K-Means avec k = 4.** Elle a les meilleurs scores internes, elle est stable, et c'est l'usage recommandé par le cours pour la « segmentation clients ».

### 4.6 Interprétation métier des segments (§12, figure `07_profil_segments.png`)

| Segment | Récence moy. (jours) | Fréquence moy. | Montant moy. (£) | Nb clients | Interprétation |
|---|---|---|---|---|---|
| 0 | 12 | 13,8 | 8 088 | 713 | **Clients champions** : récents, très fréquents, gros acheteurs. À fidéliser (programme VIP). |
| 1 | 182 | 1,3 | 341 | 1 622 | **Clients perdus / inactifs** : un seul achat il y a environ 6 mois. Campagne de reconquête, ou pas d'investissement. |
| 2 | 18 | 2,2 | 557 | 837 | **Nouveaux clients / clients récents** : potentiel à développer (offre de bienvenue, 2e achat). |
| 3 | 72 | 4,1 | 1 802 | 1 166 | **Clients réguliers à risque** : bons clients qui n'ont pas acheté depuis plus de 2 mois. À relancer en priorité. |

> Remarque : les numéros de segments (0, 1, 2, 3) peuvent changer d'une version de scikit-learn à l'autre. Les profils, eux, restent les mêmes.

---

## 5. Partie B : classification de la probabilité de réachat (Module 3)

### 5.1 Construction de la cible (§13)
Pour éviter toute fuite d'information du futur, on coupe l'historique au **9 septembre 2011** :
- **Période d'observation** (avant la coupure) : on y calcule les variables du client.
- **Période de prédiction** (les 3 derniers mois) : **Reachat = 1** si le client a acheté au moins une fois pendant cette période.

On obtient **3 362 clients**. Les classes sont assez équilibrées : **57,1 %** de réachat et 42,9 % de non-réachat.

On calcule 10 variables candidates : `Recence`, `Anciennete`, `Frequence`, `Montant`, `PanierMoyen`, `QuantiteTotale`, `NbProduits`, `PrixMoyen`, `NbAnnulations`, `RoyaumeUni`.

### 5.2 Sélection des variables (§14 à §16)

| Famille (M3) | Méthode | Résultat |
|---|---|---|
| Filtre | VarianceThreshold (seuil 0) | Aucune variable n'est constante : les 10 sont conservées. |
| Filtre | Corrélation (`08_correlation.png`) | `Montant` et `QuantiteTotale` sont presque redondantes (r = 0,92). `Frequence` est fortement liée à `NbProduits` (0,70) et à `NbAnnulations` (0,69). |
| Filtre | ANOVA (`f_classif`) | Les plus significatives sont `Recence`, `NbProduits`, `Frequence`, `NbAnnulations` et `Anciennete`. `RoyaumeUni` (p = 0,76) et `PanierMoyen` (p = 0,89) ne sont **pas significatives**. |
| Filtre | Information mutuelle | En tête : `Frequence`, `QuantiteTotale`, `Montant`, `NbProduits`. `RoyaumeUni` vaut 0. |
| Enveloppante | RFE (régression logistique, 5 variables) | `Recence`, `Frequence`, `Montant`, `PanierMoyen`, `NbProduits` |
| Enveloppante | **RFECV** (`09_rfecv.png`) | **7 variables optimales** : `Recence`, `Frequence`, `Montant`, `PanierMoyen`, `QuantiteTotale`, `NbProduits`, `NbAnnulations` |
| Intégrée | Lasso (α = 0,01) | Garde `Recence` (coefficient le plus fort, négatif), `Anciennete`, `NbProduits`, `PrixMoyen`, `NbAnnulations` |
| Intégrée | Importances Random Forest (`10_importances_rf.png`) | Classement des variables selon les splits des arbres |

**Points communs entre les méthodes :** la **Récence** est la variable la plus importante (plus le dernier achat est ancien, moins le client rachète). Viennent ensuite la fréquence et la diversité des produits. Le pays (`RoyaumeUni`) n'apporte rien.

**Choix final :** les **7 variables de RFECV**. C'est la méthode qui optimise directement la performance (F1) en validation croisée.

### 5.3 Comparaison des modèles (§17, figure `11_boxplot_modeles.png`)
Chaque modèle est placé dans un pipeline `StandardScaler → modèle`. La validation est **stratifiée, à 5 plis répétés 3 fois**, soit 15 scores par modèle.

| Modèle | Précision | Rappel | F1 moyen | F1 écart-type |
|---|---|---|---|---|
| **Régression logistique** | **0,729** | 0,702 | **0,715** | 0,015 |
| Arbre de décision | 0,708 | 0,707 | 0,705 | 0,021 |
| Random Forest | 0,694 | 0,698 | 0,696 | 0,014 |
| Gradient Boosting | 0,708 | 0,706 | 0,707 | 0,016 |

### 5.4 Tests statistiques (§18)
Test de **Wilcoxon apparié** entre le meilleur modèle (régression logistique) et chacun des autres :
- contre l'arbre de décision : p = 0,28, **pas de différence significative** ;
- contre Random Forest : p = 0,0002, **différence significative** (la régression logistique est meilleure) ;
- contre Gradient Boosting : p = 0,11, **pas de différence significative**.

**Impact de la sélection :** avec les 10 variables, F1 = 0,715 ± 0,013. Avec les 7 variables retenues, F1 = 0,715 ± 0,015. Le test de Wilcoxon donne p = 0,56 : **pas de perte de performance**. On a donc un modèle plus simple et plus interprétable pour la même qualité, comme dans l'exemple du cours (30 → 17 variables, p = 0,465).

### 5.5 Méthodes d'ensembles optimisées : CV imbriquée (§19)
- **Boucle interne** (3 plis) : `GridSearchCV` cherche les meilleurs hyperparamètres (nombre d'arbres, profondeur, taux d'apprentissage…).
- **Boucle externe** (5 plis) : mesure la performance réelle du modèle optimisé, sans biais.

| Ensemble optimisé | F1 (CV imbriquée) |
|---|---|
| Random Forest | 0,709 ± 0,018 |
| **Gradient Boosting** | **0,716 ± 0,019** |

L'optimisation améliore les deux ensembles : Random Forest passe de 0,696 à 0,709 et Gradient Boosting de 0,707 à 0,716. Le Gradient Boosting optimisé rejoint le niveau de la régression logistique.

### 5.6 Modèle final et évaluation sur un jeu de test (§20)
**Gradient Boosting** (learning_rate = 0,1, max_depth = 3, n_estimators = 100), entraîné sur 80 % des clients et testé sur les 20 % restants (673 clients jamais vus) :

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| Pas de réachat | 0,59 | 0,65 | 0,62 |
| Réachat | 0,72 | 0,66 | 0,69 |
| **Exactitude globale** | | | **0,66** |

La matrice de confusion est dans `12_matrice_confusion.png`. Le score F1 sur ce jeu de test (0,69) est un peu plus bas que celui de la validation croisée (0,716). C'est normal : un seul découpage train/test varie plus qu'une moyenne sur plusieurs plis.

Le programme affiche aussi la **probabilité de réachat** de chaque client (`predict_proba`). Les clients les plus probables (probabilité > 0,97) ont acheté très récemment, très souvent et beaucoup de produits différents. Les moins probables (< 0,2) n'ont fait qu'un achat il y a plus de 150 jours.

---

## 6. Conclusion

1. **Segmentation :** K-Means sur les variables RFM (log + standardisation) avec k = 4 donne 4 segments **stables** et **interprétables** : champions, clients récents, réguliers à risque, perdus. C'est la méthode la plus performante parmi les 6 méthodes comparées.
2. **Réachat :** la **récence** est le facteur clé. Un modèle simple (régression logistique) et un Gradient Boosting optimisé atteignent environ **F1 = 0,72**, et la différence entre eux n'est pas significative.
3. **Ensembles :** l'optimisation par CV imbriquée améliore Random Forest et Gradient Boosting. Le comportement d'achat est toutefois en grande partie expliqué par des relations simples, donc l'avantage des ensembles reste modeste.
4. **Usage métier :** on peut combiner les deux parties. Par exemple, cibler en priorité les clients du segment « réguliers à risque » qui ont une probabilité de réachat moyenne : c'est là qu'une relance a le plus d'effet.

### Limites
- Les données couvrent seulement 1 an, et la cible ne regarde que les 3 mois suivants.
- La date de coupure tombe juste avant les fêtes de fin d'année, donc il peut y avoir un effet saisonnier.
- Les clients sans identifiant (25 % des lignes) sont exclus de l'analyse.

---

## 7. Comment exécuter le projet

```bash
pip install -r requirements.txt
python projet_online_retail.py        # ou ouvrir projet_online_retail.ipynb dans Jupyter
```
Le fichier `online+retail/Online Retail.xlsx` doit se trouver à côté du script. Toutes les figures sont enregistrées dans `figures/`. La durée totale est d'environ 5 minutes : la lecture du fichier Excel et la CV imbriquée sont les étapes les plus longues.
