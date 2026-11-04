# Online Retail II: customer segmentation and repurchase prediction

Course project for Applied Machine Learning (modules 3 to 5: feature selection, introduction to clustering, advanced clustering). It uses the [Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) from UCI: about 1.07 million transaction lines from a UK online store, between 1 December 2009 and 9 December 2011.

![Customer segment profiles](figures/07_profil_segments.png)

## Objectives

| Task | What the project does |
|---|---|
| Customer clustering (RFM segmentation) | Part A: K-Means, hierarchical, DBSCAN, HDBSCAN, spectral |
| Repurchase probability classification | Part B: predict whether a customer buys again in the next 3 months |
| Ensemble methods | Random Forest and Gradient Boosting, tuned with nested cross-validation |

## Summary of results

- Customers are split into 4 segments with K-Means (k = 4). Of the six methods compared, it has the best silhouette, Davies-Bouldin and Calinski-Harabasz scores, and its segments are stable across bootstrap samples (ARI 0.92).
- Repurchase prediction reaches an F1 score of about 0.66 in cross-validation and 0.68 on a test set that was kept aside from the start. Logistic regression and tuned Random Forest perform at the same level.
- Recency is the most important variable. Three variables (recency, frequency and number of distinct products) are enough.
- All steps that learn from the data (scaling, feature selection, tuning) are done inside pipelines, so the scores are not inflated by data leakage. See the section below.

## Project structure

```
.
├── projet_online_retail.py      # full code, split into # %% cells
├── projet_online_retail.ipynb   # same code as a Jupyter notebook
├── rapport_projet.md            # detailed report (in French)
├── requirements.txt
├── figures/                     # 12 charts and 3 result tables (.csv)
└── online+retail+ii/
    └── online_retail_II.xlsx    # dataset, to download (see "Running the project")
```

Libraries: pandas, NumPy, Matplotlib, SciPy, scikit-learn (1.3 or later), openpyxl.

## Data cleaning

The Excel file has two sheets (2009-2010 and 2010-2011) that are read separately and then joined. They overlap between 1 and 9 December 2010, and the duplicated rows are removed in the cleaning step.

| Step | Reason |
|---|---|
| Remove rows without Customer ID (243,007 rows) | A customer cannot be segmented without an identifier |
| Remove exact duplicates | They distort amounts and frequencies. This also removes the overlap between the two sheets |
| Remove non-product lines (postage, manual adjustments, discounts, bank charges, test items) | They are fees or adjustments, not purchases |
| Set aside invoices starting with "C" | These are cancellations. They are reused as the `NbAnnulations` feature in Part B |
| Keep only Quantity > 0 and Price > 0 | Zero or negative values are errors or adjustments |
| Create `TotalPrice = Quantity x Price` | Amount of each line |

Raw data: 1,067,371 rows. Result: 776,579 clean rows and **5,852 customers**.

## Part A: RFM segmentation

### RFM variables

- Recency: days since the last purchase (reference date is the day after the last invoice)
- Frequency: number of distinct invoices
- Monetary: total amount spent

The distributions are very skewed: the median amount is £856 and the maximum is £580,987. The data goes through `log1p` to reduce the weight of very large customers, then `StandardScaler`. The scaler is placed inside a pipeline with each clustering method, so it is learned again every time a method is fitted, including on each bootstrap sample of the stability test.

![RFM distributions](figures/01_distribution_rfm.png)

### Choosing k for K-Means

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 2 | 8562 | 0.438 | 0.874 | 6146 |
| 3 | 6338 | 0.347 | 1.037 | 5177 |
| 4 | 4908 | 0.365 | 0.931 | 5024 |
| 5 | 4090 | 0.343 | 0.950 | 4813 |
| 6 to 8 | decreasing | 0.30 to 0.34 | 0.96 to 0.99 | decreasing |

k = 4 was chosen for three reasons. k = 2 scores best but only separates good customers from bad ones, which is of little use for marketing. Between k = 3 and k = 8, k = 4 is a local maximum of the silhouette and a local minimum of the Davies-Bouldin score. The inertia curve bends around k = 4, but inertia always decreases as k grows, so it has to be read together with the other metrics.

![Choice of k](figures/02_choix_k_kmeans.png)

### Methods compared

- Hierarchical agglomerative (Ward), cut into 4 groups (dendrogram in `figures/03_dendrogramme.png`)
- Hierarchical divisive (Bisecting K-Means)
- DBSCAN with minPts = 6 (2 x 3 dimensions) and eps = 0.35, read from the k-distance curve (`figures/04_courbe_k_distance.png`)
- HDBSCAN with `min_cluster_size = 50`
- Spectral clustering on a graph of the 15 nearest neighbors

| Method | Clusters | Noise | Silhouette | Davies-Bouldin | Calinski-H. | Dunn | Stability (ARI) |
|---|---|---|---|---|---|---|---|
| K-Means | 4 | 0 | 0.365 | 0.931 | 5024 | 0.004 | 0.922 |
| Hierarchical (Ward) | 4 | 0 | 0.335 | 1.001 | 4452 | 0.006 | 0.460 |
| Bisecting K-Means | 4 | 0 | 0.284 | 1.220 | 4328 | 0.002 | 0.727 |
| DBSCAN | 3 | 167 | 0.157 | 1.743 | 2479 | 0.057 | 0.982 |
| HDBSCAN | 6 | 1080 | 0.153 | 2.807 | 1717 | 0.019 | 0.735 |
| Spectral | 4 | 0 | 0.135 | 1.292 | 2073 | 0.001 | 0.842 |

Metrics for DBSCAN and HDBSCAN are computed without the noise points. Stability is the mean ARI over 10 bootstrap samples.

What the results show:

- K-Means is best on the three main metrics, and its stability is good.
- Ward hierarchical clustering has decent internal scores but is unstable (ARI 0.46). On a continuous cloud of points, a small change in the data changes the order of the merges.
- DBSCAN is very stable and has the best Dunn index, but its 3 groups are very unbalanced (3,161, 1,593 and 931 customers) and its silhouette is low. HDBSCAN labels 1,080 customers (18%) as noise. Density-based methods work poorly when the density varies, which is the case here.
- The Dunn index is very low for every method. Customers form a continuum, not separate groups, so the segmentation cuts this continuum into useful zones.

![PCA comparison of the methods](figures/05_comparaison_pca.png)

### The 4 segments

| Segment | Avg. recency (days) | Avg. frequency | Avg. amount (£) | Customers | Interpretation |
|---|---|---|---|---|---|
| 1 | 28 | 19.2 | 10,590 | 1,182 | Champions: recent, very frequent, big spenders. Keep them with a loyalty program |
| 3 | 28 | 3.0 | 841 | 1,246 | Recent customers still developing: active but few orders. An offer can encourage a second and third purchase |
| 0 | 227 | 5.1 | 1,978 | 1,455 | Good customers at risk: they used to buy regularly but have been inactive for about 7 months. First to re-engage |
| 2 | 393 | 1.4 | 317 | 1,969 | Lost or occasional customers: about one purchase, more than a year ago. Low-cost win-back campaign |

Segment numbers can change between scikit-learn versions. The profiles stay the same.

## Part B: Repurchase prediction

### Target and train/test split

To avoid leaking information from the future, the history is cut at 9 September 2011:

- Observation period (1 December 2009 to 8 September 2011): the 10 customer features are computed here.
- Prediction period (last 3 months): `Reachat = 1` if the customer bought at least once.

This gives 5,253 customers, with 43.7% repurchase and 56.3% no repurchase. Right after that, the data is split in a stratified way into 4,202 training customers (80%) and 1,051 test customers (20%). The test set is not touched again until the final evaluation.

The 10 candidate features are `Recence`, `Anciennete`, `Frequence`, `Montant`, `PanierMoyen`, `QuantiteTotale`, `NbProduits`, `PrixMoyen`, `NbAnnulations` and `RoyaumeUni`.

### Feature selection (training set only)

| Family | Method | Result |
|---|---|---|
| Filter | VarianceThreshold (threshold 0) | No constant variable, all 10 kept |
| Filter | Correlation | `Montant` and `QuantiteTotale` are almost redundant (r = 0.86). `Frequence` is related to `NbAnnulations` (0.76), `NbProduits` (0.67) and `Montant` (0.65) |
| Filter | ANOVA (`f_classif`) | Most significant: `Recence` (F = 962), `NbProduits`, `Frequence`, `NbAnnulations`, `Montant`. `RoyaumeUni` is not significant (p = 0.15) |
| Filter | Mutual information | Top: `Frequence`, `Recence`, `Montant`, `QuantiteTotale`. `PrixMoyen` and `RoyaumeUni` are close to 0 |
| Wrapper | RFE (logistic regression, 5 features) | `Recence`, `Frequence`, `NbProduits`, `PrixMoyen`, `NbAnnulations` |
| Wrapper | RFECV | 3 optimal features: `Recence`, `Frequence`, `NbProduits` |
| Embedded | Lasso (alpha = 0.01) | Keeps `Recence` (strongest coefficient, negative), `Anciennete`, `Frequence`, `NbProduits`, `PrixMoyen`, `NbAnnulations` |
| Embedded | Random Forest importances | `Recence` is clearly first. `RoyaumeUni` and `NbAnnulations` are last |

The methods agree that recency is the most important variable. Frequency and product variety come next, and the country adds nothing. These analyses are for understanding the variables. To evaluate the models, RFECV is part of the pipeline and is run again on each fold.

![RFECV](figures/09_rfecv.png)

### Model comparison

Each model is in a `StandardScaler -> RFECV -> model` pipeline, evaluated with stratified 5-fold cross-validation repeated 3 times (15 scores per model) on the training set.

| Model | Precision | Recall | F1 (mean) | F1 (std) |
|---|---|---|---|---|
| Logistic regression | 0.690 | 0.635 | 0.661 | 0.016 |
| Decision tree | 0.690 | 0.616 | 0.649 | 0.028 |
| Random Forest | 0.646 | 0.608 | 0.626 | 0.028 |
| Gradient Boosting | 0.708 | 0.617 | 0.659 | 0.023 |

![Model comparison](figures/11_boxplot_modeles.png)

Paired Wilcoxon tests between the best model (logistic regression) and each of the others:

- vs. decision tree: p = 0.135, no significant difference
- vs. Random Forest (not tuned): p = 0.0009, significant (logistic regression is better)
- vs. Gradient Boosting: p = 0.208, no significant difference

Effect of feature selection on logistic regression: F1 is 0.665 ± 0.015 with all 10 features and 0.661 ± 0.016 with RFECV inside the pipeline (3 features). The Wilcoxon test gives p = 0.033, so the loss is statistically significant but very small (0.004 of F1). That is the usual trade-off of module 3: a much simpler model, with 3 variables instead of 10, for almost the same performance.

### Tuned ensembles (nested cross-validation)

The inner loop (3 folds) uses `GridSearchCV` to look for the best hyperparameters. The outer loop (5 folds) measures the performance of the tuned model. Feature selection and tuning are done again inside each outer fold.

| Tuned ensemble | F1 (nested CV) |
|---|---|
| Random Forest | 0.661 ± 0.032 |
| Gradient Boosting | 0.657 ± 0.023 |

Tuning clearly improves Random Forest (0.626 to 0.661), which reaches the level of logistic regression.

### Final model and test evaluation

Random Forest (`max_depth=5`, `min_samples_leaf=5`, `n_estimators=300`), trained on the 4,202 training customers and evaluated once on the 1,051 test customers.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No repurchase | 0.75 | 0.78 | 0.77 |
| Repurchase | 0.70 | 0.66 | 0.68 |
| Overall accuracy | | | 0.73 |

![Confusion matrix](figures/12_matrice_confusion.png)

The test F1 (0.68) is consistent with the nested cross-validation (0.66 ± 0.03), which suggests the estimate is honest. The customers most likely to buy again (probability around 0.96) bought less than 15 days before the cutoff, very often and in many different products. The least likely (around 0.08) made a single purchase more than 15 months before.

## Data leakage audit

An earlier version of the project had several data leakage problems. They were found and fixed:

| Problem | Fix |
|---|---|
| Feature selection was done on 100% of the customers, then evaluated on the same customers | The test set is split off before any selection, and the selection uses the training set only |
| `StandardScaler` was fitted on the whole dataset before RFE, RFECV and Lasso | The scaler is inside a `Pipeline` and learned on training data only |
| Model comparison and nested CV used variables already chosen on all the data | RFECV is now a step of the pipeline and is run again on each fold |
| The final test set was drawn after selection and tuning | The test set is set aside first and used only once, at the end |
| Clustering used one scaler fitted on the full table, including in the bootstrap stability test | Each method is a `Pipeline(StandardScaler -> method)`, refitted on each bootstrap sample |

Because of these fixes and the change of dataset (two years of data instead of one), the results are not comparable with the earlier version.

## Conclusions

1. Segmentation: K-Means on log-scaled, standardized RFM variables with k = 4 gives four stable and interpretable segments (champions, recent customers, good customers at risk, lost customers).
2. Repurchase: recency is the key factor, and three variables are enough. Logistic regression and the tuned Random Forest reach an F1 of about 0.66 in cross-validation and 0.68 on the test set.
3. Ensembles: nested cross-validation improves Random Forest, but repurchase behavior is mostly explained by simple relationships, so the advantage of ensembles stays small.
4. Business use: both parts can be combined, for example by contacting first the "good customers at risk" who have a medium repurchase probability.

## Limitations

- The target covers only the 3 months after the cutoff, and the cutoff falls just before the holiday season, so seasonality may affect the results.
- About 23% of the rows have no customer ID and are excluded.
- Part A uses the whole history, so it is a descriptive analysis, not a predictive one. Its segments are not used as features in Part B, because they contain the prediction period.

## Running the project

1. Download the dataset from the [UCI page](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place `online_retail_II.xlsx` in a folder named `online+retail+ii/` next to the script.
2. Install the dependencies and run the script:

```bash
pip install -r requirements.txt
python projet_online_retail.py
```

The notebook `projet_online_retail.ipynb` contains the same code. A full run takes about 10 minutes, mostly for reading the Excel file (two sheets, about one million rows) and the nested cross-validation. All figures are saved in `figures/`. On Windows, `joblib_memmapping_folder` error messages may appear at the end of the run. They come from the cleanup of parallel processes and do not affect the results.
