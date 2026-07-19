# 🛡️ BAMIS Fraud Detection — Solution Data Science Explicable

## 📌 Contexte

Dans le cadre du **DATATHON ESP DATACLUB — Détection de fraude sur le Mobile Money**, nous proposons une solution intelligente permettant d'identifier les transactions suspectes et les comportements frauduleux dans un environnement Mobile Money.

Le défi principal n'est pas seulement de détecter une fraude, mais de fournir une **explication claire et exploitable** pour les analystes métiers et les investigateurs.

Une solution antifraude utilisée dans un contexte bancaire ne doit pas être une boîte noire. Chaque décision doit pouvoir être auditée, comprise et justifiée.

---

# 🎯 Objectif du projet

Construire un système de détection hybride capable de :

* Détecter les transactions anormales.
* Identifier les comportements de type compte mule.
* Détecter les réseaux suspects de transactions.
* Produire un score de risque fraude.
* Fournir une explication locale pour chaque transaction.
* Fournir une analyse globale des facteurs influençant le risque.

---

# 🏗️ Architecture globale

Notre approche repose sur une architecture **Tri-Moteur** :

```
                    Transactions Mobile Money
                              |
                              ↓
                  Data Science Pipeline
                              |
        ------------------------------------------------
        |                      |                       |
        ↓                      ↓                       ↓

   Rule Engine          Machine Learning          Graph Engine
   Business Rules       Isolation Forest          Network Analysis

        |                      |                       |
        ------------------------------------------------
                              |
                              ↓

                    Evidence Matrix
                              |
                              ↓

                  Explainability Layer
                              |
              --------------------------------
              |                              |
              ↓                              ↓

     Explication Locale             Explication Globale
     Transaction                    Modèle / Population

              |
              ↓

              Fraud Risk Score
```

---

# 🔬 Workflow Data Science

## 1. Data Understanding & EDA

Analyse complète des données :

* Structure des transactions.
* Qualité des données.
* Valeurs manquantes.
* Distribution des variables.
* Analyse des montants.
* Analyse temporelle.
* Détection des valeurs extrêmes.
* Analyse des comportements clients.

Objectif :

Comprendre les données avant toute modélisation et éviter les mauvaises interprétations.

---

## 2. Data Quality & Preprocessing

Traitement :

* Conversion des dates.
* Normalisation des montants.
* Gestion des valeurs manquantes.
* Détection des incohérences.
* Validation des types.
* Prévention du Data Leakage.

---

# ⚙️ 3. Feature Engineering comportemental

Création de variables représentant le comportement réel :

### Transactions

* Montant transactionnel.
* Fréquence des opérations.
* Volume journalier.
* Volume hebdomadaire.
* Écart par rapport aux habitudes.

### Temporel

* Activité nocturne.
* Latence transactionnelle.
* Fenêtres glissantes :

  * 1 heure
  * 24 heures
  * 7 jours
  * 30 jours

### Seuils métier

* Proximité du seuil autorisé.
* Tentative de fractionnement.
* Dépassement des limites.

---

# 📜 4. Rule Engine

Les règles métiers ne remplacent pas le Machine Learning.

Elles servent comme **preuves explicables**.

Exemples :

* Plusieurs transactions juste sous un seuil.
* Activité inhabituelle durant la nuit.
* Transferts rapides entre comptes.
* Transactions internes/externe suspectes.

Chaque règle produit :

```
Rule ID
Scenario
Evidence
Severity
Explanation
```

---

# 🤖 5. Machine Learning — Détection d'anomalies

Comme aucune étiquette fraude n'est disponible, nous utilisons une approche non supervisée.

Modèle :

## Isolation Forest

Pourquoi ?

* Adapté aux anomalies rares.
* Fonctionne sans labels.
* Performant sur grandes données transactionnelles.
* Compatible avec une approche explicable.

Le modèle produit :

```
Anomaly Score
```

Ce score représente le niveau d'écart du comportement par rapport à la population.

---

# 🕸️ 6. Graph Analysis — Détection des réseaux suspects

Les transactions sont représentées comme un graphe :

```
Client A  --------> Client B

Source              Destination
```

Analyse :

* Fan-in : beaucoup de personnes envoient vers un compte.
* Fan-out : un compte envoie vers plusieurs comptes.
* Cycles suspects.
* Structures de comptes mule.

Le graphe apporte une information impossible à voir avec une analyse transactionnelle classique.

---

# 🧩 7. Evidence Matrix (sans fusion opaque)

Contrairement aux approches classiques qui combinent directement les scores :

```
Score = 0.5 ML + 0.3 Rules + 0.2 Graph
```

notre approche conserve les preuves séparées.

Exemple :

| Transaction | Rule           | ML              | Graph |
| ----------- | -------------- | --------------- | ----- |
| TX001       | Fractionnement | Anomalie élevée | Mule  |

Avantage :

L'investigateur comprend exactement pourquoi une transaction est suspecte.

---

# 🔍 8. Explainability Layer

Notre point fort principal.

Nous proposons deux niveaux :

## Explication locale

Pourquoi cette transaction est suspecte ?

Exemple :

```
Transaction TX001 :

+ Montant proche du seuil
+ 8 transactions en 30 minutes
+ Client connecté à un réseau mule
+ Score anomalie élevé

Conclusion :
Risque élevé
```

---

## Explication globale

Quels facteurs influencent le risque général ?

Analyse :

* Importance des features.
* Distribution des anomalies.
* Analyse SHAP.
* Impact moyen des variables.

Objectif :

Comprendre le comportement global du système.

---

# 📊 Résultats attendus

Livrables :

## soumission_fraude.csv

Contient :

* Transaction ID
* Fraud Score
* Explanation

## classement_clients.csv

Contient :

* Score risque client.
* Segment client.
* Explication.

## consommation_enveloppes.csv

Contient :

* Consommation seuil.
* Niveau d'alerte.

---

# ⭐ Points forts de notre solution

## 1. Pas de Black Box

Dans un contexte bancaire, une décision non expliquée n'est pas acceptable.

Notre système fournit une justification pour chaque alerte.

---

## 2. Architecture hybride

Combinaison :

* Expertise métier (Rules)
* Intelligence statistique (ML)
* Intelligence réseau (Graph)

---

## 3. Auditabilité

Chaque décision peut être reconstruite :

Transaction → Features → Evidence → Score → Explication

---
# 🖥️ Plateforme d'aide à la décision antifraude

Afin de valoriser les résultats du pipeline Data Science, une plateforme interactive sera développée au-dessus des sorties générées par les différents moteurs de détection.

Cette plateforme permet aux analystes fraude de :

Explorer les transactions à risque.
Visualiser les scores de risque.
Consulter les explications associées à chaque alerte.
Identifier les comptes présentant des comportements suspects.
Explorer les relations entre clients grâce à l'analyse des graphes.
Faciliter la priorisation des investigations. 

# Objectif métier

La plateforme ne remplace pas l'expert fraude. Elle agit comme un système d'aide à la décision permettant de réduire le temps d'analyse et d'améliorer la compréhension des alertes.

Chaque décision est accompagnée de preuves :

Signaux métier détectés.
Facteurs comportementaux.
Anomalies statistiques.
Relations réseau suspectes.
Explication générée automatiquement.

Ainsi, l'investigateur peut passer d'une simple alerte à une analyse complète et justifiée.

# 🚀 Perspectives

* Ajout d'un modèle supervisé lorsque des labels fraude seront disponibles.
* Mise en production avec API.
* Dashboard analyste.
* Monitoring continu.
* Apprentissage des nouveaux comportements frauduleux.
