
Ce projet présente le développement complet d'un agent de poker autonome, capable de développer une stratégie quasi-optimale pour le Texas Hold'em en un-contre-un (Heads-Up). L'approche adoptée est un système hybride combinant le Machine Learning pour la simplification de l'environnement de jeu et la Théorie des Jeux pour l'apprentissage de la stratégie.

![poker_ai_demo](https://github.com/user-attachments/assets/95d1e390-3ab7-428c-a4ed-3eca765b6554)

# 🃏 Texas Hold'em Poker AI: Agent Adaptatif basé sur MCCFR

Une IA de poker avancée qui combine la **Minimisation de Regret Contrefactuel Monte Carlo (MCCFR)** avec une abstraction intelligente du jeu pour jouer au Texas Hold'em à un niveau compétitif. Ce projet relève le défi fondamental des jeux à information imparfaite, où l'espace d'états massif (~10^160 états de jeu possibles) rend les approches traditionnelles de résolution de jeux computationnellement impossibles.

---

## 🎯 Vue d'ensemble du projet

Cet agent d'IA démontre comment l'apprentissage automatique moderne peut maîtriser des environnements stratégiques complexes grâce à :
- **L'abstraction stratégique** réduisant des milliards de scénarios de poker en clusters gérables
- **L'apprentissage par renforcement auto-supervisé** convergeant vers des stratégies d'équilibre de Nash
- **L'exploitation adaptative** des styles de jeu adverses (tight/loose, passif/agressif)
- **Le raisonnement basé sur les caractéristiques** plutôt que la mémorisation

---

## 🧠 Architecture : Un Pipeline en Quatre Étapes

```
Mains de Poker Brutes → Extraction de Caractéristiques → Clustering → Entraînement Auto-Supervisé → Évaluation
                        (PotentialAware)                 (K-Means)    (MCCFR)                      (vs Slumbot)
```

### 🔬 Étape 1 : Extraction de Caractéristiques (`PotentialAwareCalculator`)

Plutôt que de traiter chaque configuration de main unique séparément, l'IA analyse des **caractéristiques stratégiques** :

- **Équité de la main** : Probabilité de gagner actuelle contre des mains adverses aléatoires
- **Potentiel de Couleur** : Probabilité de compléter un tirage couleur
- **Potentiel de Suite** : Probabilité de compléter un tirage suite
- **Force de Main** : Classement brut de la main (carte haute → quinte flush royale)
- **Texture du Board** : Analyse des interactions entre cartes communes

**Pourquoi c'est important** : A♠K♠ et A♥K♥ sont stratégiquement identiques malgré des cartes différentes. L'extraction de caractéristiques fait converger ces variations, réduisant drastiquement la complexité de l'espace d'états.

```python
# Exemple de vecteur de caractéristiques
{
    "equity": 0.72,
    "flush_potential": 0.18,
    "straight_potential": 0.12,
    "hand_strength": 0.85,
    "feature_vector": [0.72, 0.18, 0.12, 0.85, ...]
}
```

---

### 🎨 Étape 2 : Abstraction & Bucketing (`FixedAbstractionManager` + K-Means)

**Le Problème** : Même avec les caractéristiques, il y a encore trop de situations uniques pour l'entraînement direct.

**La Solution** : Utiliser le **clustering K-Means** pour regrouper les mains stratégiquement similaires en **20 buckets abstraits** :

1. Générer des vecteurs de caractéristiques pour 100 000+ scénarios de poker
2. Appliquer K-Means pour les regrouper en 20 "types de mains" stratégiques
3. Mapper chaque main de poker possible à son centroïde de cluster le plus proche

**Résultat** : L'IA voit maintenant le poker comme ayant ~20 catégories de mains distinctes au lieu de milliards de situations uniques.

```
Complexité Originale :  ~10^160 états
Après Abstraction :     ~10^6 états (traitable !)
```

---

### 🚀 Étape 3 : Entraînement par Auto-Confrontation (`OptimizedMCCFRTrainer`)

L'algorithme d'apprentissage central : **Minimisation de Regret Contrefactuel Monte Carlo**

**Comment fonctionne MCCFR** :
1. **Simulation auto-supervisée** : L'agent joue des millions de mains contre lui-même
2. **Suivi du regret** : Pour chaque point de décision, calculer le "regret" (combien de valeur a été perdue en ne choisissant pas d'actions alternatives)
3. **Mises à jour de stratégie** : Ajuster les probabilités d'action pour minimiser le regret cumulé
4. **Convergence vers l'équilibre de Nash** : Approche de manière prouvée des stratégies inexploitables

**Boucle d'Entraînement** :
```
Pour 1 000 000+ itérations :
    ├─ Simuler une main de poker avec la stratégie actuelle
    ├─ Échantillonner les actions basées sur des probabilités pondérées par le regret
    ├─ Calculer le regret contrefactuel (et si j'avais choisi différemment ?)
    ├─ Mettre à jour la stratégie pour favoriser les actions à faible regret
    └─ Répéter jusqu'à convergence
```

**Innovation Clé** : MCCFR utilise une **approximation basée sur l'échantillonnage** au lieu d'un parcours exhaustif de l'arbre, atteignant une convergence 1000x plus rapide que CFR classique tout en maintenant les garanties théoriques.

---

---

## 🛠️ Stack Technique

- **Framework Principal** : OpenSpiel (framework RL de Google DeepMind)
- **Langage** : Python 3.8+
- **Algorithmes Clés** : 
  - Monte Carlo CFR (minimisation de regret à variance réduite)
  - Clustering K-Means (scikit-learn)
  - Calcul d'équité (simulation Monte Carlo personnalisée)
- **Évaluation** : Intégration PyPokerEngine + API Slumbot

---

## 🚧 Limitations & Travaux Futurs

### Limitations Actuelles
1. **Taille de mise statique** : Utilise une abstraction de mise fixe au lieu d'un dimensionnement dynamique
2. **Contraintes computationnelles** : Entraîné avec 1M d'itérations (les agents de pointe utilisent des milliards)
3. **Modèles d'adversaires fixes** : L'exploitation nécessite un pré-entraînement pour des styles de jeu spécifiques

### Améliorations Prévues
1. **Dimensionnement dynamique des mises** : Intégrer plusieurs options de taille de mise (3-5 tailles par point de décision) dans l'entraînement MCCFR
2. **Adaptation en ligne** : Modélisation de l'adversaire en temps réel pendant le jeu au lieu de politiques d'exploitation pré-entraînées
3. **Deep CFR** : Remplacer le stockage tabulaire de stratégie par une approximation de fonction par réseau de neurones

---

## 🎓 Contexte Théorique

### Théorie des Jeux à Information Imparfaite

Le poker appartient à une classe de jeux où :
- **Information cachée** : Les cartes adverses sont inconnues
- **Résultats stochastiques** : Les tirages de cartes sont probabilistes
- **Stratégies mixtes** : L'optimalité nécessite de la randomisation (bluff)
- **Équilibre de Nash** : Aucun joueur ne peut améliorer unilatéralement son résultat

### Minimisation de Regret Contrefactuel

**Regret Contrefactuel** : La différence de valeur entre l'action choisie et la meilleure action alternative, pondérée par la probabilité d'atteindre cet état de jeu.

**Propriété de Convergence** : Si tous les joueurs minimisent leur regret, leurs stratégies convergent vers un équilibre de Nash.

**Avantage Monte Carlo** : L'échantillonnage réduit la complexité de O(|États|) à O(√|États|) tout en préservant les garanties de convergence.

---

## 📚 Références & Recherche

Ce projet s'appuie sur des recherches de pointe en IA pour le poker :
- **Counterfactual Regret Minimization** (Zinkevich et al., 2007)
- **DeepStack** (Moravčík et al., 2017) - Premier agent surhumain au poker heads-up
- **Libratus** (Brown & Sandholm, 2018) - Vainqueur contre des professionnels
- **Pluribus** (Brown & Sandholm, 2019) - Extension au poker multijoueur

---

## 💡 Applications au-delà du Poker

Les techniques développées ici s'appliquent à :
- **Finance** : Trading algorithmique avec information incomplète
- **Cybersécurité** : Stratégies défensives contre adversaires adaptatifs
- **Négociation** : Systèmes de décision multi-agents
- **Systèmes Autonomes** : Planification sous incertitude

