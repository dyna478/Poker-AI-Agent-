
Ce projet présente le développement complet d'un agent de poker autonome, capable de développer une stratégie quasi-optimale pour le Texas Hold'em en un-contre-un (Heads-Up). L'approche adoptée est un système hybride combinant le Machine Learning pour la simplification de l'environnement de jeu et la Théorie des Jeux pour l'apprentissage de la stratégie.

![poker_ai_demo](https://github.com/user-attachments/assets/95d1e390-3ab7-428c-a4ed-3eca765b6554)

🎯 Vue d'ensemble
Ce projet implémente un "cerveau" d'IA sophistiqué capable de jouer au poker Texas Hold'em à un niveau compétitif. Le défi ? Le poker possède environ 10^160 états de jeu possibles - rendant impossible pour toute IA de mémoriser toutes les situations. Notre solution combine l'abstraction intelligente du jeu avec la Minimisation de Regret Contrefactuel Monte Carlo (MCCFR) pour créer un agent de poker puissant et adaptatif.
🧠 L'Architecture
Le Problème : L'Information Imparfaite
Le poker est fondamentalement différent de jeux comme les échecs :

Information imparfaite : Vous ne voyez pas les cartes des adversaires
Résultats stochastiques : Les tirages de cartes sont probabilistes
Espace d'états massif : Scénarios de jeu quasi-infinis
Complexité psychologique : Bluff, lecture des adversaires, théorie des jeux

Les approches traditionnelles de résolution de jeux échouent spectaculairement. Nous avons besoin de quelque chose de plus intelligent.
La Solution : Un Pipeline en Quatre Étapes
Mains de Poker Brutes → Extraction de Caractéristiques → Clustering → Entraînement Auto-Supervisé → Évaluation
                        (PotentialAware)                 (K-Means)    (MCCFR)                    (Tournoi)

🔬 Étape 1 : Extraction de Caractéristiques
PotentialAwareCalculator
L'IA ne mémorise pas les mains individuelles - elle analyse des caractéristiques stratégiques.
Ce qu'il calcule :

Équité : Force actuelle de la main (probabilité de gagner)
Potentiel de Couleur : Probabilité de compléter une couleur
Potentiel de Suite : Probabilité de compléter une suite
Force de Main : Classement brut de la main
Texture du Board : Analyse des cartes communes

Pourquoi c'est important :
Au lieu de traiter A♠ K♠ et A♥ K♥ comme des mains différentes, l'IA reconnaît qu'elles sont stratégiquement identiques (même équité, même potentiel de couleur). Cela réduit considérablement la complexité.
python# Exemple de sortie
{
    "equity": 0.72,
    "flush_potential": 0.18,
    "straight_potential": 0.12,
    "hand_strength": 0.85,
    "feature_vector": [0.72, 0.18, 0.12, 0.85, ...]
}

🎨 Étape 2 : Abstraction & Bucketing
FixedAbstractionManager + Clustering K-Means
Une fois que nous avons des vecteurs de caractéristiques pour des milliers de mains, nous utilisons le clustering K-Means pour regrouper les mains similaires en buckets.
Le processus :

Générer des vecteurs de caractéristiques pour 100 000+ situations de poker
Appliquer K-Means pour les regrouper en 20 buckets stratégiques
Mapper chaque main de poker possible à son bucket le plus proche

Résultat :
L'IA voit maintenant le poker comme un jeu avec seulement 20 "types de mains" distincts au lieu de milliards. C'est ce qui rend l'apprentissage réalisable.
Complexité originale : ~10^160 états
Après abstraction :    ~10^6 états (gérable !)

🚀 Étape 3 : Entraînement par Auto-Confrontation
OptimizedMCCFRTrainer
Vient maintenant l'apprentissage proprement dit. Notre IA utilise la Minimisation de Regret Contrefactuel Monte Carlo - un algorithme de pointe qui :

Joue des millions de mains contre elle-même à vitesse fulgurante
Suit le "regret" pour chaque décision (Combien ai-je perdu en ne choisissant pas l'action X ?)
Met à jour sa stratégie pour minimiser le regret total
Converge vers l'équilibre de Nash - une stratégie prouvée comme inexploitable

La boucle d'entraînement :
Pour 1 000 000+ itérations :
    ├─ Simuler une main de poker
    ├─ L'IA choisit une action basée sur la stratégie actuelle
    ├─ Calculer le regret contrefactuel (et si j'avais choisi différemment ?)
    ├─ Mettre à jour la stratégie pour favoriser les actions à faible regret
    └─ Répéter
Innovation clé :
MCCFR est une méthode basée sur l'échantillonnage. Au lieu de calculer le regret pour chaque scénario possible (impossible), elle échantillonne les arbres de jeu et converge tout aussi efficacement - mais 1000x plus rapidement.

