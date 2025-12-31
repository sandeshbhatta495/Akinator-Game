# 🎮 Akinator Game - Character Guessing AI

A machine learning-based Akinator game that uses Decision Tree classification to guess characters based on their attributes. The game asks strategic yes/no questions about character traits and predicts which character you're thinking of.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Files Description](#files-description)
- [Requirements](#requirements)

## 🎯 Project Overview

This project implements an interactive Akinator-style guessing game using Python and scikit-learn. The model is trained on character datasets with binary attributes (yes/no features) and uses a Decision Tree algorithm to efficiently narrow down the possibilities through strategic questioning.

## ✨ Features

- **Interactive Gameplay**: Play a game where the AI guesses the character you're thinking of
- **Decision Tree Model**: Uses entropy-based decision tree classifier for intelligent questioning
- **Multiple Datasets**: Supports various character datasets including Nepali cricket players and foreign names
- **Binary Question System**: Simple yes/no questions for easy user interaction
- **Smart Predictions**: Efficiently narrows down character possibilities through a decision tree

## 📁 Project Structure

```
Akinator Game (mini)/
├── README.md                                    # This file
├── data/                                        # Dataset folder
│   ├── akinator_data.csv                       # Main character dataset
│   ├── nepali_cricket_akinator.csv             # Nepali cricket players dataset
│   ├── nepali_cricket_akinator_real.csv        # Nepali cricket players (real version)
│   ├── nepali_cricket_akinator_expanded_real.csv # Nepali cricket players (expanded)
│   ├── nepali_cricket_akinator_full.csv        # Nepali cricket players (full)
│   └── nepali_cricket_akinator_perfect.csv     # Nepali cricket players (optimized)
│
└── src/                                         # Source code folder
    ├── final.ipynb                             # Main game implementation
    ├── simple.ipynb                            # Simplified version
    ├── more.ipynb                              # Extended features version
    ├── betttter.ipynb                          # Improved implementation
    ├── new.ipynb                               # New experimental features
    └── foriengn name.ipynb                     # Foreign names version
```

## 📊 Datasets

The project includes multiple datasets for different character variations:

| Dataset | Description | Usage |
|---------|-------------|-------|
| `akinator_data.csv` | Main character dataset | Primary training data |
| `nepali_cricket_akinator.csv` | Basic cricket players | Nepali cricket guessing |
| `nepali_cricket_akinator_real.csv` | Real cricket player data | Accurate cricket player data |
| `nepali_cricket_akinator_expanded_real.csv` | Expanded cricket dataset | More cricket player attributes |
| `nepali_cricket_akinator_full.csv` | Complete cricket dataset | Full feature set |
| `nepali_cricket_akinator_perfect.csv` | Optimized cricket dataset | Best performing dataset |

Each dataset contains binary features (columns with yes/no or 1/0 values) representing character attributes.

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- pandas
- scikit-learn

### Setup Steps

1. **Clone or download the repository**
   ```bash
   cd "Akinator Game (mini)"
   ```

2. **Install required packages**
   ```bash
   pip install pandas scikit-learn jupyter
   ```

3. **Navigate to the project directory**
   ```bash
   cd "Akinator Game (mini)"
   ```

## 💻 Usage

### Running the Game

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   - Navigate to `src/final.ipynb` (recommended for best experience)

3. **Run the game**
   - Execute all cells in the notebook
   - Follow the on-screen prompts
   - Answer yes/no questions about your character
   - The AI will guess your character!

### Example Gameplay

```
Think of a character from the dataset 🤔

Is your character 'tall'? (yes/no): yes
Is your character 'brave'? (yes/no): no
Is your character 'magical'? (yes/no): yes

🎯 I guess your character is: Hermione Granger
```

## 🧠 How It Works

### Training Process

1. **Data Loading**: Character dataset is loaded from CSV file
2. **Feature Extraction**: Binary attributes are used as features
3. **Model Training**: Decision Tree Classifier is trained with:
   - Criterion: Entropy (Information Gain)
   - Max Depth: 6 (prevents overfitting)
   - Training data: Character features and names

### Prediction Process

1. **Tree Traversal**: Starting from the root node of the decision tree
2. **Feature Questions**: At each node, a binary question is asked about a feature
3. **Path Navigation**:
   - "Yes" answer → moves to right child node
   - "No" answer → moves to left child node
4. **Leaf Node**: Reaches a leaf node and predicts the character with highest confidence

### Code Example

```python
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load data
df = pd.read_csv("data/akinator_data.csv")

# Prepare features and target
X = df.drop("character", axis=1)
y = df["character"]

# Train model
model = DecisionTreeClassifier(criterion="entropy", max_depth=6)
model.fit(X, y)

# Play game
def play_akinator(model, feature_names):
    tree = model.tree_
    node = 0
    
    print("\nThink of a character from the dataset 🤔\n")
    
    while tree.feature[node] != -2:  # -2 indicates leaf node
        feature = feature_names[tree.feature[node]]
        ans = input(f"Is your character '{feature}'? (yes/no): ").lower()
        
        if ans == "yes":
            node = tree.children_right[node]
        else:
            node = tree.children_left[node]
    
    prediction = model.classes_[tree.value[node].argmax()]
    print(f"\n🎯 I guess your character is: {prediction}")

play_akinator(model, X.columns.tolist())
```

## 📂 Files Description

### Main Notebooks

- **final.ipynb** - Complete and optimized Akinator implementation with full game logic
- **simple.ipynb** - Simplified version for learning and understanding basics
- **more.ipynb** - Extended version with additional features and variations
- **betttter.ipynb** - Improved implementation with enhancements
- **new.ipynb** - Experimental features and new approaches
- **foriengn name.ipynb** - Variant using foreign character names dataset

### Data Files

- **akinator_data.csv** - Primary training dataset
- **nepali_cricket_akinator*.csv** - Various versions of Nepali cricket player datasets for specialized gameplay

## 📦 Requirements

```
pandas>=1.0.0
scikit-learn>=0.24.0
jupyter>=1.0.0
numpy>=1.19.0
```

Install all requirements:
```bash
pip install -r requirements.txt
```

## 🎓 Learning Outcomes

This project demonstrates:

- **Machine Learning**: Decision Tree classification
- **Data Processing**: Loading and preparing CSV datasets
- **Interactive Systems**: Building user-friendly games
- **Tree-based Algorithms**: Understanding decision trees and entropy
- **Python Programming**: Pandas, scikit-learn, and interactive programming

## 🤝 Contributing

Feel free to:
- Add new character datasets
- Improve the game logic
- Optimize the decision tree
- Create variations for different domains

## 📝 License

This project is open source and available for educational purposes.

## 🎯 Future Enhancements

- [ ] Web-based interface using Flask/Django
- [ ] Graphical visualization of decision tree
- [ ] Add new datasets (movies, books, historical figures)
- [ ] Implement machine learning model persistence (save/load trained models)
- [ ] Add difficulty levels (fewer questions, more questions)
- [ ] Implement feedback system to improve predictions
- [ ] Create a leaderboard system
- [ ] Support for multi-player gameplay

## 👨‍💻 Author Notes

This is a mini machine learning project demonstrating:
- Classification using decision trees
- Interactive game development
- Data-driven predictions
- Binary decision trees

Enjoy the game and feel free to experiment with different datasets and hyperparameters!

---

**Last Updated**: December 2025

**Sandesh Bhatta**
