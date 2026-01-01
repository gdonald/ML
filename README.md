# Machine Learning

This repo contains my machine learning projects.

## Projects

### [Bayesian Weather Network](https://github.com/gdonald/ML/blob/main/bayesian_weather/bayesian_weather.ipynb)

Probabilistic graphical model using discrete Bayesian networks to predict weather conditions across 36 US cities. Infers temperature, humidity, pressure, and wind categories from time of day and seasonal patterns. Implements both manual and learned network structures with pgmpy.

### CIFAR-10

- **[Corrupted Dataset Analysis](https://github.com/gdonald/ML/blob/main/cifar-10/cifar-10-1/cifar-10-1.ipynb)**
  
  Initial exploration of CIFAR-10-C dataset analyzing 19 corruption types across 5 severity levels. Extracts and organizes 950k corrupted images with detailed data summaries and class distribution analysis.

- **[Model Training](https://github.com/gdonald/ML/blob/main/cifar-10/cifar-10-2/cifar-10-2.ipynb)**
  
  Compact CNN with batch normalization and data augmentation. Trained for 60 epochs with learning rate scheduling, achieving 63.8% test accuracy. Visualizes training curves and includes full model summary.

- **[Advanced Training](https://github.com/gdonald/ML/blob/main/cifar-10/cifar-10-3/cifar-10-3.ipynb)**
  
  Improved 3-layer CNN with dropout regularization, cosine annealing learning rate schedule, and early stopping. Achieves 84.2% test accuracy with confusion matrix analysis showing strong performance across all 10 classes.

### [IMDB Sentiment Analysis](https://github.com/gdonald/ML/blob/main/imdb/imdb.ipynb)

Fine-tuned DistilBERT model for binary sentiment classification on movie reviews. Implements early stopping, sequence length optimization (320 tokens), and threshold tuning. Achieves 92.2% accuracy and 0.9219 F1 score on 25k test reviews.

### [Kuzushiji-MNIST Ensemble](https://github.com/gdonald/ML/blob/main/kuzushiji_mnist/kuzushiji_mnist.ipynb)

Ensemble learning combining three CNN architectures for Japanese Hiragana character recognition. Uses stacking with logistic regression meta-learner and k-fold cross-validation. Final ensemble achieves 96.05% accuracy with 0.1942 log loss on 10-class classification.

### [Mastodon Sentiment Analysis](https://github.com/gdonald/ML/blob/main/mastodon_sentiment/mastodon_sentiment.ipynb)

Real-time sentiment analysis pipeline for Mastodon posts using Twitter-RoBERTa. Fetches 500 English posts from mastodon.social public timeline, performs spaCy preprocessing with feature extraction, and classifies sentiment with confidence scores.

### [School Attendance](https://github.com/gdonald/ML/blob/main/school_attendance/school_attendance.ipynb)

Regression model predicting Connecticut school district attendance rates using HistGradientBoostingRegressor. Analyzes attendance patterns across student demographics and identifies at-risk districts through clustering and visualization.

### [Titanic Survivors](https://github.com/gdonald/ML/blob/main/titanic/titanic.ipynb)

Binary classification predicting passenger survival on the Titanic using RandomForestClassifier. Features engineered include family size, title extraction, and one-hot encoding. Achieved 81.7% accuracy with hyperparameter tuning via GridSearchCV.

