import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score




data_path = "m1-dataset/ratings.dat"
df = pd.read_csv(data_path, engine='python', sep="::", header=None)
df.columns = ['user_id', 'movie_id', 'rating', 'time_stamp']


# number of unique users and movies 
n_users = df['user_id'].nunique()
n_movies = df['movie_id'].nunique()


# making a matrix of shape  'n_users x n_movies' 
def load_user_rating_data(df, n_users, n_movies):
    data = np.zeros((n_users, n_movies), dtype=np.intc)
    movie_id_mapping = {}
    for user_id , movie_id , rating in zip(df['user_id'], df['movie_id'], df['rating']):
        user_id = int(user_id) - 1
        if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
        data[user_id, movie_id_mapping[movie_id]] = rating
    return data, movie_id_mapping

data, movie_id_mapping = load_user_rating_data(df, n_users, n_movies)
print(movie_id_mapping)

# number of rating for each movie id
values, counts = np.unique(data, return_counts=True)
for value, count in zip(values, counts):
    print(f"Number of Rating {value}: {count}")


# number of movie counts in primary dataset
number_of_rating_per_movie = df['movie_id'].value_counts()
target_movie_id = 2858


# changing dataset and keep those records which have valid value(1 to 5) for movie id 2858
X_raw = np.delete(data, movie_id_mapping[target_movie_id], axis=1)
Y_raw = data[:, movie_id_mapping[target_movie_id]]

X =  X_raw[Y_raw > 0]
Y =  Y_raw[Y_raw > 0]
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

# number of positive and negaive labels(class instances)
recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()


# splitting test and train data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# since each input feature can have values between 0 to 5. we use multinomial model.
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)


# getting prediction probability for teseting date
prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:10])

# predict classes for testing data
prediction = clf.predict(X_test)
print(prediction[0:10])

# calculating the accuracy of model
accuracy = clf.score(X_test, Y_test)
print(f"The accuracy of classifier is: {accuracy * 100:.1f}")


# testing model's predormance using different metrics
confusion_matrix_ = confusion_matrix(Y_test, prediction, labels=[0,1])
precision_scr = precision_score(Y_test, prediction, pos_label=1)
recall_scr = recall_score(Y_test, prediction, pos_label=1)
fone_score = f1_score(Y_test, prediction, pos_label=1)
