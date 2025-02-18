from sklearn.naive_bayes import BernoulliNB
import numpy as np

X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]])

Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[1, 1, 0]])


nb_model = BernoulliNB(alpha=1.0, fit_prior=True)
nb_model.fit(X_train, Y_train)


pred_prob = nb_model.predict_proba(X_test)
print("Prediction Probability is:", pred_prob)

pred = nb_model.predict(X_test)
print(pred)