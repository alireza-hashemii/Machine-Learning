import numpy as np
from collections import defaultdict

# providing toy data
X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]])

Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[1, 1, 0]])


def get_label_indices(labels: list):
    label_indices = defaultdict(list)

    for index, label in enumerate(labels):
        label_indices[label].append(index)
    
    return label_indices


label_ind = get_label_indices(Y_train)


def get_prior(label_indices: dict):
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total = sum(prior.values())

    for label in prior:
        prior[label] /= total
    
    return prior

p = get_prior(label_ind)


def get_likelihood(features, label_indices, smoothing=0 , /):
  
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        print(label, likelihood[label])
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
      
    return likelihood

likelihood = get_likelihood(X_train, label_ind,1 )



def get_posterior(X, prior, likelihood):

    posteriors = []
    for x in X:
        # posterior is proportional to prior * likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
        # normalize so that all sums up to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


print(get_posterior(X_test, p, likelihood))