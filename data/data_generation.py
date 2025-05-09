import numpy as np

def sigmoid(logits):
    return 1 / (1 + np.exp(-logits))

def generate_data_bias_only(num_samples, sample_dim, min_frac, 
                            min_signal, maj_signal, 
                            flip_min=0.2, flip_maj=0.05, seed=None, shuffle = True):
    """
    generates data where min and maj groups share weight vector but differ ONLY by group-specific bias terms
    """
    if seed is not None:
        np.random.seed(seed)            # optional seed
    
    X = np.random.randn(num_samples, sample_dim).astype(np.float32)
    group_tags = (np.random.rand(num_samples) < min_frac).astype(np.int64) # randomly assign tail_frac proportion of the samples to the minority group

    w = np.random.randn(sample_dim)     # generate random weight vector
    w /= np.linalg.norm(w)              # normalize it

    b_maj = maj_signal
    b_min = min_signal

    logits = X @ w                      # logit inputs to sigmoid function
    logits[group_tags == 1] += b_min    # adds bias term to minority group
    logits[group_tags == 0] += b_maj    # adds bias term to majority group

    probs = sigmoid(logits)             # get probabilities for samples
    y = np.random.binomial(1, probs)    # sample binary labels based on probabilities

    flip = np.random.rand(num_samples)              # each element is random float in [0,1)
    y[(group_tags == 1) & (flip < flip_min)] ^= 1   # flips min group labels with probability flip_min
    y[(group_tags == 0) & (flip < flip_maj)] ^= 1   # flips maj group labels with probability flip_maj

    # then shuffle if shuffle = True
    if shuffle:
        idx = np.random.permutation(num_samples)
        X = X[idx]
        y = y[idx]
        group_tags = group_tags[idx]
    
    return X, y.astype(np.float32), group_tags

def generate_data_weight_only(num_samples, sample_dim, min_frac,
                              flip_min=0.2, flip_maj=0.05, seed=None, shuffle=True):
    """
    generates data where min and maj groups differ in weight vectors but DO NOT have bias terms
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(num_samples, sample_dim).astype(np.float32)
    group_tags = (np.random.rand(num_samples) < min_frac).astype(np.int64)

    w_min = np.random.randn(sample_dim)
    w_min /= np.linalg.norm(w_min)

    w_maj = np.random.randn(sample_dim)
    w_maj /= np.linalg.norm(w_maj)

    logits = np.zeros(num_samples)
    logits[group_tags == 1] = X[group_tags == 1] @ w_min
    logits[group_tags == 0] = X[group_tags == 0] @ w_maj

    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)

    flip = np.random.rand(num_samples)
    y[(group_tags == 1) & (flip < flip_min)] ^= 1
    y[(group_tags == 0) & (flip < flip_maj)] ^= 1

    if shuffle:
        idx = np.random.permutation(num_samples)
        X = X[idx]
        y = y[idx]
        group_tags = group_tags[idx]

    return X, y.astype(np.float32), group_tags

def generate_data_weight_and_bias(num_samples, sample_dim, min_frac,
                                  min_bias, maj_bias,
                                  flip_min=0.2, flip_maj=0.05, seed=None, shuffle=True):
    """
    generates data where min and maj groups differ both in weight vectors and bias terms
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(num_samples, sample_dim).astype(np.float32)
    group_tags = (np.random.rand(num_samples) < min_frac).astype(np.int64)

    w_min = np.random.randn(sample_dim)
    w_min /= np.linalg.norm(w_min)

    w_maj = np.random.randn(sample_dim)
    w_maj /= np.linalg.norm(w_maj)

    logits = np.zeros(num_samples)
    logits[group_tags == 1] = X[group_tags == 1] @ w_min + min_bias
    logits[group_tags == 0] = X[group_tags == 0] @ w_maj + maj_bias

    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)

    flip = np.random.rand(num_samples)
    y[(group_tags == 1) & (flip < flip_min)] ^= 1
    y[(group_tags == 0) & (flip < flip_maj)] ^= 1

    if shuffle:
        idx = np.random.permutation(num_samples)
        X = X[idx]
        y = y[idx]
        group_tags = group_tags[idx]

    return X, y.astype(np.float32), group_tags


##### complicated data generation original decided not to use (ignore everything below)

def get_w_tail(w_maj, overlap_angle, sample_dim):
    w_perp = np.random.randn(sample_dim) # create random vector
    w_perp -= np.dot(w_perp, w_maj) * w_maj # remove the part that is the projection of w_perp onto w_maj
    w_perp /= np.linalg.norm(w_perp) # now this is a normalized vector perpendicular to w_maj
    w_tail = np.cos(overlap_angle) * w_maj + np.sin(overlap_angle) * w_perp # make w_tail overlap_angle away from w_maj and 90-overlap_angle from w_perp
    return w_tail

def get_logits(w_maj, w_tail, feature_vectors, tail_flags, tail_signal, majority_signal):
    num_samples = len(feature_vectors)
    logits = np.zeros(num_samples)
    for i in range(num_samples):
        if tail_flags[i] == 1: # this means it is a tail sample
            logits[i] = tail_signal * np.dot(w_tail, feature_vectors[i]) # alpha times dot product of w and x, i.t. w^T x that we plug into sigmoid
        else: # majority sample
            logits[i] = majority_signal * np.dot(w_maj, feature_vectors[i])
    return logits

def generate_data(
        num_samples, 
        sample_dim, 
        tail_fraction, 
        overlap_angle, 
        tail_signal = 1.0, 
        majority_signal = 1.0,
        flip_probability_tail = None, # can also set to 0 but this is faster?
        flip_probability_maj = None,
        seed = None
        ):
    if seed is not None:
        np.random.seed(seed)
    num_tail = int(num_samples * tail_fraction) # number of tail samples

    # we'll assume that the samples are uncorrelated, so let's have some gaussian random feature vecotrs
    feature_vectors = np.random.randn(num_samples, sample_dim) 

    # now want to assign each sample either group or majority; will shuffle at the end
    tail_flags = np.zeros(num_samples, dtype = int)
    tail_flags[:num_tail] = 1

    w_maj = np.random.randn(sample_dim)
    w_maj /= np.linalg.norm(w_maj) # vector perpendicular to the hyperplane that is the decision boundary for majority
    w_tail = get_w_tail(w_maj, overlap_angle, sample_dim)
    logits = get_logits(w_maj, w_tail, feature_vectors, tail_flags, tail_signal, majority_signal)    
    probs = sigmoid(logits)
    labels = (np.random.rand(num_samples) < probs).astype(int) 
        # for each sample, generate new random number from uniform dist, 1 if less then probs 0 if not
        # then label[i] has probs[i] chance of being 1, as we want
    
    if flip_probability_tail is not None and flip_probability_tail > 0:
        tail_mask = (tail_flags == 1)
        flip_mask_tail = (np.random.rand(num_samples) < flip_probability_tail) & tail_mask
        labels[flip_mask_tail] = 1 - labels[flip_mask_tail]

    if flip_probability_maj is not None and flip_probability_maj > 0:
        maj_mask = (tail_flags == 0)
        flip_mask_maj = (np.random.rand(num_samples) < flip_probability_maj) & maj_mask
        labels[flip_mask_maj] = 1 - labels[flip_mask_maj]
    
    indices = np.arange(num_samples) # indices 0 through num_tail-1 are tail samples
    np.random.shuffle(indices) # shuffle so tails aren't all at the beginning
    
    # get feature vectors, labels, and tail flags in new shuffled index order
    feature_vectors = feature_vectors[indices] 
    labels = labels[indices]
    tail_flags = tail_flags[indices]

    return feature_vectors, labels, tail_flags