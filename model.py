"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    # TODO: wrap a Python integer seed into a JAX PRNG key (uint32 array of shape (2,))
    return jax.random.PRNGKey(seed)

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    # TODO: split `key` into `num` independent subkeys and return them as a (num, 2) array.
    return jax.random.split(key, num)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    # TODO: return a jnp array of the given shape with i.i.d. N(0,1) samples drawn from key
    return jax.random.normal(key, shape, dtype=jnp.float32)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    # TODO: draw a batch of random input feature vectors from the PRNG key
    return sample_normal_matrix(key, (batch_size, num_features))

# Step 5 - assign_class_labels
def assign_class_labels(inputs, num_classes):
    # TODO: return an int32 label per row using the first num_classes feature columns.
    labels = jnp.argmax(inputs[:, :num_classes], axis=1)
    return labels.astype(jnp.int32)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    # TODO: Convert a 1-D array of integer class indices into a 2-D one-hot matrix of shape (batch, num_classes).
    return jax.nn.one_hot(labels, num_classes, dtype=jnp.float32)

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    # TODO: sample W from a scaled normal and set b to zeros, return as a dict.
    W = scale * sample_normal_matrix(key, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return {'W': W, 'b': b}

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    # TODO: build a list of per-layer parameter dicts from adjacent layer sizes.
    num_layers = len(layer_sizes) - 1
    keys = split_prng_key(key, num_layers)

    params = []
    for i in range(num_layers):
        layer = init_linear_layer(
            keys[i],
            layer_sizes[i],
            layer_sizes[i + 1],
            scale=scale
        )
        params.append(layer)

    return params

# Step 9 - linear_forward
def linear_forward(x, layer_params):
    # TODO: compute x @ W + b using layer_params['W'] and layer_params['b'].
    W = layer_params['W']
    b = layer_params['b']
    return jnp.matmul(x, W) + b

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    # TODO: return an array of the same shape with negatives replaced by zero.
    return jnp.maximum(x, 0)

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    # TODO: convert logits into a numerically stable softmax along the last axis
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(shifted)
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

# Step 12 - mlp_forward
def mlp_forward(params, x):
    # TODO: run x through all hidden layers with ReLU, then a final linear layer, returning logits.
    for layer in params[:-1]:
        x = linear_forward(x, layer)
        x = relu_activation(x)

    x = linear_forward(x, params[-1])
    return x

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    # TODO: return the numerically stable log-softmax of logits along the last axis.
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    return shifted - jnp.log(
        jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True)
    )

# Step 14 - cross_entropy_loss
def cross_entropy_loss(logits, one_hot_targets):
    # TODO: return the mean cross-entropy between logits and one-hot targets
    log_probs = log_softmax_logits(logits)
    return -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    # TODO: compute predicted classes from logits and compare to labels
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

