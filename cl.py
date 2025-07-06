# Contrastive Loss Function for Cross-Domain Aspect-Based Sentiment Analysis.
#
# Implementation by Johan Verschoor (2025) for the thesis:
# "Enhancing Cross-Domain Aspect-Based Sentiment Analysis with Contrastive Learning"
#
# Erasmus University Rotterdam
# Master Econometrics and Management Science
# Business Analytics and Quantitative Marketing
#
# Implements a contrastive loss function using cosine similarity for sentiment-based embeddings.
# The function ensures that instances of the same sentiment polarity are pulled together while
# pushing apart instances with different sentiment polarities.

import tensorflow as tf
import numpy as np


# Define cosine similarity function
def cosine_similarity(x, y):
    """
    Compute cosine similarity between two tensors.

    :param x: Tensor of shape [batch_size, hidden_dim]
    :param y: Tensor of shape [batch_size, hidden_dim]
    :return: Cosine similarity between x and y
    """
    x_norm = tf.nn.l2_normalize(x, axis=-1)
    y_norm = tf.nn.l2_normalize(y, axis=-1)
    return tf.reduce_sum(tf.multiply(x_norm, y_norm), axis=-1)  # Element-wise product along the hidden_dim axis


# Define contrastive loss function using TensorFlow operations
def contrastive_loss(outputs, y_src, tau):
    """
    Compute contrastive loss for the batch by taking the log of each fraction separately and summing over all logs.

    :param outputs: Feature representations (outputs_fin_source), shape [batch_size, hidden_dim]
    :param y_src: Labels (sentiment polarity), shape [batch_size, n_class]
    :param tau: Temperature parameter for contrastive loss scaling
    :return: Scalar contrastive loss value
    """
    epsilon = 1e-5  # Small value to avoid log(0) or division by zero
    batch_size = tf.shape(outputs)[0]  # Get the batch size

    def compute_loss_for_instance(i):
        vi = outputs[i]  # Anchor instance (output)
        yi = y_src[i]  # Anchor label (sentiment polarity)

        # Compute cosine similarities between the anchor (vi) and all other instances (outputs)
        sim_all = cosine_similarity(tf.expand_dims(vi, axis=0), outputs) / tau  # Shape: [batch_size]

        # Find all instances with the same sentiment polarity as the anchor
        mask_same_class = tf.equal(tf.argmax(y_src, 1), tf.argmax(yi))  # True if same class
        mask_same_class = tf.cast(mask_same_class, tf.float32)  # Convert boolean mask to float32

        # Create a mask to exclude the similarity of vi with itself (set it to 0)
        mask_not_self = 1.0 - tf.one_hot(i, depth=batch_size)  # Shape: [batch_size]

        # Exclude self from both the same class mask and similarities
        mask_pos = mask_same_class * mask_not_self  # Exclude the anchor itself from the positive mask

        # Apply the mask to exclude the self-similarity, compute sum
        sim_all_not_self = sim_all * mask_not_self  # Shape: [batch_size]
        sum_all = tf.reduce_sum(tf.exp(sim_all_not_self))  # Sum of exp(similarity) for all pairs (excluding self)

        # Extract positive similarities (those with the same sentiment polarity as vi, excluding self)
        sim_pos = tf.exp(sim_all) * mask_pos  # Apply mask to get only positive pairs

        # Check if mask_pos has any non-zero values
        sum_mask_pos = tf.reduce_sum(mask_pos)

        # Use tf.cond to check if sum_mask_pos is greater than zero and conditionally compute log_fractions
        loss_for_instance = tf.cond(
            sum_mask_pos > 0,
            lambda: -tf.reduce_sum(mask_pos * tf.math.log((sim_pos + epsilon) / (sum_all + epsilon))) / (
                        sum_mask_pos + epsilon),
            lambda: tf.constant(0.0)
        )

        return loss_for_instance

    # Use tf.map_fn to apply the function over the batch and return only the loss
    loss_per_instance = tf.map_fn(compute_loss_for_instance, tf.range(batch_size), dtype=tf.float32)

    # Return the mean loss over the batch
    return tf.reduce_mean(loss_per_instance)


# Testing the functions
def test_contrastive_loss():
    # Create random data for testing
    batch_size = 20
    hidden_dim = 768
    num_classes = 3
    tau = 0.07

    # Randomly generate feature representations (outputs)
    outputs = tf.random.normal([batch_size, hidden_dim], mean=0, stddev=1)

    # Randomly generate one-hot encoded labels for the batch
    y_src = tf.one_hot(np.random.randint(0, num_classes, size=batch_size), depth=num_classes)

    # Compute the contrastive loss
    loss_value = contrastive_loss(outputs, y_src, tau)

    # Run the session to get the result
    with tf.compat.v1.Session() as sess:  # Use tf.compat.v1 for TensorFlow 1.x functions
        sess.run(tf.compat.v1.global_variables_initializer())
        loss_value_np = sess.run(loss_value)
        print(f"Contrastive Loss: {loss_value_np}")


# Run the test
if __name__ == "__main__":
    test_contrastive_loss()
