
@jit
def train_step(lr, opt_state,  batch_data, loss_fn):
    """Implements train step.
    
    Args:
        step: Integer representing the step index
        opt_state: Current state of the optimizer
        batch_data: A batch of data (images and labels)
    Returns:
        Batch loss, batch accuracy, updated optimizer state
    """
    params = get_params(opt_state)
    batch_loss, batch_gradients = value_and_grad(loss_fn)(params, batch_data)
    batch_accuracy = calculate_accuracy(params, batch_data)
    return batch_loss, batch_accuracy, opt_update(step, batch_gradients, opt_state)

@jit
def test_step(opt_state, batch_data):
    """Implements train step.

    Args:
        opt_state: Current state of the optimizer
        batch_data: A batch of data (images and labels)
    Returns:
        Batch loss, batch accuracy
    """
    params = get_params(opt_state)
    batch_loss = loss_fn(params, batch_data)
    batch_accuracy = calculate_accuracy(params, batch_data)
    return batch_loss, batch_accuracy


def calculate_accuracy(params, batch_data):
    """Implements accuracy metric.
    
    Args:
        params: Parameters of the network
        batch_data: A batch of data (images and labels)
    Returns:
        Accuracy for the current batch
    """
    inputs, targets = batch_data
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(nn_apply(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)

def cross_loss_fn(params, batch_data):
    """Implements cross-entropy loss function.
    
    Args:
        params: Parameters of the network
        batch_data: A batch of data (images and labels)
    Returns:
        Loss calculated for the current batch
    """
    inputs, targets = batch_data
    preds = nn_apply(params, inputs)
    return -jnp.mean(jnp.sum(log_softmax(preds) * targets, axis=1))
