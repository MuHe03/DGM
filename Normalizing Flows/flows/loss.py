def likelihood(X_train, model, device):
    ##########################################################
    # YOUR CODE HERE
    
    model.eval()
    X_train = X_train.to(device)
    loss = -model.log_prob(X_train).mean()

    ##########################################################

    return loss
