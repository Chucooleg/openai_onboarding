import torch
from torch import nn
from nn_modules import LogisticRegressionLazy

# LAZY VERSION
# use default backward, loss critierion, optimizer


def train(model, train_loader, valid_loader, args):

    loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    history = {
        'train_losses': [],
        'valid_losses': [],        
        'valid_accuracy': [],
        'weights': [],
    }

    for epoch_i in range(args.num_epochs):

        # train
        model.train()
        batch_losses = []
        for batch_i, batch_data in enumerate(train_loader):
            logits, activations = model(batch_data['X'])
            loss = loss_criterion(logits, batch_data['y'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        history['train_losses'].append(sum(batch_losses) / len(batch_losses))

        # validate
        batch_val_losses, batch_val_accuracies = pred(model, valid_loader)
        history['valid_losses'].append(sum(batch_losses) / len(batch_losses))
        history['valid_losses'].append(sum(batch_accuracies) / len(batch_accuracies))

        # save weights
        weights = model.scorer.weight.data.numpy() #TODO check
        history['weights'] = weights
    

def pred(model, test_loader):

    # evaluate
    model.eval()
    batch_losses = []
    batch_accuracies = []
    for batch_i, batch_data in enumerate(test_loader):
        logits, activations = model(batch_data['X'])
        loss = loss_criterion(logits, batch_data['y'])
        batch_losses.append(loss.item())
        # TODO compute accuracy -- check
        accuracy = torch.mean((activations > 0.5).type(torch.FloatTensor) == batch_data['y'])
        batch_accuracies.append(accuracy.item())
    
    return batch_losses, batch_accuracies






torch.manual_seed(args.seed)
model = LogisticRegressionLazy(nx=nx)
train_X, train_y, valid_X, valid_y, test_X, test_y = load_dataset()
