import torch
import time
import os

from training.metric_logger import MetricLogger


def run_every_epoch(callback):
    def modified_callback(output_location, epoch, it, model, optimizer, logger, preprocessing_model):
        if it != 0:
            return
        callback(output_location, epoch, it, model, optimizer, logger, preprocessing_model)
    return modified_callback


def run_last_epoch(callback):
    pass


def save_model(output_location, epoch, it, model, optimizer, logger, preprocessing_model):
    if not os.path.exists(output_location):
        os.makedirs(output_location)
    
    torch.save(model.state_dict(), os.path.join(output_location,'checkpoint'+str(epoch)+'.pth'))


def save_logger(output_location, epoch, it, model, optimizer, logger, preprocessing_model):
    logger.save(output_location)


def create_eval_callback(eval_name: str, loader: torch.utils.data.DataLoader, verbose=False):
    """This function returns a callback."""

    time_of_last_call = None

    def eval_callback(output_location, epoch, it, model, optimizer, logger, preprocessing_model=None):
        example_count = torch.tensor(0.0).to('cuda')
        total_loss = torch.tensor(0.0).to('cuda')
        total_correct = torch.tensor(0.0).to('cuda')

        def correct(labels, outputs):
            return torch.sum(torch.eq(labels, output.argmax(dim=1)))

        model.eval()

        with torch.no_grad():
            for examples, labels in loader:
                examples = examples.to('cuda')
                labels = labels.squeeze().to('cuda')

                # preprocess
                if preprocessing_model != None:
                    examples = preprocessing_model(examples)
                output = model(examples)

                labels_size = torch.tensor(len(labels), device='cuda')
                example_count += labels_size
                total_loss += model.loss_criterion(output, labels) * labels_size
                total_correct += correct(labels, output)

        total_loss = total_loss.cpu().item()
        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()

        logger.add('{}_loss'.format(eval_name), epoch, it, total_loss / example_count)
        logger.add('{}_accuracy'.format(eval_name), epoch, it, total_correct / example_count)
        logger.add('{}_examples'.format(eval_name), epoch, it, example_count)

        if verbose:
            nonlocal time_of_last_call
            elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
            print('{}\tep {:03d}\tit {:03d}\tloss {:.3f}\tacc {:.2f}%\tex {:d}\ttime {:.2f}s'.format(
                eval_name, epoch, it, total_loss/example_count, 100 * total_correct/example_count,
                int(example_count), elapsed))
            time_of_last_call = time.time()

    return eval_callback


def train(
        model,
        optimizer,
        train_loader,
        output_location,
        preprocessing_model=None,
        callbacks = [],
        epochs = 10,
        device='cuda',
):
    """
    Train a model using PyTorch
    """
    # create logger
    logger = MetricLogger()

    # training loop
    for epoch in range(epochs):

        for it, (examples, labels) in enumerate(train_loader):
            print(epoch,it)

            # run callbacks
            for callback in callbacks:
                callback(output_location, epoch, it, model, optimizer, logger, preprocessing_model)

            # train
            examples = examples.to(device)
            labels = labels.to(device)

            # preprocess
            if preprocessing_model != None:
                with torch.no_grad():
                    examples = preprocessing_model(examples)

            # zero grad
            for param in model.parameters():
                param.grad = None

            loss = model.loss_criterion(model(examples), labels)
            loss.backward()

            optimizer.step()