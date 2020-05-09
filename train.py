"""
Actual training code

Gram style matrix dimensions:
torch.Size([10, 64, 64])
torch.Size([10, 128, 128])
torch.Size([10, 256, 256])
torch.Size([10, 512, 512])
"""
from tqdm import tqdm

from style_utils import *
from args import args


def train(model, vgg, dataloader, criterion, optimizer, epoch, gram_ix=2):
    model.train()
    total_correct = 0.
    running_correct = 0.
    total_loss = 0.
    running_loss = 0.
    
    for batch_ix, (data, labels) in enumerate(tqdm(dataloader)):
        data = data.to(args.device)
        labels = labels.to(args.device)
        
        # Do feature extractions
        gram_means, encoding_means = feature_extract(vgg, data)
        outputs = model(encoding_means, gram_means[gram_ix])
        _, predictions = torch.max(outputs.data, 1)
        total_correct += (predictions == labels).sum().item()
        running_correct += (predictions == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        running_loss += loss.item()

        if batch_ix % args.log_frequency == args.log_frequency - 1:    # print every 1000 mini-batches
            print(f'Epoch: {epoch:<2} | Batch: {batch_ix:<2} | Train accuracy: {running_correct / (args.batch_size * args.log_frequency):<.4f} | Train loss: {running_loss / (args.batch_size * args.log_frequency):<.4f}')
            # print('Epoch %2d, Batch %2d, Training Loss: %.3f' %
            #     (epoch + 1, batch_ix + 1, running_loss / 5))
            running_correct = 0.0
            running_loss = 0.0
    print(f'Epoch: {epoch:<2} | Training accuracy: {total_correct / len(dataloader.dataset):<.4f} | Training loss: {total_loss / len(dataloader.dataset):<.4f}')


def evaluate(model, vgg, dataloader, criterion, epoch, gram_ix=2, split='Val'):
    model.eval()
    with torch.no_grad():
        total_correct = 0.
        running_correct = 0.
        total_loss = 0.
        running_loss = 0.
        for batch_ix, (data, labels) in enumerate(tqdm(dataloader)):
            data = data.to(args.device)
            labels = labels.to(args.device)
            
            # Do feature extractions
            gram_means, encoding_means = feature_extract(vgg, data)
            outputs = model(encoding_means, gram_means[gram_ix])
            _, predictions = torch.max(outputs.data, 1)
            total_correct += (predictions == labels).sum().item()
            running_correct += (predictions == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            running_loss += loss.item()

            if batch_ix % args.log_frequency == args.log_frequency - 1:    # print every 1000 mini-batches
                print(f'Epoch: {epoch:<2} | Batch: {batch_ix:<2} | Train accuracy: {running_correct / (args.batch_size * args.log_frequency):<.4f} | Train loss: {running_loss / (args.batch_size * args.log_frequency):<.4f}')
                # print('Epoch %2d, Batch %2d, Training Loss: %.3f' %
                #     (epoch + 1, batch_ix + 1, running_loss / 5))
                running_correct = 0.0
                running_loss = 0.0
            
        print(f'Epoch: {epoch:<2} | {split} accuracy: {total_correct / len(dataloader.dataset):<.4f} | {split} loss: {total_loss / len(dataloader.dataset):<.4f}')
        # if batch_ix % 5 == 4:    # print every 10 mini-batches
        #     print('Epoch %2d, Batch %2d, Training Loss: %.3f' %
        #         (epoch + 1, batch_ix + 1, running_loss / 5))
        #     running_loss = 0.0

        

def feature_extract(vgg, data):
    if args.dataset_type == 'frames':
        features = vgg(data)
        encoding = vgg.encode(data)
        gram_style = [gram_matrix(y) for y in features]
        return gram_style, encoding

    gram_styles = [[] for _ in range(5)]
    encoding = []
    for i in range(24):
        frames = data[:, i, :, :, :].to(args.device)
        features = vgg(frames)
        encoding.append(vgg.encode(frames))
        gram_style = [gram_matrix(y) for y in features]
        for ix, gm_s in enumerate(gram_style):
            gram_styles[ix].append(gm_s[:frames.size(0), :, :])
        gram_styles[-1].append(features.relu2_2)  # Save content embedding
    gram_means = [torch.stack(gram_styles[i]).mean(dim=0).to(args.device) for i in range(len(gram_styles))]
    encoding_means = torch.stack(encoding).mean(dim=0).to(args.device)
    return gram_means, encoding_means
