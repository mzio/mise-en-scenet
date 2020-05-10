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

    total_accuracy = []
    
    for batch_ix, (data, labels) in enumerate(tqdm(dataloader)):
        data = data.to(args.device)
        labels = labels.to(args.device)
        
        # Do feature extractions
        if args.model_type == 'vgg-pretrained':
            gram_means, encoding_means = feature_extract(vgg, data)
            outputs = model(encoding_means, gram_means[gram_ix])
        else:
            # print(data.shape)
            outputs = model(data)
        _, predictions = torch.max(outputs.data, 1)
        if args.label_type == 'director':
            total_correct += (predictions == labels).sum().item()
            running_correct += (predictions == labels).sum().item()
        elif args.label_type == 'genre':
            accuracy = [pred_acc(labels[ix].cpu(), output.cpu()) for ix, output in enumerate(outputs, 0)]
            total_accuracy.append(np.array(accuracy).mean())

        # print(outputs.shape)
        # print(labels.shape)

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
    
    if args.label_type == 'director':
        total_acc = total_correct / len(dataloader.dataset)
    elif args.label_type == 'genre':
        total_acc = np.array(total_accuracy).mean()
    print(f'Epoch: {epoch:<2} | Training accuracy: {total_acc:<.4f} | Training loss: {total_loss / len(dataloader.dataset):<.4f}')


def evaluate(model, vgg, dataloader, criterion, epoch, gram_ix=2, split='Val', save_dict=None, genres=None):
    model.eval()
    with torch.no_grad():
        total_correct = 0.
        running_correct = 0.
        total_loss = 0.
        running_loss = 0.
        total_accuracy = []
        for batch_ix, (data, labels) in enumerate(tqdm(dataloader)):
            data = data.to(args.device)
            labels = labels.to(args.device)
            
            # Do feature extractions
            if args.model_type == 'vgg-pretrained':
                gram_means, encoding_means = feature_extract(vgg, data)
                outputs = model(encoding_means, gram_means[gram_ix])
            else:
                outputs = model(data)
            _, predictions = torch.max(outputs.data, 1)
            if args.label_type == 'director':
                total_correct += (predictions == labels).sum().item()
                running_correct += (predictions == labels).sum().item()
            elif args.label_type == 'genre':
                accuracy = [pred_acc(labels[ix].cpu(), output.cpu()) for ix, output in enumerate(outputs, 0)]
                total_accuracy.append(np.array(accuracy).mean())

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            running_loss += loss.item()

            if batch_ix % args.log_frequency == args.log_frequency - 1:    # print every 1000 mini-batches
                print(f'Epoch: {epoch:<2} | Batch: {batch_ix:<2} | Train accuracy: {running_correct / (args.batch_size * args.log_frequency):<.4f} | Train loss: {running_loss / (args.batch_size * args.log_frequency):<.4f}')
                # print('Epoch %2d, Batch %2d, Training Loss: %.3f' %
                #     (epoch + 1, batch_ix + 1, running_loss / 5))
                running_correct = 0.0
                running_loss = 0.0
 
            if save_dict is not None:
                embeddings = model.embed(data).cpu().numpy()
                dataset = dataloader.dataset
                for ix in range(embeddings.shape[0]):
                    data_row = dataset.df.iloc[batch_ix + ix]
                    save_dict['frame'].append(data_row['frame'])
                    save_dict['video'].append(data_row['video'])
                    save_dict['movie'].append(data_row['movie'])
                    save_dict['director'].append(data_row['director'])
                    for ex, embedding in enumerate(range(embeddings.shape[1])):
                        save_dict[f'e_{ex}'].append(embedding)
                    for genre in genres:
                        save_dict[genre].append(data_row[genre])
                
        if args.label_type == 'director':
            total_acc = total_correct / len(dataloader.dataset)
        elif args.label_type == 'genre':
            total_acc = np.array(total_accuracy).mean()
            
        print(f'Epoch: {epoch:<2} | {split} accuracy: {total_acc:<.4f} | {split} loss: {total_loss / len(dataloader.dataset):<.4f}')
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

def pred_acc(target, prediction):
    """
    Prediction accuracy for multi-label setting
    """
    return torch.round(prediction).eq(target).sum().numpy()/len(target)

def save_embedding_data(save_dict, data, labels):
    """
    Method to save embedding data
    """
    pass
