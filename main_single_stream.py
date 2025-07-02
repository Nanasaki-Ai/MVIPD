import os
import time
import yaml
import torch
import pickle
import random  
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from model.resnet.model_resnet import Model1s
from torch.utils.data import DataLoader
from dataset_reader import DataSetReader
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed):
    """
    Set the random seed to ensure the reproducibility of experimental results.
    """
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  # Setting PyTorch's CPU random seed
    torch.cuda.manual_seed(seed)  # Setting PyTorch's GPU random seed
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs, set the random seed for all GPUs
    # Ensure that non-deterministic behavior is eliminated when using deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_parse():
    parser = argparse.ArgumentParser(description="model arg")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test mode')
    parser.add_argument('--exp_dir', type=str, default='', help='experiment directory')
    parser.add_argument('--output', type=str, default='runs/3view_cat_input/resnet/yolov11nfci', help='output directory')
    parser.add_argument('--batch_size', type=int, default=40, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=2000, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('--weights', default=None, help='pretrained model weights')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')  # Add seed parameter

    # data
    parser.add_argument('--label_path', type=str, default='label', help='path to label files')
    parser.add_argument('--benchmark', type=str, default='mv123',
                        choices=['sv1', 'sv2', 'sv3',
                              'sv12', 'sv23', 'sv13', 'sv123',
                              'mv12', 'mv23', 'mv13', 'mv123', 'mv12_13_23'],
                        help='benchmark type')
    parser.add_argument('--data_path', type=str, default='/home/user/dataset',
                        help='data path')
    parser.add_argument('--data', type=str, default='yolo_fci',
                        choices=['yolo_fci', 'ori'],
                        help='ori means Original Images, fci represents Foreground Consistent image')
    parser.add_argument('--viewpoint', type=str, default='mv3_cat', 
                        choices=['mv', 'sv', 'mv3', 'mv3_cat', 'global_local_cat'],
                        help='mv3_cat corresponds to the H dimension splicing in the manuscript (three-view input)')

    args = parser.parse_args()
    return args


def print_log(exp_dir, msg):
    print(msg)
    with open(f'{exp_dir}/log.txt', 'a') as f:
        print(msg, file=f)
        

def evaluate(net, data_loader, criterion, device, viewpoint):
    net.eval()
    total_loss = 0.0
    labels_list = torch.empty(0).to(device)
    outputs_list = torch.empty(0).to(device)
    sample_names = []  
    pred_scores = []  

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            if viewpoint in ['sv', 'mv3_cat']:
                view = data[0].to(device)
                labels = data[1].to(device)
                sample_names.extend(data[2])  
                outputs = net(view)
            elif viewpoint in ['mv', 'global_local_cat']:
                view1 = data[0].to(device)
                view2 = data[1].to(device)
                labels = data[2].to(device)
                sample_names.extend(data[3])  
                outputs = net(view1, view2)
            elif viewpoint == 'mv3':
                view1 = data[0].to(device)
                view2 = data[1].to(device)
                view3 = data[2].to(device)
                labels = data[3].to(device)
                sample_names.extend(data[4])  
                outputs = net(view1, view2, view3)
                
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            labels_list = torch.cat((labels_list, labels), 0)
            outputs_list = torch.cat((outputs_list, outputs), 0)
            pred_scores.extend(outputs.cpu().tolist())  

    avg_loss = total_loss / len(data_loader)
    _, predicted_labels = torch.max(outputs_list.data, 1)
    accuracy = torch.mean((predicted_labels == labels_list.data).float()).item()

    # joint_acc
    id_correct_count = {}
    id_total_count = {}
    for i, name in enumerate(sample_names):
        id = name[5:10]  
        if id not in id_correct_count:
            id_correct_count[id] = 0
            id_total_count[id] = 0
        id_total_count[id] += 1
        if predicted_labels[i] == labels_list.data[i]:
            id_correct_count[id] += 1

    total_ids = len(id_correct_count)  
    completely_correct_ids = 0
    for id in id_correct_count:
        if id_correct_count[id] == id_total_count[id]:  
            completely_correct_ids += 1

    joint_acc = completely_correct_ids / total_ids

    return avg_loss, accuracy, joint_acc, sample_names, pred_scores


if __name__ == '__main__':
    args = init_parse()

    set_seed(args.seed)

    running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    exp_dir = os.path.join(args.output, args.benchmark, running_date)
    os.makedirs(exp_dir, exist_ok=True)
    args.exp_dir = exp_dir

    arg_dict = vars(args)
    with open(f'{exp_dir}/config.yaml', 'w') as f:
        yaml.dump(arg_dict, f)

    log_writer = SummaryWriter(log_dir=exp_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dynamically loading models
    if args.viewpoint in ['sv', 'mv3_cat']:  
        net = Model1s().to(device)
    elif args.viewpoint == 'mv':  
        net = Model2s().to(device)
    elif args.viewpoint == 'mv3':  
        net = Model3s().to(device)
        
    if args.weights:
        net.load_state_dict(torch.load(args.weights))
        print_log(exp_dir, f"Loaded model weights from {args.weights}")
    elif args.mode == 'test':
        raise ValueError("In test mode, weights must be provided.")
    else:
        print_log(exp_dir, "Training model from scratch...")

    dataset_test = DataSetReader(args, 'test')
    dataset_test_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                     shuffle=False, pin_memory=True)
    print_log(exp_dir, "Loaded test data successfully.")

    if args.mode == 'train':
        dataset_train = DataSetReader(args, 'train')
        print(f"Number of samples in training dataset: {len(dataset_train)}")
        dataset_train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True)
        print_log(exp_dir, "Loaded train data successfully.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=net.parameters(), lr=1e-4)
        max_acc = 0
        max_epoch = 0
        max_jacc_epoch = 0
        max_joint_acc = 0

        for epoch in range(args.num_epochs):
            net.train()
            total_loss = 0.0
            pbar = tqdm(dataset_train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
            for itern, data in enumerate(pbar):
                if args.viewpoint in ['sv', 'mv3_cat']:
                    view = data[0].to(device)
                    labels = data[1].to(device)
                    outputs = net(view)
                elif args.viewpoint in ['mv', 'global_local_cat']:
                    view1 = data[0].to(device)
                    view2 = data[1].to(device)
                    labels = data[2].to(device)
                    outputs = net(view1, view2)
                elif args.viewpoint == 'mv3':
                    view1 = data[0].to(device)
                    view2 = data[1].to(device)
                    view3 = data[2].to(device)
                    labels = data[3].to(device)
                    outputs = net(view1, view2, view3)                

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted_labels = torch.max(outputs.data, 1)
                acc = torch.mean((predicted_labels == labels.data).float())
                pbar.set_postfix(Acc=acc.item(), Loss=loss.item())

            avg_loss = total_loss / len(dataset_train_loader)
            print_log(exp_dir, f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {avg_loss:.4f}")

            # Testing
            test_loss, test_acc, joint_acc, sample_names, pred_scores = evaluate(net, dataset_test_loader, criterion, device, args.viewpoint)
            print_log(exp_dir, f"Epoch [{epoch + 1}/{args.num_epochs}] - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}, Joint Acc: {joint_acc:.2f}")

            # Update the best test accuracy
            if test_acc > max_acc:
                max_acc = test_acc
                max_epoch = epoch + 1
                save_path = os.path.join(exp_dir, 'Max_Acc.pt')
                torch.save(net.state_dict(), save_path)
                print_log(exp_dir, f"Max Accuracy model saved at: {save_path}")

                results = {'sample_names': sample_names, 'scores': pred_scores}
                with open(os.path.join(exp_dir, 'Max_Acc_Scores.pkl'), 'wb') as f:
                    pickle.dump(results, f)
                print_log(exp_dir, f"Saved scores to {os.path.join(exp_dir, 'Max_Acc_Scores.pkl')}")

            # Update the best joint accuracy
            if joint_acc > max_joint_acc:
                max_joint_acc = joint_acc
                max_jacc_epoch = epoch + 1
                save_path = os.path.join(exp_dir, 'Max_Joint_Acc.pt')
                torch.save(net.state_dict(), save_path)
                print_log(exp_dir, f"Max Joint Accuracy model saved at: {save_path}")

                results = {'sample_names': sample_names, 'scores': pred_scores}
                with open(os.path.join(exp_dir, 'Max_Joint_Acc_Scores.pkl'), 'wb') as f:
                    pickle.dump(results, f)
                print_log(exp_dir, f"Saved scores to {os.path.join(exp_dir, 'Max_Joint_Acc_Scores.pkl')}")

            print_log(exp_dir, f"Current Epoch {epoch+1} | Accuracy: {test_acc:.3f} | Joint Accuracy: {joint_acc:.3f} ")
            print_log(exp_dir, f"Best Accuracy: {max_acc:.3f} at Epoch {max_epoch} | Best Joint Accuracy: {max_joint_acc:.3f} at Epoch {max_jacc_epoch}.")

    if args.mode == 'test':
        if not args.weights:
            raise ValueError("In test mode, you must provide a valid pretrained model using the --weights argument.")
        test_loss, test_acc, joint_acc, sample_names, pred_scores = evaluate(net, dataset_test_loader, nn.CrossEntropyLoss(), device, args.viewpoint)
        print_log(exp_dir, f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}, Joint Accuracy: {joint_acc:.2f}")

        save_path_acc = os.path.join(exp_dir, 'Max_Acc.pt')
        torch.save(net.state_dict(), save_path_acc)
        print_log(exp_dir, f"Max Accuracy model saved at: {save_path_acc}")

        results_acc = {'sample_names': sample_names, 'scores': pred_scores}
        with open(os.path.join(exp_dir, 'Max_Acc_Scores.pkl'), 'wb') as f:
            pickle.dump(results_acc, f)
        print_log(exp_dir, f"Saved scores to {os.path.join(exp_dir, 'Max_Acc_Scores.pkl')}")

        save_path_joint = os.path.join(exp_dir, 'Max_Joint_Acc.pt')
        torch.save(net.state_dict(), save_path_joint)
        print_log(exp_dir, f"Max Joint Accuracy model saved at: {save_path_joint}")

        results_joint = {'sample_names': sample_names, 'scores': pred_scores}
        with open(os.path.join(exp_dir, 'Max_Joint_Acc_Scores.pkl'), 'wb') as f:
            pickle.dump(results_joint, f)
        print_log(exp_dir, f"Saved scores to {os.path.join(exp_dir, 'Max_Joint_Acc_Scores.pkl')}")
