import numpy as np
import pickle
import os
import glob
import pandas as pd
from scipy.io import loadmat
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import sys
import torchvision
import torch

import warnings
warnings.filterwarnings("ignore")


class MatrixSquareRoot(torch.autograd.Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.

    From : https://github.com/steveli/pytorch-sqrtm
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

def cca_decomp(A, B):
    """Computes CCA vectors, correlations, and transformed matrices
    requires a < n and b < n
    Args:
        A: np.array of size a x n where a is the number of neurons and n is the dataset size
        B: np.array of size b x n where b is the number of neurons and n is the dataset size
    Returns:
        u: left singular vectors for the inner SVD problem
        s: canonical correlation coefficients
        vh: right singular vectors for the inner SVD problem
        transformed_a: canonical vectors for matrix A, a x n array
        transformed_b: canonical vectors for matrix B, b x n array
    """
    assert A.shape[0] < A.shape[1]
    assert B.shape[0] < B.shape[1]

    evals_a, evecs_a = torch.linalg.eigh(A @ A.T)
    evals_a = (evals_a + torch.abs(evals_a)) / 2
    inv_a = torch.tensor([1 / torch.sqrt(x) if x > 0 else 0 for x in evals_a])

    evals_b, evecs_b = torch.linalg.eigh(B @ B.T)
    evals_b = (evals_b + torch.abs(evals_b)) / 2
    inv_b = torch.tensor([1 / torch.sqrt(x) if x > 0 else 0 for x in evals_b])

    cov_ab = A @ B.T

    temp = (
        (evecs_a @ torch.diag(inv_a) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T)
    )

    try:
        u, s, vh = torch.linalg.svd(temp)
    except:
        u, s, vh = torch.linalg.svd(temp * 100)
        s = s / 100

    transformed_a = (u.T @ (evecs_a @ torch.diag(inv_a) @ evecs_a.T) @ A).T
    transformed_b = (vh @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T) @ B).T
    return u, s, vh, transformed_a, transformed_b


def mean_sq_cca_corr(rho):
    """Compute mean squared CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return torch.sum(rho * rho) / len(rho)


def mean_cca_corr(rho):
    """Compute mean CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return torch.sum(rho) / len(rho)


def pwcca_dist(A, rho, transformed_a):
    """Computes projection weighted CCA distance between A and B given the correlation
    coefficients rho and the transformed matrices after running CCA
    :param A: np.array of size a x n where a is the number of neurons and n is the dataset size
    :param B: np.array of size b x n where b is the number of neurons and n is the dataset size
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    :param transformed_a: canonical vectors for A returned by cca_decomp(A,B)
    :param transformed_b: canonical vectors for B returned by cca_decomp(A,B)
    :return: PWCCA distance
    """
    in_prod = transformed_a.T @ A.T
    weights = torch.sum(torch.abs(in_prod), axis=1)
    weights = weights / torch.sum(weights)
    dim = min(len(weights), len(rho))
    return 1 - torch.dot(weights[:dim], rho[:dim])


def lin_cka_dist_2(A, B):
    """
    Computes Linear CKA distance bewteen representations A and B
    based on the reformulation of the Frobenius norm term from Kornblith et al. (2018)
    np.linalg.norm(A.T @ B, ord="fro") ** 2 == np.trace((A @ A.T) @ (B @ B.T))
    
    Parameters
    ----------
    A : examples x neurons
    B : examples x neurons

    Original Code from Ding et al. (2021)
    -------------
    similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
    normalization = np.linalg.norm(A @ A.T, ord="fro") * np.linalg.norm(B @ B.T, ord="fro")
    """

    similarity = torch.trace((A @ A.T) @ (B @ B.T))
    normalization = (torch.linalg.norm(A @ A.T,ord='fro') * 
                     torch.linalg.norm(B @ B.T,ord='fro'))

    return 1 - similarity / normalization


def procrustes_2(A, B):
    """
    Computes Procrustes distance bewteen representations A and B
    for when |neurons| >> |examples| and A.T @ B too large to fit in memory.
    Based on:
         np.linalg.norm(A.T @ B, ord="nuc") == np.sum(np.sqrt(np.linalg.eig(((A @ A.T) @ (B @ B.T)))[0]))
    
    Parameters
    ----------
    A : examples x neurons
    B : examples x neurons

    Original Code
    -------------    
    nuc = np.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    """
    A_sq_frob = torch.sum(A ** 2)
    B_sq_frob = torch.sum(B ** 2)
    nuc = torch.sum(torch.sqrt(torch.abs(torch.linalg.eig(((A @ A.T) @ (B @ B.T)))[0])))
    return A_sq_frob + B_sq_frob - 2 * nuc


def cca_decomp_2(A,B,pen_a=0,pen_b=0):
    """
    Computes CCA vectors, correlations, and transformed matrices
    based on Tuzhilina et al. (2021)

    Args:
        A: np.array of size n x a where a is the number of neurons and n is the dataset size
        B: np.array of size n x b where b is the number of neurons and n is the dataset size
    Returns:
        u: left singular vectors for the inner SVD problem
        s: canonical correlation coefficients
        vh: right singular vectors for the inner SVD problem
        transformed_a: canonical vectors for matrix A, n x a array
        transformed_b: canonical vectors for matrix B, n x b array
        
    Tuzhilina et al. (2021) normalizes by (1/n), but that doesn't match
    Ding et al. (2021):
    
        A_cov_inv = np.linalg.inv(sqrtm((1/n) * A_cov + pen_a * np.identity(A.shape[1])))
        B_cov_inv = np.linalg.inv(sqrtm((1/n) * B_cov + pen_b * np.identity(B.shape[1])))

        objective_matrix = (A_cov_inv @ ((1/n) * AB_cov) @ B_cov_inv)        
        
    """
    A_cov = A.T @ A
    B_cov = B.T @ B
    AB_cov = A.T @ B
    
    A_cov_inv = torch.linalg.inv(sqrtm(A_cov + pen_a * torch.eye(A.shape[1],device=A_cov.device)))
    B_cov_inv = torch.linalg.inv(sqrtm(B_cov + pen_b * torch.eye(B.shape[1],device=B_cov.device)))
    
    objective_matrix = (A_cov_inv @ (AB_cov) @ B_cov_inv)
    
    u,s,vh = torch.linalg.svd(objective_matrix,full_matrices=False)
    transformed_a = (u.T @ A_cov_inv @ A.T).T
    transformed_b = (vh  @ B_cov_inv @ B.T).T
    
    return u, s, vh, transformed_a, transformed_b


def cca_decomp_kernel_trick(A,B,pen_a=0,pen_b=0):
    """
    Computes CCA vectors, modified correlations, and transformed matrices.
    Implements the kernel trick from Tuzhilina et al. (2021). Useful for n << a,b.
    The kernel trick replaces A and B in the objective function with 
    A_R and B_R (A = A_R @ V.T). Replacing A with A_R and B with B_R, 
    reduces the size of the covariance matrices, making working in high dimensions tractable.
    The CCA vectors and modified correlations are the same for solutions based on A and A_R.
    The only caveat is that the dimension of the CCA vectors are restricted to the size of the
    dataset (n).
    
    Args:
        A: np.array of size n x a where a is the number of neurons and n is the dataset size
        B: np.array of size n x b where b is the number of neurons and n is the dataset size
        pen_a: regularization penalty for A, required when a >= n
        pen_b: regularization penalty for B, required when b >= n
    Returns:
        u: left singular vectors for the inner SVD problem
        s: canonical correlation coefficients
        vh: right singular vectors for the inner SVD problem
        transformed_a: canonical vectors for matrix A, n x a array
        transformed_b: canonical vectors for matrix B, n x b array
        
    Tuzhilina et al. (2021) normalizes by (1/n), but that doesn't match
    Ding et al. (2021):
    
        A_cov_inv = np.linalg.inv(sqrtm((1/n) * A_cov + pen_a * np.identity(A.shape[1])))
        B_cov_inv = np.linalg.inv(sqrtm((1/n) * B_cov + pen_b * np.identity(B.shape[1])))

        objective_matrix = (A_cov_inv @ ((1/n) * AB_cov) @ B_cov_inv)        
        
    """
    torch.cuda.empty_cache()
    Au,As,Av = torch.linalg.svd(A,full_matrices=False)
    As_diag = torch.diag(As)
    A_R = Au @ As_diag

    Bu,Bs,Bv = torch.linalg.svd(B,full_matrices=False)
    Bs_diag = torch.diag(Bs)
    B_R = Bu @ Bs_diag

    A_cov  = A_R.T @ A_R
    B_cov  = B_R.T @ B_R
    AB_cov = A_R.T @ B_R

    A_cov_inv = torch.linalg.inv(sqrtm(A_cov + pen_a * torch.eye(A_R.shape[1],device=A_cov.device)))
    B_cov_inv = torch.linalg.inv(sqrtm(B_cov + pen_b * torch.eye(B_R.shape[1],device=B_cov.device)))

    objective_matrix = (A_cov_inv @ (AB_cov) @ B_cov_inv)

    u,s,vh = torch.linalg.svd(objective_matrix,full_matrices=False)

    transformed_a = (u.T @ A_cov_inv @ A_R.T).T
    transformed_b = (vh @ B_cov_inv @ B_R.T).T
    
    return u, s, vh, transformed_a, transformed_b


def get_activation_matrix(activation_path,layer):
    activation_files = glob.glob(os.path.join(activation_path,'*'))
    activation_files = np.sort(activation_files)
    activations = []
    for f in activation_files:
        with open(f,'rb') as f:
            activations.append(pickle.load(f)[layer].flatten())
    return np.row_stack(activations)

def decoding_accuracy(activations,y,iterations=200,neurons=100):
    losses = []
    for _ in range(iterations):
        clf = LogisticRegression(random_state=0)
        random_indexes = np.random.choice(np.arange(activations.shape[1]),
                                            size=min(activations.shape[1],neurons),
                                            replace=False)
        clf.fit(activations[:,random_indexes],y)
        losses.append(log_loss(y,clf.predict(activations[:,random_indexes])))

    mean_loss = np.mean(losses)
    return np.exp(-mean_loss)


def rls(X,Y,penalty=0):
    return (torch.linalg.inv(
                X.T @ X + penalty * X.shape[0] * torch.eye(X.shape[1],dtype=X.dtype,device=X.device)) 
            @ X.T @ Y)

def acc(X,Y,W):
    predictions = torch.argmax(X @ W, 1)
    labels = torch.argmax(Y, 1)
    return torch.count_nonzero(predictions==labels)/len(predictions)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return (2*np.eye(num_classes)-1)[y]


def main():
    classes = ['schooner','brain_coral','junco_bird','snail','grey_whale','siberian_husky','electric_fan','bookcase','fountain_pen','toaster']
    class_wids = ['n04147183','n01917289','n01534433','n01944390','n02066245','n02110185','n03271574','n02870880','n03388183','n04442312']

    data_path = '../data/'
    model_name = 'resnet50'
    experiment_path = '../data/experiments/'
    # Load model
    model = torchvision.models.resnet50(weights='DEFAULT').to('cuda:0')
    layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]
    del model
    torch.cuda.empty_cache()

    #Load the details of all the 1000 classes and the function to conver the synset id to words
    meta_clsloc_file = data_path+'meta_clsloc.mat'
    synsets = loadmat(meta_clsloc_file)['synsets'][0]
    synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])
    corr = {}
    for j in range(1000):
        corr[synsets_imagenet_sorted[j][0]] = j

    corr_inv = {}
    for j in range(1, 1001):
        corr_inv[corr[j]] = j

    #Code snippet to load the ground truth labels to measure the performance
    truth = {}
    with open(data_path+'ILSVRC2014_clsloc_validation_ground_truth.txt') as f:
        line_num = 1
        for line in f.readlines():
            ind_ = int(line)
            temp  = None
            for i in synsets_imagenet_sorted:
                if i[0] == ind_:
                    temp = i
            #print ind_,temp
            if temp != None:
                truth[line_num] = temp
            else:
                print('##########', ind_)
                pass
            line_num += 1

    activation_files = glob.glob(os.path.join(data_path,'activations','*'))
    activation_files = np.sort(activation_files)

    # Make list of wids
    true_valid_wids = []
    for i in activation_files:
        temp1 = i.split('/')[-1]
        temp = temp1.split('.')[0]
        true_valid_wids.append(truth[int(temp)][1])
    true_valid_wids = np.asarray(true_valid_wids)

    # Using command line to process classes selectively
    args = sys.argv[1:]
    if len(args)!=0 and args[0] == '--class':
        classes_to_process = [args[1]]
    else:
        classes_to_process = classes

    if len(args)!=0 and args[2] == '--layer_start':
        layer_start = [args[3]]
    else:
        layer_start = [layers[0]]

    
    for class_index,c in enumerate(classes):
        if c not in classes_to_process:
            continue

        results = pd.DataFrame([],columns=['class','layer','tile',
                                            'decoding_accuracy_delta',
                                            'ablation_impact',
                                            'cka','procrustes','pwcca','mean_sq_cca_corr','mean_cca_corr'])

        class_indexes = [idx for idx in range(len(true_valid_wids)) if true_valid_wids[idx]==class_wids[class_index]]
        # Labels correspond to class indexes
        y = np.asarray([1 if i in class_indexes else 0 for i in range(500)])

        layer_start_seen = False        

        for layer_index,layer in enumerate(layers):
            if not layer_start_seen:
                if layer in layer_start:
                    layer_start_seen = True
                else:
                    continue
            
            print(c,layer)
            # Load entire layer's activations as layer_activations
            layer_activations = get_activation_matrix(os.path.join(data_path,'activations'),layer)
            print('layer_activations shape ',layer_activations.shape)
            if layer_activations.shape[1] < 400000 or layer_activations.shape[1] > 800000:
                continue
            # Center and normalize each activation matrix as in Ding et al. (2021)
            # center each column, so that each neuron representation has mean 0
            layer_activations = layer_activations - layer_activations.mean(axis=0, keepdims=True)
            # normalize each representation (Messes up the linear decoding)
            # layer_activations = layer_activations / np.linalg.norm(layer_activations)
        
            # Compute decoding accuracy for class across entire layer
            layer_decoding_accuracy = decoding_accuracy(layer_activations,y,
                                                        iterations=int(layer_activations.shape[1]/50),
                                                        neurons=100)
            print('decoding accuracy: ',layer_decoding_accuracy)
            
            # Load units_in_cells dictionary
            with open(os.path.join(experiment_path,'units_in_cells_'+c+'_'+layer+'.pkl'),'rb') as f:
                units_in_cells = pickle.load(f)[0]
                
            # Load ablation impacts
            with open(os.path.join(experiment_path,'srd_grid_4x4_'+c+'_'+layer+'.pkl'),'rb') as f:
                ablation_impacts = pickle.load(f)[0]
            
            for tile in units_in_cells.keys():
                # Select all the neurons that weren't ablated in this tile
                ablated_indexes = np.asarray(units_in_cells[tile])
                intact_indexes = list(set([i for i in range(layer_activations.shape[1])]) - set(ablated_indexes))
                intact_activations = layer_activations[:,intact_indexes]
                print(layer_activations.shape[1],intact_activations.shape[1])
                
                # Compute decoding accuracy change
                ablated_decoding_accuracy = decoding_accuracy(intact_activations,y,
                                                              iterations=int(intact_activations.shape[1]/50),
                                                              neurons=100)
                print('ablated accuracy: ',ablated_decoding_accuracy)

                # Regularized decoding accuracy
                # training_indices = np.random.choice(np.arange(500),400,replace=False)
                # testing_indices = np.setdiff1d(np.arange(500),training_indices)
                # y_one_hot = to_categorical(y,2)
                # w = rls(torch.from_numpy(intact_activations[training_indices]).float().to('cuda'),
                #         torch.from_numpy(y_one_hot[training_indices]).float().to('cuda'),
                #         penalty=10)
                # regularized_decoding_accuracy = acc(
                #     torch.from_numpy(intact_activations[testing_indices]).float().to('cuda'),
                #     torch.from_numpy(y_one_hot[testing_indices]).float().to('cuda'),
                #     w).cpu().numpy()
                # print('regularized ablated accuracy',regularized_decoding_accuracy)
                
                # Ablation impact
                ablation_impact = ablation_impacts[tile][1]
                print('ablation impact: ',ablation_impact)
                
                # In Ding et al. (2021) the matrices are neurons x examples, so we need to transpose
                # Compute CKA
                cka_sim = lin_cka_dist_2(torch.from_numpy(layer_activations).to('cuda'), 
                                        torch.from_numpy(intact_activations).to('cuda')).cpu().numpy()
                print('cka: ',cka_sim)
                
                # Compute Procrustes
                procrustes_sim = procrustes_2(torch.from_numpy(layer_activations/np.linalg.norm(layer_activations)).to('cuda'),
                                            torch.from_numpy(intact_activations/np.linalg.norm(intact_activations)).to('cuda')).cpu().numpy()
                print('procrustes: ',procrustes_sim)
                
                # Compute PWCCA
                _, cca_rho, _, transformed_rep1, _ = cca_decomp_kernel_trick(
                                                                            torch.from_numpy(layer_activations).to('cuda'), 
                                                                            torch.from_numpy(intact_activations).to('cuda'),
                                                                            pen_a=1e3,pen_b=1e3,
                                                                            )
                pwcca_sim = pwcca_dist(torch.from_numpy(layer_activations.T).to('cuda'), cca_rho, transformed_rep1).cpu().numpy()
                print('pwcca: ',pwcca_sim)
                
                mean_sq_cca_sim = mean_sq_cca_corr(cca_rho).cpu().numpy()
                
                mean_cca_sim = mean_cca_corr(cca_rho).cpu().numpy()
                print('mean_cca_sim',mean_cca_sim)
                
                # Add to dataframe
                results.loc[len(results)] = [c,layer,tile,
                                            layer_decoding_accuracy-ablated_decoding_accuracy,
                                            ablation_impact,
                                            cka_sim,procrustes_sim,pwcca_sim,mean_sq_cca_sim,mean_cca_sim]

                results.to_csv(os.path.join(experiment_path,'iclr_results_'+c+'_first_layers.csv'))

if __name__ == "__main__":
    main()