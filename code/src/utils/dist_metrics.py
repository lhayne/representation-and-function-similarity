import torch
import numpy as np
import scipy


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
    
    Au,As,_ = torch.linalg.svd(A,full_matrices=False)
    As_diag = torch.diag(As)
    A_R = Au @ As_diag
    del A, Au, As, As_diag

    Bu,Bs,_ = torch.linalg.svd(B,full_matrices=False)
    Bs_diag = torch.diag(Bs)
    B_R = Bu @ Bs_diag
    del B, Bu, Bs, Bs_diag

    A_cov  = A_R.T @ A_R
    B_cov  = B_R.T @ B_R
    AB_cov = A_R.T @ B_R

    A_cov_inv = torch.linalg.inv(sqrtm(A_cov + pen_a * torch.eye(A_R.shape[1],device=A_cov.device)))
    B_cov_inv = torch.linalg.inv(sqrtm(B_cov + pen_b * torch.eye(B_R.shape[1],device=B_cov.device)))

    objective_matrix = (A_cov_inv @ (AB_cov) @ B_cov_inv)

    del A_cov, B_cov, AB_cov

    u,s,vh = torch.linalg.svd(objective_matrix,full_matrices=False)

    transformed_a = (u.T @ A_cov_inv @ A_R.T).T
    transformed_b = (vh @ B_cov_inv @ B_R.T).T
    
    return u, s, vh, transformed_a, transformed_b