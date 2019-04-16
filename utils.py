import torch
import torch.nn as nn
SEloss = nn.MSELoss(reduction = 'sum')

class NMF_algorithim(nn.Module):
    '''
    implements ARD NMF from https://arxiv.org/pdf/1111.6085.pdf

    '''
    def __init__(self,Beta,H_prior,W_prior):
        super(NMF_algorithim, self).__init__()
        # Beta paramaterizes the objective function
        # Beta = 1 induces a poisson objective
        # Beta = 2 induces a gaussian objective
        # Priors on the component matrices are Exponential (L1) and half-normal (L2)

        # Set H update
        if H_prior == 'L1':
            self.update_H = update_H_L1
        elif H_prior == 'L2':
            self.update_H = update_H_L2
        else:
            raise ValueError()

        # Set W update
        if W_prior == 'L1':
            self.update_W = update_W_L1
        elif W_prior == 'L2':
            self.update_W = update_W_L2
        else:
            raise ValueError()

        # Set Lambda update
        if H_prior == 'L1' and W_prior == 'L1':
            self.lambda_update = update_lambda_L1
        elif H_prior == 'L2' and W_prior == 'L1':
            self.lambda_update = update_lambda_L1_L2
        elif H_prior == 'L1' and W_prior == 'L2':
            self.lambda_update = update_lambda_L2_L1
        elif H_prior == 'L2' and W_prior == 'L2':
            self.lambda_update = update_lambda_L2
        else:
            raise ValueError()

        #
        #
        # if Beta == 1 and H_prior == 'L1' and W_prior == 'L1' :
        #     self.update_W = update_W_poisson_L1
        #     self.update_H = update_H_poisson_L1
        #     self.lambda_update = update_lambda_L1
        #
        # elif Beta == 1 and H_prior == 'L1' and W_prior == 'L2':
        #     self.update_W = update_W_poisson_L2
        #     self.update_H = update_H_poisson_L1
        #     self.lambda_update = update_lambda_L2_L1
        #
        # elif Beta == 1 and H_prior == 'L2' and W_prior == 'L1':
        #     self.update_W = update_W_poisson_L1
        #     self.update_H = update_H_poisson_L2
        #     self.lambda_update = update_lambda_L1_L2
        #
        # elif Beta == 1 and H_prior == 'L2' and W_prior == 'L2':
        #     self.update_W = update_W_poisson_L2
        #     self.update_H = update_H_poisson_L2
        #     self.lambda_update = update_lambda_L2
        #
        # if Beta == 2 and H_prior == 'L1' and W_prior == 'L1':
        #     self.update_W = update_W_gaussian_L1
        #     self.update_H = update_H_gaussian_L1
        #     self.lambda_update = update_lambda_L1
        #
        # elif Beta == 2 and H_prior == 'L1' and W_prior == 'L2':
        #     self.update_W = update_W_gaussian_L2
        #     self.update_H = update_H_gaussian_L1
        #     self.lambda_update = update_lambda_L2_L1
        #
        # elif Beta == 2 and H_prior == 'L2' and W_prior == 'L1':
        #     self.update_W = update_W_gaussian_L1
        #     self.update_H = update_H_gaussian_L2
        #     self.lambda_update = update_lambda_L1_L2
        #
        # elif Beta == 2 and H_prior == 'L2' and W_prior == 'L2':
        #     self.update_W = update_W_gaussian_L2
        #     self.update_H = update_H_gaussian_L2
        #     self.lambda_update = update_lambda_L2

    def forward(self,W, H, V, lambda_, C, b0, eps_, phi, Beta):
        h_ = self.update_H(H, W, lambda_, phi, V, eps_, Beta)
        w_ = self.update_W(h_, W, lambda_, phi, V, eps_, Beta)
        lam_ = self.lambda_update(w_,h_,b0,C,eps_)
        return h_, w_,lam_

# --------------------------------------
# Divergence + Cost
# --------------------------------------
def beta_div(Beta,V,W,H,eps_):
    V_ap = torch.matmul(W, H).type(V.dtype) + eps_.type(V.dtype)
    if Beta == 2:
        return SEloss(V,V_ap)/2
    if Beta == 1:
        lr = torch.log(torch.div(V, V_ap))
        return torch.sum( ( (V*lr) + V_ap) - V)

def calculate_objective_function(Beta,V,W,H,lambda_,C, eps_,phi,K):
    loss = beta_div(Beta,V,W,H,eps_)
    cst = (K*C)*(1.0-torch.log(C))
    return torch.pow(phi,-1)*loss + (C*torch.sum(torch.log(lambda_ * C))) + cst

# --------------------------------------
# MM Functions - Updates
# --------------------------------------
def _zeta(Beta):
    """
    Exponent in MM-optimization of each factorized matrix.
    Used for an l2 update step.
    """
    if Beta <= 2:
        return torch.div(1,3-Beta)
    else:
        return torch.div(1,Beta-1)

def _gamma(Beta):
    """
    Exponent in MM-optimization of each factorized matrix.
    Used for an l1 update step.
    """
    if Beta < 1:
        return torch.div(1,2-Beta)
    elif Beta <= 2:
        return 1
    else:
        return torch.div(1,Beta-1)

def update_H_L1(H, W, lambda_, phi, V, eps_, Beta):
    """
    Generalized update for H matrix.
    """
    V_ap = torch.matmul(W,H) + eps_
    V_res = torch.matmul(torch.pow(V_ap, Beta-2), V))

    num = torch.matmul(torch.t(W), V_res)
    denom = torch.matmul(torch.t(W), torch.pow(V_ap,Beta-1)) + torch.div(phi, lambda_ ) + eps_
    update = torch.pow(torch.div(num, denom), _gamma(Beta))

    return H * update

def update_H_L2(H, W, lambda_, phi, V, eps_, Beta):
    """
    Generalized update for H matrix.
    """
    V_ap = torch.matmul(W,H) + eps_
    V_res = torch.matmul(torch.pow(V_ap, Beta-2), V))

    num = torch.matmul(torch.t(W), V_res)
    denom = torch.matmul(torch.t(W), torch.pow(V_ap,Beta-1)) + torch.div(phi*H, lambda_ ) + eps_
    update = torch.pow(torch.div(num, denom), _zeta(Beta))

    return H * update

def update_W_L1(H, W, lambda_, phi, V, eps_, Beta):
    """
    Generalized update for W matrix.
    """
    V_ap = torch.matmul(W, H) + eps_
    V_res = torch.matmul(torch.pow(V_ap, Beta-2), V)

    num = torch.matmul(V_res, torch.t(H))
    denom = torch.matmul(torch.pow(V_ap, Beta-1), torch.t(H)) + torch.div(phi, lambda_) + eps_
    update = torch.pow(torch.div(num, denom), _gamma(Beta))
    return H * update

def update_W_L2(H, W, lambda_, phi, V, eps_, Beta):
    """
    Generalized update for W matrix.
    """
    V_ap = torch.matmul(W, H) + eps_
    V_res = torch.matmul(torch.pow(V_ap, Beta-2), V)

    num = torch.matmul(V_res, torch.t(H))
    denom = torch.matmul(torch.pow(V_ap, Beta-1), torch.t(H)) + torch.div(phi*W, lambda_) + eps_
    update = torch.pow(torch.div(num, denom), _zeta(Beta))
    return H * update

 # Lambda Updates
def update_lambda_L1(W,H,b0,C,eps_):
    return torch.div(torch.sum(W,0) + torch.sum(H,1) + b0, C)

def update_lambda_L2(W,H,b0,C,eps_):
    return torch.div(0.5*torch.sum(W*W,0) + (0.5*torch.sum(H*H,1))+b0,C)

def update_lambda_L1_L2(W,H,b0,C,eps_):
    return torch.div(torch.sum(W,0) + 0.5*torch.sum(H*H,1)+b0,C)

def update_lambda_L2_L1(W,H,b0,C,eps_):
    return torch.div(0.5*torch.sum(torch.pow(W,2),0) + torch.sum(H,1)+b0,C)

# Tolerance Update
def update_del(lambda_, lambda_last):
    """
    Update tolerance criteria.
    """
    return torch.max(torch.div(torch.abs(lambda_ - lambda_last)), lambda_last)
