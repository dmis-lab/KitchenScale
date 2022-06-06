import math
import torch
import numpy as np

def norm_log_pdf(x, mean, stdev):
        return -0.5 * torch.pow((x-mean)/stdev, 2.0) - torch.log(np.sqrt(2.0 * np.pi) * stdev)
        # return -0.5 * tf.pow((x-loc)/scale, 2.0) - tf.log(np.sqrt(2.0 * np.pi) * scale)

def norm_pdf(value):
    constant = torch.tensor([2.0*math.pi]).to(value.device)
    return 1.0/torch.sqrt(constant) * torch.exp(-.5*(value**2))


def norm_cdf(x, mean, stdev):
    # return 0.5 + 0.5 * tf.erf((x-loc)/(scale*np.sqrt(2.0)))
    return 0.5 + 0.5 * torch.erf((x-mean)/(stdev*np.sqrt(2.0)))


def log_normal(x, means, logvars):
    """
    Returns the density of x under the supplied gaussian. Defaults to
    standard gaussian N(0, I)
    :param x: (B) torch.Tensor
    :param mean: float or torch.FloatTensor with dimensions (n_component)
    :param logvar: float or torch.FloatTensor with dimensions (n_component)
    :return: (B,n_component) elementwise log density
    """
    log_norm_constant = -0.5 * torch.log(torch.tensor(2 * math.pi))
    a = (x - means) ** 2
    log_p = -0.5 * (logvars + a / logvars.exp())
    log_p = log_p + log_norm_constant
    return log_p


def log_truncate(x, means, logvars, a=.1, b=1.0, debug=False):
    stdev = logvars.exp().sqrt()
    
    if debug:
        p, numerator, denominator = truncated_normal(x, means, stdev, a, b, debug) #bsk
    else:
        p = truncated_normal(x, means, stdev, a, b, debug) #bsk

    log_p = torch.log(p)
    return log_p


def truncated_normal(value, mean, stdev, a, b, debug=False):
    masked_values = torch.zeros_like(value).fill_(.5)
    mask_ind1 =  torch.where(value > 1.0)
    mask_ind2 =  torch.where(value < 0.1)

    value = torch.where(value > 1.0, masked_values, value)
    value = torch.where(value < 0.1, masked_values, value)

    x = (value - mean)/stdev
    numerator = norm_pdf(x)
    denominator = stdev*(norm_cdf(b, mean, stdev) - norm_cdf(a, mean, stdev))
    denominator = torch.ones_like(denominator)

    probs = numerator/denominator
    probs[mask_ind1] = 0.0
    probs[mask_ind2] = 0.0

    if debug:
        return probs, numerator, denominator
    # Not actually probabilities can be larger than 1.
    return probs



def fexp_embed(numbers):
    #returns only positive embedding
    exponents = torch.log10(numbers).long()
    exponents += 1
    return exponents

def inv_fexp_embed(exponents):
    #returns only positive embedding
    exponents = exponents - 1
    numbers = torch.pow(10, exponents)
    return numbers.float()

def fexp(numbers, ignore=False):
    #numerical can return negative numbers
    exponents = torch.log10(numbers).long()

    #predictions can be less than 1

    return exponents

def fman(numbers, ignore=False):
    exponents = fexp(numbers, ignore).float() + 1.0
    mantissas = numbers / torch.pow(10.0, exponents)
    return mantissas

def np_str(x):
    # default precision is 6.
    return f'{x:e}'

def np_float(x):
    return float(x)

def np_embed(batch):
    '''x is string'''
    b,s = batch.shape
    embed_x = np.zeros((b,s,12))
    
    for i, example in enumerate(batch):
        for j,number in enumerate(example):
            for k,digit in enumerate(number):
                embed_x[i,j,k] = digit_vocab.index(digit)

    return embed_x


digit_vocab = ['0','1','2','3','4','5','6','7','8','9','.','e','+','-']
v_np_str = np.vectorize(np_str)
v_np_float = np.vectorize(np_float)
