#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

DATA_PATH = "./data" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    train_xs, dev_xs = parse_data(args)
    cov = np.cov(train_xs, rowvar=False)
    if args.cluster_num:
        lambdas = np.random.dirichlet(np.ones(args.cluster_num), size=1).reshape(-1, 1)
        mus = 10*np.random.standard_normal((args.cluster_num, 2))
        if not args.tied:
            sigmas = np.zeros([args.cluster_num, 2, 2])
            for k in range(0, args.cluster_num):
                    sigmas[k] = 3*cov + 10
        else:
            sigmas = 3*cov + 10
        #TODO: randomly initialize clusters (lambdas, mus, and sigmas)
        #raise NotImplementedError #remove when random initialization is implemented
    else:
        lambdas = []
        mus = []
        sigmas = []
        import os
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    #TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    #NOTE: if args.tied was provided, sigmas will have a different shape
    class Model:
        def __init__(self):
            self.lambdas = lambdas
            self.mus = mus
            self.sigmas = sigmas
            self.prob = np.zeros([train_xs.shape[0], args.cluster_num])

        def estep(self, args, xn):
            from scipy.stats import multivariate_normal
            for p in range(len(xn)):
                for c in range(len(self.lambdas)):
                    if args.tied:
                        self.prob[p, c] = self.lambdas[c]*multivariate_normal(mean=self.mus[c], cov=self.sigmas).pdf(xn[p])
                    else:
                        self.prob[p, c] = self.lambdas[c]*multivariate_normal(mean=self.mus[c], cov=self.sigmas[c]).pdf(xn[p])
                self.prob[p, :] /= np.sum(self.prob[p, :])

        def mstep(self, args, xn):
            for k in range(len(self.lambdas)):
                self.lambdas[k] = np.sum(self.prob[:, k])/len(xn)
                self.mus[k] = np.dot(self.prob[:, k].T, xn)/np.sum(self.prob[:, k])
                if args.tied:
                    self.sigmas += np.dot(np.multiply(self.prob[:, k].reshape(-1, 1), (xn[:]-self.mus[k])).T, (xn[:]-self.mus[k]))/np.sum(self.prob[:, k], axis=0)/len(self.lambdas)
                else:
                    self.sigmas[k] = np.dot(np.multiply(self.prob[:, k].reshape(-1, 1), (xn[:]-self.mus[k])).T, (xn[:]-self.mus[k]))/np.sum(self.prob[:, k], axis=0)


    #NOTE: if args.tied was provided, sigmas will have a different shape
    model = Model()
    return model

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    dev_ll = float("inf")
    while args.iterations:
        model.estep(args, train_xs)
        model.mstep(args, train_xs)
        if not args.nodev:
            if dev_ll < average_log_likelihood(model, dev_xs, args):
                dev_ll = average_log_likelihood(model, dev_xs, args)
            else:
                break
        args.iterations -= 1
    return model

def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    ll = 0.0
    for n in range(len(data)):
        temp = 0.0
        for k in range(len(model.lambdas)):
            if args.tied:
                temp += model.lambdas[k] * multivariate_normal(mean=model.mus[k], cov=model.sigmas).pdf(data[n])
            else:
                temp += model.lambdas[k]*multivariate_normal(mean=model.mus[k], cov=model.sigmas[k]).pdf(data[n])
        ll += log(temp)
    ll = ll/len(data)
    #raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll

def extract_parameters(model):
    #TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas = None
    mus = None
    sigmas = None
    lambdas = model.lambdas
    mus = model.mus
    sigmas = model.sigmas
    #raise NotImplementedError #remove when parameter extraction is implemented
    return lambdas, mus, sigmas

def plot_graph(args, train_xs, dev_xs):
    import matplotlib.pyplot as plt
    args.nodev = True
    cluster_num = 7
    iterations = 20
    train_ll = np.zeros([20,6])
    dev_ll = np.zeros([20,6])

    for i in range(0, iterations):
        print(i)
        for c in range(2, cluster_num+1):
            args.iterations = i
            args.cluster_num = c
            model = init_model(args)
            model = train_model(model, train_xs, dev_xs, args)
            train_ll[i][c-2] = average_log_likelihood(model, train_xs, args)
            dev_ll[i][c-2] = average_log_likelihood(model, dev_xs, args)

    iterations = list(range(iterations))

    plt.subplot(1, 3, 1)
    plt.plot(iterations, train_ll[:, 0])
    plt.plot(iterations, dev_ll[:, 0])
    plt.xlabel('iterations')
    plt.ylabel('average_log_likelihood')
    plt.title('cluster_num = 2')
    plt.legend(['train_ll', 'dev_ll'])
    plt.subplot(1, 3, 2)
    plt.plot(iterations, train_ll[:, 1])
    plt.plot(iterations, dev_ll[:, 1])
    plt.xlabel('iterations')
    plt.ylabel('average_log_likelihood')
    plt.title('cluster_num = 3')
    plt.subplot(1, 3, 3)
    plt.plot(iterations, train_ll[:, 2])
    plt.plot(iterations, dev_ll[:, 2])
    plt.xlabel('iterations')
    plt.ylabel('average_log_likelihood')
    plt.title('cluster_num = 4')
    plt.show()

    plt.subplot(2, 3, 1)
    plt.plot(iterations, train_ll[:, 3])
    plt.plot(iterations, dev_ll[:, 3])
    plt.xlabel('iterations')
    plt.ylabel('average_log_likelihood')
    plt.title('cluster_num = 5')
    plt.subplot(2, 3, 2)
    plt.plot(iterations, train_ll[:, 4])
    plt.plot(iterations, dev_ll[:, 4])
    plt.xlabel('iterations')
    plt.ylabel('average_log_likelihood')
    plt.title('cluster_num = 6')
    plt.subplot(2, 3, 3)
    plt.plot(iterations, train_ll[:, 5])
    plt.plot(iterations, dev_ll[:, 5])
    plt.xlabel('iterations')
    plt.ylabel('average_log_likelihood')
    plt.title('cluster_num = 7')

    plt.show()

    pass

def main():
    import argparse
    import os
    print('Gaussian')  # Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    init_group.add_argument('--plot', action='store_true', default=False, help='If provided, analysis graph will be plotted. Note this should not be provided unless the matplotlib is installed.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', default=True,
                        help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied', action='store_true',
                        help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print(
            'You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    if args.plot:
        plot_graph(args, train_xs, dev_xs)

    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str, a))

        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '), mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '), map(lambda s: np.nditer(s), sigmas)))))

if __name__ == '__main__':
    main()
