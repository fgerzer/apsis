__author__ = 'Frederik Diehl'

# Code for the NN adapted from the breze library example.
# For Breze, see github.com/breze-no-salt/breze.

import cPickle
import gzip
import time

import numpy as np
import theano.tensor as T

import climin.schedule

import climin.stops
import climin.initialize

from breze.learn.mlp import Mlp
from breze.learn.data import one_hot

from apsis.models.parameter_definition import *
from apsis.assistants.lab_assistant import ValidationLabAssistant
from apsis.utilities.logging_utils import get_logger

logger = get_logger("apsis.demos.demo_MNIST_NN")

start_time = None

def load_MNIST():
    datafile = 'mnist.pkl.gz'
    # Load data.

    with gzip.open(datafile,'rb') as f:
        train_set, val_set, test_set = cPickle.load(f)

    X, Z = train_set
    VX, VZ = val_set
    TX, TZ = test_set

    Z = one_hot(Z, 10)
    VZ = one_hot(VZ, 10)
    TZ = one_hot(TZ, 10)
    image_dims = 28, 28
    return X, Z, VX, VZ, TX, TZ, image_dims

def do_one_eval(X, Z, VX, VZ, step_rate, momentum, decay, c_wd):
    max_passes = 100
    batch_size = 250
    max_iter = max_passes * X.shape[0] / batch_size
    n_report = X.shape[0] / batch_size
    optimizer = 'rmsprop', {'step_rate': step_rate, 'momentum': momentum, 'decay': decay}
    #optimizer = 'adam'
    #optimizer = 'gd', {'steprate': 0.1, 'momentum': climin.schedule.SutskeverBlend(0.99, 250), 'momentum_type': 'nesterov'}
    m = Mlp(784, [800], 10, hidden_transfers=['sigmoid'], out_transfer='softmax', loss='cat_ce',
            optimizer=optimizer, batch_size=batch_size)
    climin.initialize.randomize_normal(m.parameters.data, 0, 1e-1)
    losses = []
    weight_decay = ((m.parameters.in_to_hidden**2).sum()
                + (m.parameters.hidden_to_out**2).sum())
    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = c_wd
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay
    n_wrong = 1 - T.eq(T.argmax(m.exprs['output'], axis=1), T.argmax(m.exprs['target'], axis=1)).mean()
    f_n_wrong = m.function(['inpt', 'target'], n_wrong)
    stop = climin.stops.AfterNIterations(max_iter)
    pause = climin.stops.ModuloNIterations(n_report)

    start = time.time()
    # Set up a nice printout.
    keys = '#', 'seconds', 'loss', 'val loss', 'train emp', 'val emp'
    max_len = max(len(i) for i in keys)
    header = '\t'.join(i for i in keys)
    #print header
    #print '-' * len(header)

    for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause)):
        passed = time.time() - start
        losses.append((info['loss'], info['val_loss']))

        #img = tile_raster_images(fe.parameters['in_to_hidden'].T, image_dims, feature_dims, (1, 1))
        #save_and_display(img, 'filters-%i.png' % i)
        info.update({
            'time': passed,
            'train_emp': f_n_wrong(X, Z),
            'val_emp': f_n_wrong(VX, VZ),
        })
        row = '%(n_iter)i\t%(time)g\t%(loss)g\t%(val_loss)g\t%(train_emp)g\t%(val_emp)g' % info
        #print row
    return info["val_emp"]

def do_evaluation(LAss, opt, X, Z, VX, VZ):
    to_eval = LAss.get_next_candidate(opt)
    step_rate = to_eval.params["step_rate"]
    momentum = to_eval.params["momentum"]
    decay = to_eval.params["decay"]
    c_wd = to_eval.params["c_wd"]
    result = do_one_eval(X, Z, VX, VZ, step_rate, momentum, decay, c_wd)
    to_eval.result = result
    LAss.update(opt, to_eval)


def demo_on_MNIST(random_steps, steps, cv=1):
    X, Z, VX, VZ, TX, TZ, image_dims = load_MNIST()

    param_defs = {
        #"step_rate": MinMaxNumericParamDef(0, 1),
        "step_rate": AsymptoticNumericParamDef(0, 1),
        #"momentum": MinMaxNumericParamDef(0, 1),
        "momentum": AsymptoticNumericParamDef(1, 0),
        'decay': MinMaxNumericParamDef(0, 1),
        "c_wd": MinMaxNumericParamDef(0, 1)
    }

    LAss = ValidationLabAssistant(cv=cv)
    experiments = ["random_mnist", "bay_mnist_ei_L-BFGS-B"]#, "bay_mnist_ei_rand"]
    LAss.init_experiment("random_mnist", "RandomSearch", param_defs, minimization=True)
    #LAss.init_experiment("bay_mnist_ei_rand", "BayOpt", param_defs,
    #                     minimization=True, optimizer_arguments=
    #    {"acquisition_hyperparams":{"optimization": "random"}})
    global start_time
    start_time = time.time()
    #First, the random steps
    for i in range(random_steps*cv):
        print("%s\tBeginning with random initialization. Step %i/%i" %(str(time.time()-start_time), i, random_steps*cv))
        do_evaluation(LAss, "random_mnist", X, Z, VX, VZ)

    #clone
    #LAss.init_experiment("bay_mnist_ei_L-BFGS-B", "BayOpt", param_defs, minimization=True)
    LAss.clone_experiments_by_name(exp_name=experiments[0], new_exp_name=experiments[1],
                               optimizer="BayOpt",
                               optimizer_arguments={"initial_random_runs": random_steps})

    #learn the rest
    for i in range((steps-random_steps)*cv):
        for opt in experiments:
            print("%s\tBeginning with %s, step %i/%i" %(time.time() - start_time, opt, i+1+random_steps*cv, steps*cv))
            do_evaluation(LAss, opt, X, Z, VX, VZ)

    for opt in experiments:
        logger.info("Best %s score:  %s" %(opt, [x.result for x in LAss.get_best_candidates(opt)]))
        print("Best %s score:  %s" %(opt, [x.result for x in LAss.get_best_candidates(opt)]))
    LAss.plot_result_per_step(experiments, title="Neural Network on MNIST.", plot_min=0.0, plot_max=1.0)

if __name__ == '__main__':
    demo_on_MNIST(10, 30, 5)