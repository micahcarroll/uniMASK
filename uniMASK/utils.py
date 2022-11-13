

import cProfile
import inspect
import io
import itertools
import json
import os
import pstats
import shutil
from collections import Iterable, OrderedDict, defaultdict, Counter
from logging import warning

import numpy as np
import pickle5 as pickle
import torch
from matplotlib import pyplot as plt


def profile(fnc):
    """A decorator that uses cProfile to profile a function (from https://osf.io/upav8/)"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())

        filename = os.path.expanduser(os.path.join("~", fnc.__name__ + ".pstat"))
        print(filename)
        pr.dump_stats(filename)

        return retval

    return inner


#########
# UTILS #
#########


class imdict(dict):
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError("object is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


def item_xy_from_theta(theta):
    # Get xy of item from theta
    theta = np.array(theta)
    return np.round([np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)], 8).T


def Ut_from_theta(theta, weighing=None):
    # Get xy of human preference from theta
    theta = np.array(theta)
    theta = np.round([np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)], 8)
    if weighing is not None:
        theta *= weighing
    return theta.T


def get_item_values(theta, weighing=None):
    """
    [num_thetas x 2] x [num_items x 2].T
    returns: [num_thetas x num_items]
    """
    return Ut_from_theta(theta, weighing) @ item_xy_from_theta(theta).T


def prop_of_normal_mass_in_interval(mean, std, interval):
    # Importing within the util to keep the file imports be low-load
    from scipy import stats

    # Technically we should be using a circular distribution like the von Mises distribution,
    # but if standard deviation is small enough, it's fine.
    assert all(np.abs(interval) <= 360) and 0 <= mean < 360, "{}, {}, {}, {}, {}, {}".format(
        mean,
        std,
        interval,
        all(np.abs(interval) <= 360),
        0 <= mean < 360,
        np.abs(interval) <= 360,
    )
    ci = np.array(interval) - mean

    if not all(np.abs(ci) <= 180):
        ci = np.array([item - 360 if item > 180 else item for item in ci])
        ci = np.array([item + 360 if item < -180 else item for item in ci])

    dist = stats.norm(0, std)

    if ci[1] < ci[0]:
        assert ci[0] >= 0 >= ci[1]
        mass = prop_of_normal_mass_in_interval(0, std, (-180, ci[1])) + prop_of_normal_mass_in_interval(
            0, std, (ci[0], 180)
        )
    else:
        mass = dist.cdf(ci[1]) - dist.cdf(ci[0])

    return mass


def symm_kl_divergence(p, q):
    # Importing within the util to keep the file imports be low-load
    from scipy import special

    return np.sum(special.kl_div(p, q), axis=-1) + np.sum(special.kl_div(q, p), axis=-1)


def argmax_set(lst):
    argmaxes = np.argwhere(lst == np.max(lst)).ravel()
    return argmaxes, lst[argmaxes[0]]


def lazy_property(fn):
    """
    Decorator that makes a property lazy-evaluated.
    From https://stevenloria.com/lazy-properties/
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


def save_pickle(data, filename):
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename + ".pkl", "rb") as f:
        return pickle.load(f)


def to_numpy(x):
    if type(x) in [float, int, list, np.ndarray, np.float64]:
        return x
    return x.cpu().detach().numpy()


def delete_dir_if_exists(dir_path, verbose=False):
    if os.path.exists(dir_path):
        if verbose:
            print("Deleting old dir", dir_path)
        shutil.rmtree(dir_path)


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def clip_dict_values(d, max_val):
    return {k: max(min(max_val, v), -max_val) for k, v in d.items()}


def std_err(lst):
    """Computes the standard error"""
    sd = np.std(lst)
    n = len(lst)
    return sd / np.sqrt(n)


def mean_and_std_err(lst):
    """Mean and standard error of list"""
    mu = np.mean(lst)
    return mu, std_err(lst)


def dict_mean_and_std_err(d):
    """
    Takes in a dictionary with lists as keys, and returns a dictionary
    with mean and standard error for each list as values
    """
    assert all(isinstance(v, Iterable) for v in d.values())
    result = {}
    for k, v in d.items():
        result[k] = mean_and_std_err(v)
    return result


def have_same_keys(ds):
    return all(set(d.keys()) == set(ds[0].keys()) for d in ds)


def append_dictionaries(dictionaries):
    """
    Append many dictionaries with numbers as values into one dictionary with lists as values.
    {a: 1, b: 2}, {a: 3, b: 0}  ->  {a: [1, 3], b: [2, 0]}
    """
    assert have_same_keys(dictionaries)
    final_dict = defaultdict(list)
    for d in dictionaries:
        for k, v in d.items():
            final_dict[k].append(v)
    return dict(final_dict)


def merge_dictionaries(dictionaries):
    """
    Merge many dictionaries by extending them to one another.
    {a: [1, 7], b: [2, 5]}, {a: [3], b: [0]}  ->  {a: [1, 7, 3], b: [2, 5, 0]}
    """
    assert have_same_keys(dictionaries)
    final_dict = defaultdict(list)
    for d in dictionaries:
        for k, v in d.items():
            final_dict[k].extend(v)
    return dict(final_dict)


def cat_ordered_dicts(*ordered_dicts):
    items_n = [list(od.items()) for od in ordered_dicts]
    return OrderedDict(list(itertools.chain.from_iterable(items_n)))


def average_dictionaries(dictionaries):
    """
    Merge many dictionaries by averaging them with each other.
    """
    assert have_same_keys(dictionaries)
    n = len(dictionaries)
    final_dict = defaultdict(float)
    for d in dictionaries:
        for k, v in d.items():
            final_dict[k] += v / n
    return dict(final_dict)


def angle_to_polar(thetas):
    return (np.array(thetas) / 360) * 2 * np.pi


def fix_filetype(path, filetype):
    if path[-len(filetype) :] == filetype:
        return path
    else:
        return path + filetype


def save_as_json(data, filename):
    with open(fix_filetype(filename, ".json"), "w") as outfile:
        json.dump(data, outfile)
    return filename


def load_from_json(filename):
    with open(fix_filetype(filename, ".json"), "r") as json_file:
        return json.load(json_file)


def set_style(font_scale=1.6, tex=True):
    import matplotlib
    import seaborn

    seaborn.set(font="serif", font_scale=font_scale)
    # Make the background white, and specify the specific font family
    seaborn.set_style(
        "white",
        {
            "font.family": "serif",
            "font.weight": "normal",
            "font.serif": ["Times", "Palatino", "serif"],
            "axes.facecolor": "white",
            "lines.markeredgewidth": 1,
        },
    )
    matplotlib.rcParams["text.usetex"] = tex
    matplotlib.rc("font", family="serif", serif=["Palatino"])


def make_dot(var, params):
    """Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad
    """
    import torch
    from graphviz import Digraph

    param_map = {id(v): k for k, v in params.items()}
    print(param_map)

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
            elif hasattr(var, "variable"):
                u = var.variable
                node_name = "%s\n %s" % (
                    param_map.get(id(u)),
                    size_to_str(u.size()),
                )
                dot.node(str(id(var)), node_name, fillcolor="lightblue")
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


def logsumexp(a, b):
    # Importing torch within the util to keep the file imports be low-load
    import torch

    # Transform to log space
    a = torch.log(a)
    b = torch.log(b)

    # Sum in log space, which is equivalent to mutliplying
    c = a + b

    # Exponentiate back
    return torch.exp(c)


def manhattan_dist(pos0, pos1):
    assert len(pos0) == len(pos1) == 2
    pos0, pos1 = np.array(pos0), np.array(pos1)
    return np.abs(pos0 - pos1).sum()


def format_str_to_red(s):
    """Formats the text in such a way that it prints red in a notebook"""
    return "\x1b[31m{}\x1b[0m".format(s)


def get_class_attributes(cls):
    # https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
    attributes = inspect.getmembers(cls, lambda a: not (inspect.isroutine(a)))
    return [a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))]


def get_inheritors(klass):
    """
    Recursively lists all subclasses of a class. https://stackoverflow.com/a/5883218

    Args:
        klass (type): A class.

    Returns:
        [type]: List of all subclasses, and their subclasses, and so on (excluding klass).
    """
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


#############
# GPU UTILS #
#############
def get_freer_gpu():
    """
    This util is to try to figure out which GPU is most free, and automatically assign the current computation
    to that GPU. This probably only works with nvidia-smi GPUs.
    """
    import torch

    if torch.cuda.device_count() == 1:
        # HACK: If there's only one GPU, assign idx 0. This is mostly in order to not have errors if one is trying to
        # use CUDA_VISIBLE_DEVICES with this function. Without this if statement, the program will be able to see that
        # there are other GPUs with nvidia-smi and try to use one that is not actually available to python!
        return 0

    os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    assert len(memory_available) > 0, "This is probably due to your "
    return np.argmax(memory_available)


def determine_default_torch_device(local):
    if local:
        _device_code = "cpu"
    else:
        try:
            GPU_ID = get_freer_gpu()
        except:
            warning("Was not able to auto-assign the most free GPU to the job. Defaulting to the GPU 0")
            GPU_ID = 0
        _device_code = "cuda:{}".format(GPU_ID)
    return _device_code


class Categorical:
    """
    A class to represent categorical distributions, i.e. probabilities over buckets.

    Internally, these are represented as dictionaries where the keys constitute the domain and the values are the probs.
    """

    def __init__(self, d):
        assert np.allclose([sum(d.values())], 1)
        self.distr = d
        self.domain = np.array(list(d.keys()))
        self.probs = np.array(list(d.values()))

    def prob(self, x):
        return self.distr[x]

    def __repr__(self):
        return repr(self.distr)

    def __hash__(self):
        return hash((tuple(self.domain), tuple(self.probs)))

    def __eq__(self, other):
        return np.allclose(self.probs, other.probs) and np.array_equal(self.domain, other.domain)

    @property
    def nonzero_domain(self):
        return np.array([theta for theta, prob in self.distr.items() if prob > 0])

    def reweigh(self, weights):
        """Re-weigh categorical's probabilities"""
        return Categorical.from_unnormalized_probs(self.probs * weights)

    @staticmethod
    def average(categoricals, weights=None):
        if weights is None:
            weights = [1] * len(categoricals)
        assert len(categoricals) == len(weights)
        assert all(issubclass(type(item), Categorical) for item in categoricals), "{}".format(
            (
                [type(item) for item in categoricals],
                [issubclass(type(item), Categorical) for item in categoricals],
            )
        )
        assert all(np.array_equal(categoricals[0].domain, item.domain) for item in categoricals)
        new_distr = {}
        for k in categoricals[0].domain:
            new_distr[k] = sum([item.distr[k] * weight for item, weight in zip(categoricals, weights)])
        return Categorical._from_unnormalized(new_distr)

    @staticmethod
    def _from_unnormalized(distr):
        normaliz_const = sum(distr.values())
        return Categorical({k: v / normaliz_const for k, v in distr.items()})

    @staticmethod
    def from_domain_and_probs(domain, probs):
        return Categorical({d: p for d, p in zip(domain, probs)})

    @staticmethod
    def from_unnormalized_probs(domain, probs):
        d = {d: p for d, p in zip(domain, probs)}
        return Categorical._from_unnormalized(d)

    @staticmethod
    def from_obs(domain, obs):
        counts = {k: 0 for k in domain}
        counts.update(dict(Counter(obs)))
        return Categorical._from_unnormalized(counts)

    def sample(self, size=1):
        samples = np.random.choice(self.domain, p=self.probs, size=size, replace=True)
        if size == 1:
            return samples[0]
        return samples

    def hist_plot(self, title="", ax=None):
        n = len(self.domain)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.bar(self.domain, self.probs, width=360 / (n + 1), color="g")
        ax.set_xticks(np.arange(0, 360, 360 / 12))
        ax.set_title(title)

    def plot(self, label=None, title="", ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.domain, self.probs, label=label)
        ax.set_title(title)
        ax.set_ylim(0, None)

    @staticmethod
    def plot_multiple(categoricals, slider_label="timestep"):
        from IPython.display import display
        from ipywidgets import IntSlider, interactive

        def display_plot(**kwargs):
            categoricals[kwargs[slider_label]].plot()

        display(interactive(display_plot, **{slider_label: IntSlider(min=0, max=len(categoricals) - 1, step=1)}))

    @property
    def argmax(self):
        idx = np.argmax(self.probs)
        return self.domain[idx]


LOCAL = not torch.cuda.is_available()
DEVICE = torch.device(determine_default_torch_device(LOCAL))


def go_to_cpu():
    """Switches all computation to happen on CPU"""
    import torch

    global DEVICE
    DEVICE = torch.device("cpu")
    print("Switching to", DEVICE)


def back_to_default():
    """Switches all computation to happen on the default device"""
    assert False, "Currently this code doesn't work"
    global DEVICE
    DEVICE = torch.device("cpu" if LOCAL else "cuda:{}".format(GPU_ID))
    print("Back to default", DEVICE)
