

from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


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
