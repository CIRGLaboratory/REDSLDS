import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
_colors = [
    "#ff0000", #red
    "#00ff00", #green
    "#0000ff", #blue
    "#ffff00", #yellow
    "#13eac9", #aqua
    "#ffa500", #orange
    "#ff00ff", #fuchsia
    "#aaa662", #khaki
    "#808080", #grey
    "#e6daa6", #beige
    "#4b0082", #indigo
    "#580f41", #plum
    "#d1b26f", #tan
    "#ff6347", #tomato
    "#bbf90f", #yellowgreen
    "#fac205", #goldenrod
    "#069af3", #azure
    "#a52a2a", #brown
    "#00008b", #darkblue
    "#add8e6", #lightblue
    "#76ff7b", #lightgreen
    "#a9561e", #sienna
    "#ff796c", #salmon
    "#808000", #olive
    "#000000", #black
    "#8a2be2", #blueviolet
    "#7fff00", #chartreuse
    "#d2691e", #chocolate
    "#dc143c", #crimson
    "#00ffff", #cyan
    "#00bfff", #deepskyblue
    "#1e90ff", #dodgerblue
    "#b22222", #firebrick
    "#228b22", #forestgreen
    "#daa520", #goldenrod
    "#adff2f", #greenyellow
    "#ff69b4", #hotpink
    "#cd5c5c", #indianred
    "#f0e68c", #khaki
    "#e6e6fa", #lavender
    "#7cfc00", #lawngreen
    "#ffe4b5", #moccasin
    "#ffdead", #navajowhite
    "#fa8072", #salmon
    "#f4a460", #sandybrown
    "#2e8b57", #seagreen
    "#a0522d", #sienna
    "#87ceeb", #skyblue
    "#4682b4", #steelblue
    "#d8bfd8", #thistle
    "#40e0d0", #turquoise
    "#ee82ee", #violet
    "#f5deb3", #wheat
    "#9acd32", #yellowgreen
    "#8b4513", #saddlebrown
    "#32cd32", #limegreen
    "#800000", #maroon
    "#b0c4de", #lightsteelblue
    "#bc8f8f", #rosybrown
    "#9370db", #mediumpurple
    "#fffafa", #snow
    "#db7093", #palevioletred
    "#556b2f", #darkolivegreen
    "#00ced1", #darkturquoise
    "#8b0000", #darkred
    "#008080", #teal
    "#66cdaa", #mediumaquamarine
    "#9932cc", #darkorchid
    "#afeeee", #paleturquoise
    "#5f9ea0", #cadetblue
    "#7b68ee", #mediumslateblue
    "#6495ed", #cornflowerblue
    "#ff1493", #deeppink
    "#c71585", #mediumvioletred
    "#ff7f50", #coral
    "#ffc0cb", #pink
    "#fffacd", #lemonchiffon
    "#7fffd4", #aquamarine
    "#800080", #purple
    "#bdb76b"  #darkkhaki
]


def plot_dynamics(X, Y, Z, U, V, states_n, path, figsize=(6, 6), xlabel='x', ylabel='y', title='Dynamics'):
    plt.figure(figsize=figsize)
    minx = np.min(X)
    maxx = np.max(X)
    miny = np.min(Y)
    maxy = np.max(Y)
    plt.xlim(minx-1, maxx+1)
    plt.ylim(miny-1, maxy+1)
    for i in range(states_n):
        plt.scatter(X[Z == i], Y[Z == i], color=_colors[i])
        x, y = (X[Z == i], Y[Z == i])
        u, v = (U[Z == i], V[Z == i])
        plt.quiver(x, y, u, v, color=_colors[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path)
    if "pdf" not in path:
        path = path[:-3] + "pdf"
        plt.savefig(path)
    plt.close()


def plot_emission(Y, Z, states_n, path, figsize=(6, 6), xlabel='x', ylabel='y', title='Emission', x0=0, x1=1):
    plt.figure(figsize=figsize)
    Y = Y[:,[x0, x1]]
    minx = np.min(Y, axis=0)
    maxx = np.max(Y, axis=0)
    plt.xlim(minx[0]-1, maxx[0]+1)
    plt.ylim(minx[1]-1, maxx[1]+1)
    for i in range(states_n):
        plt.scatter(Y[Z == i, 0], Y[Z == i, 1], color=_colors[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path)
    if "pdf" not in path:
        path = path[:-3] + "pdf"
        plt.savefig(path)
    plt.close()


def plot_phase_portrait(X, Z, As, states_n, path, figsize=(6, 6), xlabel='x', ylabel='y', title='Dynamics', x0=0,
                        x1=1):
    U = np.zeros((X.shape[0], As[0].shape[0]))
    for i in range(states_n):
        indices = Z == i
        U[indices] = X[indices] @ As[i].T - X[indices,:As[i].shape[0]]
    X, Y = X[:, x0], X[:, x1]
    U, V = U[:, x0], U[:, x1]
    plot_dynamics(X, Y, Z, U, V, states_n, path, figsize=figsize, xlabel=xlabel, ylabel=ylabel, title=title)


def plot_actual_shifts(X, Z, states_n, path, figsize=(6, 6), xlabel='x', ylabel='y', title='Dynamics', x0=0, x1=1):
    Z = Z[:-1]
    U, V = X[1:, x0], X[1:, x1]
    X, Y = X[:-1, x0], X[:-1, x1]
    plot_dynamics(X, Y, Z, U, V, states_n, path, figsize=figsize, xlabel=xlabel, ylabel=ylabel, title=title)


def plot_distribution_shift(X, Y, path, name_X="X", name_Y="Y", title="PairPlot"):
    df = [
        pd.DataFrame(X, columns=[f"X_{i}" for i in range(X.shape[1])]),
        pd.DataFrame(Y, columns=[f"X_{i}" for i in range(Y.shape[1])])
    ]
    df[0]["name"] = name_X
    df[1]["name"] = name_Y
    df = pd.concat(df).reset_index(drop=True)
    fig = sns.pairplot(df, hue="name")
    fig.set(title=title)
    fig.savefig(path)
    if "pdf" not in path:
        path = path[:-3] + "pdf"
        plt.savefig(path)
    plt.close(fig.fig)


def plot_durations(Z, D, states_num, title="Count"):
    counts = [[] for _ in range(states_num)]
    prev = None
    for i, z in enumerate(Z):
        if z != prev:
            counts[z].append(D[i])
        prev = z
    fig, ax = plt.subplots(states_num, 1)
    for i, count in enumerate(counts):
        ax[i].hist(count)
    fig.suptitle(title)
    return fig, ax

def plot_observations(z, y, title=None, fname=None, ax=None, ls="-", lw=1):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(20, 10), dpi=80)
        ax = fig.gca()
    if len(y.shape) == 1:
        y = y[:, np.newaxis]
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=lw, ls=ls,
                    color=_colors[z[start] % len(_colors)],
                    alpha=1.0)
    if title:
        ax.set_title(title)
    if fname:
        plt.savefig(fname)
    if "pdf" not in fname:
        path = fname[:-3] + "pdf"
        plt.savefig(path)
    return ax
