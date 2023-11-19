import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display


class Animator:
    def __init__(self, xlabel=None,
                 ylabel=None,
                 legend=None,
                 xlim=None,
                 ylim=None,
                 xscale="linear",
                 yscale="linear",
                 title=None,
                 fmts=('-', 'm--', 'g-.', 'r:'), # line styles
                 nrows=1,
                 ncols=1,
                 figsize=(3.5, 2.5)):
        
        # set legends
        if legend is None:  
            legend = []

        # set svg display
        backend_inline.set_matplotlib_formats("svg")

        # set figure and axes
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)  

        # convert axes to list
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        # set attributes
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.title = title
        self.legend = legend

        self.X, self.Y, self.fmts = None, None, fmts
    

    def config_axes(self, axes):
        """Set the axes for matplotlib."""
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.set_xlim(self.xlim),
        axes.set_ylim(self.ylim)
        
        if self.title:
            axes.set_title(self.title + "..")
        if self.legend:
            axes.legend(self.legend)

        axes.grid()

    def add(self, x, y):
        """
        Add multiple data points into the figure.
        x: scalar or list
        y: scalar or list
        """
        # convert y to list
        if not hasattr(y, "__len__"): 
            y = [y]

        n = len(y)

        # convert x to list
        if not hasattr(x, "__len__"):
            x = [x] * n

        if not self.X:
            self.X = [[] for _ in range(n)]

        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        self.axes[0].cla()

        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)

        self.config_axes(self.axes[0])

        display.display(self.fig)
        display.clear_output(wait=True)
