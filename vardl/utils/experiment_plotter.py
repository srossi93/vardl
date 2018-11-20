import matplotlib
import tensorflow as tf

matplotlib.use('agg')
import matplotlib.pylab as plt
import glob
import pandas as pd

import matplotlib2tikz


class ExperimentPlotter():
    def __init__(self, name, basedir, methods, savepath):
        self.name = name
        self.basedir = basedir
        self.methods = methods
        self.raw_data = {}
        self.aggr_data = {}
        self.savepath = savepath
        self.upper_clip = 100

    def parse(self):
        for method in self.methods:
            filename_runs = self._get_runs_in_dir(self.basedir + '/%s/' % method)

            df = self._get_results_runs(filename_runs)
            self.raw_data[method] = df

    def plot_tag(self, method, tag, ax, label, linestyle='-'):
        data = self.raw_data[method]

        filtered_data = data[data.tag == tag]


        filtered_data.value = filtered_data.value.where(filtered_data.value <= self.upper_clip, self.upper_clip)

        #print(filtered_data)

        aggr_data = filtered_data.groupby('step')

#        aggr_data['value'] = aggr_data['value'].rolling(window=10, min_periods=2, center=True).mean()

#        print(aggr_data)
        #logscale = np.logspace(0, 5, 1000, endpoint=True)
        #print(logscale)
        
        #print(aggr_data.mean())
        if tag == 'model/dkl':
            window = 1000
        else:
            window = 1
        steps = aggr_data.mean().rolling(window, min_periods=1, center=True).mean().index.tolist()
        #print(steps)
        means = aggr_data.mean().rolling(window, min_periods=1, center=True).mean()['value']
        stds = aggr_data.std().rolling(window, min_periods=1, center=True).mean()['value']

        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(steps, means, label=label, color=color, alpha=1, linestyle=linestyle)
        ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.1)

        # return means.tolist(), stds.tolist()

    def plot(self, tags, xlim, ylims, logx: bool, save: bool):

        fig, axs = plt.subplots(len(tags), 1)

        for i, tag in enumerate(tags):

            for method in self.methods:
                if method == 'mcd':
                    linestyle = '--'
                else:
                    linestyle = '-'
                self.plot_tag(method=method,
                              tag=tag,
                              ax=axs[i],
                              label=method.upper(),
                              linestyle=linestyle)
            axs[i].set_ylim(*ylims[i])
            axs[i].set_xlim(*xlim)
            axs[i].semilogx() if logx else 0
            axs[i].set_ylabel(tag.upper())

        fig.tight_layout()
        axs[0].set_title(self.name.upper())
        axs[-1].set_xlabel('Step')

        if save:
            path = self.savepath + self.name
            self.savefig(path, 'pdf')
            self.savefig(path, 'tex')

        return fig, axs


    def write_summary_tex(self, tags, step=-1):
        summary = ''
        for tag in tags:
            summary += '==== %s ====\n' % tag.upper()

            for method in self.methods:
                data = self.raw_data[method]

                aggr_data = data[data.tag == tag].groupby('step')

                steps = aggr_data.mean().index.tolist()
                final_mean = aggr_data.mean()['value'].iloc[step]
                final_std = aggr_data.std()['value'].iloc[step]

                summary += "%s: $%.4f \\pm %.3f$ at %d step\n" % (method, final_mean, final_std, steps[step])

            summary += '\n\n'

        savepath = self.savepath + '/' + self.name + '-summary.txt'
        with open(savepath, 'w') as fd:
            fd.write(summary)

        print(summary)
        return


    def _get_runs_in_dir(self, path: str):
        return glob.glob('%s/**/events.out.tfevents*' % path, recursive=True)

    def _get_results_runs(self, filename_runs):
        frames = []

        for i, filename_run in enumerate(filename_runs):
            df = self._parse_tfevents_file(filename_run)
            df['run'] = i
            frames.append(df)

        df = pd.concat(frames)
        return df

    def _parse_tfevents_file(self, filename: str) -> pd.DataFrame:
        print('INFO - Parsing %s' % filename)
        events = []
        for i, event in enumerate(tf.train.summary_iterator(filename)):
            if i == 0:
                continue

            scalar = list(event.summary.value)[0]

            dicti = {'wall_time': event.wall_time,
                     'step': event.step + 1,
                     'tag': str(scalar.tag),
                     'value': scalar.simple_value}
            events.append(dicti)
        return pd.DataFrame(events)


    @staticmethod
    def savefig(path, ext='pdf', close=False, verbose=True):
        #path = self.savepath + self.name
        import os
        # Extract the directory and filename from the given path
        directory = os.path.split(path)[0]
        filename = "%s.%s" % (os.path.split(path)[1], ext)
        if directory == '':
            directory = '.'

        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # The final path to save to
        savepath = os.path.join(directory, filename)

        if verbose:
            print("INFO - Saving figure to '%s'..." % savepath)

        # Actually save the figure
        if ext == 'tex':
            tikz_code = matplotlib2tikz.get_tikz_code(savepath, figureheight='\\figureheight',
                                                      figurewidth='\\figurewidth')

            wide_to_ascii = dict((i, chr(i - 0xfee0)) for i in range(0xff01, 0xff5f))
            wide_to_ascii.update({0x3000: u' ', 0x2212: u'-'})  # space and minus
            tikz_code = tikz_code.translate(wide_to_ascii)


            tikz_code = tikz_code.encode('ascii')
            with open(savepath, 'wb') as fd:
                fd.write(tikz_code)

            # tikz_save(savepath, figureheight='\\figureheight', figurewidth='\\figurewidth')
        else:
            plt.savefig(savepath)

        # Close it
        if close:
            plt.close()