import os
from matplotlib import pyplot as plt

class Plotter():
	def __init__(self):
		self.stats = {}

	def add(self, **kwargs):
		for k, v in kwargs.items():
			if k not in self.stats:
				self.stats[k] = []

			self.stats[k].append(v)

	def build_plot(self, suptitle=None, subplots=None, figsize=(10, 10), **kwargs):
		if subplots == None:
			fig, ax = plt.subplots(1, len(self.stats), figsize=figsize)
		else:
			fig, ax_array = plt.subplots(*subplots, figsize=figsize)
			ax = ax_array.flatten()

		if suptitle != None:
			fig.suptitle(suptitle)

		for i, (k, v) in enumerate(self.stats.items()):
			ax[i].plot(v, label=k, **kwargs)

			ax[i].legend(loc="upper left")

	def output(self, fp="results.png", suptitle=None, subplots=None, figsize=(10, 10), **kwargs):
		self.build_plot(suptitle=suptitle, subplots=subplots, figsize=figsize, **kwargs)

		plt.savefig(fp)
		plt.close()

	def output_show(self, suptitle=None, subplots=None, figsize=(10, 10), **kwargs):
		self.build_plot(suptitle=suptitle, subplots=subplots, figsize=figsize, **kwargs)

		plt.show()

class Experiment():
	def __init__(self, experiment_name, allow_replace=False, config_file='config.txt', plot_file='plot.png', folder='experiments', **kwargs):
		self.experiment_name = experiment_name

		self.folder = folder

		if not os.path.isdir(folder):
			os.mkdir(folder)

		self.experiment_path = os.path.join(self.folder, self.experiment_name)

		if os.path.isdir(self.experiment_path) and not allow_replace:
			raise Exception("Experiment at %s already exists" % self.experiment_path)

		if not os.path.isdir(self.experiment_path):
			os.mkdir(self.experiment_path)

		with open(os.path.join(self.experiment_path, config_file), 'w') as file_out:
			file_out.write(str(kwargs))
		
		self.plotter = Plotter()
		self.plot_file = plot_file

	def log(self, *args, log_file='log.txt', cmd=True, cwd_func=print):
		log_path = os.path.join(self.experiment_path, log_file)

		line = ' '.join([str(x) for x in args])

		with open(log_path, 'a') as file_out:
			file_out.write(line + "\n")

		if cmd:
			cwd_func(line)
	
	def log_stats(self, num, log_file='log.txt', cmd=True, plot=True, cwd_func=print, plot_kwargs={}, **kwargs):
		log_string = f'{num}: {str(kwargs)}'
		
		if log_file != None:
			self.log(log_string, log_file=log_file, cmd=cmd, cwd_func=cwd_func)
		
		if plot:
			self.plotter.add(**kwargs)
			self.plotter.output(fp=os.path.join(self.experiment_path, self.plot_file), **plot_kwargs)	
			
class RunningAvg:
	def __init__(self, buffer_size, default=None):
		self.buffer = [default] * buffer_size
		self.idx = 0
		self.buffer_size = buffer_size
	
	def __call__(self, x):
		self.buffer[self.idx] = x
		self.idx = (self.idx + 1) % self.buffer_size
		return self.none_avg()
	
	def none_avg(self):
		return sum([b for b in self.buffer if b != None]) / self.buffer_size