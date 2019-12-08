import numpy as np
from matplotlib import pyplot as plt
import  argparse


def main():
	folder = args.folder

	q_nway_accuracies = np.genfromtxt(folder + '/q_nway_train_accuracies.txt')[:,-1]
	q_nway_epochs = np.genfromtxt(folder + '/q_nway_train_accuracies.txt')[:,0]
	q_discrim_accuracies = np.genfromtxt(folder + '/q_discrim_train_accuracies.txt')[:,-1]
	q_discrim_epochs = np.genfromtxt(folder + '/q_nway_train_accuracies.txt')[:,0]
	gen_nway_accuracies = np.genfromtxt(folder + '/gen_nway_train_accuracies.txt')[:,-1]
	gen_nway_epochs = np.genfromtxt(folder + '/q_nway_train_accuracies.txt')[:,0]
	gen_discrim_accuracies = np.genfromtxt(folder + '/gen_discrim_train_accuracies.txt')[:,-1]
	gen_discrim_epochs = np.genfromtxt(folder + '/q_nway_train_accuracies.txt')[:,0]
	if args.nway_test:
		nway_test_accuracies = np.genfromtxt(folder + '/q_nway_test_accuracies.txt')[:,-1]
		nway_test_epochs = np.genfromtxt(folder + '/q_nway_train_accuracies.txt')[:,0]


	if args.q_nway:
		plt.plot(q_nway_epochs, q_nway_accuracies, label = 'query nway')
		# plt.label("q_nway")
	if args.q_discrim:
		plt.plot(q_nway_epochs, q_discrim_accuracies, label = 'query discrim')
		# plt.label("q_nway")
	if args.gen_nway:
		plt.plot(q_nway_epochs, gen_nway_accuracies, label = 'gen nway')
		# plt.label("q_nway")
	if args.gen_discrim:
		plt.plot(q_nway_epochs, gen_discrim_accuracies, label = 'gen discrim')
		# plt.label("q_nway")
	if args.nway_test:
		plt.plot(nway_test_epochs, nway_test_accuracies, label = 'nway_test')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	# eg. python plot_results.py --folder='2019-12-01 06:21:00_omni' --q_nway --q_discrim --gen_nway --gen_discrim
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--folder', type=str, help='folder with model accs to plot', required=True)
    argparser.add_argument('--q_nway', default=False, action='store_true', help='Bool type. Pass to plot q_nway accs')
    argparser.add_argument('--q_discrim', default=False, action='store_true', help='Bool type. Pass to plot q_discrim accs')
    argparser.add_argument('--gen_nway', default=False, action='store_true', help='Bool type. Pass to plot gen_nway accs')
    argparser.add_argument('--gen_discrim', default=False, action='store_true', help='Bool type. Pass to plot gen_discrim accs')
    argparser.add_argument('--nway_test', default=False, action='store_true', help='Bool type. Pass to plot gen_discrim accs')
    args = argparser.parse_args()

    main()