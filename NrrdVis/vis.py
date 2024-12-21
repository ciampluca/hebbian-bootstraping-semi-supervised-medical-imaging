import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nrrd

DATA_PATH = 'data'
SAVE_PATH = 'results'
THR = (1, 50)
TARGET_THR = 0.5
SHAPE = (32, 32, 32)
COLOR = (1, 0, 0)
TARGET_COLOR = (0, 1, 0)
BOUNDS = None #(20, 30)
SHADE = (0.1, 0.9)
ALPHA = (0.6, 0.8)
#FILTER = (32, 1)

def resize(data, out_shape=SHAPE):
	in_shape = data.shape
	bin_size = [sh // out_shape[i] for i, sh in enumerate(in_shape)]
	data = data[:out_shape[0]*bin_size[0], :out_shape[1]*bin_size[1], :out_shape[2]*bin_size[2]]
	return data.reshape((out_shape[0], bin_size[0], out_shape[1], bin_size[1], out_shape[2], bin_size[2])).mean((1, 3, 5))

def normalize(data, alpha=ALPHA, bounds=None):
	if bounds is None: bounds = (data.min(), data.max())
	#return (np.tanh((data - FILTER[0]) / FILTER[1]) + 1 )/ 2
	#return (1 - (data - data.min())/(data.max() - data.min())) * (alpha[1] - alpha[0]) + alpha[0]
	res = ((data - bounds[0]) / (bounds[1] - bounds[0]))
	res[res <= 0] = 0
	res[res >= 1] = 1
	return res * (alpha[1] - alpha[0]) + alpha[0]

def plot_volumes(data_path, mask_path, pred_path, r, save_path=SAVE_PATH):
	data, _ = nrrd.read(data_path)
	mask, _ = nrrd.read(mask_path)
	pred, _ = nrrd.read(pred_path)
	#print("Original data shape {}, min {},  max {}, mean {}, std {}".format(data.shape, data.min(), data.max(), data.mean(), data.std()))
	#print("Mask data shape {}, min {},  max {}, mean {}, std {}".format(mask.shape, mask.min(), mask.max(), mask.mean(), mask.std()))
	
	# Plot slices
	fig = plt.figure()
	ax = fig.add_subplot()
	img = ax.imshow(np.rot90(data[:, data.shape[1] // 2]))
	img.set_cmap('hot')
	ax.axis('off')
	fig.tight_layout()
	fig.savefig(os.path.join(save_path, 'r{}_slice.png'.format(r)), bbox_inches='tight')
	plt.close(fig)
	fig = plt.figure()
	ax = fig.add_subplot()
	img = ax.imshow(np.rot90(mask[:, mask.shape[1] // 2]))
	img.set_cmap('hot')
	ax.axis('off')
	fig.tight_layout()
	fig.savefig(os.path.join(save_path, 'r{}_slice_mask.png'.format(r)), bbox_inches='tight')
	plt.close(fig)
	fig = plt.figure()
	ax =fig.add_subplot()
	img = ax.imshow(np.rot90(pred[:, pred.shape[1] // 2]))
	img.set_cmap('hot')
	ax.axis('off')
	fig.tight_layout()
	fig.savefig(os.path.join(save_path, 'r{}_slice_pred.png'.format(r)), bbox_inches='tight')
	plt.close(fig)
	
	data, mask, pred = resize(data, out_shape=SHAPE), resize(mask, out_shape=SHAPE), resize(pred, out_shape=SHAPE)
	#data = data.max() - data
	#print("Resized data shape {}, min {},  max {}, mean {}, std {}".format(data.shape, data.min(), data.max(), data.mean(), data.std()))
	
	#voxels = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), np.arange(SHAPE[2]))
	#center = np.array(SHAPE) / 2
	#radii = ((center**2).sum() * THR[0], (center**2).sum() * THR[1])
	#voxels = ((voxels - center.reshape(-1, 1, 1, 1)) ** 2).sum(0)
	#voxels = (voxels <= radii[1]) & (voxels >= radii[0])
	voxels = ((data <= THR[1]) & (data >= THR[0]))
	mask = mask >= TARGET_THR
	pred = pred >= TARGET_THR
	#voxels = voxels & np.logical_not(voxels & np.random.binomial(1, p=min((pred.astype(int).sum()/voxels.astype(int).sum()).item(), 1), size=(voxels.shape)).astype(bool))
	#voxels = voxels | mask
	#segm = data[mask]
	#if len(segm > 0): print("Segmented data shape {}, min {},  max {}, mean {}, std {}".format(segm.shape, segm.min(), segm.max(), segm.mean(), segm.std()))
	#plt.hist(data[voxels])
	#plt.hist(segm)
	#plt.show()
	
	coords = np.meshgrid(np.linspace(0, 1, SHAPE[0]), np.linspace(0, 1, SHAPE[1]), np.linspace(0, 1, SHAPE[2]))
	voxels = voxels & (coords[0] >= 0.5)
	mask = mask & (coords[0] >= 0.5)
	pred = pred & (coords[0] >= 0.5)
	colors = np.zeros((*voxels.shape, 4))
	#colors[:, :, :, :-1] = coords[0].reshape(*SHAPE, 1) * np.array(COLOR).reshape(1, -1) + coords[1].reshape(*SHAPE, 1) * 0 + coords[2].reshape(*SHAPE, 1) * 0
	colors[voxels | mask | pred, :-1] = normalize(data[voxels | mask | pred], alpha=SHADE, bounds=BOUNDS).reshape(-1, 1) * np.array(COLOR).reshape(1, -1)
	#colors[mask, :-1] = np.array(COLOR)
	colors[:, :, :, -1] = coords[0] * (ALPHA[1] - ALPHA[0]) + ALPHA[0]
	#colors[voxels, -1] = normalize(data[voxels], alpha=ALPHA, bounds=BOUNDS)
	colors[mask, -1] = 1
	
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.voxels(voxels | mask | pred, facecolors=colors, linewidth=0.5, shade=True)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	fig.tight_layout()
	fig.savefig(os.path.join(save_path, 'r{}_volume.png'.format(r)), bbox_inches='tight')
	plt.close(fig)
	
	edges = np.zeros((*voxels.shape, 4))
	edges[mask, :-1] = np.array(TARGET_COLOR)
	edges[mask, -1] = 1
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.voxels(mask, facecolors=colors, edgecolors=edges, linewidth=0.5, shade=False)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	fig.tight_layout()
	fig.savefig(os.path.join(save_path, 'r{}_volume_mask.png'.format(r)), bbox_inches='tight')
	plt.close(fig)
	
	edges = np.zeros((*voxels.shape, 4))
	edges[pred, :-1] = np.array(TARGET_COLOR)
	edges[pred, -1] = 1
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.voxels(pred, facecolors=colors, edgecolors=edges, linewidth=0.5, shade=False)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	fig.tight_layout()
	fig.savefig(os.path.join(save_path, 'r{}_volume_pred.png'.format(r)), bbox_inches='tight')
	plt.close(fig)
	
	#plt.show()


if __name__ == '__main__':
	
	files = [f for f in os.listdir(DATA_PATH) if '-mask' not in f and '-pred' not in f]
	for i, f in enumerate(files):
		fname = f.rsplit('.', 1)[0]
		mask_f, pred_f, r = None, None, None
		for f1 in os.listdir(DATA_PATH):
			if f1.startswith(fname + '-mask'): mask_f = f1
			if f1.startswith(fname + '-pred'): pred_f, r = f1, int(f1.rsplit('.', 1)[0].rsplit('r', 1)[1])

		print("Plotting {}/{}...".format(i+1, len(files)))
		os.makedirs(SAVE_PATH, exist_ok=True)
		plot_volumes(os.path.join(DATA_PATH, f), os.path.join(DATA_PATH, mask_f), os.path.join(DATA_PATH, pred_f), r, save_path=SAVE_PATH)
