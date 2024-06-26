import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


PARALLEL_CHANNELS = 32 # How many channels to process in parallel to compute Hebbian updates. Higher values require more GPU memory.

def normalize(x, dim=None):
	nrm = (x**2).sum(dim=dim, keepdim=True)**0.5
	nrm[nrm == 0] = 1.
	return x / nrm


class HebbianConv3d(nn.Module):
	"""
	A 2d convolutional layer that learns through Hebbian plasticity
	"""
	
	MODE_SWTA = 'swta'
	MODE_HPCA = 'hpca'
	MODE_CONTRASTIVE = 'contrastive'

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
	             w_nrm=True, act=nn.Identity(),
	             mode=MODE_SWTA, k=1, patchwise=True,
	             contrast=1., uniformity=False, alpha=0.):
		"""

		:param out_channels: output channels of the convolutional kernel
		:param in_channels: input channels of the convolutional kernel
		:param kernel_size: size of the convolutional kernel (int or tuple)
		:param stride: stride of the convolutional kernel (int or tuple)
		:param stride: padding to apply before convolution (int or tuple)
		:param w_nrm: whether to normalize the weight vectors before computing outputs
		:param act: the nonlinear activation function after convolution
		:param mode: the learning mode, either 'swta' or 'hpca'
		:param k: softmax inverse temperature parameter used for swta-type learning
		:param patchwise: whether updates for each convolutional patch should be computed separately,
		and then aggregated
		:param contrast: coefficient that rescales negative compared to positive updates in contrastive-type learning
		:param uniformity: whether to use uniformity weighting in contrastive-type learning.
		:param alpha: weighting coefficient between hebbian and backprop updates (0 means fully backprop, 1 means fully hebbian).
		
		"""
		
		super().__init__()
		self.mode = mode
		self.out_channels = out_channels
		self.in_channels = in_channels
		self.kernel_size = _triple(kernel_size)
		self.padding = padding
		self.stride = _triple(stride)

		self.weight = nn.Parameter(torch.empty((self.out_channels, self.in_channels, *self.kernel_size)), requires_grad=True)
		nn.init.xavier_normal_(self.weight)
		self.w_nrm = w_nrm
		self.bias = nn.Parameter(torch.zeros(self.out_channels), requires_grad=bias)
		self.act = act
		self.register_buffer('delta_w', torch.zeros_like(self.weight))

		self.k = k
		self.patchwise = patchwise
		self.contrast = contrast
		self.uniformity = uniformity
		
		self.alpha = alpha

	def apply_weights(self, x, w):
		"""
		This function provides the logic for combining input x and weight w
		"""
		
		return torch.conv3d(x, w, bias=self.bias, stride=self.stride)
	
	def compute_activation(self, x):
		w = self.weight
		if self.w_nrm: w = normalize(w, dim=(1, 2, 3, 4))
		y = self.act(self.apply_weights(x, w))
		return y
	
	def pad(self, x):
		pad = [self.padding]*6 if isinstance(self.padding, int) else [self.padding[0], self.padding[0], self.padding[1], self.padding[1], self.padding[2], self.padding[2]] if len(self.padding) == 3 else self.padding
		return F.pad(x, pad)
	
	def forward(self, x):
		x = self.pad(x)
		y = self.compute_activation(x)
		if self.training and self.alpha != 0: self.compute_update(x, y)
		return y
	
	def unfold3d(self, x, kernel_size, stride=(0, 0, 0), padding=0):
		x = F.pad(x, [padding]*6)
		B, C, D, H, W = x.shape
		x_unf = F.unfold(x.reshape(B, C*D, H, W), kernel_size=kernel_size[-2:], stride=stride[-2:])
		x_unf = x_unf.reshape(B, C, D, -1, x_unf.shape[-1])
		K, L = x_unf.shape[3], x_unf.shape[4]
		x_unf = x_unf.permute(0, 4, 1, 3, 2).reshape(B, L, C, K, D)
		x_unf = x_unf.unfold(4, kernel_size[0], stride[0])
		x_unf = x_unf.permute(0, 2, 5, 3, 4, 1).reshape(B, -1, x_unf.shape[-2]*L)
		return x_unf
	
	def compute_update(self, x, y):
		"""
		This function implements the logic that computes local plasticity rules from input x and output y. The
		resulting weight update is stored in buffer self.delta_w for later use.
		"""
		
		if self.mode not in [self.MODE_SWTA, self.MODE_HPCA, self.MODE_CONTRASTIVE]:
			raise NotImplementedError("Learning mode {} unavailable for {} layer".format(self.mode, self.__class__.__name__))
		
		if self.mode == self.MODE_SWTA:
			with torch.no_grad():
				# Logic for swta-type learning
				if self.patchwise:
					r = (y * self.k).softmax(dim=1).permute(1, 0, 2, 3, 4).reshape(y.shape[1], -1)
					for i in range((self.weight.shape[1] // PARALLEL_CHANNELS) + (1 if self.weight.shape[1] % PARALLEL_CHANNELS != 0 else 0)):
						start = i * PARALLEL_CHANNELS
						end = min((i + 1) * PARALLEL_CHANNELS, self.weight.shape[1])
						x_unf = self.unfold3d(x[:, start:end], self.kernel_size, self.stride)
						x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
						w_i = self.weight[:, start:end]
						r_i = r[:]
						dec = r_i.sum(1, keepdim=True) * w_i.reshape(w_i.shape[0], -1)
						self.delta_w[:, start:end] += (r_i.matmul(x_unf) - dec).reshape_as(w_i)
				else:
					r = (y * self.k).softmax(dim=1).permute(2, 3, 4, 1, 0)
					for i in range((self.weight.shape[1] // PARALLEL_CHANNELS) + (1 if self.weight.shape[1] % PARALLEL_CHANNELS != 0 else 0)):
						start = i * PARALLEL_CHANNELS
						end = min((i + 1) * PARALLEL_CHANNELS, self.weight.shape[1])
						x_unf = self.unfold3d(x[:, start:end], self.kernel_size, self.stride)
						x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
						w_i = self.weight[:, start:end]
						r_i = r[:]
						krn = torch.eye(len(w_i[0].reshape(-1)), device=x.device, dtype=x.dtype).reshape(len(w_i[0].reshape(-1)), w_i.shape[1], *self.kernel_size)
						dec = torch.conv_transpose3d((r_i.sum(dim=-1, keepdim=True) * w_i.reshape(1, 1, 1, w_i.shape[0], -1)).permute(3, 4, 0, 1, 2), krn, stride=self.stride)
						self.delta_w[:, start:end] += (r_i.permute(3, 4, 0, 1, 2).reshape(r_i.shape[3], -1).matmul(x_unf) - self.unfold3d(dec, self.kernel_size, self.stride).sum(dim=-1)).reshape_as(w_i)
				
		if self.mode == self.MODE_HPCA:
			with torch.no_grad():
				# Logic for hpca-type learning
				if self.patchwise:
					r = y.permute(1, 0, 2, 3, 4).reshape(y.shape[1], -1)
					for i in range((self.weight.shape[1] // PARALLEL_CHANNELS) + (1 if self.weight.shape[1] % PARALLEL_CHANNELS != 0 else 0)):
						start = i * PARALLEL_CHANNELS
						end = min((i + 1) * PARALLEL_CHANNELS, self.weight.shape[1])
						x_unf = self.unfold3d(x[:, start:end], self.kernel_size, self.stride)
						x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.shape[1])
						w_i = self.weight[:, start:end]
						r_i = r[:]
						l = (torch.arange(w_i.shape[0], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(w_i.shape[0], 1) <= torch.arange(w_i.shape[0], device=x.device, dtype=x.dtype).unsqueeze(1)).to(dtype=x.dtype)
						dec = (r_i.matmul(r_i.transpose(-2, -1)) * l).matmul(w_i.reshape(w_i.shape[0], -1))
						self.delta_w[:, start:end] += (r_i.matmul(x_unf) - dec).reshape_as(w_i)
				else:
					r = y.permute(2, 3, 4, 1, 0)
					for i in range((self.weight.shape[1] // PARALLEL_CHANNELS) + (1 if self.weight.shape[1] % PARALLEL_CHANNELS != 0 else 0)):
						start = i * PARALLEL_CHANNELS
						end = min((i + 1) * PARALLEL_CHANNELS, self.weight.shape[1])
						x_unf = self.unfold3d(x[:, start:end], self.kernel_size, self.stride)
						x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.shape[1])
						w_i = self.weight[:, start:end]
						r_i = r[:]
						l = (torch.arange(w_i.shape[0], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(w_i.shape[0], 1) <= torch.arange(w_i.shape[0], device=x.device, dtype=x.dtype).unsqueeze(1)).to(dtype=x.dtype)
						dec = torch.conv_transpose3d((r_i.matmul(r_i.transpose(-2, -1)) * l.unsqueeze(0).unsqueeze(1).unsqueeze(2)).permute(4, 3, 0, 1, 2), w_i, stride=self.stride)
						self.delta_w[:, start:end] += (r_i.permute(3, 4, 0, 1, 2).reshape(r_i.shape[3], -1).matmul(x_unf) - self.unfold3d(dec, self.kernel_size, self.stride).sum(dim=-1)).reshape_as(w_i)
			
		if self.mode == self.MODE_CONTRASTIVE:
			y = self.compute_activation(x.clone().detach())
			y = normalize(y, dim=1)
			y_unf = self.unfold3d(y, _triple(3), padding=1)
			y_unf = y_unf.permute(0, 2, 1).reshape(y_unf.size(0), y_unf.size(2), y.size(1), 27)
			
			# Positive contribution
			L = - y_unf.sum(-1).reshape(-1, y.size(1)) * y.permute(0, 2, 3, 4, 1).reshape(-1, y.size(1))
			
			# Uniformity weighting, if required
			if self.uniformity:
				with torch.no_grad():
					x = normalize(x, dim=1)
					x_unf = self.unfold3d(x, _triple(3), padding=1)
					x_unf = x_unf.permute(0, 2, 1).reshape(x_unf.size(0), x_unf.size(2), x.size(1), 27)
					uniformity_map = (x_unf.sum(-1).reshape(-1, x.size(1)) * x.permute(0, 2, 3, 4, 1).reshape(-1, x.size(1))).sum(dim=-1, keepdim=True)
					uniformity_map = self.apply_weights(uniformity_map.reshape(x.size(0), 1, *x.shape[2:]), torch.ones([1, 1, *self.kernel_size], device=x.device, dtype=x.dtype)).reshape(-1, 1)
				L = L * uniformity_map
			
			# Negative contribution
			idx = torch.randperm(y_unf.size(0), device=y_unf.device)
			L = L + self.contrast * y_unf[idx].sum(-1).reshape(-1, y.size(1)) * y.permute(0, 2, 3, 4, 1).reshape(-1, y.size(1))
			
			# Compute update
			L = L.sum()
			prev_grad = self.weight.grad
			self.weight.grad = None
			L.backward()
			self.delta_w += self.weight.grad.clone().detach()
			self.weight.grad = prev_grad
	
	@torch.no_grad()
	def local_update(self):
		"""
		
		This function transfers a previously computed weight update, stored in buffer self.delta_w, to the gradient
		self.weight.grad of the weight parameter.
		
		This function should be called before optimizer.step(), so that the optimizer will use the locally computed
		update as optimization direction. Local updates can also be combined with end-to-end updates by calling this
		function between loss.backward() and optimizer.step(). loss.backward will store the end-to-end gradient in
		self.weight.grad, and this function combines this value with self.delta_w as
		self.weight.grad = (1 - alpha) * self.weight.grad - alpha * self.delta_w
		Parameter alpha determines the scale of the local update compared to the end-to-end gradient in the combination.
		
		"""
		
		if self.weight.grad is None: self.weight.grad = -self.alpha * self.delta_w
		else: self.weight.grad = (1 - self.alpha) * self.weight.grad - self.alpha * self.delta_w
		self.delta_w.zero_()


class HebbianConvTranspose3d(HebbianConv3d):
	"""
	A 2d convolutional layer that learns through Hebbian plasticity
	"""
	
	MODE_SWTA_T = 'swta_t'
	MODE_HPCA_T = 'hpca_t'

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, w_nrm=True, act=nn.Identity(),
	             mode=MODE_SWTA_T, k=1, patchwise=True, contrast=1., uniformity=False, alpha=0.):
		"""
		
		:param out_channels: output channels of the convolutional kernel
		:param in_channels: input channels of the convolutional kernel
		:param kernel_size: size of the convolutional kernel (int or tuple)
		:param stride: stride of the convolutional kernel (int or tuple)
		:param w_nrm: whether to normalize the weight vectors before computing outputs
		:param act: the nonlinear activation function after convolution
		:param mode: the learning mode, either 'swta', 'swta_t, 'hpca', 'hpca_t', or 'contrastive'
		:param k: softmax inverse temperature parameter used for swta-type learning
		:param patchwise: whether updates for each convolutional patch should be computed separately,
		and then aggregated
		:param contrast: coefficient that rescales negative compared to positive updates in contrastive-type learning
		:param uniformity: whether to use uniformity weighting in contrastive-type learning.
		"""
		
		super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, w_nrm, act, mode, k, patchwise, contrast, uniformity, alpha)
		with torch.no_grad():
			self.weight.transpose_(0, 1)
			self.delta_w.transpose_(0, 1)

	def apply_weights(self, x, w):
		"""
		This function implements the logic that computes Hebbian plasticity rules from input x and output y. The
		resulting weight update is stored in buffer self.delta_w for later use.
		"""
		
		return torch.conv_transpose3d(x, w, bias=self.bias, stride=self.stride)

	def compute_update(self, x, y):
		"""
		This function implements the logic that computes local plasticity rules from input x and output y. The
		resulting weight update is stored in buffer self.delta_w for later use.
		"""
		
		if self.mode not in [self.MODE_SWTA, self.MODE_HPCA, self.MODE_SWTA_T, self.MODE_HPCA_T, self.MODE_CONTRASTIVE]:
			raise NotImplementedError("Learning mode {} unavailable for {} layer".format(self.mode, self.__class__.__name__))
		
		if self.mode in [self.MODE_SWTA, self.MODE_HPCA]:
			# In case of swta-type or hpca-type learning, use the learning rules for ordinary convolution,
			# but exchanging x and y
			super().compute_update(y, x)
		
		if self.mode in [self.MODE_CONTRASTIVE]:
			# Reuse update from base conv class
			super().compute_update(x, y)
		
		if self.mode == self.MODE_SWTA_T:
			with torch.no_grad():
				# Logic for swta-type learning in transpose convolutional layers
				r = (y * self.k).softmax(dim=1)
				for i in range((self.weight.shape[1] // PARALLEL_CHANNELS) + (1 if self.weight.shape[1] % PARALLEL_CHANNELS != 0 else 0)):
					start = i * PARALLEL_CHANNELS
					end = min((i + 1) * PARALLEL_CHANNELS, self.weight.shape[1])
					w_i = self.weight[:, start:end]
					r_i = r[:, start:end]
					r_i = self.unfold3d(r_i, kernel_size=self.kernel_size, stride=self.stride)
					r_i = r_i.permute(0, 2, 1).reshape(-1, w_i.shape[1], self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]).permute(2, 1, 0)
					dec = r_i.sum(2, keepdim=True) * w_i.permute(2, 3, 4, 1, 0).reshape(-1, w_i.shape[1], w_i.shape[0])
					if self.patchwise: dec = dec.sum(dim=0, keepdim=True)
					self.delta_w[:, start:end] += (r_i.matmul(x.permute(0, 2, 3, 4, 1).reshape(1, -1, x.size(1))) - dec).permute(2, 1, 0).reshape_as(w_i)
			
		if self.mode == self.MODE_HPCA_T:
			with torch.no_grad():
				# Logic for hpca-type learning in transpose convolutional layers
				r = y
				for i in range((self.weight.shape[1] // PARALLEL_CHANNELS) + (1 if self.weight.shape[1] % PARALLEL_CHANNELS != 0 else 0)):
					start = i * PARALLEL_CHANNELS
					end = min((i + 1) * PARALLEL_CHANNELS, self.weight.shape[1])
					w_i = self.weight[:, start:end]
					r_i = r[:, start:end]
					r_i = self.unfold3d(r_i, kernel_size=self.kernel_size, stride=self.stride)
					r_i = r_i.permute(0, 2, 1).reshape(-1, w_i.shape[1], self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]).permute(2, 1, 0)
					l = (torch.arange(w_i.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(w_i.shape[1], 1) <= torch.arange(w_i.shape[1], device=x.device, dtype=x.dtype).unsqueeze(1)).to(dtype=x.dtype)
					dec = (r_i.matmul(r_i.transpose(-2, -1)) * l.unsqueeze(0)).matmul(w_i.permute(2, 3, 4, 1, 0).reshape(-1, w_i.shape[1], w_i.shape[0]))
					if self.patchwise: dec = dec.sum(dim=0, keepdim=True)
					self.delta_w[:, start:end] += (r_i.matmul(x.permute(0, 2, 3, 4, 1).reshape(1, -1, x.size(1))) - dec).permute(2, 1, 0).reshape_as(w_i)
			
		