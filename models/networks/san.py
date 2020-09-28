import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from collections import namedtuple
from string import Template
import cupy
from utils.registry import *

Stream = namedtuple('Stream', ['ptr'])
CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N):
		return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

#############cuda implementation################
kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
			i < (n);                                       \
			i += blockDim.x * gridDim.x)
'''

_subtraction_zeropad_forward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction_zeropad_forward_kernel(
const ${Dtype}* bottom_data, ${Dtype}* top_data) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${top_height} / ${top_width};
		const int c = (index / ${top_height} / ${top_width}) % ${input_channels};
		const int h = (index / ${top_width}) % ${top_height};
		const int w = index % ${top_width};
		const int h_in_center = -${pad_h} + h * ${stride_h} + (${kernel_h} - 1) / 2 * ${dilation_h};
		const int w_in_center = -${pad_w} + w * ${stride_w} + (${kernel_w} - 1) / 2 * ${dilation_w};
		const int offset_center = ((n * ${input_channels} + c) * ${bottom_height} + h_in_center) * ${bottom_width} + w_in_center;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
				const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
				const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
				if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
					const int offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
					top_data[offset_top] = bottom_data[offset_center] - bottom_data[offset_bottom];
				}
				else
					top_data[offset_top] = bottom_data[offset_center];
			}
		}
	}
}
'''

_subtraction_zeropad_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction_zeropad_input_backward_kernel(
		const ${Dtype}* const top_diff, ${Dtype}* bottom_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${bottom_height} / ${bottom_width};
		const int c = (index / ${bottom_height} / ${bottom_width}) % ${input_channels};
		const int h = (index / ${bottom_width}) % ${bottom_height};
		const int w = index % ${bottom_width};
		${Dtype} value = 0;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
				const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
				if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
					const int h_out = h_out_s / ${stride_h};
					const int w_out = w_out_s / ${stride_w};
					if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
						const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
						value += -top_diff[offset_top];
					}
				}
			}
		}
		if (((h % ${stride_h}) == 0) && ((w % ${stride_w}) == 0)) {
			const int h_out = h / ${stride_h};
			const int w_out = w / ${stride_w};
			for (int kh = 0; kh < ${kernel_h}; ++kh) {
				for (int kw = 0; kw < ${kernel_w}; ++kw) {
					const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
					value += top_diff[offset_top];
				}
			}
		}
		bottom_diff[index] = value;
	}
}
'''

def Dtype(t):
	if isinstance(t, torch.cuda.FloatTensor):
		return 'float'
	elif isinstance(t, torch.cuda.DoubleTensor):
		return 'double'

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
	code = Template(code).substitute(**kwargs)
	kernel_code = cupy.cuda.compile_with_cache(code)
	return kernel_code.get_function(kernel_name)

class SubtractionZeropad(Function):
	@staticmethod
	def forward(ctx, input, kernel_size, stride, padding, dilation):
		kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
		ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
		assert input.dim() == 4 and input.is_cuda
		batch_size, input_channels, input_height, input_width = input.size()
		output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] -1) + 1)) / stride[0] + 1)
		output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
		output = input.new(batch_size, input_channels, kernel_size[0] * kernel_size[1], output_height * output_width)
		n = output.numel() // output.shape[2]
		with torch.cuda.device_of(input):
			f = load_kernel('subtraction_zeropad_forward_kernel', _subtraction_zeropad_forward_kernel, Dtype=Dtype(input), nthreads=n,
							num=batch_size, input_channels=input_channels,
							bottom_height=input_height, bottom_width=input_width,
							top_height=output_height, top_width=output_width,
							kernel_h=kernel_size[0], kernel_w=kernel_size[1],
							stride_h=stride[0], stride_w=stride[1],
							dilation_h=dilation[0], dilation_w=dilation[1],
							pad_h=padding[0], pad_w=padding[1])
			f(block=(CUDA_NUM_THREADS, 1, 1),
				grid=(GET_BLOCKS(n), 1, 1),
				args=[input.data_ptr(), output.data_ptr()],
				stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
		ctx.save_for_backward(input)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
		input, = ctx.saved_tensors
		assert grad_output.is_cuda
		if not grad_output.is_contiguous():
			grad_output = grad_output.contiguous()
		batch_size, input_channels, input_height, input_width = input.size()
		output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
		output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
		grad_input = None
		opt = dict(Dtype=Dtype(grad_output),
							 num=batch_size, input_channels=input_channels,
							 bottom_height=input_height, bottom_width=input_width,
							 top_height=output_height, top_width=output_width,
							 kernel_h=kernel_size[0], kernel_w=kernel_size[1],
							 stride_h=stride[0], stride_w=stride[1],
							 dilation_h=dilation[0], dilation_w=dilation[1],
							 pad_h=padding[0], pad_w=padding[1])
		with torch.cuda.device_of(input):
			if ctx.needs_input_grad[0]:
				grad_input = input.new(input.size())
				n = grad_input.numel()
				opt['nthreads'] = n
				f = load_kernel('subtraction_zeropad_input_backward_kernel', _subtraction_zeropad_input_backward_kernel, **opt)
				f(block=(CUDA_NUM_THREADS, 1, 1),
					grid=(GET_BLOCKS(n), 1, 1),
					args=[grad_output.data_ptr(), grad_input.data_ptr()],
					stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
		return grad_input, None, None, None, None

_subtraction_refpad_forward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction_refpad_forward_kernel(
const ${Dtype}* bottom_data, ${Dtype}* top_data) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${top_height} / ${top_width};
		const int c = (index / ${top_height} / ${top_width}) % ${input_channels};
		const int h = (index / ${top_width}) % ${top_height};
		const int w = index % ${top_width};
		const int h_in_center = -${pad_h} + h * ${stride_h} + (${kernel_h} - 1) / 2 * ${dilation_h};
		const int w_in_center = -${pad_w} + w * ${stride_w} + (${kernel_w} - 1) / 2 * ${dilation_w};
		const int offset_center = ((n * ${input_channels} + c) * ${bottom_height} + h_in_center) * ${bottom_width} + w_in_center;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
				int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
				const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
				int offset_bottom;
				if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
					offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
				}
				else {
					if (h_in < 0) h_in = -h_in;
					if (h_in >= ${bottom_height}) h_in = 2 * (${bottom_height} - 1) - h_in;
					if (w_in < 0) w_in = -w_in;
					if (w_in >= ${bottom_width}) w_in = 2 * (${bottom_width} - 1) - w_in;
					offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
				}
				top_data[offset_top] = bottom_data[offset_center] - bottom_data[offset_bottom];
			}
		}
	}
}
'''

_subtraction_refpad_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction_refpad_input_backward_kernel(
		const ${Dtype}* const top_diff, ${Dtype}* bottom_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / (${bottom_height} + 2 * ${pad_h}) / (${bottom_width} + 2 * ${pad_w});
		const int c = (index / (${bottom_height} + 2 * ${pad_h}) / (${bottom_width} + 2 * ${pad_w})) % ${input_channels};
		const int h = (index / (${bottom_width} + 2 * ${pad_w})) % (${bottom_height} + 2 * ${pad_h});
		const int w = index % (${bottom_width} + 2 * ${pad_w});
		${Dtype} value = 0;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_out_s = h - kh * ${dilation_h};
				const int w_out_s = w - kw * ${dilation_w};
				if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
					const int h_out = h_out_s / ${stride_h};
					const int w_out = w_out_s / ${stride_w};
					if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
						const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
						value += -top_diff[offset_top];
					}
				}
			}
		}
		const int hh = h - ${pad_h};
		const int ww = w - ${pad_w};
		if ((hh >= 0) && (hh < ${bottom_height}) && (ww >= 0) && (ww < ${bottom_width})) {
			if (((hh % ${stride_h}) == 0) && ((ww % ${stride_w}) == 0)) {
				const int h_out = hh / ${stride_h};
				const int w_out = ww / ${stride_w};
				for (int kh = 0; kh < ${kernel_h}; ++kh) {
					for (int kw = 0; kw < ${kernel_w}; ++kw) {
						const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
						value += top_diff[offset_top];
					}
				}
			}
		}
		bottom_diff[index] = value;
	}
}
'''

class SubtractionRefpad(Function):
		@staticmethod
		def forward(ctx, input, kernel_size, stride, padding, dilation):
				kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
				ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
				assert input.dim() == 4 and input.is_cuda
				batch_size, input_channels, input_height, input_width = input.size()
				output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
				output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
				output = input.new(batch_size, input_channels, kernel_size[0] * kernel_size[1], output_height * output_width)
				n = output.numel() // output.shape[2]
				with torch.cuda.device_of(input):
						f = load_kernel('subtraction_refpad_forward_kernel', _subtraction_refpad_forward_kernel, Dtype=Dtype(input), nthreads=n,
														num=batch_size, input_channels=input_channels,
														bottom_height=input_height, bottom_width=input_width,
														top_height=output_height, top_width=output_width,
														kernel_h=kernel_size[0], kernel_w=kernel_size[1],
														stride_h=stride[0], stride_w=stride[1],
														dilation_h=dilation[0], dilation_w=dilation[1],
														pad_h=padding[0], pad_w=padding[1])
						f(block=(CUDA_NUM_THREADS, 1, 1),
							grid=(GET_BLOCKS(n), 1, 1),
							args=[input.data_ptr(), output.data_ptr()],
							stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				ctx.save_for_backward(input)
				return output

		@staticmethod
		def backward(ctx, grad_output):
				kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
				input, = ctx.saved_tensors
				assert grad_output.is_cuda
				if not grad_output.is_contiguous():
						grad_output = grad_output.contiguous()
				batch_size, input_channels, input_height, input_width = input.size()
				output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
				output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
				grad_input = None
				opt = dict(Dtype=Dtype(grad_output),
									 num=batch_size, input_channels=input_channels,
									 bottom_height=input_height, bottom_width=input_width,
									 top_height=output_height, top_width=output_width,
									 kernel_h=kernel_size[0], kernel_w=kernel_size[1],
									 stride_h=stride[0], stride_w=stride[1],
									 dilation_h=dilation[0], dilation_w=dilation[1],
									 pad_h=padding[0], pad_w=padding[1])
				with torch.cuda.device_of(input):
						if ctx.needs_input_grad[0]:
								grad_input = input.new(batch_size, input_channels, input_height + 2 * padding[0], input_width + 2 * padding[1])
								n = grad_input.numel()
								opt['nthreads'] = n
								f = load_kernel('subtraction_refpad_input_backward_kernel', _subtraction_refpad_input_backward_kernel, **opt)
								f(block=(CUDA_NUM_THREADS, 1, 1),
									grid=(GET_BLOCKS(n), 1, 1),
									args=[grad_output.data_ptr(), grad_input.data_ptr()],
									stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
								grad_input[:, :, padding[0] + 1:2 * padding[0] + 1, :] += torch.flip(grad_input[:, :, :padding[0], :], dims=[2])
								grad_input[:, :, input_height - 1:input_height + padding[0] - 1, :] += torch.flip(grad_input[:, :, input_height + padding[0]:, :], dims=[2])
								grad_input[:, :, :, padding[1] + 1:2 * padding[1] + 1] += torch.flip(grad_input[:, :, :, :padding[1]], dims=[3])
								grad_input[:, :, :, input_width - 1:input_width + padding[1] - 1] += torch.flip(grad_input[:, :, :, input_width + padding[1]:], dims=[3])
								grad_input = grad_input[:, :, padding[0]:padding[0] + input_height, padding[1]:padding[1] + input_width]
				return grad_input, None, None, None, None

_subtraction2_refpad_forward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction2_refpad_forward_kernel(
const ${Dtype}* bottom1_data, const ${Dtype}* bottom2_data, ${Dtype}* top_data) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${top_height} / ${top_width};
		const int c = (index / ${top_height} / ${top_width}) % ${input_channels};
		const int h = (index / ${top_width}) % ${top_height};
		const int w = index % ${top_width};
		const int h_in_center = -${pad_h} + h * ${stride_h} + (${kernel_h} - 1) / 2 * ${dilation_h};
		const int w_in_center = -${pad_w} + w * ${stride_w} + (${kernel_w} - 1) / 2 * ${dilation_w};
		const int offset_center = ((n * ${input_channels} + c) * ${bottom_height} + h_in_center) * ${bottom_width} + w_in_center;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
				int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
				const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
				int offset_bottom;
				if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
					offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
				}
				else {
					if (h_in < 0) h_in = -h_in;
					if (h_in >= ${bottom_height}) h_in = 2 * (${bottom_height} - 1) - h_in;
					if (w_in < 0) w_in = -w_in;
					if (w_in >= ${bottom_width}) w_in = 2 * (${bottom_width} - 1) - w_in;
					offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
				}
				top_data[offset_top] = bottom1_data[offset_center] - bottom2_data[offset_bottom];
			}
		}
	}
}
'''


_subtraction2_refpad_input1_backward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction2_refpad_input1_backward_kernel(
		const ${Dtype}* const top_diff, ${Dtype}* bottom_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${bottom_height} / ${bottom_width};
		const int c = (index / ${bottom_height} / ${bottom_width}) % ${input_channels};
		const int h = (index / ${bottom_width}) % ${bottom_height};
		const int w = index % ${bottom_width};
		${Dtype} value = 0;
		if (((h % ${stride_h}) == 0) && ((w % ${stride_w}) == 0)) {
			const int h_out = h / ${stride_h};
			const int w_out = w / ${stride_w};
			for (int kh = 0; kh < ${kernel_h}; ++kh) {
				for (int kw = 0; kw < ${kernel_w}; ++kw) {
					const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
					value += top_diff[offset_top];
				}
			}
		}
		bottom_diff[index] = value;
	}
}
'''


_subtraction2_refpad_input2_backward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction2_refpad_input2_backward_kernel(
		const ${Dtype}* const top_diff, ${Dtype}* bottom_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / (${bottom_height} + 2 * ${pad_h}) / (${bottom_width} + 2 * ${pad_w});
		const int c = (index / (${bottom_height} + 2 * ${pad_h}) / (${bottom_width} + 2 * ${pad_w})) % ${input_channels};
		const int h = (index / (${bottom_width} + 2 * ${pad_w})) % (${bottom_height} + 2 * ${pad_h});
		const int w = index % (${bottom_width} + 2 * ${pad_w});
		${Dtype} value = 0;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_out_s = h - kh * ${dilation_h};
				const int w_out_s = w - kw * ${dilation_w};
				if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
					const int h_out = h_out_s / ${stride_h};
					const int w_out = w_out_s / ${stride_w};
					if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
						const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
						value += -top_diff[offset_top];
					}
				}
			}
		}
		bottom_diff[index] = value;
	}
}
'''


class Subtraction2Refpad(Function):
		@staticmethod
		def forward(ctx, input1, input2, kernel_size, stride, padding, dilation):
				kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
				ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
				assert input1.dim() == 4 and input1.is_cuda
				batch_size, input_channels, input_height, input_width = input1.size()
				output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
				output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
				output = input1.new(batch_size, input_channels, kernel_size[0] * kernel_size[1], output_height * output_width)
				n = output.numel() // output.shape[2]
				with torch.cuda.device_of(input1):
						f = load_kernel('subtraction2_refpad_forward_kernel', _subtraction2_refpad_forward_kernel, Dtype=Dtype(input1), nthreads=n,
														num=batch_size, input_channels=input_channels,
														bottom_height=input_height, bottom_width=input_width,
														top_height=output_height, top_width=output_width,
														kernel_h=kernel_size[0], kernel_w=kernel_size[1],
														stride_h=stride[0], stride_w=stride[1],
														dilation_h=dilation[0], dilation_w=dilation[1],
														pad_h=padding[0], pad_w=padding[1])
						f(block=(CUDA_NUM_THREADS, 1, 1),
							grid=(GET_BLOCKS(n), 1, 1),
							args=[input1.data_ptr(), input2.data_ptr(), output.data_ptr()],
							stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				ctx.save_for_backward(input1, input2)
				return output

		@staticmethod
		def backward(ctx, grad_output):
				kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
				input1, input2 = ctx.saved_tensors
				assert grad_output.is_cuda
				if not grad_output.is_contiguous():
						grad_output = grad_output.contiguous()
				batch_size, input_channels, input_height, input_width = input1.size()
				output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
				output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
				grad_input1, grad_input2 = None, None
				opt = dict(Dtype=Dtype(grad_output),
									 num=batch_size, input_channels=input_channels,
									 bottom_height=input_height, bottom_width=input_width,
									 top_height=output_height, top_width=output_width,
									 kernel_h=kernel_size[0], kernel_w=kernel_size[1],
									 stride_h=stride[0], stride_w=stride[1],
									 dilation_h=dilation[0], dilation_w=dilation[1],
									 pad_h=padding[0], pad_w=padding[1])
				with torch.cuda.device_of(input1):
						if ctx.needs_input_grad[0]:
								grad_input1 = input1.new(input1.size())
								n = grad_input1.numel()
								opt['nthreads'] = n
								f = load_kernel('subtraction2_refpad_input1_backward_kernel', _subtraction2_refpad_input1_backward_kernel, **opt)
								f(block=(CUDA_NUM_THREADS, 1, 1),
									grid=(GET_BLOCKS(n), 1, 1),
									args=[grad_output.data_ptr(), grad_input1.data_ptr()],
									stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				with torch.cuda.device_of(input2):
						if ctx.needs_input_grad[1]:
								grad_input2 = input2.new(batch_size, input_channels, input_height + 2 * padding[0], input_width + 2 * padding[1])
								n = grad_input2.numel()
								opt['nthreads'] = n
								f = load_kernel('subtraction2_refpad_input2_backward_kernel', _subtraction2_refpad_input2_backward_kernel, **opt)
								f(block=(CUDA_NUM_THREADS, 1, 1),
									grid=(GET_BLOCKS(n), 1, 1),
									args=[grad_output.data_ptr(), grad_input2.data_ptr()],
									stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
								grad_input2[:, :, padding[0] + 1:2 * padding[0] + 1, :] += torch.flip(grad_input2[:, :, :padding[0], :], dims=[2])
								grad_input2[:, :, input_height - 1:input_height + padding[0] - 1, :] += torch.flip(grad_input2[:, :, input_height + padding[0]:, :], dims=[2])
								grad_input2[:, :, :, padding[1] + 1:2 * padding[1] + 1] += torch.flip(grad_input2[:, :, :, :padding[1]], dims=[3])
								grad_input2[:, :, :, input_width - 1:input_width + padding[1] - 1] += torch.flip(grad_input2[:, :, :, input_width + padding[1]:], dims=[3])
								grad_input2 = grad_input2[:, :, padding[0]:padding[0] + input_height, padding[1]:padding[1] + input_width]
				return grad_input1, grad_input2, None, None, None, None

_subtraction2_zeropad_forward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction2_zeropad_forward_kernel(
const ${Dtype}* bottom1_data, const ${Dtype}* bottom2_data, ${Dtype}* top_data) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${top_height} / ${top_width};
		const int c = (index / ${top_height} / ${top_width}) % ${input_channels};
		const int h = (index / ${top_width}) % ${top_height};
		const int w = index % ${top_width};
		const int h_in_center = -${pad_h} + h * ${stride_h} + (${kernel_h} - 1) / 2 * ${dilation_h};
		const int w_in_center = -${pad_w} + w * ${stride_w} + (${kernel_w} - 1) / 2 * ${dilation_w};
		const int offset_center = ((n * ${input_channels} + c) * ${bottom_height} + h_in_center) * ${bottom_width} + w_in_center;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
				const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
				const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
				if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
					const int offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
					top_data[offset_top] = bottom1_data[offset_center] - bottom2_data[offset_bottom];
				}
				else
					top_data[offset_top] = bottom1_data[offset_center];
			}
		}
	}
}
'''


_subtraction2_zeropad_input1_backward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction2_zeropad_input1_backward_kernel(
		const ${Dtype}* const top_diff, ${Dtype}* bottom_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${bottom_height} / ${bottom_width};
		const int c = (index / ${bottom_height} / ${bottom_width}) % ${input_channels};
		const int h = (index / ${bottom_width}) % ${bottom_height};
		const int w = index % ${bottom_width};
		${Dtype} value = 0;
		if (((h % ${stride_h}) == 0) && ((w % ${stride_w}) == 0)) {
			const int h_out = h / ${stride_h};
			const int w_out = w / ${stride_w};
			for (int kh = 0; kh < ${kernel_h}; ++kh) {
				for (int kw = 0; kw < ${kernel_w}; ++kw) {
					const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
					value += top_diff[offset_top];
				}
			}
		}
		bottom_diff[index] = value;
	}
}
'''

_subtraction2_zeropad_input2_backward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction2_zeropad_input2_backward_kernel(
		const ${Dtype}* const top_diff, ${Dtype}* bottom_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${bottom_height} / ${bottom_width};
		const int c = (index / ${bottom_height} / ${bottom_width}) % ${input_channels};
		const int h = (index / ${bottom_width}) % ${bottom_height};
		const int w = index % ${bottom_width};
		${Dtype} value = 0;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
				const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
				if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
					const int h_out = h_out_s / ${stride_h};
					const int w_out = w_out_s / ${stride_w};
					if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
						const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
						value += -top_diff[offset_top];
					}
				}
			}
		}
		bottom_diff[index] = value;
	}
}
'''

class Subtraction2Zeropad(Function):
		@staticmethod
		def forward(ctx, input1, input2, kernel_size, stride, padding, dilation):
				kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
				ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
				assert input1.dim() == 4 and input1.is_cuda
				batch_size, input_channels, input_height, input_width = input1.size()
				output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
				output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
				output = input1.new(batch_size, input_channels, kernel_size[0] * kernel_size[1], output_height * output_width)
				n = output.numel() // output.shape[2]
				with torch.cuda.device_of(input1):
						f = load_kernel('subtraction2_zeropad_forward_kernel', _subtraction2_zeropad_forward_kernel, Dtype=Dtype(input1), nthreads=n,
														num=batch_size, input_channels=input_channels,
														bottom_height=input_height, bottom_width=input_width,
														top_height=output_height, top_width=output_width,
														kernel_h=kernel_size[0], kernel_w=kernel_size[1],
														stride_h=stride[0], stride_w=stride[1],
														dilation_h=dilation[0], dilation_w=dilation[1],
														pad_h=padding[0], pad_w=padding[1])
						f(block=(CUDA_NUM_THREADS, 1, 1),
							grid=(GET_BLOCKS(n), 1, 1),
							args=[input1.data_ptr(), input2.data_ptr(), output.data_ptr()],
							stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				ctx.save_for_backward(input1, input2)
				return output

		@staticmethod
		def backward(ctx, grad_output):
				kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
				input1, input2 = ctx.saved_tensors
				assert grad_output.is_cuda
				if not grad_output.is_contiguous():
						grad_output = grad_output.contiguous()
				batch_size, input_channels, input_height, input_width = input1.size()
				output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
				output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
				grad_input1, grad_input2 = None, None
				opt = dict(Dtype=Dtype(grad_output),
									 num=batch_size, input_channels=input_channels,
									 bottom_height=input_height, bottom_width=input_width,
									 top_height=output_height, top_width=output_width,
									 kernel_h=kernel_size[0], kernel_w=kernel_size[1],
									 stride_h=stride[0], stride_w=stride[1],
									 dilation_h=dilation[0], dilation_w=dilation[1],
									 pad_h=padding[0], pad_w=padding[1])
				with torch.cuda.device_of(input1):
						if ctx.needs_input_grad[0]:
								grad_input1 = input1.new(input1.size())
								n = grad_input1.numel()
								opt['nthreads'] = n
								f = load_kernel('subtraction2_zeropad_input1_backward_kernel', _subtraction2_zeropad_input1_backward_kernel, **opt)
								f(block=(CUDA_NUM_THREADS, 1, 1),
									grid=(GET_BLOCKS(n), 1, 1),
									args=[grad_output.data_ptr(), grad_input1.data_ptr()],
									stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				with torch.cuda.device_of(input2):
						if ctx.needs_input_grad[1]:
								grad_input2 = input2.new(input2.size())
								n = grad_input2.numel()
								opt['nthreads'] = n
								f = load_kernel('subtraction2_zeropad_input2_backward_kernel', _subtraction2_zeropad_input2_backward_kernel, **opt)
								f(block=(CUDA_NUM_THREADS, 1, 1),
									grid=(GET_BLOCKS(n), 1, 1),
									args=[grad_output.data_ptr(), grad_input2.data_ptr()],
									stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				return grad_input1, grad_input2, None, None, None, None

_aggregation_zeropad_forward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${top_height} / ${top_width};
		const int c = (index / ${top_height} / ${top_width}) % ${input_channels};
		const int h = (index / ${top_width}) % ${top_height};
		const int w = index % ${top_width};
		${Dtype} value = 0;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
				const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
				if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
					const int offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
					const int offset_weight = ((n * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
					value += weight_data[offset_weight] * bottom_data[offset_bottom];
				}
			}
		}
		top_data[index] = value;
	}
}
'''


_aggregation_zeropad_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_input_backward_kernel(
		const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* bottom_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${bottom_height} / ${bottom_width};
		const int c = (index / ${bottom_height} / ${bottom_width}) % ${input_channels};
		const int h = (index / ${bottom_width}) % ${bottom_height};
		const int w = index % ${bottom_width};
		${Dtype} value = 0;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
				const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
				if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
					const int h_out = h_out_s / ${stride_h};
					const int w_out = w_out_s / ${stride_w};
					if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
						const int offset_top = ((n * ${input_channels} + c) * ${top_height} + h_out) * ${top_width} + w_out;
						const int offset_weight = ((n * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
						value += weight_data[offset_weight] * top_diff[offset_top];
					}
				}
			}
		}
		bottom_diff[index] = value;
	}
}
'''


_aggregation_zeropad_weight_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_weight_backward_kernel(
		const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* weight_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${weight_channels} / ${top_height} / ${top_width};
		const int c = (index / ${top_height} / ${top_width}) % ${weight_channels};
		const int h = (index / ${top_width}) % ${top_height};
		const int w = index % ${top_width};
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
				const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
				const int offset_weight = ((n * ${weight_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
				${Dtype} value = 0;
				if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
					for (int cc = c; cc < ${input_channels}; cc += ${weight_channels}) {
						const int offset_bottom = ((n * ${input_channels} + cc) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
						const int offset_top = ((n * ${input_channels} + cc) * ${top_height} + h) * ${top_width} + w;
						value += bottom_data[offset_bottom] * top_diff[offset_top];
					}
				}
				weight_diff[offset_weight] = value;
			}
		}
	}
}
'''


class AggregationZeropad(Function):
		@staticmethod
		def forward(ctx, input, weight, kernel_size, stride, padding, dilation):
				kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
				ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
				assert input.dim() == 4 and input.is_cuda and weight.is_cuda
				batch_size, input_channels, input_height, input_width = input.size()
				_, weight_channels, weight_height, weight_width = weight.size()
				output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
				output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
				assert output_height * output_width == weight_width
				output = input.new(batch_size, input_channels, output_height, output_width)
				n = output.numel()
				with torch.cuda.device_of(input):
						f = load_kernel('aggregation_zeropad_forward_kernel', _aggregation_zeropad_forward_kernel, Dtype=Dtype(input), nthreads=n,
														num=batch_size, input_channels=input_channels, weight_channels=weight_channels,
														bottom_height=input_height, bottom_width=input_width,
														top_height=output_height, top_width=output_width,
														kernel_h=kernel_size[0], kernel_w=kernel_size[1],
														stride_h=stride[0], stride_w=stride[1],
														dilation_h=dilation[0], dilation_w=dilation[1],
														pad_h=padding[0], pad_w=padding[1])
						f(block=(CUDA_NUM_THREADS, 1, 1),
							grid=(GET_BLOCKS(n), 1, 1),
							args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
							stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				ctx.save_for_backward(input, weight)
				return output

		@staticmethod
		def backward(ctx, grad_output):
				kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
				input, weight = ctx.saved_tensors
				assert grad_output.is_cuda
				if not grad_output.is_contiguous():
						grad_output = grad_output.contiguous()
				batch_size, input_channels, input_height, input_width = input.size()
				_, weight_channels, weight_height, weight_width = weight.size()
				output_height, output_width = grad_output.size()[2:]
				grad_input, grad_weight = None, None
				opt = dict(Dtype=Dtype(grad_output),
									 num=batch_size, input_channels=input_channels, weight_channels=weight_channels,
									 bottom_height=input_height, bottom_width=input_width,
									 top_height=output_height, top_width=output_width,
									 kernel_h=kernel_size[0], kernel_w=kernel_size[1],
									 stride_h=stride[0], stride_w=stride[1],
									 dilation_h=dilation[0], dilation_w=dilation[1],
									 pad_h=padding[0], pad_w=padding[1])
				with torch.cuda.device_of(input):
						if ctx.needs_input_grad[0]:
								grad_input = input.new(input.size())
								n = grad_input.numel()
								opt['nthreads'] = n
								f = load_kernel('aggregation_zeropad_input_backward_kernel', _aggregation_zeropad_input_backward_kernel, **opt)
								f(block=(CUDA_NUM_THREADS, 1, 1),
									grid=(GET_BLOCKS(n), 1, 1),
									args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
									stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
						if ctx.needs_input_grad[1]:
								grad_weight = weight.new(weight.size())
								n = grad_weight.numel() // weight.shape[2]
								opt['nthreads'] = n
								f = load_kernel('aggregation_zeropad_weight_backward_kernel', _aggregation_zeropad_weight_backward_kernel, **opt)
								f(block=(CUDA_NUM_THREADS, 1, 1),
									grid=(GET_BLOCKS(n), 1, 1),
									args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
									stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				return grad_input, grad_weight, None, None, None, None

_aggregation_refpad_forward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_refpad_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / ${top_height} / ${top_width};
		const int c = (index / ${top_height} / ${top_width}) % ${input_channels};
		const int h = (index / ${top_width}) % ${top_height};
		const int w = index % ${top_width};
		${Dtype} value = 0;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
				int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
				const int offset_weight = ((n * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
				int offset_bottom;
				if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
					offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
				}
				else {
					if (h_in < 0) h_in = -h_in;
					if (h_in >= ${bottom_height}) h_in = 2 * (${bottom_height} - 1) - h_in;
					if (w_in < 0) w_in = -w_in;
					if (w_in >= ${bottom_width}) w_in = 2 * (${bottom_width} - 1) - w_in;
					offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
				}
				value += weight_data[offset_weight] * bottom_data[offset_bottom];
			}
		}
		top_data[index] = value;
	}
}
'''


_aggregation_refpad_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_refpad_input_backward_kernel(
		const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* bottom_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${input_channels} / (${bottom_height} + 2 * ${pad_h}) / (${bottom_width} + 2 * ${pad_w});
		const int c = (index / (${bottom_height} + 2 * ${pad_h}) / (${bottom_width} + 2 * ${pad_w})) % ${input_channels};
		const int h = (index / (${bottom_width} + 2 * ${pad_w})) % (${bottom_height} + 2 * ${pad_h});
		const int w = index % (${bottom_width} + 2 * ${pad_w});
		${Dtype} value = 0;
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				const int h_out_s = h - kh * ${dilation_h};
				const int w_out_s = w - kw * ${dilation_w};
				if ((h_out_s % ${stride_h} == 0) && (w_out_s % ${stride_w} == 0)) {
					const int h_out = h_out_s / ${stride_h};
					const int w_out = w_out_s / ${stride_w};
					if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
						const int offset_top = ((n * ${input_channels} + c) * ${top_height} + h_out) * ${top_width} + w_out;
						const int offset_weight = ((n * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
						value += weight_data[offset_weight] * top_diff[offset_top];
					}
				}
			}
		}
		bottom_diff[index] = value;
	}
}
'''


_aggregation_refpad_weight_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_refpad_weight_backward_kernel(
		const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* weight_diff) {
	CUDA_KERNEL_LOOP(index, ${nthreads}) {
		const int n = index / ${weight_channels} / ${top_height} / ${top_width};
		const int c = (index / ${top_height} / ${top_width}) % ${weight_channels};
		const int h = (index / ${top_width}) % ${top_height};
		const int w = index % ${top_width};
		for (int kh = 0; kh < ${kernel_h}; ++kh) {
			for (int kw = 0; kw < ${kernel_w}; ++kw) {
				int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
				int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
				const int offset_weight = ((n * ${weight_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
				${Dtype} value = 0;
				for (int cc = c; cc < ${input_channels}; cc += ${weight_channels}) {
					const int offset_top = ((n * ${input_channels} + cc) * ${top_height} + h) * ${top_width} + w;
					int offset_bottom;
					if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
						offset_bottom = ((n * ${input_channels} + cc) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
					}
					else {
						if (h_in < 0) h_in = -h_in;
						if (h_in >= ${bottom_height}) h_in = 2 * (${bottom_height} - 1) - h_in;
						if (w_in < 0) w_in = -w_in;
						if (w_in >= ${bottom_width}) w_in = 2 * (${bottom_width} - 1) - w_in;
						offset_bottom = ((n * ${input_channels} + cc) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
					}
					value += bottom_data[offset_bottom] * top_diff[offset_top];
				}
				weight_diff[offset_weight] = value;
			}
		}
	}
}
'''

class AggregationRefpad(Function):
		@staticmethod
		def forward(ctx, input, weight, kernel_size, stride, padding, dilation):
				kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
				ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
				assert input.dim() == 4 and input.is_cuda and weight.is_cuda
				batch_size, input_channels, input_height, input_width = input.size()
				_, weight_channels, weight_height, weight_width = weight.size()
				output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
				output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
				assert output_height * output_width == weight_width
				output = input.new(batch_size, input_channels, output_height, output_width)
				n = output.numel()
				with torch.cuda.device_of(input):
						f = load_kernel('aggregation_refpad_forward_kernel', _aggregation_refpad_forward_kernel, Dtype=Dtype(input), nthreads=n,
														num=batch_size, input_channels=input_channels, weight_channels=weight_channels,
														bottom_height=input_height, bottom_width=input_width,
														top_height=output_height, top_width=output_width,
														kernel_h=kernel_size[0], kernel_w=kernel_size[1],
														stride_h=stride[0], stride_w=stride[1],
														dilation_h=dilation[0], dilation_w=dilation[1],
														pad_h=padding[0], pad_w=padding[1])
						f(block=(CUDA_NUM_THREADS, 1, 1),
							grid=(GET_BLOCKS(n), 1, 1),
							args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
							stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				ctx.save_for_backward(input, weight)
				return output

		@staticmethod
		def backward(ctx, grad_output):
				kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
				input, weight = ctx.saved_tensors
				assert grad_output.is_cuda
				if not grad_output.is_contiguous():
						grad_output = grad_output.contiguous()
				batch_size, input_channels, input_height, input_width = input.size()
				_, weight_channels, weight_height, weight_width = weight.size()
				output_height, output_width = grad_output.size()[2:]
				grad_input, grad_weight = None, None
				opt = dict(Dtype=Dtype(grad_output),
									 num=batch_size, input_channels=input_channels, weight_channels=weight_channels,
									 bottom_height=input_height, bottom_width=input_width,
									 top_height=output_height, top_width=output_width,
									 kernel_h=kernel_size[0], kernel_w=kernel_size[1],
									 stride_h=stride[0], stride_w=stride[1],
									 dilation_h=dilation[0], dilation_w=dilation[1],
									 pad_h=padding[0], pad_w=padding[1])
				with torch.cuda.device_of(input):
						if ctx.needs_input_grad[0]:
								grad_input = input.new(batch_size, input_channels, input_height + 2 * padding[0], input_width + 2 * padding[1])
								n = grad_input.numel()
								opt['nthreads'] = n
								f = load_kernel('aggregation_refpad_input_backward_kernel', _aggregation_refpad_input_backward_kernel, **opt)
								f(block=(CUDA_NUM_THREADS, 1, 1),
									grid=(GET_BLOCKS(n), 1, 1),
									args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
									stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
								grad_input[:, :, padding[0] + 1:2 * padding[0] + 1, :] += torch.flip(grad_input[:, :, :padding[0], :], dims=[2])
								grad_input[:, :, input_height - 1:input_height + padding[0] - 1, :] += torch.flip(grad_input[:, :, input_height + padding[0]:, :], dims=[2])
								grad_input[:, :, :, padding[1] + 1:2 * padding[1] + 1] += torch.flip(grad_input[:, :, :, :padding[1]], dims=[3])
								grad_input[:, :, :, input_width - 1:input_width + padding[1] - 1] += torch.flip(grad_input[:, :, :, input_width + padding[1]:], dims=[3])
								grad_input = grad_input[:, :, padding[0]:padding[0]+input_height, padding[1]:padding[1]+input_width]

						if ctx.needs_input_grad[1]:
								grad_weight = weight.new(weight.size())
								n = grad_weight.numel() // weight.shape[2]
								opt['nthreads'] = n
								f = load_kernel('aggregation_refpad_weight_backward_kernel', _aggregation_refpad_weight_backward_kernel, **opt)
								f(block=(CUDA_NUM_THREADS, 1, 1),
									grid=(GET_BLOCKS(n), 1, 1),
									args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
									stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
				return grad_input, grad_weight, None, None, None, None
#############cuda implementation################

class Subtraction(nn.Module):
	def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
		super(Subtraction, self).__init__()
		self.kernel_size = _pair(kernel_size)
		self.stride = _pair(stride)
		self.padding = _pair(padding)
		self.dilation = _pair(dilation)
		self.pad_mode = pad_mode

	def forward(self, input):
		assert input.dim() == 4 and pad_mode in [0, 1]
		if input.is_cuda:
			if self.pad_mode == 0:
				out = SubtractionZeropad.apply(input, self.kernel_size, self.stride, self.padding, self.dilation)
			elif self.pad_mode == 1:
				out = SubtractionRefpad.apply(input, self.kernel_size, self.stride, self.padding, self.dilation)
		else:
			raise NotImplementedError
		return out

class Subtraction2(nn.Module):
	def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
		super(Subtraction2, self).__init__()
		self.kernel_size = _pair(kernel_size)
		self.stride = _pair(stride)
		self.padding = _pair(padding)
		self.dilation = _pair(dilation)
		self.pad_mode = pad_mode

	def forward(self, input1, input2):
		assert input1.dim() == 4 and input2.dim() == 4 and pad_mode in [0, 1]
		if input1.is_cuda:
			if self.pad_mode == 0:
				out = Subtraction2Zeropad.apply(input1, input2, self.kernel_size, self.stride, self.padding, self.dilation)
			elif self.pad_mode == 1:
				out = Subtraction2Refpad.apply(input1, input2, self.kernel_size, self.stride, self.padding, self.dilation)
		else:
			raise NotImplementedError
		return out

class Aggregation(nn.Module):
	def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
		super(Aggregation, self).__init__()
		self.kernel_size = _pair(kernel_size)
		self.stride = _pair(stride)
		self.padding = _pair(padding)
		self.dilation = _pair(dilation)
		self.pad_mode = pad_mode

	def forward(self, input, weight):
		assert input.shape[0] == weight.shape[0] and (input.shape[1] % weight.shape[1] == 0) and self.pad_mode in [0, 1]
		if input.is_cuda:
			if self.pad_mode == 0:
				out = AggregationZeropad.apply(input, weight, self.kernel_size, self.stride, self.padding, self.dilation)
			elif self.pad_mode == 1:
				out = AggregationRefpad.apply(input, weight, self.kernel_size, self.stride, self.padding, self.dilation)
		else:
			raise NotImplementedError
		return out

def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def position(H, W, is_cuda=True):
	if is_cuda:
		loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
		loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
	else:
		loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
		loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
	loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
	return loc

class SAM(nn.Module):
	def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3,
				stride=1, dilation=1):
		super(SAM, self).__init__()
		self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
		self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
		self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
		self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
		if sa_type == 0:
			self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
									nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
									nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
									nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
			self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
			self.subtraction = Subtraction(kernel_size, stride, (dilation*(kernel_size-1)+1) // 2,
											dilation, pad_mode=1)
			self.subtraction2 = Subtraction2(kernel_size, stride, (dilation*(kernel_size-1)+1) // 2,
											dilation, pad_mode=1)
			self.softmax = nn.Softmax(dim=-2)
		else:
			self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
										nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
										nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
										nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
			self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
			self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
			self.pad = nn.ReflectionPad2d(kernel_size // 2)
		self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

	def forward(self, x):
		x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
		if self.sa_type == 0: # pairwise
			p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
			w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0],1,1,1)], 1)))
		else: # patchwise
			if self.stride != 1:
				x1 = self.unfold_i(x1)
			x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
			x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
			w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
		x = self.aggregation(x3, w)
		return x


class Bottleneck(nn.Module):
	def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8,
					kernel_size=7, stride=1):
		super(Bottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
		self.bn2 = nn.BatchNorm2d(mid_planes)
		self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
		self.relu = nn.ReLU(inplace=True)
		self.stride = stride

	def forward(self, x):
		identity = x
		out = self.relu(self.bn1(x))
		out = self.relu(self.bn2(self.sam(out)))
		out = self.conv(out)
		out += identity
		return out

class _SAN(nn.Module):
	def __init__(self, sa_type, block, layers, kernels, num_classes):
		super(_SAN, self).__init__()
		c = 64
		self.conv_in, self.bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
		self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
		self.layer0 = self._make_layer(sa_type, block, c, layers[0], kernels[0])

		c *= 4
		self.conv1, self.bn1 = conv1x1(c // 4, c), nn.BatchNorm2d(c)
		self.layer1 = self._make_layer(sa_type, block, c, layers[1], kernels[1])

		c *= 2
		self.conv2, self.bn2 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
		self.layer2 = self._make_layer(sa_type, block, c, layers[2], kernels[2])

		c *= 2
		self.conv3, self.bn3 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
		self.layer3 = self._make_layer(sa_type, block, c, layers[3], kernels[3])

		c *= 2
		self.conv4, self.bn4 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
		self.layer4 = self._make_layer(sa_type, block, c, layers[4], kernels[4])

		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(c, num_classes)

	def _make_layer(self, sa_type, block, planes, blocks, kernel_size=7, stride=1):
		layers = []
		for _ in range(0, blocks):
			layers.append(block(sa_type, planes, planes//16, planes//4, planes, 8, kernel_size, stride))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.relu(self.bn_in(self.conv_in(x)))
		x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
		x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
		x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
		x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
		x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

@register_model
def SAN_model(opt):
	model = _SAN(opt.sa_type, Bottleneck, opt.layers, opt.kernels, opt.num_classes)
	return model.cuda()