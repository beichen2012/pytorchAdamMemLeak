#include "BasicConv.hpp"

BasicConvImpl::BasicConvImpl(BasicConvOptions options) : 
	conv(torch::nn::Conv2dOptions(
			options.ch_in_, options.ch_out_, options.ksize_).stride(options.stride_).padding(options.padding_)),
	bn(options.ch_out_)
{
	register_module("conv", conv);
	register_module("bn", bn);
}
torch::Tensor BasicConvImpl::forward(const torch::Tensor& input)
{
	auto x = conv->forward(input);
	x = bn->forward(x);
	x = torch::relu(x);
	return x;
}

void BasicConvImpl::reset()
{
	conv->reset();
	bn->reset();
}

/// Pretty prints the `BatchNorm` module into the given `stream`.
void BasicConvImpl::pretty_print(std::ostream& stream) const
{
	conv->pretty_print(stream);
	bn->pretty_print(stream);
}
