#pragma once 
#include <torch/torch.h>
#include <torch/arg.h>

struct BasicConvOptions
{
	BasicConvOptions(int ch_in, int ch_out, int ksize) :
		ch_in_(ch_in), ch_out_(ch_out), ksize_(ksize) {}
	TORCH_ARG(int, ch_in);
	TORCH_ARG(int, ch_out);
	TORCH_ARG(int, ksize);
	TORCH_ARG(int, stride) = 1;
	TORCH_ARG(int, padding) = 0;
};

class BasicConvImpl : public torch::nn::Cloneable<BasicConvImpl>
{
public:
	explicit BasicConvImpl(BasicConvOptions options);

	torch::Tensor forward(const torch::Tensor& input);

	void reset() override;

	/// Pretty prints the `BatchNorm` module into the given `stream`.
	void pretty_print(std::ostream& stream) const override;

protected:
	//torch::nn::Sequential seq = nullptr;
	torch::nn::Conv2d conv = nullptr;
	torch::nn::BatchNorm bn = nullptr;
};

TORCH_MODULE(BasicConv);
