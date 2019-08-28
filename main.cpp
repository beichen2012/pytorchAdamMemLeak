#include <iostream>
#include <torch/torch.h>
#include "BasicConv.hpp"


int main()
{
	try
	{
		auto device = torch::Device(torch::kCUDA, 0);

		BasicConv net(BasicConvOptions(1, 2, 3).padding(1));
		std::shared_ptr<torch::optim::Optimizer> optim_;
#if 0
		optim_ = std::shared_ptr<torch::optim::Optimizer>(
			dynamic_cast<torch::optim::Optimizer*>(
				new torch::optim::Adam(net->parameters(),
					torch::optim::AdamOptions(0.001)
					.weight_decay(0.0005))
				)
			);
#else
		optim_ = std::shared_ptr<torch::optim::Optimizer>(
			dynamic_cast<torch::optim::Optimizer*>(
				new torch::optim::RMSprop(net->parameters(),
					torch::optim::RMSpropOptions(0.001)
					.weight_decay(0.0005)
					.momentum(0.9))
				)
			);
#endif 
		net->to(device);
		while (true)
		{
			net->train();
			torch::Tensor images = torch::randn({ 1, 1, 176, 112 });
			torch::Tensor targets = torch::randint(1, { 1,176, 112 });
			images = images.to(device);
			targets = targets.to(device).toType(torch::kLong);
			auto pred = net->forward(images);

			auto prob = torch::log_softmax(pred, 1);
			auto loss = torch::nll_loss2d(prob, targets, {}, Reduction::Mean, -1);

			optim_->zero_grad();
			loss.backward();
			optim_->step();

			std::cout << loss.item<float>() << std::endl;
		}
	}
	catch (c10::Error& er)
	{
		std::string msg = std::string("c10::error->") + er.what();
		std::cout << msg << std::endl;
	}
	catch (std::runtime_error& er)
	{
		std::string msg = std::string("std::runtime_error->") + er.what();
		std::cout << msg << std::endl;
	}

	return 0;

}
