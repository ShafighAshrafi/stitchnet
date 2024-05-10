import torch.optim as optim


def create_optimizer(model_to_fit, feature_extract=False):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_to_fit.parameters()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_to_fit.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                # print("\t", name)
    # else:
    #     for name, param in model_to_fit.named_parameters():
    #         if param.requires_grad:
    #             print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer