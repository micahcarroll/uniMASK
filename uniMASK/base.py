import torch

from uniMASK.utils import determine_default_torch_device

# TODO: clean up all the .to(DEVICE) calls. Probably can be done all upstream in a single point
#  https://stackoverflow.com/questions/53325418/pytorch-speed-comparison-gpu-slower-than-cpu
LOCAL = not torch.cuda.is_available()
DEVICE = torch.device(determine_default_torch_device(LOCAL))


def go_to_cpu():
    """Switches all computation to happen on CPU"""
    import torch

    global DEVICE
    DEVICE = torch.device("cpu")
    print("Switching to", DEVICE)


def back_to_default():
    """Switches all computation to happen on the default device"""
    assert False, "Currently this code doesn't work"
    global DEVICE
    DEVICE = torch.device("cpu" if LOCAL else "cuda:{}".format(GPU_ID))
    print("Back to default", DEVICE)
