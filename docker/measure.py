"""
Start a background thread to measure the GPU usage of the segmentation code

Create a thread in the background and catch the invocation of the GPU

"""

import nvidia_smi

print(dir(nvidia_smi))

nvidia_smi.nvmlInit()
nvidia_smi.nvmlDeviceGetAccountingMode(0)#0, 1)