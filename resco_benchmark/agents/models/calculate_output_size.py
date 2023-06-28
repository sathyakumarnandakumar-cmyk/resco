def conv2d_size_out(size, kernel_size=2, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1    
