import numpy as np

try:
    import pycuda.driver as cuda
    import pycuda.autoinit

    from pycuda.compiler import SourceModule
    CUDA = True

except ImportError:
    CUDA = False
    print("No PyCuda functionality.")
    pass

def make_context(devicenum, log):

    if log is not None:
        log.info("======")
        log.info("=== Available devices:")
        for dnum in range(cuda.Device.count()):
            device=cuda.Device(dnum)
            attrs=device.get_attributes()
            #Beyond this point is just pretty printing
            log.info("== Device ({}): {}".format(dnum, device.name()))

            #print("\n===Attributes for device ({}): {}".format(devicenum, device.name()))
            #for (key,value) in attrs.items():
            #    print("%s:%s"%(str(key),str(value)))
        log.info("======")
        log.info("======")
        log.info("== Chosen device: ({}) {}".format(devicenum, cuda.Device(devicenum).name()))
        log.info("======")
        cuda.Context.pop()
        cuda.Device(devicenum).make_context()
    else:
        cuda.Context.pop()
        cuda.Device(devicenum).make_context()


def complex_to_float(x_complex):

    x_float = np.zeros(x_complex.shape + (2,))
    x_float[... , 0] = np.real(x_complex)
    x_float[... , 1] = np.imag(x_complex)

    return x_float.astype(np.float32)

def float_to_complex(x_float):

    x_complex = np.zeros(x_float.shape[:-1])
    x_complex = (x_float[..., 0] + 1j * x_float[..., 1])

    return x_complex

def alloc_complex(x):

    # Reshape
    x_float = complex_to_float(x)

    # Allocate memory on device
    x_gpu = cuda.mem_alloc(x_float.nbytes)

    return x_gpu, x_float

def complex_to_cuda(x):

    # Reshape
    x_float = complex_to_float(x)

    # Allocate memory on device
    x_gpu = cuda.mem_alloc(x_float.nbytes)

    # Transfer data to device
    cuda.memcpy_htod(x_gpu, x_float)

    return x_gpu

def mem_alloc_htod(x, dtype=None):

    # Allocate memory on device
    x_gpu = cuda.mem_alloc(np.array(x, dtype=dtype).nbytes)
    # Transfer data to device
    cuda.memcpy_htod(x_gpu, np.array(x, dtype=dtype))

    return x_gpu

def float_to_complex_cpu(x_float, x_gpu):

    cuda.memcpy_dtoh(x_float, x_gpu)
    x = float_to_complex(x_float)

    return x 

def mem_alloc(nbytes):

    out_gpu = cuda.mem_alloc(nbytes)

    return out_gpu

def host_mem_alloc(out):

    return cuda.pagelocked_zeros_like(out)



def memcpy_htod(out_gpu, out):

    cuda.memcpy_htod(out_gpu, np.array(out))

    return out_gpu

def memcpy_dtoh(out, out_gpu):

    cuda.memcpy_dtoh(out, out_gpu)

    return out
