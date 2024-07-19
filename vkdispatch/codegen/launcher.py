
class BufferKernelArgument:
    def __init__(self, var_type) -> None:
        self.var_type = var_type

class ImageKernelArgument:
    def __init__(self, var_type, dimentions: int, layers: int) -> None:
        self.var_type = var_type
        self.dimensions = dimentions
        self.layers = layers

