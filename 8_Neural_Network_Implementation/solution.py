from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return inputs * (inputs >= 0)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * (self.forward_inputs >= 0)
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        max_power = np.max(inputs, axis=1)
        inputs = inputs - max_power[..., None]
        denom = np.sum(np.exp(inputs), axis=1)
        return np.exp(inputs) / denom[..., None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        res = np.sum(grad_outputs * self.forward_outputs, axis=1) 
        return self.forward_outputs * (grad_outputs - res[..., None])
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        return inputs @ self.weights + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        self.biases_grad = np.sum(grad_outputs, axis=0)
        self.weights_grad = self.forward_inputs.T @ grad_outputs
        return grad_outputs @ self.weights.T
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        # your code here \/
        return -np.mean(np.sum(y_gt * np.log(y_pred), axis=1), axis=0).reshape(-1)
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        # your code here \/
        return -y_gt/np.maximum(y_pred, eps) / y_pred.shape[0]
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(0.001, momentum=0.9))
    


    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(50, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(100))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=16, epochs=2, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # your code here \/
    #print(f"{inputs.shape=}")
    n, d = inputs.shape[:2]
    c = kernels.shape[0]
    kh, kw = kernels.shape[2:]
    ih, iw = inputs.shape[2:]
    oh, ow = ih + 2 * padding - kh + 1, iw + 2 * padding - kw + 1
    h, w = ih + 2 * (kh - 1), iw + 2 * (kh - 1)
    result = np.zeros((n, c, h - kh + 1, w - kw + 1))
    for img_num, input in enumerate(inputs):
        img = np.zeros((d, ih + 2 * (kh - 1), iw + 2 * (kw - 1)))
        img[..., kh-1:ih + kh - 1, kh-1:iw + kh - 1] = input 
        for i in range(h - kh + 1):
            for j in range(w - kh + 1):
                result[img_num, :, i, j] = np.sum(img[:, i:i + kh, j:j + kh][:,::-1,::-1] * kernels, axis = (1,2,3))
    
    # img = np.zeros((n, d, ih + 2 * (kh - 1), iw + 2 * (kw - 1)))
    # img[:, :, kh-1:ih + kh - 1, kh-1:iw + kh - 1] = inputs
    # for i in range(ih + kh - 1):
    #     for j in range(iw + kw - 1):
    #         result[:, :, i, j] = np.sum(
    #                     np.expand_dims(img[:, :, i:i + kh, j:j + kh][:,::-1,::-1], axis=1) * \
    #                     np.expand_dims(kernels, axis=0), 
    #                     axis = (2,3,4))
    
    dx, dy = kh - 1 - padding, kh - 1 - padding
    if dx == 0:
        return result
    elif dx > 0:
        #print(f"{result.shape}, {result[..., dx:-dx, dy:-dy].shape=}, {dx=}, {dy=}")
        return result[..., dx:-dx, dy:-dy]
    total_result = np.zeros((n, c, oh, ow))
    total_result[..., -dx:dx, -dy:dy] = result
    return total_result
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        self.forward_inputs = inputs
        self.p = (self.kernel_size) // 2
        return convolve(inputs, self.kernels, padding=self.p) + self.biases[None, ..., None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        self.kernels_grad = np.transpose(
                convolve(
                        np.transpose(self.forward_inputs[..., ::-1, ::-1], axes=(1, 0, 2, 3)), 
                        np.transpose(grad_outputs, axes=(1, 0, 2, 3)), padding=self.p
                        ),
                axes=(1, 0, 2, 3))
        return convolve(grad_outputs, np.transpose(self.kernels[..., ::-1, ::-1], axes=(1, 0, 2, 3)), padding=self.p)
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        n, d, ih, iw = inputs.shape
        oh, ow = ih // self.pool_size, iw // self.pool_size
        if self.pool_mode == 'max':
            max_matrix = np.max(inputs.reshape(n, d, oh, self.pool_size, ow, self.pool_size), axis=(3, 5))
            max_mask = (inputs.reshape(n, d, oh, self.pool_size, ow, self.pool_size) == max_matrix[:, :, :, None, :, None])
            dif_adder = np.arange(self.pool_size * self.pool_size, 0, -1).reshape(self.pool_size, self.pool_size)
            # assert(np.sum(max_mask) + np.sum(1 - max_mask) == n * d * ih * iw)
            max_mask = max_mask.astype(np.int32) * dif_adder[:, None, :]
            self.max_mask = (max_mask == np.max(max_mask, axis=(3, 5))[:, :, :, None, :, None])
            # assert(np.sum(max_mask) == n * d * oh * ow)
            return max_matrix
        else:
            mean_matrix = np.mean(inputs.reshape(n, d, oh, self.pool_size, ow, self.pool_size), axis=(3, 5))
            return mean_matrix
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        if self.pool_mode == 'max':
            return (self.max_mask * grad_outputs[:, :, :, None, :, None]).reshape(self.forward_inputs.shape)
        else:
            mask = np.repeat(np.repeat(grad_outputs[:, :, None, :, None], self.pool_size, axis=3), self.pool_size, axis=5)
            return mask.reshape(self.forward_inputs.shape) / (self.pool_size ** 2) 
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            self.mean = np.mean(inputs, axis=(0, 2, 3))
            self.var = np.var(inputs, axis = (0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.mean + (self.momentum) * self.running_mean
            self.running_var = (1 - self.momentum) * self.var + (self.momentum) * self.running_var 
            self.forward_centered_inputs = inputs - self.mean[None, :, None, None]
            self.forward_normalized_inputs =  self.forward_centered_inputs /  np.sqrt(eps + self.var[None, ..., None, None])     
            #
            #
            return self.gamma[None, :, None, None] * self.forward_normalized_inputs + self.beta[None, :, None, None]
        else:

            norm = (inputs - self.running_mean[None, ..., None, None]) / np.sqrt((eps + self.running_var[None, ..., None, None]))
            return self.gamma[None, :, None, None] * norm + self.beta[None, :, None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        n, d, h, w = grad_outputs.shape
        N = n * h * w
        self.beta_grad = grad_outputs.sum(axis=(0, 2, 3))
        self.gamma_grad = np.sum(self.forward_normalized_inputs * grad_outputs, axis=(0, 2, 3))
        std = np.sqrt(eps + self.var) #(n,)
        grad_x_norm = grad_outputs * self.gamma[None, :, None, None]
        grad_var =  -1 / (std ** 3) / 2 * np.sum(grad_x_norm * self.forward_centered_inputs, axis=(0, 2, 3))
        grad_x1 = grad_x_norm / std[None, :, None, None] + 2 * self.forward_centered_inputs * grad_var[None, :, None, None] / N
        grad_x2 =- 1 / N * np.sum(grad_x1, axis=(0, 2, 3))[None, :, None, None]
        return grad_x1 + grad_x2
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        n, d, h, w = inputs.shape
        self.size = inputs.shape
        return inputs.reshape(n, d * h * w)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs.reshape(self.size)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        ind = np.random.uniform(0, 1, size=inputs.shape)
        inputs_copy = np.copy(inputs)
        if self.is_training:
            self.forward_mask = ind < self.p
        else:
            self.forward_mask = ind > self.p
        inputs_copy[self.forward_mask] = 0
        return inputs_copy
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        grad_outputs_copy = np.copy(grad_outputs)
        grad_outputs_copy[self.forward_mask] = 0
        return grad_outputs_copy
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr=0.001, momentum=0.9))
    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(16, kernel_size=5, input_shape=(3, 32, 32)))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(2, 'max'))
    #(16, 16)
    model.add(Conv2D(32, kernel_size=5))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(2, 'max'))
    # #(8,8)
    model.add(Conv2D(64, kernel_size=3))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(2, 'avg'))
    # #(4,4)

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Softmax())
    
    
    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=64, epochs=8, shuffle=True)#,x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model

# ============================================================================
