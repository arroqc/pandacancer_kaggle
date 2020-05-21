
import os
from joblib import Parallel, delayed
import torch.distributions as dist
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_patch(img_b, coord, patch_size):
    """ Extract a single patch """
    x_start = int(coord[0])
    x_end = x_start + int(patch_size[0])
    y_start = int(coord[1])
    y_end = y_start + int(patch_size[1])

    patch = img_b[:, x_start:x_end, y_start:y_end]
    return patch


def _extract_patches_batch(b, img, offsets, patch_size, num_patches, extract_patch_parallel=False):
    """ Extract patches for a single batch. This function can be called in a for loop or in parallel.
        This functions returns a tensor of patches of size [num_patches, channels, width, height] """
    patches = []

    # Extracting in parallel is more expensive than doing it sequentially. This I left it in here
    if extract_patch_parallel:
        num_jobs = min(os.cpu_count(), num_patches)
        patches = Parallel(n_jobs=num_jobs)(
            delayed(_extract_patch)(img[b], offsets[b, p], patch_size) for p in range(num_patches))

    else:
        # Run extraction sequentially
        for p in range(num_patches):
            patch = _extract_patch(img[b], offsets[b, p], patch_size)
            patches.append(patch)

    return torch.stack(patches)


def extract_patches(img, offsets, patch_size, extract_batch_parallel=False):
    img = img.permute(0, 3, 1, 2)

    num_patches = offsets.shape[1]
    batch_size = img.shape[0]

    # I pad the images with zeros for the cases that a part of the patch falls outside the image
    pad_const = int(patch_size[0].item() / 2)
    pad_func = torch.nn.ConstantPad2d(pad_const, 0.0)
    img = pad_func(img)

    # Add the pad_const to the offsets, because everything is now shifted by pad_const
    offsets = offsets + pad_const

    all_patches = []

    # Extracting in parallel is more expensive than doing it sequentially. This I left it in here
    if extract_batch_parallel:
        num_jobs = min(os.cpu_count(), batch_size)
        all_patches = Parallel(n_jobs=num_jobs)(
            delayed(_extract_patches_batch)(b, img, offsets, patch_size, num_patches) for b in range(batch_size))

    else:
        # Run sequentially over the elements in the batch
        for b in range(batch_size):
            patches = _extract_patches_batch(b, img, offsets, patch_size, num_patches)
            all_patches.append(patches)

    return torch.stack(all_patches)


class FromTensors:
    def __init__(self, xs, y):
        """Given input tensors for each level of resolution provide the patches.
        Arguments
        ---------
        xs: list of tensors, one tensor per resolution in ascending
            resolutions, namely the lowest resolution is 0 and the highest
            is len(xs)-1
        y: tensor or list of tensors or None, the targets can be anything
           since it is simply returned as is
        """
        self._xs = xs
        self._y = y

    def targets(self):
        # Since the xs were also given to us the y is also given to us
        return self._y

    def inputs(self):
        # We leave it to the caller to add xs and y to the input list if they
        # are placeholders
        return []

    def patches(self, samples, offsets, sample_space, previous_patch_size,
                patch_size, fromlevel, tolevel):
        device = samples.device

        # Make sure everything is a tensor
        sample_space = to_tensor(sample_space, device=device)
        previous_patch_size = to_tensor(previous_patch_size, device=device)
        patch_size = to_tensor(patch_size, device=device)
        shape_from = self._shape(fromlevel)
        shape_to = self._shape(tolevel)

        # Compute the scales
        scale_samples = self._scale(sample_space, tolevel).to(device)
        scale_offsets = self._scale(shape_from, shape_to).to(device)

        # Steps is the offset per pixel of the sample space. Pixel zero should
        # be at position steps/2 and the last pixel should be at
        # space_available - steps/2.
        space_available = to_float32(previous_patch_size) * scale_offsets
        steps = space_available / to_float32(sample_space)

        # Compute the patch start which are also the offsets to be returned
        offsets = to_int32(torch.round(
            to_float32(offsets) * expand_many(scale_offsets, [0, 0]) +
            to_float32(samples) * expand_many(steps, [0, 0]) +
            expand_many(steps / 2, [0, 0]) -
            expand_many(to_float32(patch_size) / 2, [0, 0])
        ))

        # Extract the patches
        patches = extract_patches(
            self._xs[tolevel],
            offsets,
            patch_size
        )

        return patches, offsets

    def data(self, level):
        return self._xs[level]

    def _scale(self, shape_from, shape_to):
        # Compute the tensor that needs to be multiplied with `shape_from` to
        # get `shape_to`
        shape_from = to_float32(to_tensor(shape_from))
        shape_to = to_float32(to_tensor(shape_to))

        return shape_to / shape_from

    def _shape(self, level):
        x = self._xs[level]
        int_shape = x.shape[1:-1]
        if not any(s is None for s in int_shape):
            return int_shape

        return x.shape[1:-1]


def to_tensor(x, dtype=torch.int32, device=None):
    """If x is a Tensor return it as is otherwise return a constant tensor of
    type dtype."""
    device = torch.device('cpu') if device is None else device
    if torch.is_tensor(x):
        return x.to(device)

    return torch.tensor(x, dtype=dtype, device=device)


def to_dtype(x, dtype):
    """Cast Tensor x to the dtype """
    return x.type(dtype)


to_float16 = partial(to_dtype, dtype=torch.float16)
to_float32 = partial(to_dtype, dtype=torch.float32)
to_float64 = partial(to_dtype, dtype=torch.float64)
to_double = to_float64
to_int8 = partial(to_dtype, dtype=torch.int8)
to_int16 = partial(to_dtype, dtype=torch.int16)
to_int32 = partial(to_dtype, dtype=torch.int32)
to_int64 = partial(to_dtype, dtype=torch.int64)


def expand_many(x, axes):
    """Call expand_dims many times on x once for each item in axes."""
    for ax in axes:
        x = torch.unsqueeze(x, ax)
    return x


def _sample_with_replacement(logits, n_samples):
    """Sample with replacement using the pytorch categorical distribution op."""
    distribution = dist.categorical.Categorical(logits=logits)
    return distribution.sample(sample_shape=torch.Size([n_samples])).transpose(0, 1)


def _sample_without_replacement(logits, n_samples):
    """Sample without replacement using the Gumbel-max trick.
    See lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
    """
    z = -torch.log(-torch.log(torch.rand_like(logits)))
    return torch.topk(logits+z, k=n_samples)[1]


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return torch.stack(tuple(reversed(out)))


def sample(n_samples, attention, sample_space, replace=False,
           use_logits=False):
    """Sample from the passed in attention distribution.
    Arguments
    ---------
    n_samples: int, the number of samples per datapoint
    attention: tensor, the attention distribution per datapoint (could be logits
               or normalized)
    sample_space: This should always equal K.shape(attention)[1:]
    replace: bool, sample with replacement if set to True (defaults to False)
    use_logits: bool, assume the input is logits if set to True (defaults to False)
    """
    # Make sure we have logits and choose replacement or not
    logits = attention if use_logits else torch.log(attention)
    sampling_function = (
        _sample_with_replacement if replace
        else _sample_without_replacement
    )

    # Flatten the attention distribution and sample from it
    logits = logits.reshape(-1, sample_space[0]*sample_space[1])
    samples = sampling_function(logits, n_samples)

    # Unravel the indices into sample_space
    batch_size = attention.shape[0]
    n_dims = len(sample_space)

    # Gather the attention
    attention = attention.view(batch_size, 1, -1).expand(batch_size, n_samples, -1)
    sampled_attention = torch.gather(attention, -1, samples[:, :, None])[:, :, 0]

    samples = unravel_index(samples.reshape(-1, ), sample_space)
    samples = torch.reshape(samples.transpose(1, 0), (batch_size, n_samples, n_dims))

    return samples, sampled_attention


class ExpectWithReplacement(torch.autograd.Function):
    """ Custom pytorch layer for calculating the expectation of the sampled patches
        with replacement.
    """
    @staticmethod
    def forward(ctx, weights, attention, features):

        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)

        F = torch.sum(wf * features, dim=1)

        ctx.save_for_backward(weights, attention, features, F)
        return F

    @staticmethod
    def backward(ctx, grad_output):
        weights, attention, features, F = ctx.saved_tensors
        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)

        grad = torch.unsqueeze(grad_output, 1)

        # Gradient wrt to the attention
        ga = grad * features
        ga = torch.sum(ga, axis=list(range(2, len(ga.shape))))
        ga = ga * weights / attention

        # Gradient wrt to the features
        gf = wf * grad

        return None, ga, gf


class ExpectWithoutReplacement(torch.autograd.Function):
    """ Custom pytorch layer for calculating the expectation of the sampled patches
        without replacement.
    """

    @staticmethod
    def forward(ctx, weights, attention, features):
        # Reshape the passed weights and attention in feature compatible shapes
        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)
        af = expand_many(attention, axes)

        # Compute how much of the probablity mass was available for each sample
        pm = 1 - torch.cumsum(attention, axis=1)
        pmf = expand_many(pm, axes)

        # Compute the features
        Fa = af * features
        Fpm = pmf * features
        Fa_cumsum = torch.cumsum(Fa, axis=1)
        F_estimator = Fa_cumsum + Fpm

        F = torch.sum(wf * F_estimator, axis=1)

        ctx.save_for_backward(weights, attention, features, pm, pmf, Fa, Fpm, Fa_cumsum, F_estimator)

        return F

    @staticmethod
    def backward(ctx, grad_output):
        weights, attention, features, pm, pmf, Fa, Fpm, Fa_cumsum, F_estimator = ctx.saved_tensors
        device = weights.device

        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)
        af = expand_many(attention, axes)

        N = attention.shape[1]
        probs = attention / pm
        probsf = expand_many(probs, axes)
        grad = torch.unsqueeze(grad_output, 1)

        # Gradient wrt to the attention
        ga1 = F_estimator / probsf
        ga2 = (
                torch.cumsum(features, axis=1) -
                expand_many(to_float32(torch.arange(0, N, device=device)), [0] + axes) * features
        )
        ga = grad * (ga1 + ga2)
        ga = torch.sum(ga, axis=list(range(2, len(ga.shape))))
        ga = ga * weights

        # Gradient wrt to the features
        gf = expand_many(to_float32(torch.arange(N-1, -1, -1, device=device)), [0] + axes)
        gf = pmf + gf * af
        gf = wf * gf
        gf = gf * grad

        return None, ga, gf


class Expectation(nn.Module):
    """ Approximate the expectation of all the features under the attention
        distribution (and its gradient) given a sampled set.
        Arguments
        ---------
        attention: Tensor of shape (B, N) containing the attention values that
                   correspond to the sampled features
        features: Tensor of shape (B, N, ...) containing the sampled features
        replace: bool describing if we sampled with or without replacement
        weights: Tensor of shape (B, N) or None to weigh the samples in case of
                 multiple samplings of the same position. If None it defaults
                 o torch.ones(B, N)
        """

    def __init__(self, replace=False):
        super(Expectation, self).__init__()
        self._replace = replace

        self.E = ExpectWithReplacement() if replace else ExpectWithoutReplacement()

    def forward(self, features, attention, weights=None):
        if weights is None:
            weights = torch.ones_like(attention) / float(attention.shape[1])

        return self.E.apply(weights, attention, features)


class SamplePatches(nn.Module):
    """SamplePatches samples from a high resolution image using an attention
    map. The layer expects the following inputs when called `x_low`, `x_high`,
    `attention`. `x_low` corresponds to the low resolution view of the image
    which is used to derive the mapping from low resolution to high. `x_high`
    is the tensor from which we extract patches. `attention` is an attention
    map that is computed from `x_low`.
    Arguments
    ---------
        n_patches: int, how many patches should be sampled
        patch_size: int, the size of the patches to be sampled (squared)
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.
        replace: bool, whether we should sample with replacement or without
        use_logits: bool, whether of not logits are used in the attention map
    """

    def __init__(self, n_patches, patch_size, receptive_field=0, replace=False,
                 use_logits=False, **kwargs):
        self._n_patches = n_patches
        self._patch_size = (patch_size, patch_size)
        self._receptive_field = receptive_field
        self._replace = replace
        self._use_logits = use_logits

        super(SamplePatches, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Legacy function of the pytorch implementation """
        shape_low, shape_high, shape_att = input_shape

        # Figure out the shape of the patches
        patch_shape = (shape_high[1], *self._patch_size)

        patches_shape = (shape_high[0], self._n_patches, *patch_shape)

        # Sampled attention
        att_shape = (shape_high[0], self._n_patches)

        return [patches_shape, att_shape]

    def forward(self, x_low, x_high, attention):
        sample_space = attention.shape[1:]
        samples, sampled_attention = sample(
            self._n_patches,
            attention,
            sample_space,
            replace=self._replace,
            use_logits=self._use_logits
        )

        offsets = torch.zeros_like(samples).float()
        if self._receptive_field > 0:
            offsets = offsets + self._receptive_field / 2

        # Get the patches from the high resolution data
        # Make sure that below works
        x_low = x_low.permute(0, 2, 3, 1)
        x_high = x_high.permute(0, 2, 3, 1)
        assert x_low.shape[-1] == x_high.shape[-1], "Channels should be last for now"
        patches, _ = FromTensors([x_low, x_high], None).patches(
            samples,
            offsets,
            sample_space,
            torch.Tensor([x_low.shape[1:-1]]).view(-1) - self._receptive_field,
            self._patch_size,
            0,
            1
        )

        return [patches, sampled_attention]


class ATSModel(nn.Module):
    """ Attention sampling model that perform the entire process of calculating the
        attention map, sampling the patches, calculating the features of the patches,
        the expectation and classifices the features.
        Arguments
        ---------
        attention_model: pytorch model, that calculated the attention map given a low
                         resolution input image
        feature_model: pytorch model, that takes the patches and calculated features
                       of the patches
        classifier: pytorch model, that can do a classification into the number of
                    classes for the specific problem
        n_patches: int, the number of patches to sample
        patch_size: int, the patch size (squared)
        receptive_field: int, how large is the receptive field of the attention network.
                         It is used to map the attention to high resolution patches.
        replace: bool, if to sample with our without replacment
        use_logts: bool, if to use logits when sampling
    """

    def __init__(self, attention_model, feature_model, classifier, n_patches, patch_size, receptive_field=0,
                 replace=False, use_logits=False):
        super(ATSModel, self).__init__()

        self.attention_model = attention_model
        self.feature_model = feature_model
        self.classifier = classifier

        self.sampler = SamplePatches(n_patches, patch_size, receptive_field, replace, use_logits)
        self.expectation = Expectation(replace=replace)

        self.patch_size = patch_size
        self.n_patches = n_patches

    def forward(self, x_low, x_high):
        # First we compute our attention map
        attention_map = self.attention_model(x_low)

        # Then we sample patches based on the attention
        patches, sampled_attention = self.sampler(x_low, x_high, attention_map)

        # We compute the features of the sampled patches
        channels = patches.shape[2]
        patches_flat = patches.view(-1, channels, self.patch_size, self.patch_size)
        patch_features = self.feature_model(patches_flat)
        dims = patch_features.shape[-1]
        patch_features = patch_features.view(-1, self.n_patches, dims)

        sample_features = self.expectation(patch_features, sampled_attention)

        y = self.classifier(sample_features)

        return y, attention_map, patches, x_low


class ClassificationHead(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()

        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.classifier(x)


class AttentionModelTrafficSigns(nn.Module):
    """ Base class for calculating the attention map of a low resolution image """

    def __init__(self,
                 squeeze_channels=False,
                 softmax_smoothing=0.0):
        super(AttentionModelTrafficSigns, self).__init__()

        conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding_mode='valid')
        relu1 = nn.ReLU()

        conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding_mode='valid')
        relu2 = nn.ReLU()

        conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding_mode='valid')
        relu3 = nn.ReLU()

        conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding_mode='valid')

        pool = nn.MaxPool2d(kernel_size=8)
        sample_softmax = SampleSoftmax(squeeze_channels, softmax_smoothing)

        self.part1 = nn.Sequential(conv1, relu1, conv2, relu2, conv3, relu3)
        self.part2 = nn.Sequential(conv4, pool, sample_softmax)

    def forward(self, x_low):
        out = self.part1(x_low)

        out = self.part2(out)

        return out


class SampleSoftmax(nn.Module):
    """ Apply softmax to the whole sample not just the last dimension.
        Arguments
        ---------
        squeeze_channels: bool, if True then squeeze the channel dimension of the input
        """

    def __init__(self, squeeze_channels=False, smooth=0):
        self.squeeze_channels = squeeze_channels
        self.smooth = smooth
        super(SampleSoftmax, self).__init__()

    def forward(self, x):
        # Apply softmax to the whole x (per sample)
        s = x.shape
        x = F.softmax(x.reshape(s[0], -1), dim=-1)

        # Smooth the distribution
        if 0 < self.smooth < 1:
            x = x * (1 - self.smooth)
            x = x + self.smooth / float(x.shape[1])

        # Finally reshape to the original shape
        x = x.reshape(s)

        # Squeeze the channels dimension if set
        if self.squeeze_channels:
            x = torch.squeeze(x, 1)

        return x


class AttentionModelMNIST(nn.Module):
    """ Base class for calculating the attention map of a low resolution image """

    def __init__(self,
                 squeeze_channels=False,
                 softmax_smoothing=0.0):
        super(AttentionModelMNIST, self).__init__()

        self.squeeze_channels = squeeze_channels
        self.softmax_smoothing = softmax_smoothing

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, padding_mode='reflect')
        self.tanh1 = nn.Tanh()

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, padding_mode='reflect')
        self.tanh2 = nn.Tanh()

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect')

        self.sample_softmax = SampleSoftmax(squeeze_channels, softmax_smoothing)

    def forward(self, x_low):
        out = self.conv1(x_low)
        out = self.tanh1(out)

        out = self.conv2(out)
        out = self.tanh2(out)

        out = self.conv3(out)
        out = self.sample_softmax(out)

        return out


class AttentionModelColonCancer(nn.Module):
    """ Base class for calculating the attention map of a low resolution image """

    def __init__(self,
                 squeeze_channels=False,
                 softmax_smoothing=0.0):
        super(AttentionModelColonCancer, self).__init__()

        conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding_mode='zeros', padding=1)
        relu1 = nn.ReLU()

        conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding_mode='zeros', padding=1)
        relu2 = nn.ReLU()

        conv3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding_mode='zeros', padding=1)

        sample_softmax = SampleSoftmax(squeeze_channels, softmax_smoothing)

        self.forward_pass = nn.Sequential(conv1, relu1, conv2, relu2, conv3, sample_softmax)

    def forward(self, x_low):
        out = self.forward_pass(x_low)
        return out



def conv_layer(in_channels, out_channels, kernel, strides, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=strides, padding_mode="zeros", bias=False,
                     padding=padding)


def batch_norm(filters):
    return nn.BatchNorm2d(filters)


def relu():
    return nn.ReLU()


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, short):
        super(Block, self).__init__()

        self.short = short
        self.bn1 = batch_norm(in_channels)
        self.relu1 = relu()
        self.conv1 = conv_layer(in_channels, out_channels, 1, stride, padding=0)

        self.conv2 = conv_layer(in_channels, out_channels, kernel_size, stride)
        self.bn2 = batch_norm(out_channels)
        self.relu2 = relu()
        self.conv3 = conv_layer(out_channels, out_channels, kernel_size, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)

        x_short = x
        if self.short:
            x_short = self.conv1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        out = x + x_short
        return out


class FeatureModelTrafficSigns(nn.Module):

    def __init__(self, in_channels, strides=[1, 2, 2, 2], filters=[32, 32, 32, 32]):
        super(FeatureModelTrafficSigns, self).__init__()

        stride_prev = strides.pop(0)
        filters_prev = filters.pop(0)

        self.conv1 = conv_layer(in_channels, filters_prev, 3, stride_prev)

        module_list = nn.ModuleList()
        for s, f in zip(strides, filters):
            module_list.append(Block(filters_prev, f, s, 3, s != 1 or f != filters_prev))

            stride_prev = s
            filters_prev = f

        self.module_list = nn.Sequential(*module_list)

        self.bn1 = batch_norm(filters_prev)
        self.relu1 = relu()
        self.pool = nn.AvgPool2d(kernel_size=(13, 13))

    def forward(self, x):
        out = self.conv1(x)
        out = self.module_list(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool(out)
        out = out.view(out.shape[0], out.shape[1])
        out = F.normalize(out, p=2, dim=-1)
        return out


class FeatureModelMNIST(nn.Module):

    def __init__(self, in_channels):
        super(FeatureModelMNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7)
        self.relu1 = relu()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu2 = relu()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu3 = relu()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu4 = relu()

        self.pool = nn.AvgPool2d(kernel_size=(38, 38))

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.relu4(out)

        out = self.pool(out)
        out = out.view(out.shape[0], out.shape[1])
        out = F.normalize(out, p=2, dim=-1)

        return out


class FeatureModelColonCancer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FeatureModelColonCancer, self).__init__()

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 2, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.feature_extractor_part1(x)
        out = torch.cat([F.adaptive_avg_pool2d(out, output_size=(1, 1)),
                         F.adaptive_max_pool2d(out, output_size=(1, 1))], dim=1)
        out = out.view(out.shape[0], -1)

        out = self.feature_extractor_part2(out)

        out = F.normalize(out, p=2, dim=-1)
        return out


class MultinomialEntropy(nn.Module):
    """Increase or decrease the entropy of a multinomial distribution.
    Arguments
    ---------
    strength: A float that defines the strength and direction of the
              regularizer. A positive number increases the entropy, a
              negative number decreases the entropy.
    eps: A small float to avoid numerical errors when computing the entropy
    """

    def __init__(self, strength=1, eps=1e-6):
        super(MultinomialEntropy, self).__init__()
        if strength is None:
            self.strength = float(0)
        else:
            self.strength = float(strength)
        self.eps = float(eps)

    def forward(self, x):
        logx = torch.log(x + self.eps)
        # Formally the minus sign should be here
        return - self.strength * torch.sum(x * logx) / float(x.shape[0])
