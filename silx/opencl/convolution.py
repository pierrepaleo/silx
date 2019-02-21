#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""Module for convolution on CPU/GPU."""

from __future__ import absolute_import, print_function, with_statement, division

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "11/02/2019"

import numpy as np

from .common import pyopencl as cl
import pyopencl.array as parray
from .processing import OpenclProcessing, EventDescription


class ConvolutionInfos(object):
    allowed_axes = {
        "1D": [None],
        "separable_2D_1D_2D": [None, (0, 1), (1, 0)],
        "batched_1D_2D": [(0,), (1,)],
        "separable_3D_1D_3D": [
            None,
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
            (1, 0, 2),
            (0, 2, 1)
        ],
        "batched_1D_3D": [(0,), (1,), (2,)],
        "batched_separable_2D_1D_3D": [(0,), (1,), (2,)], # unsupported (?)
        "2D": [None],
        "batched_2D_3D": [(0,), (1,), (2,)],
        "separable_3D_2D_3D": [
            (1, 0),
            (0, 1),
            (2, 0),
            (0, 2),
            (1, 2),
            (2, 1),
        ],
        "3D": [None],
    }
    use_cases = {
        (1, 1): {
            "1D": {
                "name": "1D convolution on 1D data",
                "kernels": ["convol_1D_X"],
            },
        },
        (2, 2): {
            "2D": {
                "name": "2D convolution on 2D data",
                "kernels": ["convol_2D_XY"],
            },
        },
        (3, 3): {
            "3D": {
                "name": "3D convolution on 3D data",
                "kernels": ["convol_3D_XYZ"],
            },
        },
        (2, 1): {
            "separable_2D_1D_2D": {
                "name": "Separable (2D->1D) convolution on 2D data",
                "kernels": ["convol_1D_X", "convol_1D_Y"],
            },
            "batched_1D_2D": {
                "name": "Batched 1D convolution on 2D data",
                "kernels": ["convol_1D_X", "convol_1D_Y"],
            },
        },
        (3, 1): {
            "separable_3D_1D_3D": {
                "name": "Separable (3D->1D) convolution on 3D data",
                "kernels": ["convol_1D_X", "convol_1D_Y", "convol_1D_Z"],
            },
            "batched_1D_3D": {
                "name": "Batched 1D convolution on 3D data",
                "kernels": ["convol_1D_X", "convol_1D_Y", "convol_1D_Z"],
            },
            "batched_separable_2D_1D_3D": {
                "name": "Batched separable (2D->1D) convolution on 3D data",
                "kernels": ["convol_1D_X", "convol_1D_Y", "convol_1D_Z"],
            },
        },
        (3, 2): {
            "separable_3D_2D_3D": {
                "name": "Separable (3D->2D) convolution on 3D data",
                "kernels": ["convol_2D_XY", "convol_2D_XZ", "convol_2D_YZ"],
            },
            "batched_2D_3D": {
                "name": "Batched 2D convolution on 3D data",
                "kernels": ["convol_2D_XY", "convol_2D_XZ", "convol_2D_YZ"],
            },
        },
    }


class Convolution(OpenclProcessing):
    """
    A class for performing convolution on CPU/GPU with OpenCL.
    """

    def __init__(self, shape, kernel, axes=None, ctx=None,
                 devicetype="all", platformid=None, deviceid=None,
                 profile=False, extra_options=None):
        """Constructor of OpenCL Convolution.

        :param shape: shape of the array.
        :param kernel: convolution kernel (1D, 2D or 3D).
        :param axes: axes along which the convolution is performed,
            for batched convolutions.
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by
                           clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param profile: switch on profiling to be able to profile at the kernel
                        level, store profiling elements (makes code slightly
                        slower)
        :param extra_options: Advanced options (dict). Current options are:
            "allocate_input_array": True,
            "allocate_output_array": True,
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)

        self._configure_extra_options(extra_options)
        self._determine_use_case(shape, kernel, axes)
        self._allocate_memory()
        self._init_kernels()


    def _configure_extra_options(self, extra_options):
        self.extra_options = {
            "allocate_input_array": True,
            "allocate_output_array": True,
            "allocate_tmp_array": True,
            "dont_use_textures": False,
        }
        extra_opts = extra_options or {}
        self.extra_options.update(extra_opts)
        self.is_cpu = (self.device.type == "CPU")
        self.use_textures = not(self.extra_options["dont_use_textures"])
        self.use_textures *= not(self.is_cpu)


    def _get_dimensions(self, shape, kernel):
        self.shape = shape
        self.data_ndim = self._check_dimensions(shape=shape, name="Data")
        self.kernel_ndim = self._check_dimensions(arr=kernel, name="Kernel")
        Nx = shape[-1]
        if self.data_ndim >= 2:
            Ny = shape[-2]
        else:
            Ny = 1
        if self.data_ndim >= 3:
            Nz = shape[-3]
        else:
            Nz = 1
        self.Nx = np.int32(Nx)
        self.Ny = np.int32(Ny)
        self.Nz = np.int32(Nz)


    def _determine_use_case(self, shape, kernel, axes):
        """
        Determine the convolution use case from the input/kernel shape, and axes.
        """
        self._get_dimensions(shape, kernel)
        if self.kernel_ndim > self.data_ndim:
            raise ValueError("Kernel dimensions cannot exceed data dimensions")
        data_ndim = self.data_ndim
        kernel_ndim = self.kernel_ndim
        self.kernel = kernel.astype("f")

        convol_infos = ConvolutionInfos()
        k = (data_ndim, kernel_ndim)
        if k not in convol_infos.use_cases:
            raise ValueError(
                "Cannot find a use case for data ndim = %d and kernel ndim = %d"
                % (data_ndim, kernel_ndim)
            )
        possible_use_cases = convol_infos.use_cases[k]

        self.use_case_name = None
        for uc_name, uc_params in possible_use_cases.items():
            if axes in convol_infos.allowed_axes[uc_name]:
                self.use_case_name = uc_name
                self.use_case_desc = uc_params["name"]
                self.use_case_kernels = uc_params["kernels"].copy()
        if self.use_case_name is None:
            raise ValueError(
                "Cannot find a use case for data ndim = %d, kernel ndim = %d and axes=%s"
                % (data_ndim, kernel_ndim, str(axes))
            )
        # TODO implement this use case
        if self.use_case_name == "batched_separable_2D_1D_3D":
            raise NotImplementedError(
                "The use case %s is not implemented"
                % self.use_case_name
            )
        #
        self.axes = axes
        # Replace "axes=None" with an actual value (except for ND-ND)
        allowed_axes = convol_infos.allowed_axes[self.use_case_name]
        if len(allowed_axes) > 1:
            # The default choice might impact perfs
            self.axes = allowed_axes[0] or allowed_axes[1]
        self.separable = self.use_case_name.startswith("separable")
        self.batched = self.use_case_name.startswith("batched")
        # Update kernel names when using textures
        if self.use_textures:
            for i, kern_name in enumerate(self.use_case_kernels):
                self.use_case_kernels[i] = kern_name + "_tex"


    def _allocate_memory(self):
        option_array_names = {
            "allocate_input_array": "data_in",
            "allocate_output_array": "data_out",
            "allocate_tmp_array": "data_tmp",
        }
        for option_name, array_name in option_array_names.items():
            if self.extra_options[option_name]:
                value = parray.zeros(self.queue, self.shape, "f")
            else:
                value = None
            setattr(self, array_name, value)

        if isinstance(self.kernel, np.ndarray):
            self.d_kernel = parray.to_device(self.queue, self.kernel)
        else:
            if not(isinstance(self.kernel, parray.Array)):
                raise ValueError("kernel must be either numpy array or pyopencl array")
            self.d_kernel = self.kernel
        self._old_input_ref = None
        self._old_output_ref = None
        if self.use_textures:
            self._allocate_textures()


    def _allocate_tex(self, shape):
        return cl.Image(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
            cl.ImageFormat(
                cl.channel_order.INTENSITY,
                cl.channel_type.FLOAT
            ),
            hostbuf=np.zeros(shape[::-1], dtype=np.float32)
        )


    def _allocate_textures(self):
        self.data_in_tex = self._allocate_tex(self.shape)
        self.d_kernel_tex = self._allocate_tex(self.kernel.shape)
        self.transfer_to_texture(self.d_kernel, self.d_kernel_tex)


    def _init_kernels(self):
        if self.kernel_ndim > 1:
            if np.abs(np.diff(self.kernel.shape)).max() > 0:
                raise NotImplementedError(
                    "Non-separable convolution with non-square kernels is not implemented yet"
                )
        compile_options = None
        if self.use_textures:
            kernel_files = ["convolution_textures.cl"]
            data_in_ref = self.data_in_tex
            d_kernel_ref = self.d_kernel_tex
        else:
            kernel_files = ["convolution.cl"]
            data_in_ref = self.data_in.data
            d_kernel_ref = self.d_kernel.data
        self.compile_kernels(
            kernel_files=kernel_files,
            compile_options=compile_options
        )
        self.ndrange = np.int32(self.shape)[::-1]
        self.wg = None
        kernel_args = [
            self.queue,
            self.ndrange, self.wg,
            data_in_ref,
            self.data_out.data,
            d_kernel_ref,
            np.int32(self.kernel.shape[0]),
            self.Nx, self.Ny, self.Nz
        ]
        if self.kernel_ndim == 2:
            kernel_args.insert(6, np.int32(self.kernel.shape[1]))
        if self.kernel_ndim == 3:
            kernel_args.insert(6, np.int32(self.kernel.shape[2]))
            kernel_args.insert(7, np.int32(self.kernel.shape[1]))
        self.kernel_args = tuple(kernel_args)
        # If self.data_tmp is allocated, separable transforms can be performed
        # by a series of batched transforms, without any copy, by swapping refs.
        self.swap_pattern = None
        if self.data_tmp is not None:
            self.swap_pattern = {
                2: [
                    ("data_in", "data_tmp"),
                    ("data_tmp", "data_out")
                ],
                3: [
                    ("data_in", "data_out"),
                    ("data_out", "data_tmp"),
                    ("data_tmp", "data_out"),
                ],
            }
        else:
            # TODO
            raise NotImplementedError("For now, data_tmp has to be allocated")


    def _get_swapped_arrays(self, i):
        n_batchs = len(self.axes)
        in_ref, out_ref = self.swap_pattern[n_batchs][i]
        d_in = getattr(self, in_ref)
        d_out = getattr(self, out_ref)
        return d_in, d_out


    def _configure_kernel_args(self, opencl_kernel_args, input_ref, output_ref):
        # TODO more elegant
        if isinstance(input_ref, parray.Array):
            input_ref = input_ref.data
        if isinstance(output_ref, parray.Array):
            output_ref = output_ref.data
        if input_ref is not None or output_ref is not None:
            opencl_kernel_args = list(opencl_kernel_args)
            if input_ref is not None:
                opencl_kernel_args[3] = input_ref
            if output_ref is not None:
                opencl_kernel_args[4] = output_ref
            opencl_kernel_args = tuple(opencl_kernel_args)
        return opencl_kernel_args


    @staticmethod
    def _check_dimensions(arr=None, shape=None, name="", dim_min=1, dim_max=3):
        if shape is not None:
            ndim = len(shape)
        elif arr is not None:
            ndim = arr.ndim
        else:
            raise ValueError("Please provide either arr= or shape=")
        if ndim < dim_min or ndim > dim_max:
            raise ValueError("%s dimensions should be between %d and %d"
                % (name, dim_min, dim_max)
            )
        return ndim


    def _check_array(self, arr):
        # TODO allow cl.Buffer
        if not(isinstance(arr, parray.Array) or isinstance(arr, np.ndarray)):
            raise TypeError("Expected either pyopencl.array.Array or numpy.ndarray")
        # TODO composition with ImageProcessing/cast
        if arr.dtype != np.float32:
            raise TypeError("Data must be float32")
        if arr.shape != self.shape:
            raise ValueError("Expected data shape = %s" % str(self.shape))


    def _set_arrays(self, array, output=None):
        # When using textures: copy
        if self.use_textures:
            self.transfer_to_texture(array, self.data_in_tex)
            data_in_ref = self.data_in_tex
        else:
            # Otherwise: copy H->D or update references.
            if isinstance(array, np.ndarray):
                self.data_in[:] = array[:]
            else:
                self._old_input_ref = self.data_in
                self.data_in = array
            data_in_ref = self.data_in
        if output is not None:
            if not(isinstance(output, np.ndarray)):
                self._old_output_ref = self.data_out
                self.data_out = output
        # Update OpenCL kernel arguments with new array references
        self.kernel_args = self._configure_kernel_args(
            self.kernel_args,
            data_in_ref,
            self.data_out
        )


    def _separable_convolution(self):
        assert len(self.axes) == len(self.use_case_kernels)
        # Separable: one kernel call per data dimension
        for i, axis in enumerate(self.axes):
            in_ref, out_ref = self._get_swapped_arrays(i)
            self._batched_convolution(axis, input_ref=in_ref, output_ref=out_ref)


    def _batched_convolution(self, axis, input_ref=None, output_ref=None):
        # Batched: one kernel call in total
        opencl_kernel = self.kernels.get_kernel(self.use_case_kernels[axis])
        opencl_kernel_args = self._configure_kernel_args(
            self.kernel_args,
            input_ref,
            output_ref
        )
        ev = opencl_kernel(*opencl_kernel_args)
        if self.profile:
            self.events.append(EventDescription("batched convolution", ev))


    def _nd_convolution(self):
        assert len(self.use_case_kernels) == 1
        opencl_kernel = self.kernels.get_kernel(self.use_case_kernels[0])
        ev = opencl_kernel(*self.kernel_args)
        if self.profile:
            self.events.append(EventDescription("ND convolution", ev))


    def _recover_arrays_references(self):
        if self._old_input_ref is not None:
            self.data_in = self._old_input_ref
            self._old_input_ref = None
        if self._old_output_ref is not None:
            self.data_out = self._old_output_ref
            self._old_output_ref = None
        self.kernel_args = self._configure_kernel_args(
            self.kernel_args,
            self.data_in,
            self.data_out
        )


    def _get_output(self, output):
        if output is None:
            res = self.data_out.get()
        else:
            res = output
            if isinstance(output, np.ndarray):
                output[:] = self.data_out[:]
        self._recover_arrays_references()
        return res


    def convolve(self, array, output=None):
        """
        Convolve an array with the class kernel.

        :param array: Input array. Can be numpy.ndarray or pyopencl.array.Array.
        :param output: Output array. Can be numpy.ndarray or pyopencl.array.Array.
        """
        self._check_array(array)
        self._set_arrays(array, output=output)
        if self.axes is not None:
            if self.separable:
                self._separable_convolution()
            elif self.batched:
                assert len(self.axes) == 1 #
                self._batched_convolution(self.axes[0])
            # else: ND-ND convol
        else:
            # ND-ND convol
            self._nd_convolution()

        res = self._get_output(output)
        return res


    __call__ = convolve

