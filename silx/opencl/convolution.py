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
from .processing import OpenclProcessing


class ConvolutionInfos(object):
    allowed_axes = {
        "1D": [None],
        "separable_2D_1D_2D": [None, (1, 0), (0, 1)],
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
    It supports:
      - 1D, 2D, 3D convolutions
      - batched 1D and 2D
    """
    kernel_files = ["convolution.cl"]#, "convolution_batched.cl"]

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
        self._configure_axes(shape, kernel, axes)
        self._allocate_memory()
        self._init_kernels()


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


    # This function might also be useful in other processings
    def _configure_axes(self, shape, kernel, axes):
        self._get_dimensions(shape, kernel)
        data_ndim = self.data_ndim
        kernel_ndim = self.kernel_ndim
        self.kernel = kernel.astype("f")
        if self.kernel_ndim > self.data_ndim:
            raise ValueError("Kernel dimensions cannot exceed data dimensions")
        if axes is None:
            # By default, convolve along all axes (as for FFT)
            default_separable_axes = {
                (1, 1): None,
                (2, 1): (1, 0),
                (2, 2): None,
                (3, 1): (2, 1, 0),
                (3, 2): (1, 0),
                (3, 3): None,
            }
            axes = default_separable_axes[(data_ndim, kernel_ndim)]
        axes_ndim = len(axes)
        # Handle negative values of axes
        axes = tuple(np.array(axes) % data_ndim)

        if self.kernel_ndim == self.data_ndim:
            # "Regular" non-separable case
            tr_name = str("nonseparable_%dD" % data_ndim)
        if self.kernel_ndim < self.data_ndim:
            # Separable/batched case
            if axes_ndim > data_ndim:
                raise ValueError("Axes dimensions cannot exceed data dimensions")
            if axes_ndim == data_ndim:
                # Separable case
                allowed_axes = {
                    # 2D data, 1D kernel (separable 2D)
                    (2, 1): [(1, 0), (0, 1)],
                    # 3D data, 1D kernel (separable 3D)
                    (3, 1): [
                        (0, 1, 2),
                        (1, 2, 0),
                        (2, 0, 1),
                        (2, 1, 0),
                        (1, 0, 2),
                        (0, 2, 1)
                    ],
                }
                tr_name = str("separable_%dD" % data_ndim)
            if axes_ndim < data_ndim:
                # Batched case
                allowed_axes = {
                    # 2D data, 1D kernel
                    (2, 1): [(0,), (1,)],
                    # 3D data, 1D kernel
                    (3, 1): [
                        # batched 1D on 3D data
                        (0,), (1,), (2,),
                        # batched separable 2D on 3D data
                        (1, 0), (0, 1), (2, 0), (0, 2), (1, 2), (2, 1),
                    ],
                    # 3D data, 2D kernel
                    (3, 2): [(0,), (1,), (2,)],
                }
                tr_name = str("batched_%dD" % data_ndim)
            k = (data_ndim, kernel_ndim)
            if k not in allowed_axes:
                raise ValueError(
                    "Could not find valid axes for %dD convolution on %dD data with axes=%s"
                    % (kernel_ndim, data_ndim, str(axes)),
                )
            if axes not in allowed_axes[k]:
                raise ValueError(
                    "Allowed axes for %dD convolution on %dD data are %s"
                    % (kernel_ndim, data_ndim, str(allowed_axes[k]))
                )
        self.axes = axes
        self.transform_name = tr_name


    # TODO for separable transform, "allocate_tmp_array"
    # for swapping references instead of copying data_out to data_in
    def _configure_extra_options(self, extra_options):
        self.extra_options = {
            "allocate_input_array": True,
            "allocate_output_array": True,
            "allocate_tmp_array": True,
        }
        extra_opts = extra_options or {}
        self.extra_options.update(extra_opts)


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


    def _init_kernels(self):
        compile_options = None
        self.compile_kernels(
            kernel_files=None,
            compile_options=compile_options
        )
        self.ndrange = np.int32(self.shape)[::-1]
        self.wg = None
        self.kernel_args = {
            "1D": (
                self.queue,
                self.ndrange,
                self.wg,
                self.data_in.data,
                self.data_out.data,
                self.d_kernel.data,
                np.int32(self.kernel.shape[0]),
                self.Nx,
                self.Ny,
                self.Nz
            )
        }



    def convolve(self, array, output=None):

        # todo check array (shape, dtype)

        ####
        kern = self.kernels.convol_1D_X
        kern(*self.kernel_args["1D"])

        self.data_in[:] = self.data_out[:]
        self.data_out.fill(0)
        kern = self.kernels.convol_1D_Y
        kern(*self.kernel_args["1D"])

        return self.data_out.get()
        ####





"""
Wanted:
 - 1D, 2D, 3D convol => one kernel for each dimension
 - batched 2D and 3D => other kernels...
 - Use textures when possible => still other kernels
It should be possible to make one class for all these use cases

 - compose with "ImageProcessing" ?
   if template= or dtype=   in the constructor => instantiate an ImageProcessing
   and do casts under the hood

 - Gaussian convolution => class inheriting from Convolution
  (used for blurring, ex. in sift)

 - [Bonus] utility for 3D on "big" volume, with
   H<->D transfers performed under the hood, + necessary overlap

  - Input strides and output strides ? This implies a big modification in the code


Use case name                       Kernel name
------------------------------------------------------------------
1D convol on 1D data                  convol_1D_X
batched 1D convol on 2D data          convol_1D_X or convol_1D_Y
separable (2,1)D convol on 2D data    convol_1D_X and convol_1D_X

batched 1D convol on 3D data          convol_1D_X or convol_1D_Y or convol_1D_Z
separable (3,1) 1D convol on 3D data  convol_1D_X and convol_1D_Y and convol_1D_Z
[batched separable 2D on 3D data]     convol_1D_X and convol_1D_Y and convol_1D_Z

2D convol on 2D data                  convol_2D_XY
batched 2D convol on 3D data          convol_2D_XY or convol_2D_XZ or convol_2D_YZ
separable (3, 2)D convol on 3D data   convol_2D_XY and convol_2D_XZ and convol_2D_YZ

3D convol on 3D data                  convol_3D_XYZ



(1, 1)
(2, 1), axes in [None, (1, 0), (0, 1)] => separable (2D->1D) on 2D data
(2, 1), axes in [(0,), (1,)]   => batched 1D on 2D data

(3, 1), axes in [None, valid 3-tuple] => separable (3D->1D) on 3D data
(3, 1), axes in [1-tuple] => batched 1D on 3D
(3, 1), axes in [valid 2-tuple] => batched (along 1 axis) separable (2D->1D) [same as (2, 1, axes=None) if Nz==1]

(2, 2) => (nonseparable) 2D on 2D data
(3, 2), axes in [None, valid 2-tuple] => separable (3D->2D) on 3D data   (along y then z  or  along x then z   or   along x then y)
(3, 2), axes in [1-tuple] => batched 2D convol on 3D data

(3, 3) => (nonseparable) 3D convol

"""







