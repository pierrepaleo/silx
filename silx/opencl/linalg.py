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
"""Module for basic linear algebra in OpenCL"""

from __future__ import absolute_import, print_function, with_statement, division

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "10/08/2017"

import logging
import numpy as np

from .common import pyopencl, kernel_workgroup_size
from .processing import EventDescription, OpenclProcessing, BufferDescription
from .backprojection import _sizeof, _idivup
cl = pyopencl



"""
d_sino = P(sino, onlygpu=True)

d_gradient = linalg.allocate(shp, dtype)
linalg.gradient(d_sino, dst=d_gradient)
# or:
d_grad = linalg.gradient(d_sino) # allocation + computation
"""


class Linalg(OpenclProcessing):

    kernel_files = ["linalg.cl"]

    def __init__(self, shape, do_checks=False, ctx=None, devicetype="all", platformid=None, deviceid=None, profile=False):
        """
        Create a "Linear Algebra" plan for a given image shape.

        :param shape: shape of the image (num_rows, num_columns)
        :param do_checks (optional): if True, memory and data type checks are performed when possible.
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)

        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)

        self.d_gradient = parray.zeros(self.queue, shape, np.complex64)
        self.d_image = parray.zeros(self.queue, shape, np.float32)

        self.wg2D = None
        self.shape = shape
        self.ndrange2D = (
            self.shape[1],  # TODO pitch ?
            self.shape[0]
        )
        self.do_check = bool(do_checks)



        OpenclProcessing.compile_kernels(self, self.kernel_files)


    @staticmethod
    def check_array(array, dtype, shape, arg_name):
        if array.shape != shape or array.dtype != dtype:
            raise ValueError("%s should be a %s array of type %s" %(arg_name, str(shape), str(dtype)))


    def get_refs(self, src, dst, default_src_ref, default_dst_ref):
        """
        From various types of src and dst arrays,
        returns the references that will be used by the OpenCL kernels.

        This function will make a copy host->device if the input is on host (eg. numpy array)
        """
        if dst:
            if isinstance(dst, cl.array.Array):
                dst_ref = dst.data
            elif isinstance(image, cl.Buffer):
                dst_ref = dst
            else:
                raise ValueError("dst should be either pyopencl.array.Array or pyopencl.Buffer")
        else:
            dst_ref = default_dst_ref

        if isinstance(image, cl.array.Array):
            src_ref = src.data
        elif isinstance(image, cl.Buffer):
            src_ref = src
        else: # assuming numpy.ndarray
            evt = cl.enqueue_copy(self.queue, self.d_image.data, image)
            self.events.append(EventDescription("copy H->D", evt))
            src_ref = default_src_ref
        return src_ref, dst_ref


    def gradient(self, image, dst=None, return_to_host=False):
        """

        if dst is provided, it should be of type numpy.complex64 !
        """
        # call gradient kernel with self.d_gradient if not dst, with dst otherwise
        # return parray/buffer to dst or self.d_gradient

        n_y, n_x = np.int32(image.shape)
        events = []
        if self.do_checks:
            self.check_array(image, np.float32, self.shape, "image")
            if dst:
                self.check_array(dst, np.complex64, self.shape, "dst")
        img_ref, grad_ref = self.get_refs(image, dst, self.d_gradient.data, self.d_image.data)

        # Prepare the kernel call
        kernel_args = [
            img_ref,
            grad_ref,
            n_x,
            n_y
        ]
        # Call the gradient kernel
        evt = self.program.kern_gradient2D(
            self.queue,
            self.ndrange2D,
            self.wg2D,
            *kernel_args
        )
        events.append(EventDescription("gradient2D", evt))

        if return_to_host:
            return grad_ref.get()
        else:
            return grad_ref



    #
    # TODO:
    #   - Gradient
    #   - Divergence
    #   - mul_add, add_scaled, shrink, ...-> straightforward with parray
    #       Benchmark custom opencl kernels  against pyopencl.array functions ?
    #   - projection onto the L-infinity ball
    #   - soft thresholding
    #
    #  - modify projector and backprojector  for a  "onlygpu" computation, returning a parray
    #



    def __del__(self):
        # todo: delete parrays which were not added in cl_mem
        OpenclProcessing.__del__(self)
        self.d_gradient = None
        self.d_image = None



































