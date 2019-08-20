#  Copyright (c) 2019
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#  Authors:
#      Simone Rossi <simone.rossi@eurecom.fr>
#      Maurizio Filippone <maurizio.filippone@eurecom.fr>
#

import gc
import inspect
import linecache
import os.path
import sys
import time
import threading
import traceback as tb
from collections import namedtuple
from functools import lru_cache, partial

import torch

import inspect
import string


def f(f_string):
    frame = inspect.stack()[1][0]
    return Formatter(frame.f_globals, frame.f_locals).format(f_string)


class Formatter(string.Formatter):
    def __init__(self, globals_, locals_):
        self.globals = globals_
        self.locals = locals_

    def _vformat(self, *args, **kwargs):
        _vformat = super(Formatter, self)._vformat
        if 'auto_arg_index' in inspect.getargspec(_vformat)[0]:
            kwargs['auto_arg_index'] = False
        return _vformat(*args, **kwargs)

    def get_field(self, field_name, args, kwargs):
        if not field_name.strip():
            raise ValueError('empty expression not allowed')
        return eval('(' + field_name + ')', self.globals, self.locals), None


def mem_stat(stats=('allocated', 'cached'), device_ids=None):
    ''' Return a dictionary of CUDA memory stats '''
    mem_stats = {}
    device_ids = device_ids or range(torch.cuda.device_count())
    for device in [torch.cuda.device(i) for i in device_ids]:
        with device:
            device_stats = {}
            for stat in stats:
                stat_name = 'memory_' + str(stat)
                max_stat_name = 'max_' + str(stat_name)
                device_stats[stat_name] = torch.cuda.__dict__[stat_name]()
                device_stats[max_stat_name] = torch.cuda.__dict__[max_stat_name]()
            mem_stats[device.idx] = device_stats

    return mem_stats


def mem_stat_string(stats=('allocated', 'cached'), sep=' ', device_ids=None):
    ''' Return a formatted string of the mem stats '''
    mem_stats = []
    device_ids = device_ids or range(torch.cuda.device_count())
    for device in [torch.cuda.device(i) for i in device_ids]:
        with device:
            mem_stats.append('cuda:%s' % device.idx)
            for stat in stats:
                stat_name = 'memory_%s' % stat
                max_stat_name = 'max_%s' % stat_name
                stat_value = torch.cuda.__dict__[stat_name]() / 1024**2
                max_stat_value = torch.cuda.__dict__[max_stat_name]() / 1024**2
                mem_stats.append('%s=%.2f(%.2f)MiB' % (stat[:5], stat_value, max_stat_value))

    return sep.join(mem_stats)


@lru_cache(maxsize=None)
def get_function(code):
    ''' Get the function from the given code object '''
    # Function lookups can be VERY slow if there are lots of references in a tight loop, so use an
    # lr_cache
    for obj in gc.get_referrers(code):
        if inspect.isfunction(obj):
            return obj


def cuda_tensors():
    ''' A generator for CUDA tensors '''
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        # pylint:disable=broad-except
        except Exception:
        # pylint:enable=broad-except
            pass


def collect_modules(module, modules=None):
    ''' A function that collects all modules in the module hierarchy '''
    if not isinstance(module, torch.nn.Module):
        return set()

    modules = modules or set()
    modules.add(module)
    for child in module.children():
        modules = collect_modules(child, modules)

    return modules


def collect_scopes(module, scopes=None):
    ''' A function that collects all profile scopes in the module '''
    if not isinstance(module, torch.nn.Module):
        return set()

    scopes = scopes or set()
    scopes.add(module.forward.__qualname__)
    for child in module.children():
        scopes = collect_scopes(child, scopes)

    return scopes


class CUDAMemoryProfiler(object):
    ''' A class that does implements CUDA memory profiling '''
    AllocInfo = namedtuple('AllocInfo', ['function', 'lineno', 'device', 'creation_time', 'parent'])

    def __init__(self, models, filename='cuda.prof'):
        ''' Initialize the CUDA profiler with scopes you want to trace '''
        super(CUDAMemoryProfiler, self).__init__()

        self.stacks = {}
        self.tensors = set()
        self.last_stat = None
        self.filename = filename
        self.lock = threading.Lock()

        self.scopes = None
        self.modules = None
        for model in models:
            self.scopes = collect_scopes(model, self.scopes)
            self.modules = collect_modules(model, self.modules)

    def __call__(self, frame, event, arg):
        ''' Entry point into the profiler '''
        if event == 'exception':
            self.dump_exception(*arg)
        elif event != 'call':
            return self.trap_exception

        function = get_function(frame.f_code)
        if not function:
            return self.trap_exception

        # only trace the desired scopes (or those nested within)
        qualname = function.__qualname__
        thread = threading.get_ident()
        stack = self.stacks.get(thread, [])
        if not stack and qualname not in self.scopes:
            return self.trap_exception

        # We are now tracing a new function
        stack.append((qualname, -1))
        self.stacks[thread] = stack
        return self.profile_scope

    def dump_exception(self, exception, value, traceback):
        ''' Output exception information '''
        if not issubclass(exception, RuntimeError):
            return

        lines = ['*** %s(%s) ***\n' % (exception.__name__, value)]
        lines.extend(tb.extract_tb(traceback).format())
        lines.append('[%s]\n' % mem_stat_string())
        lines.append('Current tensors:\n')
        total_bytes = {d: 0 for d in range(torch.cuda.device_count())}
        device_lines = {d: [] for d in range(torch.cuda.device_count())}
        for tensor, size, nbytes, alloc in self.tensors:
            total_bytes[alloc.device] += nbytes
            line = '  %s:%s:%s ' % (alloc.function, alloc.lineno, alloc.device)
            if alloc.parent:
                _, _, _, parent = alloc.parent
                line += 'from %s:%s:%s ' % (parent.function, parent.lineno, parent.device)
            line += '%s%s %.3fKiB\n' % (tensor, str(size), nbytes/1024)

            device_lines[alloc.device].append((alloc.creation_time, line))

        for device in range(torch.cuda.device_count()):
            lines.append(' cuda:%s\n' % device)
            # sort by creation time
            sorted_lines = sorted(device_lines[device], key=lambda k: k[0])
            #_, sorted_lines = zip(*sorted_lines)
            lines.extend(sorted_lines)
            lines.append(' Total=%.3fMiB\n' % (total_bytes[device]/1024**2))

        with open(self.filename, 'a+') as file:
            file.writelines(lines)

    def trap_exception(self, frame, event, arg): # pylint:disable=unused-argument
        ''' Trace function that only looks for exceptions '''
        if event == 'exception':
            self.dump_exception(*arg)

    def get_tensor_info(self, tensor, parent=None, function=None):
        ''' Get the tracking information for the tensor '''
        return (
            torch.typename(tensor),
            tuple(tensor.size()),
            tensor.storage().element_size() * tensor.numel(),
            self.get_tensor_alloc_info(tensor, parent, function)
        )

    def get_tensor_alloc_info(self, tensor, parent=None, function=None):
        ''' Return the allocation info for a tensor '''
        if not hasattr(tensor, '__alloc_info'):
            thread = threading.get_ident()
            stack = self.stacks.get(thread, [])
            if stack:
                function, lineno = self.stacks[thread][-1]
            else:
                frame = sys._getframe() # pylint:disable=protected-access
                current_file = frame.f_globals['__file__']
                while frame and current_file == frame.f_globals['__file__']:
                    frame = frame.f_back

                if frame:
                    lineno = frame.f_lineno
                    function = get_function(frame.f_code)
                else:
                    lineno = -1
                    function = function or '<unknown>'

            setattr(tensor, '__alloc_info', CUDAMemoryProfiler.AllocInfo(
                function, lineno, tensor.get_device(), time.perf_counter(), parent=parent))

        return getattr(tensor, '__alloc_info')

    def profile_scope(self, frame, event, arg):
        ''' Memory profiling of the current scope '''
        if event == 'exception':
            self.dump_exception(*arg)

        thread = threading.get_ident()
        if event == 'line':
            lines = []
            lineno = frame.f_lineno
            filename = frame.f_globals["__file__"]
            line = linecache.getline(filename, lineno).strip()
            function, _ = self.stacks[thread][-1]

            # update the current line number
            self.stacks[thread][-1] = (function, lineno)

            if self.lock.acquire():
                stat = mem_stat()
                if stat != self.last_stat:
                    self.last_stat = stat
                    path = os.path.relpath(filename)
                    if len(path) > len(filename):
                        path = filename

                    if len(line) > 100:
                        line = line[:50] + '...' + line[-50:]

                    lines.append('[%s]\n %s:%s %s\n' % (mem_stat_string(), path, lineno, line))

                    tensors = set()
                    for tensor in cuda_tensors():
                        tensor_info = self.get_tensor_info(tensor)
                        tensors.add(tensor_info)
                        if tensor.requires_grad and tensor_info not in self.tensors:
                            # register hook to gather memory stats during backward
                            tensor.register_hook(partial(self.profile_grad, tensor_info))

                    for tensor, size, nbytes, alloc in tensors - self.tensors:
                        lines.append(
                            ' + %s:%s:%s %s%s %.2fKiB\n' %(alloc.function, alloc.lineno, alloc.device, tensor, str(size), nbytes/1024)
                        )
                    for tensor, size, nbytes, alloc in self.tensors - tensors:
                        lines.append(
                            ' - %s:%s:%s %s%s %.2fKiB\n' %(alloc.function, alloc.lineno, alloc.device, tensor, str(size), nbytes/1024)
                        )
                    self.tensors = tensors
                self.lock.release()

            if lines:
                with open(self.filename, 'a+') as file:
                    file.writelines(lines)
        elif event == 'return':
            self.stacks[thread].pop()

    def profile_grad(self, parent_tensor_info, grad):
        ''' Memory profiling for the backward pass '''
        if not grad.is_cuda:
            return grad

        tensor_info = self.get_tensor_info(grad, parent_tensor_info, 'torch.autograd.backward')
        if self.lock.acquire():
            self.tensors.add(tensor_info)
            self.lock.release()

        tensor, size, nbytes, alloc = tensor_info
        _, _, _, parent_alloc = parent_tensor_info
        with open(self.filename, 'a+') as file:
            file.write(
                '[Grad: %s\n + %s - %s:%s:%s %s%s %.3fKiB\n' % (
                    mem_stat_string(), alloc.function, parent_alloc.function, parent_alloc.lineno, parent_alloc.device, tensor, str(size), nbytes/1024)
            )

        return grad

