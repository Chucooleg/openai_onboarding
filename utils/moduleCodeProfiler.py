import os
import re 
import inspect
from collections import defaultdict
from functools import partial
import torch
import torch.autograd.profiler as profiler
from torch.autograd.profiler import format_memory, format_time_share, format_time

import linecache as allLines

def map_to_module_code(pathToModuleMapping, pathToFuncCodeMappingFunc, include_external, topOfStackKey, evt_name):
    '''
    Map stack lines to module, function, codeline and line number
    return (module, function, codeline, lineNum)
    '''
    module_code_keys = []
    found = False
    for stackln in topOfStackKey:
        if stackln in pathToModuleMapping:
            found = True
            module = pathToModuleMapping[stackln]
            function, line_num, codeline = pathToFuncCodeMappingFunc(stackln)
            module_code_key = (module, function, evt_name, codeline, line_num)
            module_code_keys.append(module_code_key)

    # lets you see code lines that aren't in the module, can be useful if you have some external loss
    if include_external and len(topOfStackKey) > 0 and not found:
        module = "external"
        for i in range(min(3, len(topOfStackKey))):
            function, line_num, codeline = pathToFuncCodeMappingFunc(topOfStackKey[i])
            module_code_key = (module, function, evt_name, codeline, line_num)
            module_code_keys.append(module_code_key)

    return tuple(module_code_keys)


def stackLineToCode(stackLine):
    '''
    convert stack line to code line
    stackLine: 
        e.g. '/usr/lib/python3.6/runpy.py(193): _run_module_as_main'
    returns:
        e.g. ('_run_module_as_main', '193', '                     "__main__", mod_spec)')
    '''
    for (path, lineNum, funcName) in re.findall("(.*)\((\d+)\)\: (.*)", stackLine):
        return funcName, lineNum, allLines.getline(path, int(lineNum)).strip('\n')


def makePathModuleMapping(model):
    """Returns a dict that can take a path string that looks like
    /home/azureuser/openai_learning/customTransformer.py(206): forward
    And that dict will return the module name, using .named_modles()
    "model." is appended to the front of all entries so there is no empty string
    """
    mapping = {}
    for mn, m in model.named_modules():
        # add model. to front so we don't have empty string for model
        if mn == "": mn = "model"
        else: mn = "model." + mn
        for forwardPath, lineNum, line in getFunctionLines(m.forward):
            mapping[forwardPath] = mn
    return mapping


def getFunctionLines(func):
    """Extracts the lines of the given function
    Arguments:
        func: The function we are extracting from
    Returns:
        This is an enumerator, it yields tuples of things that look like
        ('pathToForwardFile.py(206): forward', 206, '        embeddings = torch.cat([embs, posEmbs], axis=3)')
        Which is
        (Path(lineNum): func.__name__, lineNum, codeOnThatLine)
    """
    lines, lineNum = inspect.getsourcelines(func)
    filePath = inspect.getsourcefile(func)
    for i, line in enumerate(lines):
        yield f"{filePath}({lineNum+i}): {func.__name__}", (lineNum+i), line


def groupEventsByModuleCode(events, stack_to_modulecode_mapfunc):
    '''
    Find the group of events items that belong to a stack.

    events: an array of function_events returned from a torch.autograd.profiler handle

            e.g.with torch.autograd.profiler(...) as prof: 
                    #(run code ...)
                prof.function_events 

            e.g.[
                    <FunctionEvent id=2 node_id=-1 
                    cpu_time=3.905ms cpu_start=464.424 cpu_end=4369.267 
                    cpu_children=[4, 5] cuda_time=0.000us name=aten::zeros thread=1 
                    input_shapes=[[], [], [], [], []] cpu_memory_usage=4 
                    cuda_memory_usage=0 is_async=False is_remote=False seq_nr=-1>,

                    <FunctionEvent id=3 node_i
                    ...>,

                    ...
                ]

    returns:
            a dictionary 
            key: tuple of multiple tuples (module, function, codeline, lineNum)
            value: an torch.autograd.profiler.function_events array
    '''
    eventsByModuleCode = defaultdict(lambda: [])
    for j, e in enumerate(events):
        existing = set()

        for i in range(1, len(e.stack)+1):

            # extract top i items in stack (0 is lowest level)
            topOfStack = e.stack[-i:] 
            topOfStackKey = tuple(topOfStack)

            # Map from stackLines to (module, function, evt_name, codeline, lineNum)
            module_code_key = stack_to_modulecode_mapfunc(topOfStackKey, e.name)

            if module_code_key in existing:
                continue
            else:
                existing.add(module_code_key)

            eventsByModuleCode[module_code_key].append(e)

    return eventsByModuleCode


def sumStats(eventsByModuleCode):
    '''
    Sum up cpu & cuda time & memoru usage from all events under each module-code
    '''  

    allStats = defaultdict(lambda: defaultdict(float))

    for moduleCodeKey in eventsByModuleCode.keys():
        events = eventsByModuleCode[moduleCodeKey]
        for evt in events:
            composite_key = (moduleCodeKey, evt.thread, str(evt.input_shapes))
            allStats['cpu_time'][composite_key] += evt.cpu_time
            allStats['cuda_time'][composite_key] += evt.cuda_time
            allStats['cpu_memory_usage'][composite_key] += evt.cpu_memory_usage
            allStats['cuda_memory_usage'][composite_key] += evt.cuda_memory_usage

    return allStats


def rankStats(stats, criteria='cpu_memory_usage', per_thread=False, per_inp_shapes=False):
    '''
    rank by states by criteria and print results
    '''
    unit_formatter = {'cpu_memory_usage':format_memory, 'cuda_memory_usage':format_memory, 
                      'cpu_time':format_time, 'cuda_time':format_time}
    
    crit_stats = stats[criteria]

    summed_crit_stats = defaultdict(float)

    # sum per thread or sum per input shape
    if not per_thread and not per_inp_shapes:
        for (moduleCodeKey, thread_i, inp_shape), val in crit_stats.items():
            summed_crit_stats[(moduleCodeKey, None, None)] += val
    elif per_thread and not per_inp_shapes:
        for (moduleCodeKey, thread_i, inp_shape), val in crit_stats.items():
            summed_crit_stats[(moduleCodeKey, thread_i, None)] += val
    elif per_inp_shapes and not per_thread:
        for (moduleCodeKey, thread_i, inp_shape), vdeal in crit_stats.items():
            summed_crit_stats[(moduleCodeKey, None, inp_shape)] += val
    else:
        for (moduleCodeKey, thread_i, inp_shape), val in crit_stats.items():
            summed_crit_stats[(moduleCodeKey, thread_i, inp_shape)] += val

    ranked_stats = sorted(summed_crit_stats.items(), key=lambda x:x[1], reverse=True)

    printStats(ranked_stats, unit_formatter[criteria])


def printStats(stats, unit_formatter):
    for (key_lines, thread_i, inp_shapes), val in stats:
        for key in key_lines:
            module, function, evt_name, code, line_num = key
            print('{}, {}, {}, ({}) {}'.format(module, evt_name, function, line_num, code.strip()))
        if thread_i or inp_shapes:
            print(f'thread:{thread_i}, input shapes {inp_shapes}')
        print(unit_formatter(val))
        print('##############################################')   


def rankByCriteria(profiler_handler, model, criteria='cpu_time', per_thread=False, per_inp_shapes=False, include_external=True):
    '''
    rank modules and code written in pytorch by criteria
    'cpu_time', 'cpu_memory_usage', 'cuda_time', 'cuda_memory_usage'
    '''

    print(f'Ranked by {criteria}\n')

    stack_to_modulecode_mapfunc = partial(map_to_module_code, makePathModuleMapping(model), stackLineToCode, include_external)
    eventsByModuleCode = groupEventsByModuleCode(profiler_handler.function_events, stack_to_modulecode_mapfunc)
    allStats = sumStats(eventsByModuleCode)
    rankStats(allStats, criteria, per_thread, per_inp_shapes)