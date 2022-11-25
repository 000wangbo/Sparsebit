import torch
import torch.fx as fx
import sys
from enum import Enum
from pathlib import Path

from tabulate import tabulate
from sparsebit.quantization.modules import *

class ErrorCode(Enum):
    E001 = 1
    E002 = 2
    E003 = 3
    E004 = 4
    E005 = 5
    E006 = 6
    E007 = 7

class TipCode(Enum):
    T001 = 1
    T002 = 2
    T003 = 3
    T004 = 4
    T005 = 3
    T006 = 4

class RulesQuant:
    '''
    Quant 算子相关限制
    '''
    def __init__(self):
        self.err_desc = {
            ErrorCode.E001: "The input data type of the layer is ",
            ErrorCode.E002: "The layer should have an output quantizer",
            ErrorCode.E003: "The output data type of the layer is ",
            ErrorCode.E004: "Inputs and outputs must all have same data_type",
            ErrorCode.E005: "The weight data type of the layer is ",
            ErrorCode.E006: "The BatchNormalization layer's parent should be Conv、ConvTranspose、Gemm",
            ErrorCode.E007: "The Relu layer shouldn't have a qdq block between Conv、ConvTranspose、Gemm、Add",
        }

        self.tip_desc = {
            TipCode.T001: "E001:please cut the input channel",
            TipCode.T002: "E002:please cut the output channel",
            TipCode.T003: "E003:please reduce the bit of input feature",
            TipCode.T004: "E004:please reduce the bit of weight",
            TipCode.T005: "E005:please reduce the number of parent",
            TipCode.T006: "E006:please reduce the number of unpool parent",
        }


class Validator:
    def __init__(self, model):
        self.model = model
        self.rules_quant = RulesQuant()
        self.output = []
        self.headers_width = [20, 10, 80]
        self.no_input_quantizer

    def no_input_quantizer(self, output_module):
        if isinstance(output_module, Flatten):
            return 1
        return 0
    
    def wrap_lines(self, words, line_length):
        curr_line = ''
        warp_cnt = 0
        for word in words:
            warp_cnt += 1
            if warp_cnt == line_length:
                warp_cnt = 0
                curr_line += word + '\n'
            else:
                curr_line += word
        return curr_line

    def pretty_print(self):
        print("Validation result:")
        for log in self.output:
            for i, item in enumerate(log):
                if i == 0:  # 第一栏一般是layer object，跳过
                    continue
                if item:
                    log[i] = self.wrap_lines(item, self.headers_width[i])

        strresult = tabulate(self.output, headers=("Layer name", 'type', "Value"), tablefmt="grid")
        if len(self.output) == 0:
            print("--No violation!--")
        else:
            print(strresult)
        if len(self.output) > 0:
            sys.exit(1)

    def add_violation(self, layer_name, info, err):
        message = self.rules_quant.err_desc[err] + str(info),
        message = f'{sys._getframe(2).f_code.co_filename}, Line {sys._getframe(2).f_lineno}: {message}'
        self.output.append([
            layer_name,
            'ERROR',
            message
        ])

    def check_issue(self):
        named_modules = dict(self.model.model.named_modules(remove_duplicate=False))
        traced = self.model.model
        modules_viewed = {}
        qnodes = []  # 用于避免重复遍历
        for n in traced.graph.nodes:
            if not isinstance(n, fx.Node) or n in qnodes:
                continue
            elif n.op == "call_module":
                org_module = named_modules[n.target]
                if isinstance(org_module, QAdaptiveAvgPool2d):
                    input_data_type = org_module.input_quantizer.qdesc._type
                    if input_data_type not in ['uint8', 'uint16']:
                        self.add_violation(n.name, input_data_type, ErrorCode.E001)
                    children = n._next
                    children_module = named_modules[n._next.target]
                    if self.no_input_quantizer(children_module):
                        self.add_violation(n.name, '', ErrorCode.E002)
                    else:
                        output_data_type = children_module.input_quantizer.qdesc._type
                        assert output_data_type in ['uint8', 'uint16'], "error in {} output data type".format(n.name)
                    
                elif isinstance(org_module, MaxPool2d):
                    pass
                
                elif isinstance(org_module, Concat):
                    pass

                elif isinstance(org_module, QConv2d):
                    input_data_type = org_module.input_quantizer.qdesc._type
                    if input_data_type not in ['uint4', 'uint8', 'uint16']:
                        self.add_violation(n.name, input_data_type, ErrorCode.E001)
                    children = n._next
                    children_module = named_modules[n._next.target]
                    if self.no_input_quantizer(children_module):
                        self.add_violation(n.name, '', ErrorCode.E002)
                    else:
                        output_data_type = children_module.input_quantizer.qdesc._type
                        if output_data_type not in ['uint4', 'uint8', 'uint16']:
                            self.add_violation(n.name, output_data_type, ErrorCode.E003)
                        if input_data_type != output_data_type:
                            self.add_violation(n.name, '', ErrorCode.E004)
                    weight_data_type = org_module.weight_quantizer.qdesc._type
                    if weight_data_type not in ['int2', 'int4', 'int8']:
                        self.add_violation(n.name, weight_data_type, ErrorCode.E005)
                        
                elif isinstance(org_module, QConvTranspose2d):
                    input_data_type = org_module.input_quantizer.qdesc._type
                    if input_data_type not in ['uint4', 'uint8', 'uint16']:
                        self.add_violation(n.name, input_data_type, ErrorCode.E001)
                    children = n._next
                    children_module = named_modules[n._next.target]
                    if self.no_input_quantizer(children_module):
                        self.add_violation(n.name, '', ErrorCode.E002)
                    else:
                        output_data_type = children_module.input_quantizer.qdesc._type
                        if output_data_type not in ['uint4', 'uint8', 'uint16']:
                            self.add_violation(n.name, output_data_type, ErrorCode.E003)
                        if input_data_type != output_data_type:
                            self.add_violation(n.name, '', ErrorCode.E004)
                    weight_data_type = org_module.weight_quantizer.qdesc._type
                    if weight_data_type not in ['int2', 'int4', 'int8']:
                        self.add_violation(n.name, weight_data_type, ErrorCode.E005)
                        
                elif isinstance(org_module, QLinear):
                    input_data_type = org_module.input_quantizer.qdesc._type
                    if input_data_type not in ['uint4', 'uint8', 'uint16']:
                        self.add_violation(n.name, input_data_type, ErrorCode.E001)
                    children = n._next
                    if children.op != 'output':
                        children_module = named_modules[n._next.target]
                        if self.no_input_quantizer(children_module):
                            self.add_violation(n.name, '', ErrorCode.E002)
                        else:
                            output_data_type = children_module.input_quantizer.qdesc._type
                            if output_data_type not in ['uint4', 'uint8', 'uint16']:
                                self.add_violation(n.name, output_data_type, ErrorCode.E003)
                            if input_data_type != output_data_type:
                                self.add_violation(n.name, '', ErrorCode.E004)
                    weight_data_type = org_module.weight_quantizer.qdesc._type
                    if weight_data_type not in ['int2', 'int4', 'int8']:
                        self.add_violation(n.name, weight_data_type, ErrorCode.E005)
                        
                elif isinstance(org_module, QBatchNorm2d):
                    parent = n._prev
                    parent_module = named_modules[n._prev.target]
                    if not isinstance(parent_module, (QConv2d, QConvTranspose2d, QLinear)):
                        self.add_violation(n.name, '', ErrorCode.E006)
                    
                elif isinstance(org_module, QReLU):
                    parent = n._prev
                    parent_module = named_modules[n._prev.target]
                    if not isinstance(parent_module, (QConv2d, QConvTranspose2d, QLinear, QAdd)):
                        self.add_violation(n.name, '', ErrorCode.E007)
                    input_data_type = org_module.input_quantizer.qdesc._type
                    if input_data_type not in ['uint4', 'uint8', 'uint16']:
                        self.add_violation(n.name, input_data_type, ErrorCode.E001)
                    children = n._next
                    children_module = named_modules[n._next.target]
                    if self.no_input_quantizer(children_module):
                        self.add_violation(n.name, '', ErrorCode.E002)
                    else:
                        output_data_type = children_module.input_quantizer.qdesc._type
                        if output_data_type not in ['uint4', 'uint8', 'uint16']:
                            self.add_violation(n.name, output_data_type, ErrorCode.E003)
                            
                elif isinstance(org_module, (QReLU6, QLeakyReLU, QSigmoid, QSiLU, QGELU, QMish)):
                    input_data_type = org_module.input_quantizer.qdesc._type
                    if input_data_type not in ['uint8', 'uint16']:
                        self.add_violation(n.name, input_data_type, ErrorCode.E001)
                    children = n._next
                    children_module = named_modules[n._next.target]
                    if self.no_input_quantizer(children_module):
                        self.add_violation(n.name, '', ErrorCode.E002)
                    else:
                        output_data_type = children_module.input_quantizer.qdesc._type
                        if output_data_type not in ['uint8', 'uint16']:
                            self.add_violation(n.name, output_data_type, ErrorCode.E003)

                elif isinstance(org_module, (QAdd, QSubtract, QMul, QDivide, QGELU, QMish)):
                    pass
                
                elif isinstance(org_module, (Flatten, Reshape, Transpose)):
                    pass