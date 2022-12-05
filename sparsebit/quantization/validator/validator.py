import torch.fx as fx
import sys
from enum import Enum

from tabulate import tabulate
from sparsebit.quantization.modules import (Flatten, MaxPool2d, Reshape, Transpose, Concat, Permute, QIdentity, QReLU, QAdaptiveAvgPool2d, QConv2d, QConvTranspose2d, QLinear, QBatchNorm2d, QAdd, QSubtract, QMul, QDivide, QReLU6, QLeakyReLU, QSigmoid, QSiLU, QGELU, QMish)

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
    def __init__(self, model, qconfig):
        self.model = model
        self.qconfig = qconfig
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

    def add_violation(self, layer_name, info, err):
        message = self.rules_quant.err_desc[err] + str(info),
        message = f'{sys._getframe(2).f_code.co_filename}, Line {sys._getframe(2).f_lineno}: {message}'
        self.output.append([
            layer_name,
            'ERROR',
            message
        ])

    def ir_transform(self):
        named_modules = dict(self.model.model.named_modules(remove_duplicate=False))
        qnodes = []  # 用于避免重复遍历
        qnodes_invert = []
        # 给特定算子补上 input_quantizer
        for n in self.model.model.graph.nodes:
            qnodes_invert.append(n)
            if not isinstance(n, fx.Node) or n in qnodes:
                continue
            elif n.op == "call_module":
                org_module = named_modules[n.target]
                if isinstance(org_module, (MaxPool2d, Flatten, Reshape, Transpose, Concat, Permute)):
                    input_nodes_cache = list(n.all_input_nodes)
                    for idx, input_node in enumerate(input_nodes_cache):
                        new_module_name = n.name + "_identity{}".format(idx)
                        new_module = QIdentity()
                        new_module.build_quantizer(self.qconfig)
                        self.model.model.add_module(new_module_name, new_module)
                        with self.model.model.graph.inserting_before(n):
                            identity_node = self.model.model.graph.create_node(
                                op="call_module",
                                target=new_module_name,
                                args=(input_node,),
                                kwargs={},
                                name=new_module_name,
                            )
                        n.replace_input_with(input_node, identity_node)
        self.model.model.recompile()

        # 按照算子规范拉齐算子输入输出以及了多个分支的 scale 的 ZP
        named_modules = dict(self.model.model.named_modules(remove_duplicate=False))
        for n in qnodes_invert[::-1]:
            children = n._next
            if children.op != 'output':
                if not isinstance(n, fx.Node) or n in qnodes:
                    continue
                elif n.op == "call_module":
                    org_module = named_modules[n.target]
                    if isinstance(org_module, (MaxPool2d, Flatten, Reshape, Transpose, Permute)):
                        children_module = named_modules[n._next.target]
                        output_quantizer = children_module.input_quantizer
                        parent_module = named_modules[n._prev.target]
                        parent_module.input_quantizer = output_quantizer
                    if isinstance(org_module, QReLU):
                        children = n.users
                        for idx, child in enumerate(children):
                            children_module = named_modules[child.target]
                            if idx == 0:
                                children_module = named_modules[n._next.target]
                                output_quantizer = children_module.input_quantizer
                            else:
                                children_module = named_modules[n._next.target]
                                children_module.input_quantizer = output_quantizer
        return self.model

    def check_issue(self):
        named_modules = dict(self.model.model.named_modules(remove_duplicate=False))
        traced = self.model.model
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
                    if input_data_type not in ['uint4', 'uint8', 'uint16', 'int4', 'int8', 'int16']:
                        self.add_violation(n.name, input_data_type, ErrorCode.E001)
                    children = n._next
                    children_module = named_modules[n._next.target]
                    if self.no_input_quantizer(children_module):
                        self.add_violation(n.name, '', ErrorCode.E002)
                    else:
                        output_data_type = children_module.input_quantizer.qdesc._type
                        if output_data_type not in ['uint4', 'uint8', 'uint16', 'int4', 'int8', 'int16']:
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
                    parent_module = named_modules[n._prev.target]
                    if not isinstance(parent_module, (QConv2d, QConvTranspose2d, QLinear)):
                        self.add_violation(n.name, '', ErrorCode.E006)

                elif isinstance(org_module, QReLU):
                    parent_module = named_modules[n._prev.target]
                    if not isinstance(parent_module, (QConv2d, QConvTranspose2d, QLinear, QAdd)):
                        self.add_violation(n.name, '', ErrorCode.E007)
                    input_data_type = org_module.input_quantizer.qdesc._type
                    if input_data_type not in ['uint4', 'uint8', 'uint16', 'int4', 'int8', 'int16']:
                        self.add_violation(n.name, input_data_type, ErrorCode.E001)
                    children = n._next
                    children_module = named_modules[n._next.target]
                    if self.no_input_quantizer(children_module):
                        self.add_violation(n.name, '', ErrorCode.E002)
                    else:
                        output_data_type = children_module.input_quantizer.qdesc._type
                        if output_data_type not in ['uint4', 'uint8', 'uint16', 'int4', 'int8', 'int16']:
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

                elif isinstance(org_module, (QAdd, QSubtract, QMul, QDivide)):
                    pass

                elif isinstance(org_module, (Flatten, Reshape, Transpose)):
                    pass
        return
