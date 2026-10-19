"""Bounded native Gemma tool protocol and local read-only tool registry."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import ast
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
import json
import math
from pathlib import Path,PureWindowsPath
import re

START='<|tool_call>'
END='<tool_call|>'
HANDOFF='<|tool_response>'
QUOTE='<|"|>'


class ToolProtocolError(ValueError):pass


@dataclass(frozen=True)
class ToolCall:
    name:str
    arguments:dict


class _Reader:
    def __init__(self,text):self.text=text;self.at=0;self.nodes=0
    def ws(self):
        while self.at<len(self.text) and self.text[self.at].isspace():self.at+=1
    def take(self,token):
        self.ws()
        if not self.text.startswith(token,self.at):raise ToolProtocolError('Malformed native tool turn')
        self.at+=len(token)
    def word(self):
        self.ws();match=re.match(r'[A-Za-z_][A-Za-z0-9_]{0,63}',self.text[self.at:])
        if not match:raise ToolProtocolError('Invalid tool name or argument key')
        self.at+=len(match[0]);return match[0]
    def value(self,depth=0):
        self.ws();self.nodes+=1
        if depth>8 or self.nodes>128:raise ToolProtocolError('Tool argument complexity bound')
        if self.text.startswith(QUOTE,self.at):
            self.at+=len(QUOTE);end=self.text.find(QUOTE,self.at)
            if end<0:raise ToolProtocolError('Unclosed native string')
            value=self.text[self.at:end];self.at=end+len(QUOTE)
            return value
        if self.text.startswith('{',self.at):
            self.take('{');value={};self.ws()
            while not self.text.startswith('}',self.at):
                key=self.value(depth+1) if self.text.startswith(QUOTE,self.at) else self.word()
                if not isinstance(key,str) or key in value:raise ToolProtocolError('Duplicate or invalid argument key')
                self.take(':');value[key]=self.value(depth+1);self.ws()
                if self.text.startswith('}',self.at):break
                self.take(',');self.ws()
                if self.text.startswith('}',self.at):raise ToolProtocolError('Trailing object comma')
            self.take('}');return value
        if self.text.startswith('[',self.at):
            self.take('[');value=[];self.ws()
            while not self.text.startswith(']',self.at):
                value.append(self.value(depth+1));self.ws()
                if self.text.startswith(']',self.at):break
                self.take(',');self.ws()
                if self.text.startswith(']',self.at):raise ToolProtocolError('Trailing array comma')
            self.take(']');return value
        match=re.match(r'(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)',self.text[self.at:])
        if not match:raise ToolProtocolError('Unsupported native argument value')
        self.at+=len(match[0]);value=json.loads(match[0])
        if isinstance(value,float) and not math.isfinite(value):raise ToolProtocolError('Nonfinite argument')
        return value


def parse_turn(raw):
    """Return (calls, final_text); never execute an incomplete or mixed turn."""
    if not isinstance(raw,str) or len(raw)>16384:raise ToolProtocolError('Model turn bound')
    raw=raw.strip()
    if raw.startswith(START):
        if not raw.endswith(HANDOFF):raise ToolProtocolError('Tool turn lacks completed handoff')
        reader=_Reader(raw[:-len(HANDOFF)]);calls=[];reader.ws()
        while reader.at<len(reader.text):
            if len(calls)>=4:raise ToolProtocolError('Too many calls in one turn')
            reader.take(START);reader.take('call:');name=reader.word();arguments=reader.value()
            if not isinstance(arguments,dict):raise ToolProtocolError('Tool arguments must be an object')
            reader.take(END);reader.ws();calls.append(ToolCall(name,arguments))
        return calls,None
    if START in raw or HANDOFF in raw:raise ToolProtocolError('Mixed text and tool protocol is not executable')
    for ending in ('<turn|>','<eos>'):
        if raw.endswith(ending):return [],raw[:-len(ending)].strip()
    raise ToolProtocolError('Final answer is incomplete')


def calculate(expression):
    if not isinstance(expression,str) or not 1<=len(expression)<=256:raise ValueError('Expression length bound')
    tree=ast.parse(expression,mode='eval')
    if len(list(ast.walk(tree)))>64:raise ValueError('Expression complexity bound')
    def checked(value):
        if not value.is_finite() or value and abs(value.adjusted())>128:raise ValueError('Numeric range bound')
        return value
    def evaluate(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            text=ast.get_source_segment(expression,node)
            if not re.fullmatch(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',text):raise ValueError('Unsupported numeric literal')
            return checked(Decimal(text))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=evaluate(node.operand);return checked(-value if isinstance(node.op,ast.USub) else value)
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=evaluate(node.left),evaluate(node.right)
            if isinstance(node.op,ast.Add):value=a+b
            elif isinstance(node.op,ast.Sub):value=a-b
            elif isinstance(node.op,ast.Mult):value=a*b
            else:value=a/b
            return checked(value)
        raise ValueError('Only numeric +, -, *, / and parentheses are supported')
    with localcontext() as context:
        context.prec=28
        return str(evaluate(tree.body))


class WorkspaceTools:
    def __init__(self,workspace):
        self.root=Path(workspace).resolve(strict=True)
        if not self.root.is_dir():raise ValueError('Select a workspace directory')
    def declarations(self):
        descriptions=[('calculate','Evaluate bounded arithmetic; returns a decimal result.','expression','Numbers, +, -, *, / and parentheses.'),
                      ('read_text','Read a 1024-character page of a UTF-8 workspace file.','path','Relative file path inside the workspace.'),
                      ('list_files','List up to 100 entries inside a workspace directory.','path','Relative directory path; use . for the workspace root.')]
        declarations=[dict(type='function',function=dict(name=name,description=description,parameters=dict(type='object',
                properties={key:dict(type='string',description=arg)},required=[key]))) for name,description,key,arg in descriptions]
        declarations[1]['function']['parameters']['properties']['offset']=dict(type='integer',description='Character offset, default zero; use next_offset for another page.')
        return declarations
    def _path(self,value):
        if not isinstance(value,str) or not value or len(value)>512 or '\x00' in value:raise ValueError('Relative path required')
        if PureWindowsPath(value).anchor or Path(value).is_absolute() or ':' in value:raise ValueError('Absolute paths and alternate streams are not allowed')
        path=(self.root/value).resolve(strict=True)
        if not path.is_relative_to(self.root):raise ValueError('Path leaves the selected workspace')
        return path
    def execute(self,call):
        expected={'calculate':'expression','read_text':'path','list_files':'path'}
        allowed={'path','offset'} if call.name=='read_text' else {expected.get(call.name)}
        if call.name not in expected or expected[call.name] not in call.arguments or not set(call.arguments)<=allowed:
            raise ValueError('Unregistered tool or invalid arguments')
        value=call.arguments[expected[call.name]]
        if call.name=='calculate':return dict(result=calculate(value))
        path=self._path(value)
        if call.name=='read_text':
            if not path.is_file() or path.stat().st_size>1024*1024:raise ValueError('Read requires a UTF-8 file up to 1 MiB')
            with path.open('rb') as stream:data=stream.read(1024*1024+1)
            if len(data)>1024*1024:raise ValueError('File grew beyond the read bound')
            text=data.decode('utf-8-sig');offset=call.arguments.get('offset',0)
            if type(offset) is not int or not 0<=offset<=len(text):raise ValueError('Invalid character offset')
            page=text[offset:offset+1024];end=offset+len(page)
            return dict(text=page,next_offset=end if end<len(text) else None,total_characters=len(text))
        if not path.is_dir():raise ValueError('Directory required')
        entries=[]
        for item in path.iterdir():
            if len(entries)==100:raise ValueError('Directory exceeds 100 entries; choose a smaller directory')
            entries.append(item.name)
        return dict(entries=sorted(entries))
