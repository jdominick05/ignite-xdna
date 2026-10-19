"""Native protocol, tool bounds and incomplete-turn execution checks."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from decimal import DecimalException
from types import SimpleNamespace

import pytest

from npu.gemma4.tooling import ToolCall,ToolProtocolError,WorkspaceTools,calculate,parse_turn
from npu.gemma4.agent import AgentIncomplete,BoundedAgent
from npu.gemma4.runtime import HardwareBusy


def test_native_call_and_final():
    calls,answer=parse_turn('<|tool_call>call:calculate{expression:<|"|>(137 + 29) * 7<|"|>}<tool_call|><|tool_response>')
    assert answer is None and calls==[ToolCall('calculate',{'expression':'(137 + 29) * 7'})]
    assert calculate(calls[0].arguments['expression'])=='1162'
    assert calculate('0.1 + 0.2')=='0.3'
    assert parse_turn('1162<turn|>')==([], '1162')


@pytest.mark.parametrize('text',[
    '<|tool_call>call:calculate{expression:<|"|>1+2<|"|>}<tool_call|>',
    'Example: <|tool_call>call:calculate{}<tool_call|><|tool_response>',
    '<|tool_call>call:calculate{expression:<|"|>1<|"|>,expression:<|"|>2<|"|>}<tool_call|><|tool_response>',
    '<|tool_call>call:calculate{expression:<|"|>unclosed}<tool_call|><|tool_response>',
    '<|tool_call>call:calculate{value:1e999}<tool_call|><|tool_response>',
    '<|tool_call>call:calculate{expression:<|"|>1<|"|>,}<tool_call|><|tool_response>',
    'An incomplete answer',
])
def test_ambiguous_or_incomplete_turn_rejected(text):
    with pytest.raises(ToolProtocolError):parse_turn(text)


@pytest.mark.parametrize('expression',["__import__('os').system('echo bad')",'2**999999','[1,2]','True+1','1/0','1e999'])
def test_calculator_rejects_code_and_unbounded_work(expression):
    with pytest.raises((ValueError,DecimalException)):calculate(expression)


def test_workspace_tools_and_pagination(tmp_path):
    root=tmp_path/'workspace';root.mkdir();file=root/'notes.txt';file.write_text('a'*1030,encoding='utf-8')
    tools=WorkspaceTools(root)
    first=tools.execute(ToolCall('read_text',{'path':'notes.txt'}))
    assert len(first['text'])==1024 and first['next_offset']==1024
    assert tools.execute(ToolCall('read_text',{'path':'notes.txt','offset':1024}))['text']=='aaaaaa'
    assert tools.execute(ToolCall('list_files',{'path':'.'}))['entries']==['notes.txt']
    outside=tmp_path/'outside.txt';outside.write_text('outside',encoding='utf-8')
    for path in ('../outside.txt',str(outside),'C:outside.txt','notes.txt:stream','\\\\server\\share\\file'):
        with pytest.raises(ValueError):tools.execute(ToolCall('read_text',{'path':path}))
    with pytest.raises(ValueError):tools.execute(ToolCall('write_text',{'path':'notes.txt'}))
    assert file.read_text()=='a'*1030


def test_truncated_generation_cannot_execute_tool(tmp_path):
    assistant=SimpleNamespace(last_result=dict(finish_reason='length',tokens=[1],cancelled=False),cancel=lambda:None)
    # A closeable empty stream, like the hardware API.
    def stream(*a,**k):
        yield from ()
    assistant.stream_tokens=stream
    tokenizer=SimpleNamespace(encode=lambda *a,**k:[1],decode=lambda ids:pytest.fail('Truncated tokens must not be parsed'))
    agent=BoundedAgent(assistant,tokenizer,tmp_path)
    with pytest.raises(AgentIncomplete,match='output budget'):agent.run('Task')
    assert agent.last_result['tool_calls']==[]


def test_cancel_event_prevents_tool_execution(tmp_path,monkeypatch):
    assistant=SimpleNamespace(last_result=dict(finish_reason='eos',tokens=[1],cancelled=False,npu_released=True,npu_idle=True),cancel=lambda:None)
    def stream(*a,**k):yield from ()
    assistant.stream_tokens=stream
    tokenizer=SimpleNamespace(encode=lambda *a,**k:[1],decode=lambda ids:'<|tool_call>call:calculate{expression:<|"|>1+2<|"|>}<tool_call|><|tool_response>')
    agent=BoundedAgent(assistant,tokenizer,tmp_path)
    monkeypatch.setattr(agent.tools,'execute',lambda call:pytest.fail('A cancelled task must not execute a tool'))
    with pytest.raises(AgentIncomplete,match='cancelled'):
        agent.run('Task',on_event=lambda event:agent.cancel())
    assert agent.last_result['cancelled'] and not agent.last_result['tool_calls']


@pytest.mark.parametrize('cancel_wait',[False,True])
def test_busy_admission_wait_is_cancellable_and_keeps_prompt(tmp_path,cancel_wait):
    attempts=[];encodes=[]
    assistant=SimpleNamespace(cancel=lambda:None)
    def stream(*a,**k):
        attempts.append(a)
        if len(attempts)==1:raise HardwareBusy('busy',{'test':'busy'})
        assistant.last_result=dict(finish_reason='eos',tokens=[1],cancelled=False,npu_released=True,npu_idle=True)
        yield from ()
    assistant.stream_tokens=stream
    def encode(messages,**kwargs):encodes.append(list(messages));return [1]
    tokenizer=SimpleNamespace(encode=encode,decode=lambda ids:'done<turn|>')
    agent=BoundedAgent(assistant,tokenizer,tmp_path,max_hardware_wait=.01)
    if cancel_wait:
        with pytest.raises(AgentIncomplete,match='cancelled'):
            agent.run('Task',on_event=lambda event:agent.cancel())
        assert len(attempts)==1
    else:
        assert agent.run('Task')=='done' and len(attempts)==2
    assert len(encodes)==1 and len(encodes[0])==2
    assert len(agent.last_result['admission_rejections'])==1
