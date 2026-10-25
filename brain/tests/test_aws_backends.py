"""AWS chat contracts without network calls or workstation credentials."""
import json
import base64
from unittest.mock import Mock

import httpx
import pytest

from brain.agent import llm


def test_bedrock_roundtrip_tools_images_and_iam(monkeypatch):
    import boto3
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(boto3, "client", factory)
    monkeypatch.setenv("AGENT_MODEL", "in.anthropic.claude-haiku-4-5-20251001-v1:0")
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    client.converse.return_value = {"output": {"message": {"content": [
        {"text": "Inspect the frame."},
        {"toolUse": {"toolUseId": "remote-1", "name": "scan_thermal", "input": {}}}
    ]}}, "usage": {"inputTokens": 11, "outputTokens": 7}}
    backend = llm.make_backend("bedrock")
    tool = {"type": "function", "function": {"name": "scan_thermal", "description": "Inspect heat",
            "parameters": {"type": "object", "properties": {}}}}
    reply = backend.chat([{"role": "system", "content": "Be careful"},
                          {"role": "user", "content": "Look", "images": [base64.b64encode(b"png-data").decode()]}], [tool])
    assert reply.text == "Inspect the frame."
    assert reply.tool_calls == [llm.ToolCall("scan_thermal", {})]
    assert reply.raw == {"usage": {"inputTokens": 11, "outputTokens": 7}}
    kwargs = factory.call_args.kwargs
    assert kwargs["region_name"] == "ap-south-1"
    assert kwargs["config"].signature_version == "v4"
    assert kwargs["config"].connect_timeout == 3
    assert kwargs["config"].read_timeout == 20
    assert kwargs["config"].retries["total_max_attempts"] == 1
    assert not any("key" in k or "token" in k for k in kwargs)
    args = client.converse.call_args.kwargs
    assert args["modelId"] == "in.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert args["messages"][0]["content"][1]["image"]["source"]["bytes"] == b"png-data"
    # A second call must answer both tool requests together with their reconstructed IDs.
    messages = [{"role": "user", "content": "Look"}, {"role": "assistant", "content": "", "tool_calls": [
        {"name": "scan_thermal", "arguments": {}}, {"name": "assess_visibility", "arguments": {}}]},
        {"role": "tool", "content": "heat"}, {"role": "tool", "content": "clear"}]
    backend.chat(messages, [tool])
    convo = client.converse.call_args.kwargs["messages"]
    assert [b["toolResult"]["toolUseId"] for b in convo[2]["content"]] == ["t1_0", "t1_1"]
    backend.chat([{"role": "user", "content": "Hello"}], [])
    assert "toolConfig" not in client.converse.call_args.kwargs


def test_azure_environment_never_reads_home_file(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-test-key")
    monkeypatch.setenv("AZURE_OPENAI_TIMEOUT_S", "15")
    monkeypatch.setenv("AZURE_OPENAI_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(llm, "_load_env_file", lambda _: pytest.fail("Must not open a home credential file"))
    backend = llm.make_backend("azure", model="test-deployment")
    def respond(request):
        assert request.headers["api-key"] == "fake-test-key"
        assert "/deployments/test-deployment/" in str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    backend.client.close()
    backend.client = httpx.Client(transport=httpx.MockTransport(respond))
    assert backend.chat([{"role": "user", "content": "Hello"}], []).text == "ok"
    backend.client.close()


def test_bedrock_sdk_schema_and_robot_loop(tmp_path, monkeypatch):
    """Catch invalid Converse message shapes and verify the agent consumes Bedrock tool calls."""
    import boto3
    from botocore.stub import Stubber, ANY
    from brain.agent.robot_agent import RobotAgent
    sdk = boto3.client("bedrock-runtime", region_name="us-east-1",
                       aws_access_key_id="fake", aws_secret_access_key="fake")
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: sdk)
    backend = llm.BedrockBackend("test-profile")
    params = {"modelId": "test-profile", "system": ANY, "messages": ANY,
              "toolConfig": ANY, "inferenceConfig": {"maxTokens": 1024, "temperature": 0}}
    response = {"output": {"message": {"role": "assistant", "content": [
        {"toolUse": {"toolUseId": "finish-1", "name": "finish", "input": {"summary": "Wait for sensor upload"}}}]}},
        "stopReason": "tool_use", "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
        "metrics": {"latencyMs": 3}}
    with Stubber(sdk) as stub:
        stub.add_response("converse", response, params)
        result = RobotAgent(backend, tmp_path / "trace.jsonl").step({"robot": "robot_A", "robot_dir": str(tmp_path),
            "t": 0., "pose": {"x": 0., "y": 0., "heading_deg": 0.}, "event": {"kind": "periodic"}})
        assert result["summary"] == "Wait for sensor upload"
        stub.assert_no_pending_responses()


def test_azure_uses_configured_deployment_and_aws_retry_bound(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("AGENT_MODEL", "secret-backed-deployment")
    monkeypatch.setenv("AZURE_OPENAI_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("AZURE_OPENAI_TIMEOUT_S", "15")
    backend = llm.make_backend("azure")
    assert backend.model == "secret-backed-deployment"
    assert backend.client.timeout.read == 15
    backend.client.close()
    requests = []
    def throttle(request):
        requests.append(request)
        return httpx.Response(429)
    backend.client = httpx.Client(transport=httpx.MockTransport(throttle))
    with pytest.raises(httpx.HTTPStatusError):
        backend.chat([{"role": "user", "content": "Hello"}], [])
    assert len(requests) == 1
    backend.client.close()


def test_azure_reasoning_effort_refusal_is_resent_even_with_one_attempt(monkeypatch):
    """The AWS task sets AZURE_OPENAI_MAX_ATTEMPTS=1. gpt-5.6-terra answers 400 to function tools with reasoning_effort;
    the resend without it must still happen (25 Oct: every AWS agent and commander step ended in model_error)."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("AZURE_OPENAI_MAX_ATTEMPTS", "1")
    backend = llm.make_backend("azure", model="test-deployment")
    backend.client.close()
    bodies = []
    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "Function tools with reasoning_effort are not supported",
                                                       "param": "reasoning_effort"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    backend.client = httpx.Client(transport=httpx.MockTransport(respond))
    assert backend.chat([{"role": "user", "content": "Hello"}], [{"type": "function", "function": {"name": "finish"}}]).text == "ok"
    assert len(bodies) == 2 and "reasoning_effort" not in bodies[1]
    backend.client.close()
