from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

import pytest
from inspect_swe import gemini_cli_instrumentation as instrumentation


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _test_contract(
    source: str, patched: str
) -> instrumentation.GeminiCliTraceContextContract:
    return instrumentation.GeminiCliTraceContextContract(
        package="@google/gemini-cli",
        version="0.58.0",
        upstream_commit="test",
        target_relative_path=PurePosixPath(
            "node_modules/@google/gemini-cli/bundle/chunk.js"
        ),
        patch_sha256=_sha256(_test_patch().encode()),
        preimage_sha256=_sha256(source.encode()),
        postimage_sha256=_sha256(patched.encode()),
    )


def _test_patch() -> str:
    return """--- a/node_modules/@google/gemini-cli/bundle/chunk.js
+++ b/node_modules/@google/gemini-cli/bundle/chunk.js
@@ -1,2 +1,3 @@
 const first = true;
+const traceContext = true;
 const second = true;
"""


def _write_npm_tree(
    tmp_path: Path,
    contract: instrumentation.GeminiCliTraceContextContract,
    source: str,
) -> Path:
    package_root = tmp_path / "node_modules" / contract.package
    package_root.mkdir(parents=True)
    (package_root / "package.json").write_text(
        json.dumps({"version": contract.version}), encoding="utf-8"
    )
    target = tmp_path / contract.target_relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return tmp_path


def test_patch_gemini_cli_tree_applies_a_verified_exact_patch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = "const first = true;\nconst second = true;\n"
    patched = "const first = true;\nconst traceContext = true;\nconst second = true;\n"
    contract = _test_contract(source, patched)
    tree = _write_npm_tree(tmp_path, contract, source)
    monkeypatch.setattr(instrumentation, "GEMINI_CLI_TRACE_CONTEXT_CONTRACT", contract)
    monkeypatch.setattr(
        instrumentation, "_load_patch_resource", lambda: _test_patch().encode()
    )

    instrumentation.patch_gemini_cli_tree(tree, contract.version)

    assert (tree / contract.target_relative_path).read_text(encoding="utf-8") == patched


def test_patch_gemini_cli_tree_rejects_an_unexpected_preimage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = "const first = true;\nconst second = true;\n"
    patched = "const first = true;\nconst traceContext = true;\nconst second = true;\n"
    contract = _test_contract(source, patched)
    tree = _write_npm_tree(tmp_path, contract, "const modified = true;\n")
    monkeypatch.setattr(instrumentation, "GEMINI_CLI_TRACE_CONTEXT_CONTRACT", contract)
    monkeypatch.setattr(
        instrumentation, "_load_patch_resource", lambda: _test_patch().encode()
    )

    with pytest.raises(
        instrumentation.GeminiCliInstrumentationError,
        match="unexpected preimage",
    ):
        instrumentation.patch_gemini_cli_tree(tree, contract.version)


def test_patch_gemini_cli_tree_rejects_an_unsupported_version_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        instrumentation,
        "_load_patch_resource",
        lambda: pytest.fail("unsupported versions must not load a patch"),
    )

    with pytest.raises(
        instrumentation.GeminiCliInstrumentationError,
        match="supports only",
    ):
        instrumentation.patch_gemini_cli_tree(tmp_path, "0.57.0")


def test_canonical_patch_resource_and_cache_identity_are_exact() -> None:
    contract = instrumentation.GEMINI_CLI_TRACE_CONTEXT_CONTRACT

    assert contract.package == "@google/gemini-cli"
    assert (
        contract.version == instrumentation.GEMINI_CLI_INSTRUMENTED_VERSION == "0.58.0"
    )
    assert str(contract.target_relative_path) == (
        "node_modules/@google/gemini-cli/bundle/chunk-MFLFXOVQ.js"
    )
    assert _sha256(instrumentation._load_patch_resource()) == contract.patch_sha256
    assert instrumentation.GEMINI_CLI_TRACE_CONTEXT_CACHE_REVISION == (
        f"w3c-trace-context-{contract.patch_sha256[:12]}"
    )


_PATCH_FIXTURE = """    mcp_servers: mcpServers
  };
}
var LoggingContentGenerator = class {
  wrapped;
  config;
      const serverDetails = this._getEndpointUrl(req, "generateContent");
      this.logApiRequest(contents, req.model, userPromptId, role, req.config, serverDetails);
      try {
        const response = await this.wrapped.generateContent(req, userPromptId, role);
        spanMetadata.output = response.candidates?.[0]?.content ?? null;
        spanMetadata.attributes[GEN_AI_USAGE_INPUT_TOKENS] = response.usageMetadata?.promptTokenCount ?? 0;
        spanMetadata.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = response.usageMetadata?.candidatesTokenCount ?? 0;
      this.logApiRequest(toContents(req.contents), req.model, userPromptId, role, req.config, serverDetails);
      let stream2;
      try {
        stream2 = await this.wrapped.generateContentStream(req, userPromptId, role);
      } catch (error40) {
        const durationMs = Date.now() - startTime;
        this._fixGaxiosErrorData(error40);
var McpComplianceTransport = class extends EventEmitter8 {
  transport;
  constructor(transport) {
    super();
    this.transport = transport;
    this.transport.onmessage = (message) => {
      this.handleMessage(message);
    };
    this.transport.onclose = () => {
      this.onclose?.();
    };
    this.transport.onerror = (error40) => {
      this.onerror?.(error40);
    };
  }
  onclose;
  onerror;
  onmessage;
  async start() {
    await this.transport.start();
  }
  async close() {
    await this.transport.close();
  }
  async send(message) {
    await this.transport.send(message);
  }
  handleMessage(message) {
    if (this.isJsonResponse(message)) {
      this.fixStructuredContent(message);
    }
    this.onmessage?.(message);
  }
  isJsonResponse(message) {
    return "result" in message || "error" in message;
  }
  fixStructuredContent(response) {
    if (!("result" in response))
      return;
    const result2 = response.result;
    if (result2.content && Array.isArray(result2.content) && result2.content.length > 0 && !result2.structuredContent) {
      const firstItem = result2.content[0];
      if (firstItem.type === "text" && typeof firstItem.text === "string") {
        try {
          const parsed = JSON.parse(firstItem.text);
          result2.structuredContent = parsed;
        } catch {
        }
      }
    }
  }
};
"""


def _patched_transport_class() -> str:
    contract = instrumentation.GEMINI_CLI_TRACE_CONTEXT_CONTRACT
    patched = instrumentation._apply_exact_unified_patch(
        _PATCH_FIXTURE,
        instrumentation._load_patch_resource().decode("utf-8"),
        contract.target_relative_path,
    )
    start = patched.index("var McpComplianceTransport = class extends EventEmitter8 {")
    end = patched.index("\n};", start) + len("\n};")
    return patched[start:end]


def test_patched_transport_keeps_nonrecord_json_out_of_structured_content(
    tmp_path: Path,
) -> None:
    """Run the exact bundle patch over the real transport normalization behavior."""
    transport = _patched_transport_class()
    script = tmp_path / "mcp_compliance_transport.mjs"
    script.write_text(
        "class EventEmitter8 {}\n"
        + transport
        + """
const transport = new McpComplianceTransport({
  start: async () => {},
  close: async () => {},
  send: async () => {},
});
const cases = [
  {
    name: "object",
    text: '{"price":34.99,"rating":4.6}',
    expected: { price: 34.99, rating: 4.6 },
    isError: false,
  },
  {
    name: "array",
    text: '["not","a","record"]',
    expected: undefined,
    isError: false,
  },
  {
    name: "primitive",
    text: "42",
    expected: undefined,
    isError: false,
  },
  {
    name: "null",
    text: "null",
    expected: undefined,
    isError: false,
  },
  {
    name: "existing",
    text: '["ignored"]',
    structuredContent: { nested: { preserved: true } },
    expected: { nested: { preserved: true } },
    isError: true,
  },
];
let passed = true;
for (const item of cases) {
  const response = {
    result: {
      content: [{ type: "text", text: item.text }],
      ...(item.structuredContent === undefined
        ? {}
        : { structuredContent: item.structuredContent }),
      isError: item.isError,
    },
  };
  const originalContent = JSON.stringify(response.result.content);
  transport.handleMessage(response);
  const contentPreserved =
    JSON.stringify(response.result.content) === originalContent;
  const structuredContentPreserved =
    item.expected === undefined
      ? response.result.structuredContent === undefined
      : JSON.stringify(response.result.structuredContent) ===
        JSON.stringify(item.expected);
  const errorPreserved = response.result.isError === item.isError;
  const result = contentPreserved && structuredContentPreserved && errorPreserved;
  console.log(`${item.name}: ${result}`);
  passed &&= result;
}
process.exitCode = passed ? 0 : 1;
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["node", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "object: true\narray: true\nprimitive: true\nnull: true\nexisting: true\n"
    )
