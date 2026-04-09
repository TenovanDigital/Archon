# Vercel AI SDK Integration Plan

## Context

The user wants to run Archon on Linux with both Claude (via subscription) and local Ollama models for smaller tasks (classification, summarization, doc writing with tools/MCP). Rather than raw Ollama HTTP calls, we use Vercel AI SDK (`ai` package) which provides a unified interface to 20+ providers (Ollama, Groq, OpenAI, Mistral, etc.) with built-in tool calling, MCP support, and agentic loops.

Linux compatibility is already confirmed — no changes needed. This plan covers adding `vercel-ai` as a third provider alongside `claude` and `codex`.

## Key Design Decisions

1. **Provider name**: `'vercel-ai'` — distinguishes from raw provider names
2. **Model format**: Slash-delimited `provider/model` string (e.g., `ollama/llama3`, `groq/mixtral-8x7b`) — the Vercel AI SDK natural format
3. **MCP**: Reuse existing `mcp:` node field; translate MCP server configs into `createMCPClient()` calls
4. **Tools**: MCP-based only (no custom in-process tool definitions for v1). The `allowed_tools`/`denied_tools` fields are not applicable; tools come from MCP servers
5. **Session resumption**: Not supported (Vercel AI SDK is stateless) — `resumeSessionId` silently ignored
6. **Config**: `assistants.vercel-ai` block with `model` default and optional `providers` map for baseURL/apiKey overrides per sub-provider

## Example Usage

```yaml
# .archon/config.yaml
assistants:
  vercel-ai:
    model: ollama/llama3
    maxSteps: 10
    providers:
      ollama:
        baseURL: http://localhost:11434/api
```

```yaml
# .archon/workflows/mixed.yaml
nodes:
  - id: classify
    provider: vercel-ai
    model: ollama/llama3
    prompt: 'Classify: bug, feature, or docs? Output JSON.'
    output_format: { type: object, properties: { type: { type: string } } }

  - id: write-docs
    provider: vercel-ai
    model: ollama/codestral
    depends_on: [classify]
    when: "$classify.output.type == 'docs'"
    mcp: mcp-servers.yaml
    prompt: 'Write documentation for...'

  - id: implement
    provider: claude
    model: sonnet
    depends_on: [classify]
    when: "$classify.output.type == 'feature'"
    command: implement
```

---

## Phase 1: Type System (widen `'claude' | 'codex'` union)

### Step 1: Define `AssistantProvider` type alias

**File**: `packages/core/src/types/index.ts`

- Add `export type AssistantProvider = 'claude' | 'codex' | 'vercel-ai';` near top
- Export it from the package

### Step 2: Update config types

**File**: `packages/core/src/config/config-types.ts`

- Import `AssistantProvider` from `../types`
- `GlobalConfig.defaultAssistant` (line 41): `'claude' | 'codex'` → `AssistantProvider`
- `GlobalConfig.assistants` (line 46-49): Add `'vercel-ai'?: VercelAiAssistantDefaults`
- Add new interface:
  ```typescript
  export interface VercelAiAssistantDefaults {
    model?: string; // e.g., "ollama/llama3"
    maxSteps?: number; // agentic loop max steps (default 10)
    providers?: Record<string, { baseURL?: string; apiKey?: string }>;
  }
  ```
- `RepoConfig.assistant` (line 112): → `AssistantProvider`
- `MergedConfig.assistant` (line 215): → `AssistantProvider`
- `MergedConfig.assistants` (line 216-219): Add `'vercel-ai': VercelAiAssistantDefaults`
- `SafeConfig.assistant` (line 279): → `AssistantProvider`
- `SafeConfig.assistants` (line 280-283): Add `'vercel-ai': Pick<VercelAiAssistantDefaults, 'model'>`

### Step 3: Update workflow-side types

**File**: `packages/workflows/src/deps.ts`

- Line 229: `AssistantClientFactory` param → `'claude' | 'codex' | 'vercel-ai'`
- Line 241: `WorkflowConfig.assistant` → `'claude' | 'codex' | 'vercel-ai'`
- Lines 254-266: `WorkflowConfig.assistants` → add `'vercel-ai': { model?: string; maxSteps?: number }`

### Step 4: Update Zod schemas

**File**: `packages/workflows/src/schemas/workflow.ts` (line 32)

- `z.enum(['claude', 'codex'])` → `z.enum(['claude', 'codex', 'vercel-ai'])`

**File**: `packages/workflows/src/schemas/dag-node.ts` (line 119)

- Same change

**File**: `packages/server/src/routes/schemas/config.schemas.ts`

- Line 10: `z.enum(['claude', 'codex'])` → `z.enum(['claude', 'codex', 'vercel-ai'])`
- Line 37: Same
- Add `'vercel-ai'` block to `updateAssistantConfigBodySchema`

### Step 5: Update model validation

**File**: `packages/workflows/src/model-validation.ts`

- Add `isVercelAiModel(model: string): boolean` — returns true if model contains `/` and doesn't start with `claude-`
- Update `isModelCompatible()` signature: `provider: 'claude' | 'codex' | 'vercel-ai'`
- Add `vercel-ai` case: accepts models matching `isVercelAiModel()`

### Step 6: Update provider resolution in executors

**File**: `packages/workflows/src/executor.ts` (lines 281-296)

- Line 281: Widen `resolvedProvider` type to `'claude' | 'codex' | 'vercel-ai'`
- Add inference branch: `else if (workflow.model && isVercelAiModel(workflow.model))` → `'vercel-ai'`
- Inserted between the Claude check and the fallback-to-codex branch

**File**: `packages/workflows/src/dag-executor.ts`

- `resolveNodeProviderAndModel()` (line 353-388): Same pattern — widen types, add inference branch
- All other typed `'claude' | 'codex'` locals (~lines 709, 1445, 1482, 1970, 2140, 2377): Widen to include `'vercel-ai'`

---

## Phase 2: Client Implementation

### Step 7: Add dependencies

**File**: `packages/core/package.json`

```json
"ai": "^4.x",
"@ai-sdk/mcp": "^0.x",
"ai-sdk-ollama": "^x.x"
```

Note: Additional provider packages (`@ai-sdk/openai`, `@ai-sdk/anthropic`, etc.) can be added as needed. Ollama can also work via `@ai-sdk/openai` with `baseURL` set to `http://localhost:11434/v1`.

### Step 8: Create VercelAiClient

**New file**: `packages/core/src/clients/vercel-ai.ts`

Structure:

```typescript
import { streamText, tool, type CoreMessage } from 'ai';
import { createMCPClient, Experimental_StdioMCPTransport } from '@ai-sdk/mcp';
import type { IAssistantClient, MessageChunk, AssistantRequestOptions } from '../types';

export class VercelAiClient implements IAssistantClient {
  private static readonly MAX_RETRIES = 3;
  private static readonly RETRY_BASE_DELAY_MS = 2000;

  async *sendQuery(
    prompt: string,
    cwd: string,
    resumeSessionId?: string,
    options?: AssistantRequestOptions
  ): AsyncGenerator<MessageChunk> {
    // 1. Parse model: "ollama/llama3" → providerName="ollama", modelName="llama3"
    // 2. Resolve provider instance via getVercelProvider(providerName, config)
    // 3. If options.mcpServers, create MCP clients → gather tools
    // 4. Call streamText({ model, messages, tools, abortSignal, maxSteps })
    // 5. Iterate result.fullStream:
    //    - 'text-delta' → yield { type: 'assistant', content }
    //    - 'tool-call' → yield { type: 'tool', toolName, toolInput }
    //    - 'tool-result' → yield { type: 'tool_result', toolName, toolOutput }
    // 6. On finish → yield { type: 'result', tokens }
    // 7. Close MCP clients in finally block
  }

  getType(): string {
    return 'vercel-ai';
  }
}
```

Key behaviors:

- **Provider map**: Static map from provider name → SDK create function. Start with ollama, openai, groq, mistral. Unknown → clear error.
- **MCP**: Read MCP config YAML (same format as Claude's `mcp:` field), create clients via `createMCPClient()` + `Experimental_StdioMCPTransport`, get tools, pass to `streamText()`. Close all clients in `finally`.
- **Structured output**: If `options.outputFormat` set, pass schema to `streamText()` as output config.
- **Retry**: Same exponential backoff pattern as Claude/Codex (3 retries, 2s base delay).
- **Abort**: Pass `options.abortSignal` directly to `streamText()`.
- **No session resume**: `resumeSessionId` is silently ignored (log at debug level).

### Step 9: Register in factory

**File**: `packages/core/src/clients/factory.ts`

- Import `VercelAiClient`
- Add case `'vercel-ai'` to switch
- Update error message

---

## Phase 3: Configuration

### Step 10: Update config loader

**File**: `packages/core/src/config/config-loader.ts`

- `getDefaults()` (line 190): Add `'vercel-ai': {}` to `assistants`
- `applyEnvOverrides()` (line 235): Add `|| envAssistant === 'vercel-ai'` to the check
- Ensure merge logic handles `assistants['vercel-ai']` in both global and repo configs

### Step 11: Update store-adapter

**File**: `packages/core/src/workflows/store-adapter.ts`

- The compile-time assertion `MergedConfig ← WorkflowConfig` will auto-catch any drift. After updating both types consistently, no manual fix needed — just verify it compiles.

---

## Phase 4: UI and API

### Step 12: Update Web UI provider selectors

**Files** (add `'vercel-ai'` option to dropdowns):

- `packages/web/src/routes/SettingsPage.tsx` — default assistant selector + add Vercel AI model input
- `packages/web/src/components/workflows/BuilderToolbar.tsx` — workflow provider dropdown
- `packages/web/src/components/workflows/NodeInspector.tsx` — node provider dropdown
- `packages/web/src/components/workflows/WorkflowBuilder.tsx` — widen state type

### Step 13: Update server config route handler

**File**: `packages/server/src/routes/api.ts` (around line 2529)

- Handle `body['vercel-ai']` in the PATCH config endpoint

### Step 14: Regenerate API types

```bash
bun run dev:server &
bun --filter @archon/web generate:types
```

---

## Phase 5: CLI

### Step 15: Update CLI setup

**File**: `packages/cli/src/commands/setup.ts`

- Add `'vercel-ai'` to default assistant selection (lines 680-698)
- Add provider setup questions (Ollama URL, optional API keys for cloud providers)

---

## Phase 6: Testing

### Step 16: Unit tests

**New file**: `packages/core/src/clients/vercel-ai.test.ts`

- Model string parsing
- Unknown sub-provider error
- Stream event → MessageChunk mapping
- Abort signal handling
- MCP tool integration
- Session resume silently ignored

**File**: `packages/workflows/src/model-validation.test.ts`

- `isVercelAiModel()` tests
- `isModelCompatible('vercel-ai', ...)` tests

### Step 17: Integration test

- Ensure existing Claude/Codex workflows still work (no regressions)
- Test vercel-ai provider resolution from model inference

---

## Verification

1. **Type check**: `bun run type-check` — all 20+ changed files compile
2. **Lint**: `bun run lint` — no new warnings
3. **Tests**: `bun run test` — all existing + new tests pass
4. **Manual test** (Ollama): Start Ollama locally, create a workflow with `provider: vercel-ai, model: ollama/llama3`, run via CLI:
   ```bash
   bun run cli workflow run <workflow-name> "test prompt"
   ```
5. **Manual test** (mixed workflow): Run a workflow mixing Claude and Vercel AI nodes
6. **Web UI**: Verify provider dropdowns show "Vercel AI", settings page allows model config
7. **Full validation**: `bun run validate`

## Files Changed (Summary)

| File                                                        | Change                                            |
| ----------------------------------------------------------- | ------------------------------------------------- |
| `packages/core/src/types/index.ts`                          | Add `AssistantProvider` type                      |
| `packages/core/src/config/config-types.ts`                  | Widen unions, add `VercelAiAssistantDefaults`     |
| `packages/core/src/config/config-loader.ts`                 | Add defaults, env override                        |
| `packages/core/src/clients/vercel-ai.ts`                    | **NEW** — main client                             |
| `packages/core/src/clients/factory.ts`                      | Register new client                               |
| `packages/core/package.json`                                | Add `ai`, `@ai-sdk/mcp`, `ai-sdk-ollama` deps     |
| `packages/workflows/src/deps.ts`                            | Widen factory type, config type                   |
| `packages/workflows/src/model-validation.ts`                | Add `isVercelAiModel`, update `isModelCompatible` |
| `packages/workflows/src/schemas/workflow.ts`                | Widen provider enum                               |
| `packages/workflows/src/schemas/dag-node.ts`                | Widen provider enum                               |
| `packages/workflows/src/executor.ts`                        | Widen types, add inference branch                 |
| `packages/workflows/src/dag-executor.ts`                    | Widen types, add inference branch                 |
| `packages/server/src/routes/schemas/config.schemas.ts`      | Widen enums, add vercel-ai schema                 |
| `packages/server/src/routes/api.ts`                         | Handle vercel-ai in config PATCH                  |
| `packages/web/src/routes/SettingsPage.tsx`                  | Add provider option + config UI                   |
| `packages/web/src/components/workflows/BuilderToolbar.tsx`  | Add provider option                               |
| `packages/web/src/components/workflows/NodeInspector.tsx`   | Add provider option                               |
| `packages/web/src/components/workflows/WorkflowBuilder.tsx` | Widen state type                                  |
| `packages/cli/src/commands/setup.ts`                        | Add vercel-ai to setup flow                       |
| `packages/core/src/clients/vercel-ai.test.ts`               | **NEW** — tests                                   |
| `packages/workflows/src/model-validation.test.ts`           | Add vercel-ai tests                               |
