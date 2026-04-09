/**
 * Vercel AI SDK wrapper
 * Provides async generator interface for streaming responses from any
 * Vercel AI SDK-compatible provider (Ollama, OpenAI, Groq, Mistral, etc.)
 *
 * Model format: "provider/model" (e.g., "ollama/llama3", "openai/gpt-4o")
 * MCP: Reuses existing mcp: node field; translates configs into createMCPClient() calls
 * Session resumption: Not supported (stateless) — resumeSessionId silently ignored
 */
import { streamText, type LanguageModel, type ToolSet } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import {
  type AssistantRequestOptions,
  type IAssistantClient,
  type MessageChunk,
  type TokenUsage,
} from '../types';
import type { VercelAiAssistantDefaults } from '../config/config-types';
import { createLogger } from '@archon/paths';
import { loadConfig } from '../config/config-loader';

/** Lazy-initialized logger (deferred so test mocks can intercept createLogger) */
let cachedLog: ReturnType<typeof createLogger> | undefined;
function getLog(): ReturnType<typeof createLogger> {
  if (!cachedLog) cachedLog = createLogger('client.vercel-ai');
  return cachedLog;
}

/** Max retries for transient failures */
const MAX_RETRIES = 3;

/** Delay between retries in milliseconds */
const RETRY_BASE_DELAY_MS = 2000;

/** Rate limit patterns in error messages */
const RATE_LIMIT_PATTERNS = ['rate limit', 'too many requests', '429', 'overloaded'];

function isRateLimitError(message: string): boolean {
  const lower = message.toLowerCase();
  return RATE_LIMIT_PATTERNS.some(p => lower.includes(p));
}

/**
 * Parse "provider/model" format into provider name and model name.
 * E.g., "ollama/llama3" → { providerName: "ollama", modelName: "llama3" }
 * E.g., "openai/gpt-4o" → { providerName: "openai", modelName: "gpt-4o" }
 */
function parseModelString(model: string): { providerName: string; modelName: string } {
  const slashIndex = model.indexOf('/');
  if (slashIndex === -1 || slashIndex === 0 || slashIndex === model.length - 1) {
    throw new Error(
      `Invalid Vercel AI model format: "${model}". Expected "provider/model" (e.g., "ollama/llama3").`
    );
  }
  return {
    providerName: model.slice(0, slashIndex),
    modelName: model.slice(slashIndex + 1),
  };
}

/**
 * Resolve a provider name + model name to a Vercel AI SDK LanguageModel instance.
 * All providers use @ai-sdk/openai with appropriate baseURL — this gives us
 * LanguageModelV2/V3 compatibility required by AI SDK v6's LanguageModel type.
 */
function resolveModel(
  providerName: string,
  modelName: string,
  providerConfig?: VercelAiAssistantDefaults['providers']
): LanguageModel {
  const config = providerConfig?.[providerName];

  switch (providerName) {
    case 'ollama': {
      // Ollama exposes an OpenAI-compatible endpoint at /v1
      const provider = createOpenAI({
        baseURL: config?.baseURL ?? 'http://localhost:11434/v1',
        apiKey: config?.apiKey ?? 'ollama', // Ollama doesn't need a real key
      });
      return provider(modelName);
    }

    case 'openai': {
      const provider = createOpenAI({
        ...(config?.baseURL ? { baseURL: config.baseURL } : {}),
        ...(config?.apiKey ? { apiKey: config.apiKey } : {}),
      });
      return provider(modelName);
    }

    case 'groq': {
      const provider = createOpenAI({
        baseURL: config?.baseURL ?? 'https://api.groq.com/openai/v1',
        apiKey: config?.apiKey ?? process.env.GROQ_API_KEY ?? '',
      });
      return provider(modelName);
    }

    case 'mistral': {
      const provider = createOpenAI({
        baseURL: config?.baseURL ?? 'https://api.mistral.ai/v1',
        apiKey: config?.apiKey ?? process.env.MISTRAL_API_KEY ?? '',
      });
      return provider(modelName);
    }

    default: {
      if (config?.baseURL) {
        const provider = createOpenAI({
          baseURL: config.baseURL,
          apiKey: config.apiKey ?? '',
        });
        return provider(modelName);
      }
      throw new Error(
        `Unknown Vercel AI sub-provider: "${providerName}". ` +
          'Supported: ollama, openai, groq, mistral. ' +
          `For other providers, add a baseURL in config: assistants.vercel-ai.providers.${providerName}.baseURL`
      );
    }
  }
}

/**
 * Load MCP tools from config and create Vercel AI SDK tool instances.
 * Returns tools and cleanup function.
 */
async function loadMcpTools(
  mcpServers: NonNullable<AssistantRequestOptions['mcpServers']>
): Promise<{ tools: ToolSet; cleanup: () => Promise<void> }> {
  const { createMCPClient } = await import('@ai-sdk/mcp');

  const clients: { close: () => Promise<void> }[] = [];
  const allTools: ToolSet = {};

  for (const [serverName, serverConfig] of Object.entries(mcpServers)) {
    try {
      let client: Awaited<ReturnType<typeof createMCPClient>>;

      if (serverConfig.type === 'sse' || serverConfig.type === 'http') {
        client = await createMCPClient({
          transport: {
            type: serverConfig.type,
            url: serverConfig.url,
            ...(serverConfig.headers ? { headers: serverConfig.headers } : {}),
          },
        });
      } else if ('command' in serverConfig && typeof serverConfig.command === 'string') {
        // stdio transport — use StdioMCPTransport from the mcp-stdio subpath
        // eslint-disable-next-line @typescript-eslint/naming-convention -- SDK export uses PascalCase with underscore prefix
        const { Experimental_StdioMCPTransport: StdioTransport } =
          await import('@ai-sdk/mcp/mcp-stdio');
        const transport = new StdioTransport({
          command: serverConfig.command,
          args: serverConfig.args,
          env: serverConfig.env,
        });
        client = await createMCPClient({ transport });
      } else {
        getLog().warn({ serverName }, 'mcp.unsupported_transport_type');
        continue;
      }

      clients.push(client);
      const serverTools = await client.tools();
      Object.assign(allTools, serverTools);

      getLog().debug(
        { serverName, toolCount: Object.keys(serverTools).length },
        'mcp.tools_loaded'
      );
    } catch (err) {
      getLog().error({ serverName, error: (err as Error).message }, 'mcp.client_create_failed');
    }
  }

  return {
    tools: allTools,
    cleanup: async (): Promise<void> => {
      for (const client of clients) {
        try {
          await client.close();
        } catch {
          // Best-effort cleanup
        }
      }
    },
  };
}

export class VercelAiClient implements IAssistantClient {
  async *sendQuery(
    prompt: string,
    cwd: string,
    resumeSessionId?: string,
    options?: AssistantRequestOptions
  ): AsyncGenerator<MessageChunk> {
    if (resumeSessionId && !options?.conversationHistory?.length) {
      getLog().debug({ resumeSessionId }, 'session_resume_ignored (Vercel AI SDK is stateless)');
    } else if (resumeSessionId && options?.conversationHistory?.length) {
      getLog().debug(
        { resumeSessionId, historyLength: options.conversationHistory.length },
        'session_resume_via_history'
      );
    }

    const model = options?.model;
    if (!model) {
      throw new Error(
        'Vercel AI client requires a model in "provider/model" format (e.g., "ollama/llama3"). ' +
          'Set it in the workflow node, workflow-level, or assistants.vercel-ai.model in config.'
      );
    }

    const { providerName, modelName } = parseModelString(model);

    // Load config for provider overrides
    const config = await loadConfig(cwd);
    const vercelConfig = config.assistants['vercel-ai'] as VercelAiAssistantDefaults | undefined;
    const maxSteps = vercelConfig?.maxSteps ?? 10;

    const languageModel = resolveModel(providerName, modelName, vercelConfig?.providers);

    // Set up MCP tools if configured
    let tools: ToolSet | undefined;
    let mcpCleanup: (() => Promise<void>) | undefined;

    if (options?.mcpServers && Object.keys(options.mcpServers).length > 0) {
      const result = await loadMcpTools(options.mcpServers);
      tools = result.tools;
      mcpCleanup = result.cleanup;
    }

    // Build messages array with conversation history for session continuity
    const historyMessages = (options?.conversationHistory ?? []).map(msg => {
      if (msg.role === 'user' || msg.role === 'assistant') {
        return { role: msg.role, content: msg.content };
      }
      if (msg.role === 'tool_call') {
        return {
          role: 'assistant' as const,
          content: `[Tool: ${msg.toolName}] ${msg.content}`,
        };
      }
      // tool_result
      return {
        role: 'user' as const,
        content: `[Tool result: ${msg.toolName}] ${msg.content}`,
      };
    });
    const messages = [...historyMessages, { role: 'user' as const, content: prompt }];

    // Add system prompt if provided
    const systemPrompt = options?.systemPrompt;

    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (attempt > 0) {
        const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt - 1);
        getLog().info({ attempt, delay, error: lastError?.message }, 'vercel_ai.retry_attempt');
        await new Promise(resolve => setTimeout(resolve, delay));
      }

      try {
        const result = streamText({
          model: languageModel,
          messages,
          ...(systemPrompt ? { system: systemPrompt } : {}),
          ...(tools && Object.keys(tools).length > 0 ? { tools, maxSteps } : {}),
          ...(options?.abortSignal ? { abortSignal: options.abortSignal } : {}),
        });

        for await (const part of result.fullStream) {
          switch (part.type) {
            case 'text-delta':
              yield { type: 'assistant', content: part.text };
              break;

            case 'tool-call':
              yield {
                type: 'tool',
                toolName: part.toolName,
                toolInput: part.input as Record<string, unknown>,
              };
              break;

            case 'tool-result':
              yield {
                type: 'tool_result',
                toolName: part.toolName,
                toolOutput:
                  typeof part.output === 'string' ? part.output : JSON.stringify(part.output),
              };
              break;

            case 'error':
              getLog().error({ error: part.error }, 'vercel_ai.stream_error');
              break;
          }
        }

        // Get usage after stream completes
        const usage = await result.usage;
        const tokens: TokenUsage | undefined = usage
          ? {
              input: usage.inputTokens ?? 0,
              output: usage.outputTokens ?? 0,
              total: (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0),
            }
          : undefined;

        // Get structured output if output_format was requested
        const text = await result.text;
        let structuredOutput: unknown;
        if (options?.outputFormat && text) {
          try {
            structuredOutput = JSON.parse(text);
          } catch {
            // Not valid JSON — that's fine, return as text
          }
        }

        yield {
          type: 'result',
          tokens,
          ...(structuredOutput !== undefined ? { structuredOutput } : {}),
        };

        // Success — break retry loop
        if (mcpCleanup) await mcpCleanup();
        return;
      } catch (err) {
        lastError = err as Error;

        if (options?.abortSignal?.aborted) {
          getLog().info('vercel_ai.aborted');
          if (mcpCleanup) await mcpCleanup();
          yield {
            type: 'result',
            isError: true,
            errorSubtype: 'aborted',
          };
          return;
        }

        if (isRateLimitError(lastError.message)) {
          yield {
            type: 'rate_limit',
            rateLimitInfo: { message: lastError.message },
          };
          continue;
        }

        // Non-retryable error on last attempt
        if (attempt === MAX_RETRIES) {
          getLog().error(
            { error: lastError.message, attempts: attempt + 1 },
            'vercel_ai.query_failed'
          );
          if (mcpCleanup) await mcpCleanup();
          yield {
            type: 'result',
            isError: true,
            errorSubtype: 'provider_error',
          };
          yield {
            type: 'system',
            content: `Vercel AI error (${providerName}/${modelName}): ${lastError.message}`,
          };
          return;
        }
      }
    }

    // Should not reach here, but safety net
    if (mcpCleanup) await mcpCleanup();
  }

  getType(): string {
    return 'vercel-ai';
  }
}
