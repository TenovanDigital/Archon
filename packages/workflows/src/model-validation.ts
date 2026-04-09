export function isClaudeModel(model: string): boolean {
  return (
    model === 'sonnet' ||
    model === 'opus' ||
    model === 'haiku' ||
    model === 'inherit' ||
    model.startsWith('claude-')
  );
}

/** Returns true if model uses Vercel AI SDK's `provider/model` format (e.g., `ollama/llama3`). */
export function isVercelAiModel(model: string): boolean {
  return model.includes('/') && !model.startsWith('claude-');
}

export function isModelCompatible(
  provider: 'claude' | 'codex' | 'vercel-ai',
  model?: string
): boolean {
  if (!model) return true;
  if (provider === 'claude') return isClaudeModel(model);
  if (provider === 'vercel-ai') return isVercelAiModel(model);
  // Codex: accept most models, but reject obvious Claude aliases/prefixes
  return !isClaudeModel(model);
}
