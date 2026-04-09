import { describe, it, expect } from 'bun:test';
import { isClaudeModel, isVercelAiModel, isModelCompatible } from './model-validation';

describe('model-validation', () => {
  describe('isClaudeModel', () => {
    it('should recognize Claude aliases', () => {
      expect(isClaudeModel('sonnet')).toBe(true);
      expect(isClaudeModel('opus')).toBe(true);
      expect(isClaudeModel('haiku')).toBe(true);
      expect(isClaudeModel('inherit')).toBe(true);
    });

    it('should recognize claude- prefixed models', () => {
      expect(isClaudeModel('claude-sonnet-4-5-20250929')).toBe(true);
      expect(isClaudeModel('claude-opus-4-6')).toBe(true);
      expect(isClaudeModel('claude-3-5-sonnet-20241022')).toBe(true);
    });

    it('should reject non-Claude models', () => {
      expect(isClaudeModel('gpt-5.3-codex')).toBe(false);
      expect(isClaudeModel('gpt-5.2-codex')).toBe(false);
      expect(isClaudeModel('gpt-4')).toBe(false);
      expect(isClaudeModel('o1-mini')).toBe(false);
    });

    it('should reject empty string', () => {
      expect(isClaudeModel('')).toBe(false);
    });
  });

  describe('isVercelAiModel', () => {
    it('should recognize provider/model format', () => {
      expect(isVercelAiModel('ollama/llama3')).toBe(true);
      expect(isVercelAiModel('openai/gpt-4o')).toBe(true);
      expect(isVercelAiModel('groq/mixtral-8x7b')).toBe(true);
      expect(isVercelAiModel('mistral/mistral-large')).toBe(true);
    });

    it('should reject models without slash', () => {
      expect(isVercelAiModel('sonnet')).toBe(false);
      expect(isVercelAiModel('gpt-4')).toBe(false);
      expect(isVercelAiModel('llama3')).toBe(false);
    });

    it('should reject claude- prefixed models even with slash', () => {
      expect(isVercelAiModel('claude-sonnet-4-5/latest')).toBe(false);
    });

    it('should reject empty string', () => {
      expect(isVercelAiModel('')).toBe(false);
    });
  });

  describe('isModelCompatible', () => {
    it('should accept any model when model is undefined', () => {
      expect(isModelCompatible('claude')).toBe(true);
      expect(isModelCompatible('codex')).toBe(true);
    });

    it('should accept Claude models with claude provider', () => {
      expect(isModelCompatible('claude', 'sonnet')).toBe(true);
      expect(isModelCompatible('claude', 'opus')).toBe(true);
      expect(isModelCompatible('claude', 'haiku')).toBe(true);
      expect(isModelCompatible('claude', 'inherit')).toBe(true);
      expect(isModelCompatible('claude', 'claude-opus-4-6')).toBe(true);
    });

    it('should reject non-Claude models with claude provider', () => {
      expect(isModelCompatible('claude', 'gpt-5.3-codex')).toBe(false);
      expect(isModelCompatible('claude', 'gpt-4')).toBe(false);
    });

    it('should accept Codex/OpenAI models with codex provider', () => {
      expect(isModelCompatible('codex', 'gpt-5.3-codex')).toBe(true);
      expect(isModelCompatible('codex', 'gpt-5.2-codex')).toBe(true);
      expect(isModelCompatible('codex', 'gpt-4')).toBe(true);
      expect(isModelCompatible('codex', 'o1-mini')).toBe(true);
    });

    it('should reject Claude models with codex provider', () => {
      expect(isModelCompatible('codex', 'sonnet')).toBe(false);
      expect(isModelCompatible('codex', 'opus')).toBe(false);
      expect(isModelCompatible('codex', 'claude-opus-4-6')).toBe(false);
    });

    it('should handle empty string model', () => {
      // Empty string is falsy, so treated as "no model specified"
      expect(isModelCompatible('claude', '')).toBe(true);
      expect(isModelCompatible('codex', '')).toBe(true);
    });

    it('should accept vercel-ai models with vercel-ai provider', () => {
      expect(isModelCompatible('vercel-ai', 'ollama/llama3')).toBe(true);
      expect(isModelCompatible('vercel-ai', 'openai/gpt-4o')).toBe(true);
      expect(isModelCompatible('vercel-ai', 'groq/mixtral-8x7b')).toBe(true);
    });

    it('should reject non-slash models with vercel-ai provider', () => {
      expect(isModelCompatible('vercel-ai', 'sonnet')).toBe(false);
      expect(isModelCompatible('vercel-ai', 'gpt-4')).toBe(false);
    });

    it('should accept any model when vercel-ai has no model', () => {
      expect(isModelCompatible('vercel-ai')).toBe(true);
    });
  });
});
