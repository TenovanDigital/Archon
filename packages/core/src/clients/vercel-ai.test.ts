import { describe, test, expect } from 'bun:test';
import { VercelAiClient } from './vercel-ai';

describe('VercelAiClient', () => {
  test('getType returns vercel-ai', () => {
    const client = new VercelAiClient();
    expect(client.getType()).toBe('vercel-ai');
  });

  test('sendQuery throws without model', async () => {
    const client = new VercelAiClient();
    const gen = client.sendQuery('test prompt', '/tmp');

    await expect(gen.next()).rejects.toThrow(
      'Vercel AI client requires a model in "provider/model" format'
    );
  });

  test('sendQuery throws with invalid model format (no slash)', async () => {
    const client = new VercelAiClient();
    const gen = client.sendQuery('test prompt', '/tmp', undefined, {
      model: 'llama3',
    });

    await expect(gen.next()).rejects.toThrow('Invalid Vercel AI model format');
  });

  test('sendQuery throws with invalid model format (leading slash)', async () => {
    const client = new VercelAiClient();
    const gen = client.sendQuery('test prompt', '/tmp', undefined, {
      model: '/llama3',
    });

    await expect(gen.next()).rejects.toThrow('Invalid Vercel AI model format');
  });

  test('sendQuery throws with invalid model format (trailing slash)', async () => {
    const client = new VercelAiClient();
    const gen = client.sendQuery('test prompt', '/tmp', undefined, {
      model: 'ollama/',
    });

    await expect(gen.next()).rejects.toThrow('Invalid Vercel AI model format');
  });
});
