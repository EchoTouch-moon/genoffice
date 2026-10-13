/**
 * Stream-level integration for the output budget: a turn the model cuts off
 * (finish_reason=length) is recorded, and the next turn on the same
 * endpoint+model asks for a bigger budget. A document-sized request also asks
 * for more than the caller's cap, and a declared model ceiling still wins.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { streamForProvider } from '../src/stream'
import { resetOutputBudgets } from '../src/output-budget'
import { resetOutputCaps } from '../src/output-cap'
import { jsonResponse, okResponse, sseStream } from './test-utils'
import type { AiProviderConfig } from '../src/types'

afterEach(() => {
  vi.unstubAllGlobals()
  resetOutputBudgets()
  resetOutputCaps()
})

function collector() {
  const stopReasons: string[] = []
  return {
    stopReasons,
    cb: {
      signal: new AbortController().signal,
      onDelta: () => {},
      onToolCall: () => {},
      onStopReason: (r: string) => stopReasons.push(r),
    },
  }
}

const BASE: AiProviderConfig = {
  apiKey: 'k',
  model: 'local-model',
  baseUrl: 'https://gw.example.com/v1',
  // the budget policy is opt-in: a profile is what enrolls the endpoint
  profile: { api: 'openai-compatible' },
}

/** An OpenAI-compatible stream that reports finish_reason=length. */
function truncatedStream() {
  return okResponse(
    sseStream([
      'data: ' + JSON.stringify({ choices: [{ delta: { content: 'partial' } }] }),
      'data: ' + JSON.stringify({ choices: [{ delta: {}, finish_reason: 'length' }] }),
      'data: [DONE]',
    ]),
  )
}

function bodyOf(fetchMock: ReturnType<typeof vi.fn>, call = 0): Record<string, unknown> {
  return JSON.parse(fetchMock.mock.calls[call]![1].body as string) as Record<string, unknown>
}

describe('streamForProvider: output budget', () => {
  it('a truncated turn raises the budget for the next one', async () => {
    // a fresh response per call: one Response object's body is consumed once
    const fetchMock = vi.fn().mockImplementation(async () => truncatedStream())
    vi.stubGlobal('fetch', fetchMock)
    const { cb } = collector()

    // first turn: cut off at the caller's cap
    await streamForProvider('custom', BASE, 'sys', [], [], 32768, cb).catch(() => {})
    expect(bodyOf(fetchMock).max_tokens).toBe(32768)

    // second turn: still the cap (one truncation is not a pattern)
    await streamForProvider('custom', BASE, 'sys', [], [], 32768, cb).catch(() => {})
    expect(bodyOf(fetchMock, 1).max_tokens).toBe(32768)

    // third turn: the pattern is established, the budget doubles
    await streamForProvider('custom', BASE, 'sys', [], [], 32768, cb).catch(() => {})
    expect(bodyOf(fetchMock, 2).max_tokens).toBe(65536)
  })

  it('a provider without a profile keeps the caller budget exactly', async () => {
    // the policy is opt-in: a built-in vendor API must not have its spend
    // raised by a document-size heuristic or a truncation
    const fetchMock = vi.fn().mockImplementation(async () => truncatedStream())
    vi.stubGlobal('fetch', fetchMock)
    const { cb } = collector()
    const doc = 'x'.repeat(160_000)
    for (let i = 0; i < 4; i += 1) {
      await streamForProvider(
        'openai',
        { apiKey: 'k', model: 'gpt-5.6' },
        doc,
        [],
        [],
        8192,
        cb,
      ).catch(() => {})
    }
    // every turn asked for the caller's number, truncated or not — under the
    // renamed field this provider uses
    for (let i = 0; i < 4; i += 1) {
      expect(bodyOf(fetchMock, i).max_completion_tokens).toBe(8192)
    }
  })

  it('a complete turn resets the counter so a one-off does not raise anything', async () => {
    const done = () =>
      okResponse(
        sseStream([
          'data: ' + JSON.stringify({ choices: [{ delta: { content: 'hi' } }] }),
          'data: ' + JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] }),
          'data: [DONE]',
        ]),
      )
    const fetchMock = vi.fn().mockImplementation(async () => done())
    vi.stubGlobal('fetch', fetchMock)
    const { cb } = collector()

    await streamForProvider('custom', BASE, 'sys', [], [], 32768, cb).catch(() => {})
    await streamForProvider('custom', BASE, 'sys', [], [], 32768, cb).catch(() => {})
    await streamForProvider('custom', BASE, 'sys', [], [], 32768, cb).catch(() => {})
    expect(bodyOf(fetchMock, 0).max_tokens).toBe(32768)
    expect(bodyOf(fetchMock, 1).max_tokens).toBe(32768)
    expect(bodyOf(fetchMock, 2).max_tokens).toBe(32768)
  })

  it('a document-sized request asks for more than the caller cap', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse(
          sseStream([
            'data: ' + JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] }),
            'data: [DONE]',
          ]),
        ),
      )
    vi.stubGlobal('fetch', fetchMock)
    const { cb } = collector()
    // 40k tokens of document context, an 8k caller cap
    const doc = 'x'.repeat(160_000)
    await streamForProvider('custom', BASE, doc, [], [], 8192, cb).catch(() => {})
    expect(bodyOf(fetchMock).max_tokens).toBe(20_000)
  })

  it('a declared model ceiling beats the document-size raise', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse(
          sseStream([
            'data: ' + JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] }),
            'data: [DONE]',
          ]),
        ),
      )
    vi.stubGlobal('fetch', fetchMock)
    const { cb } = collector()
    const doc = 'x'.repeat(160_000)
    await streamForProvider(
      'custom',
      { ...BASE, profile: { modelProfiles: [{ id: 'local-model', maxTokens: 16384 }] } },
      doc,
      [],
      [],
      8192,
      cb,
    ).catch(() => {})
    expect(bodyOf(fetchMock).max_tokens).toBe(16384)
  })

  it('the caller still learns a real ceiling from a 400', async () => {
    let calls = 0
    const fetchMock = vi.fn().mockImplementation(async () => {
      calls += 1
      if (calls === 1) {
        return jsonResponse({ error: { message: 'max_tokens must be at most 16384' } }, 400)
      }
      return okResponse(sseStream(['data: [DONE]']))
    })
    vi.stubGlobal('fetch', fetchMock)
    const { cb } = collector()
    await streamForProvider('custom', BASE, 'sys', [], [], 32768, cb).catch(() => {})
    // first attempt at 32768, retried at the parsed ceiling
    expect(bodyOf(fetchMock, 0).max_tokens).toBe(32768)
    expect(bodyOf(fetchMock, 1).max_tokens).toBe(16384)
  })
})
