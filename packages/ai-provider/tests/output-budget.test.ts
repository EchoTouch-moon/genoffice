import { afterEach, describe, expect, it } from 'vitest'
import type { AgentMessage } from '@genoffice/agent-core'
import {
  estimateInputTokens,
  escalatedBudget,
  recordTurnOutcome,
  resetOutputBudgets,
  resolveOutputBudget,
  type OutputBudgetRequest,
} from '../src/output-budget'
import { resetOutputCaps, withOutputCapFallback } from '../src/output-cap'
import { MAX_MAX_OUTPUT_TOKENS } from '../src/providers'

const BASE = 'https://gw.example.com/v1'
const MODEL = 'm'

/** a request of roughly `tokens` input tokens (4 chars each) */
function sized(tokens: number): Pick<OutputBudgetRequest, 'system' | 'messages'> {
  return { system: 'x'.repeat(tokens * 4), messages: [] }
}

const req = (over: Partial<OutputBudgetRequest> = {}): OutputBudgetRequest => ({
  baseUrl: BASE,
  model: MODEL,
  requested: 32768,
  system: '',
  messages: [],
  ...over,
})

/** Seed the learned ceiling the way a real 400 does: one failed turn, one retry. */
async function learnCeiling(ceiling: number): Promise<void> {
  await withOutputCapFallback(BASE, MODEL, 32768, (tokens) => {
    if (tokens > ceiling) {
      throw new Error(`HTTP 400: max_tokens must be at most ${ceiling}`)
    }
    return Promise.resolve('ok')
  })
}

afterEach(() => {
  resetOutputBudgets()
  resetOutputCaps()
})

describe('estimateInputTokens', () => {
  it('counts the system prompt, messages, tool results and tool inputs', () => {
    const messages: AgentMessage[] = [
      { role: 'user', text: 'a'.repeat(400) },
      {
        role: 'assistant',
        text: 'b'.repeat(400),
        toolCalls: [{ id: 't', name: 'f', input: { k: 'v' } }],
      },
      { role: 'tool', results: [{ id: 't', name: 'f', output: 'c'.repeat(400), isError: false }] },
    ]
    // 1200 chars of prose over 4 chars per token, plus the tool input json
    expect(estimateInputTokens('', messages)).toBeGreaterThanOrEqual(300)
    expect(estimateInputTokens('d'.repeat(400), [])).toBe(100)
  })

  it('counts an image at a nominal cost, not its base64 length', () => {
    // a screenshot is ~1MB of base64 that is not text: charging it at 4
    // chars/token made one attachment look like a 250k-token document
    const withImage = estimateInputTokens('', [
      { role: 'user', text: 'hi', images: [{ mime: 'image/png', base64: 'A'.repeat(1_000_000) }] },
    ])
    expect(withImage).toBe(1024 + 1) // one nominal tile plus the two chars of text
  })
})

describe('resolveOutputBudget', () => {
  it('a question-sized request keeps the caller cap', () => {
    expect(resolveOutputBudget(req())).toBe(32768)
  })

  it('a document-sized request raises the budget to about half its size', () => {
    // 40k tokens of context → 20k output budget, above the 8k caller cap
    expect(resolveOutputBudget(req({ requested: 8192, ...sized(40_000) }))).toBe(20_000)
  })

  it('never lowers the caller cap, however small the request', () => {
    expect(resolveOutputBudget(req({ requested: 131072, ...sized(100) }))).toBe(131072)
  })

  it('clamps to the global maximum', () => {
    expect(resolveOutputBudget(req({ requested: 8192, ...sized(2_000_000) }))).toBe(
      MAX_MAX_OUTPUT_TOKENS,
    )
  })

  it('a declared model maxTokens is a hard ceiling no heuristic may pass', () => {
    expect(resolveOutputBudget(req({ requested: 8192, ceiling: 16384, ...sized(200_000) }))).toBe(
      16384,
    )
  })

  it('a learned ceiling from a 400 rejection also clamps', async () => {
    await learnCeiling(16384)
    expect(resolveOutputBudget(req({ requested: 32768 }))).toBe(16384)
  })

  it('honors a small caller budget as given', () => {
    // the settings bounds clamped this when it was read; a per-turn floor here
    // would rewrite a deliberate 16-token classification request
    expect(resolveOutputBudget(req({ requested: 16 }))).toBe(16)
  })
})

describe('recordTurnOutcome', () => {
  it('one truncation changes nothing; the second raises the budget', () => {
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    expect(escalatedBudget(BASE, MODEL)).toBeUndefined()
    expect(resolveOutputBudget(req())).toBe(32768)

    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    expect(escalatedBudget(BASE, MODEL)).toBe(65536)
    expect(resolveOutputBudget(req())).toBe(65536)
  })

  it('a complete turn resets the counter but keeps the raised budget', () => {
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    recordTurnOutcome(BASE, MODEL, 'stop', 65536)
    expect(escalatedBudget(BASE, MODEL)).toBe(65536)
    // the next truncation counts from one again, so it takes two more to raise
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 65536)
    expect(escalatedBudget(BASE, MODEL)).toBe(65536)
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 65536)
    expect(escalatedBudget(BASE, MODEL)).toBe(MAX_MAX_OUTPUT_TOKENS)
  })

  it('raises from the last budget, not from the caller cap', () => {
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    expect(escalatedBudget(BASE, MODEL)).toBe(65536)
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 65536)
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 65536)
    expect(escalatedBudget(BASE, MODEL)).toBe(131072)
  })

  it('tracks endpoints and models separately', () => {
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    recordTurnOutcome(BASE, 'other', 'max_tokens', 32768)
    expect(escalatedBudget(BASE, MODEL)).toBeUndefined()
    expect(escalatedBudget(BASE, 'other')).toBeUndefined()
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    expect(escalatedBudget(BASE, MODEL)).toBe(65536)
    expect(escalatedBudget(BASE, 'other')).toBeUndefined()
  })

  it('never raises past a learned ceiling', async () => {
    await learnCeiling(16384)
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    recordTurnOutcome(BASE, MODEL, 'max_tokens', 32768)
    expect(escalatedBudget(BASE, MODEL)).toBe(16384)
    expect(resolveOutputBudget(req({ requested: 32768 }))).toBe(16384)
  })

  it('an error the caller already handled is not a truncation', () => {
    recordTurnOutcome(BASE, MODEL, undefined, 32768)
    recordTurnOutcome(BASE, MODEL, undefined, 32768)
    expect(escalatedBudget(BASE, MODEL)).toBeUndefined()
  })
})
