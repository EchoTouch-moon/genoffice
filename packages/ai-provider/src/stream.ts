import type { AgentMessage, AgentToolDef } from '@genoffice/agent-core'
import { resolveOutputBudget, recordTurnOutcome } from './output-budget'
import { withOutputCapFallback } from './output-cap'
import { streamAnthropic } from './protocols/anthropic'
import { streamGemini } from './protocols/gemini'
import { streamOpenAiCompatible } from './protocols/openai-compatible'
import { streamCodexAppServer } from './codex-app-server'
import type { StreamCallbacks } from './protocols/shared'
import { getProviderAdapter, type AiProtocol } from './registry'
import type { AiProviderConfig, AiProviderId } from './types'

export { streamAnthropic } from './protocols/anthropic'
export { streamGemini } from './protocols/gemini'
export { streamOpenAiCompatible } from './protocols/openai-compatible'
export { AiCreditsError, sseLines } from './protocols/shared'
export type { StreamCallbacks } from './protocols/shared'

/** route a streaming, tool-calling-capable turn by provider id */
export async function streamForProvider(
  provider: AiProviderId,
  config: AiProviderConfig,
  system: string,
  messages: AgentMessage[],
  tools: AgentToolDef[],
  maxTokens: number,
  cb: StreamCallbacks,
): Promise<void> {
  const endpoint = getProviderAdapter(provider).resolveEndpoint(config)
  const { baseUrl } = endpoint
  if (endpoint.model) config = { ...config, model: endpoint.model }
  if (endpoint.protocol === 'codex-app-server') {
    return streamCodexAppServer(config, system, messages, tools, maxTokens, cb)
  }
  // A model the user declared a ceiling for outranks the global cap; the
  // endpoint's own 400 would otherwise cost a billed round trip. It stays a
  // hard ceiling: no heuristic below may raise the budget past what the model
  // itself accepts.
  const declared =
    endpoint.maxTokens !== undefined ? Math.min(maxTokens, endpoint.maxTokens) : maxTokens
  // The budget policy is opt-in per endpoint: only a user-hosted endpoint that
  // carries a profile takes part, so every built-in provider keeps the caller's
  // number exactly as it was. Raising spend for a vendor API is a decision the
  // user makes by declaring a profile, not one this layer makes for them.
  const optedIn = config.profile !== undefined
  const budget = optedIn
    ? resolveOutputBudget({
        baseUrl,
        model: config.model,
        requested: declared,
        ...(endpoint.maxTokens !== undefined ? { ceiling: endpoint.maxTokens } : {}),
        system,
        messages,
      })
    : declared
  // Observe the turn's ending so the next one can ask for a bigger budget.
  const observed: StreamCallbacks = optedIn
    ? {
        ...cb,
        onStopReason: (reason) => {
          recordTurnOutcome(baseUrl, config.model, reason, budget)
          cb.onStopReason?.(reason)
        },
      }
    : cb
  const protocol: Exclude<AiProtocol, 'codex-app-server'> = endpoint.protocol
  return withOutputCapFallback(baseUrl, config.model, budget, (tokens) => {
    switch (protocol) {
      case 'anthropic':
        return streamAnthropic(config, system, messages, tools, tokens, observed, baseUrl, {
          headers: endpoint.headers,
          idleTimeoutMs: endpoint.idleTimeoutMs,
        })
      case 'gemini':
        return streamGemini(config, system, messages, tools, tokens, observed, baseUrl, {
          omitTemperature: endpoint.omitTemperature,
          headers: endpoint.headers,
          idleTimeoutMs: endpoint.idleTimeoutMs,
        })
      case 'openai-compatible':
        return streamOpenAiCompatible(baseUrl, config, system, messages, tools, tokens, observed, {
          omitTemperature: endpoint.omitTemperature,
          useMaxCompletionTokens: endpoint.useMaxCompletionTokens,
          bodyExtras: endpoint.bodyExtras,
          echoReasoning: endpoint.echoReasoning,
          headers: endpoint.headers,
          idleTimeoutMs: endpoint.idleTimeoutMs,
        })
    }
  })
}
