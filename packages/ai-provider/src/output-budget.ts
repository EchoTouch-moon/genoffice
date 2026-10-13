/**
 * The per-turn output budget.
 *
 * Three signals decide how many tokens a turn may spend, in this order of
 * authority:
 *
 *   1. the model's own ceiling — either the user declared it in the endpoint
 *      profile, or a 400 rejection named it (see ./output-cap.ts);
 *   2. the work in front of the model — regenerating a whole document costs
 *      roughly half its size in output, so a large context raises the budget
 *      above the user's cap instead of truncating mid-document;
 *   3. the user's cap — the floor everything else may only raise, never lower.
 *
 * On top of those, a turn that comes back cut off (`finish_reason=length`)
 * counts against its endpoint+model: repeated truncation raises the budget for
 * the rest of the session, because a user who keeps hitting the ceiling is
 * telling us the cap is wrong for their work. A complete turn resets the
 * counter but keeps the raised budget, so one large document does not
 * re-truncate on the next edit.
 *
 * Everything here is per-process state (like the learned ceilings): a restart
 * starts from the settings file again, which is the honest default.
 *
 * The policy is opt-in per endpoint (see stream.ts): a built-in provider never
 * consults this module, so declaring a profile is what raises a budget.
 */
import type { AgentMessage } from '@genoffice/agent-core'
import { outputCeilingFor } from './output-cap'
import { MAX_MAX_OUTPUT_TOKENS } from './providers'

/** rough characters per token for a budget estimate; the request path has no tokenizer */
const CHARS_PER_TOKEN = 4
/**
 * What one attached image costs in the size estimate. Its base64 is thousands
 * of characters that are not text: charging them at 4 chars/token made a single
 * screenshot look like a 250k-token document and pushed the budget to the
 * ceiling. Providers bill images by area, and a nominal tile is the honest
 * stand-in here.
 */
const IMAGE_TOKENS = 1024
/** regenerating a document costs about half of it in output */
const DOCUMENT_OUTPUT_RATIO = 0.5
/** a context below this is a question, not a document: no size budget */
const DOCUMENT_SIZE_FLOOR_TOKENS = 4096
/** consecutive truncations before the budget is raised */
const ESCALATION_TRUNCATIONS = 2
/** how much one raise multiplies the current budget by */
const ESCALATION_FACTOR = 2

/** per endpoint+model: consecutive turns that ended cut off */
const truncations = new Map<string, number>()
/** per endpoint+model: the budget a raise settled on */
const escalated = new Map<string, number>()

/** Rough input size in tokens: every provider bills characters, none exposes a counter here. */
export function estimateInputTokens(system: string, messages: AgentMessage[]): number {
  let chars = system.length
  let imageTokens = 0
  for (const m of messages) {
    // the union puts text/images on user turns, tool calls on assistant turns,
    // and results on tool turns — count whatever the role actually carries
    if (m.role === 'tool') {
      for (const r of m.results) chars += r.name.length + r.output.length
      continue
    }
    chars += m.text.length
    if (m.role === 'assistant') {
      for (const c of m.toolCalls ?? []) chars += JSON.stringify(c.input).length
    } else {
      for (const _img of m.images ?? []) imageTokens += IMAGE_TOKENS
    }
  }
  return Math.ceil(chars / CHARS_PER_TOKEN) + imageTokens
}

/**
 * The budget a document-sized context deserves. Below the floor this is 0, so
 * a question ("what does this sheet do?") keeps the user's cap untouched.
 */
function documentSizeBudget(system: string, messages: AgentMessage[]): number {
  const tokens = estimateInputTokens(system, messages)
  if (tokens < DOCUMENT_SIZE_FLOOR_TOKENS) return 0
  return Math.round(tokens * DOCUMENT_OUTPUT_RATIO)
}

export interface OutputBudgetRequest {
  baseUrl: string
  model: string
  /** the caller's budget: the user's cap, already clamped by a declared model maxTokens */
  requested: number
  /**
   * A hard ceiling the model itself enforces (the user's declared maxTokens):
   * unlike the learned ceiling this one is a fact about the model, so no
   * heuristic may raise the budget past it.
   */
  ceiling?: number | undefined
  system: string
  messages: AgentMessage[]
}

/**
 * The budget one turn should ask for: the caller's, raised by the document
 * size and by any learned escalation, then clamped by every ceiling we know.
 *
 * The caller's number is honored as given — it is the app's per-turn decision
 * (a classification turn may want 16 tokens), and the settings bounds already
 * clamped it when it was read. Only the raises are capped, at the global
 * maximum, because an escalation or a size estimate must never invent a budget
 * the app never asked for beyond that.
 */
export function resolveOutputBudget(req: OutputBudgetRequest): number {
  const learned = outputCeilingFor(req.baseUrl, req.model)
  const size = documentSizeBudget(req.system, req.messages)
  const raised = escalated.get(`${req.baseUrl} ${req.model}`)
  let budget = Math.max(req.requested, size, raised ?? 0)
  budget = Math.min(budget, MAX_MAX_OUTPUT_TOKENS)
  const ceilings = [req.ceiling, learned].filter((c): c is number => c !== undefined)
  for (const ceiling of ceilings) budget = Math.min(budget, ceiling)
  return budget
}

/**
 * Record how a turn ended. A cut-off turn counts toward a raise; anything
 * else (a complete answer, a tool turn, an error the caller already handled)
 * clears the counter but leaves a raised budget in place.
 */
export function recordTurnOutcome(
  baseUrl: string,
  model: string,
  stopReason: string | undefined,
  budget: number,
): void {
  const key = `${baseUrl} ${model}`
  if (stopReason !== 'max_tokens') {
    truncations.delete(key)
    return
  }
  const seen = (truncations.get(key) ?? 0) + 1
  truncations.set(key, seen)
  if (seen < ESCALATION_TRUNCATIONS) return
  const ceiling = outputCeilingFor(baseUrl, model)
  const current = escalated.get(key) ?? budget
  const raised = Math.min(current * ESCALATION_FACTOR, MAX_MAX_OUTPUT_TOKENS)
  escalated.set(key, ceiling === undefined ? raised : Math.min(raised, ceiling))
}

/** The budget a raise settled on for one endpoint+model, when there is one. */
export function escalatedBudget(baseUrl: string, model: string): number | undefined {
  return escalated.get(`${baseUrl} ${model}`)
}

/** Test seam: forget every learned budget. */
export function resetOutputBudgets(): void {
  truncations.clear()
  escalated.clear()
}
