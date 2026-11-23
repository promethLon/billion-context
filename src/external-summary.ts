import type { CoreMessage, MessageContentStore, MessageRefMap } from "acp-kernel";

export interface SummaryWork {
    readonly content: string;
    readonly instructions: string;
    readonly reference?: string;
    readonly minSummaryChars?: number;
    readonly maxSummaryChars?: number;
}

export interface SummaryCandidate {
    summarize(work: Readonly<SummaryWork>, signal: AbortSignal): Promise<string>;
}

export interface SummaryBudget {
    readonly totalTimeoutMs: number;
    readonly targetTimeoutMs: number;
    readonly maxSummaryBytes: number;
}

/** #4833: provenance of an abort raised by the executor itself. A candidate
 *  receives ONLY the executor's per-attempt signal, so a bare `signal.aborted`
 *  cannot tell "our per-target budget expired" from "the whole batch hit its
 *  total deadline" from "the caller's request was torn down" — all three used
 *  to surface as client-abort in the HTTP layer's warn line. The marker rides
 *  `signal.reason` from the aborting controller down to the fetch site. */
type SummaryAbortSource = "target-timeout" | "batch-deadline" | "cancelled";

export class SummaryAbortError extends Error {
    readonly source: SummaryAbortSource;
    constructor(source: SummaryAbortSource) {
        super(`external summary aborted (${source})`);
        this.name = "SummaryAbortError";
        this.source = source;
    }
}

/** Read the executor's abort provenance off a signal. Undefined when the signal
 *  was never aborted or was aborted without our marker (foreign signals keep
 *  the legacy transport classification). */
export function summaryAbortSource(signal: AbortSignal): SummaryAbortSource | undefined {
    const reason: unknown = signal.reason;
    return reason instanceof SummaryAbortError ? reason.source : undefined;
}

/** #4833: human-readable WHY for a timed-out attempt — the source, the budget
 *  it hit, and how much wall-clock the range had burned. Pure numbers and fixed
 *  words; never echoes payload. */
function timeoutDetail(source: SummaryAbortSource, limits: SummaryBudget, elapsedMs: number): string {
    const budgetMs = source === "target-timeout" ? limits.targetTimeoutMs : limits.totalTimeoutMs;
    return `${source} ${budgetMs}ms (elapsed ${Math.round(elapsedMs)}ms)`;
}

/** [#2657] Per-batch pool-size request for the shared queue. The pool grows
 *  to the LARGEST request among in-flight work and returns to its base size
 *  once that work settles — one batch never permanently resizes the pool for
 *  everyone else, so chains with different values can coexist safely. */
interface SummaryPoolOptions {
    readonly concurrency?: number;
}

interface SummaryAttempt {
    readonly targetIndex: number;
    readonly outcome: "success" | "error" | "timeout" | "invalid_summary" | "cancelled";
    /** #3635: WHY an invalid_summary reply was rejected wholesale — before, a
     *  discarded over-cap summary left only this bare outcome as a trace. */
    readonly detail?: string;
}

type ExternalSummaryResult =
    | { status: "success"; summary: string; targetIndex: number; attempts: SummaryAttempt[] }
    | { status: "failed"; reason: "invalid_plan" | "exhausted"; attempts: SummaryAttempt[] }
    | { status: "cancelled" | "deadline"; attempts: SummaryAttempt[] };

export type ExternalSummaryBatchResult =
    | { status: "finished" | "cancelled" | "deadline"; results: ExternalSummaryResult[] }
    | { status: "failed"; reason: "invalid_plan"; results: [] };

type AttemptResult = { kind: "value"; value: string } | { kind: "error" } | { kind: "aborted" };
type Release = () => void;
interface Waiter {
    signal: AbortSignal;
    resolve: (release: Release | undefined) => void;
    abort: () => void;
}

function positiveInteger(value: number): boolean {
    return Number.isSafeInteger(value) && value > 0;
}

/** #3635: name WHY a candidate reply failed the length/shape gate. Mirrors the
 *  validator's check order below so the stated reason always matches the cause
 *  that rejected the reply. Pure numbers + fixed words — never echoes payload. */
function invalidSummaryDetail(value: unknown, limits: SummaryBudget, request: Readonly<SummaryWork>): string {
    if (typeof value !== "string" || !value.trim()) return "empty or non-string response";
    const bytes = Buffer.byteLength(value, "utf8");
    if (bytes > limits.maxSummaryBytes) return `response ${bytes} bytes exceeds byte cap ${limits.maxSummaryBytes}`;
    const chars = value.trim().length;
    const floor = request.minSummaryChars ?? 1;
    if (chars < floor) return `summary ${chars} chars below floor ${floor}`;
    return `summary ${chars} chars exceeds cap ${request.maxSummaryChars}`;
}

/** #1775/#3635: the length-budget clause every external-summarizer prompt
 *  carries — rendered up front with the EXACT cap the validator enforces, so
 *  the first attempt fits instead of paying full latency for a wholesale-
 *  discarded overshoot. Omitted below the minimum-summary floor (the kernel
 *  minimum-summary gate rejects such output anyway, so the instruction would
 *  be counterproductive) and when the cap is unbounded (<= 0 / Infinity). */
export function summaryLengthBudgetClause(budget: number | undefined, minChars: number): string {
    if (budget === undefined || !Number.isFinite(budget) || budget <= 0 || budget < minChars) return "";
    return `\n\nLENGTH BUDGET: Your ENTIRE response must be AT MOST ${budget} characters total — longer output is rejected by the pipeline. Be dense: compact bullets, no filler or repetition.`;
}

/** Canonical message-ref citations (`m00001`–`m99999`, `m100000+`, or zero-padded
 *  4-digit drift like `m0538`), bounded away from ASCII identifier chars so CJK
 *  and English prose (`见m01538附`, `(m01538)`, `m01636–m01672`) match while
 *  identifiers (`foo_m01538`, `m01538a`) and 4-digit non-zero-padded tokens
 *  (`m2026`, `m1024`) stay untouched. */
const MESSAGE_REF_CITATION_RE = /(?<![A-Za-z0-9_])m(0\d{3,6}|[1-9]\d{4,6})(?![A-Za-z0-9_])/g;
const SINGLE_MESSAGE_REF_RE = /^m(0\d{3,6}|[1-9]\d{4,6})$/;

/** Citation group linking adjacent refs such as `m00005/m00009` or
 *  `m01636–m01672` so paired citations (e.g. user approval + assistant plan)
 *  are evaluated together instead of falsely rejecting the partner ref. */
const MESSAGE_REF_GROUP_RE = /(?<![A-Za-z0-9_])m(?:0\d{3,6}|[1-9]\d{4,6})(?:\s*(?:\/|、|,|–|—|-|~|\.\.)\s*m(?:0\d{3,6}|[1-9]\d{4,6}))*(?![A-Za-z0-9_])/g;

const QUOTE_SPANS_RE = /「([^「」\n]{1,400})」|『([^『』\n]{1,400})』|“([^“”\n]{1,400})”|"([^"\n]{1,400})"/g;
const CODE_SPAN_RE = /```[\s\S]*?```|`[^`\n]+`/g;
const CLAUSE_BREAK_RE = /[\n。；;!?！？→⇒]|->|=>/;
const NORMALIZE_QUOTE_RE = /[\s\u3000,.;:!?，。；：！？、"'「」『』“”‘’()（）[\]【】—–\-`/*+＋]+/g;
const CJK_CHAR_RE = /[\u3400-\u9fff\uf900-\ufaff]/g;
const ELLIPSIS_SPLIT_RE = /\.{3,}|…+/;
const QUOTE_TRIGRAM_MIN_OVERLAP = 0.45;

const CHRONOLOGICAL_AFTER_RE = /^\s*(?:之后|以后|之前|以前|after\b|before\b)/i;
const CHRONOLOGICAL_BEFORE_RE = /(?:之后|以后|之前|以前|\bafter|\bbefore)\s*$/i;

const DIRECT_USER_BEFORE_ZH_RE = /(?:^|[^\u4e00-\u9fffA-Za-z0-9_])用户(?:此前|先前|已经?|明确|直接|正式|原话)?(?:授权|批准|同意|确认|要求|指示|指令|原话|发言|提问|回复|说|表示)\s*(?:[（([\s：:]*)?(?:原文见|原文|详见|参见|见|于|在)?\s*$/;
const DIRECT_USER_BEFORE_EN_RE = /\buser(?:'s)?\s+(?:previously\s+|explicitly\s+)?(?:said|asked|requested|authorized|authorised|approved|confirmed|instructed|replied|stated|message|prompt|authorization|authorisation|approval|quote)\s*(?:[（([\s：:]*)?(?:in|at|see|from|ref)?\s*$/i;
const USER_SUBJECT_BEFORE_ZH_RE = /(?:^|[^\u4e00-\u9fffA-Za-z0-9_])用户(?:此前|先前|先)?(?:在|于)?\s*$/;
const USER_ACTION_AFTER_ZH_RE = /^\s*(?:[（(]?[^）)\n]{0,6}[）)]?)?\s*(?:明确|直接|正式|已)?(?:授权|批准|同意|确认|要求|指示|指令|原话|说|表示)/;
const DIRECT_USER_AFTER_ZH_RE = /^\s*(?:[，,、]\s*)?用户(?:此前|先前|已经?|明确|直接|正式)?(?:授权|批准|同意|确认|要求|指示|指令|原话|说|表示)/;
const DIRECT_USER_AFTER_EN_RE = /^\s*(?:,\s*)?user(?:'s)?\s+(?:previously\s+|explicitly\s+)?(?:said|asked|requested|authorized|authorised|approved|confirmed|instructed|authorization|authorisation|approval)\b/i;

const MAX_QUOTE_AFTER_GAP = 20;
const MAX_QUOTE_BEFORE_GAP = 14;
const MAX_RANGE_EXPANSION = 50;

export const UNVERIFIED_REF_PLACEHOLDER = "[unverified-ref]";
const COMPACT_UNVERIFIED_REF_PLACEHOLDER = "[ref?]";
const SHORT_UNVERIFIED_REF_PLACEHOLDER = "[?]";

interface CitationMessageFact {
    readonly role?: string;
    readonly text?: string;
    normalizedText?: string;
}

/** Collect every numeric message-ref index appearing verbatim in authoritative
 *  summarizer inputs (selected content, covered message refs, reference context,
 *  sibling summaries). Excludes prompt examples and non-authoritative hints. */
export function collectGroundedMessageRefs(sources: readonly (string | undefined)[]): Set<number> {
    const refs = new Set<number>();
    for (const source of sources) {
        if (!source) continue;
        MESSAGE_REF_CITATION_RE.lastIndex = 0;
        let match: RegExpExecArray | null;
        while ((match = MESSAGE_REF_CITATION_RE.exec(source)) !== null) {
            refs.add(Number(match[1]));
        }
    }
    return refs;
}

/** Build an in-memory map from numeric message ref index to known original
 *  `role` and `text` using the current session messages and (when already loaded)
 *  the append-only content store. Pure in-memory lookup; never triggers disk I/O. */
export function buildCitationMessageIndex(
    messages: readonly CoreMessage[],
    byRaw: MessageRefMap["byRaw"],
    contentStore?: MessageContentStore,
): Map<number, CitationMessageFact> {
    const facts = new Map<number, CitationMessageFact>();
    for (const message of messages) {
        const ref = byRaw[message.id];
        if (!ref) continue;
        const m = SINGLE_MESSAGE_REF_RE.exec(ref);
        if (!m) continue;
        facts.set(Number(m[1]), { role: message.role, text: message.text ?? "" });
    }
    if (contentStore) {
        for (const [ref, entry] of Object.entries(contentStore.byRef)) {
            const m = SINGLE_MESSAGE_REF_RE.exec(ref);
            if (!m) continue;
            const index = Number(m[1]);
            if (facts.has(index)) continue;
            const text = contentStore.byHash[entry.hash];
            const role = entry.toolName !== undefined || entry.command !== undefined ? "tool" : undefined;
            if (text !== undefined || role !== undefined) {
                facts.set(index, { role, text });
            }
        }
    }
    return facts;
}

/** Highest numeric ref index allocated in a session ref map + known message facts. */
export function highestAllocatedMessageRef(
    byRef: MessageRefMap["byRef"],
    messageFacts?: ReadonlyMap<number, CitationMessageFact>,
): number {
    let highest = 0;
    for (const ref of Object.keys(byRef)) {
        const m = SINGLE_MESSAGE_REF_RE.exec(ref);
        if (m) highest = Math.max(highest, Number(m[1]));
    }
    if (messageFacts) {
        for (const index of messageFacts.keys()) {
            if (index > highest) highest = index;
        }
    }
    return highest;
}

function normalizeForQuoteMatch(text: string): string {
    return text.toLowerCase().replace(NORMALIZE_QUOTE_RE, "");
}

function normalizedFactText(fact: CitationMessageFact): string | undefined {
    if (typeof fact.text !== "string" || fact.text.length === 0) return undefined;
    if (fact.normalizedText === undefined) {
        fact.normalizedText = normalizeForQuoteMatch(fact.text);
    }
    return fact.normalizedText;
}

function extractVerifiableQuoteSegments(rawQuote: string): string[] | null {
    const segments = rawQuote
        .split(ELLIPSIS_SPLIT_RE)
        .map((part) => normalizeForQuoteMatch(part))
        .filter((part) => part.length >= 4);
    if (segments.length === 0) return null;
    const combined = segments.join("");
    const cjkCount = (combined.match(CJK_CHAR_RE) ?? []).length;
    if (cjkCount >= 6 || combined.length >= 12) return segments;
    return null;
}

function clauseBefore(text: string, pos: number, window: number): string {
    const slice = text.slice(Math.max(0, pos - window), pos);
    const re = new RegExp(CLAUSE_BREAK_RE.source, "g");
    let lastEnd = -1;
    let m: RegExpExecArray | null;
    while ((m = re.exec(slice)) !== null) lastEnd = m.index + m[0].length;
    return lastEnd >= 0 ? slice.slice(lastEnd) : slice;
}

function clauseAfter(text: string, pos: number, window: number): string {
    const slice = text.slice(pos, Math.min(text.length, pos + window));
    const breakIdx = slice.search(CLAUSE_BREAK_RE);
    return breakIdx >= 0 ? slice.slice(0, breakIdx) : slice;
}

function quoteSupportedByText(segments: readonly string[], normalizedOriginal: string): boolean {
    if (segments.every((seg) => normalizedOriginal.includes(seg))) return true;
    let totalGrams = 0;
    let hitGrams = 0;
    for (const seg of segments) {
        if (seg.length < 3) continue;
        for (let i = 0; i <= seg.length - 3; i++) {
            totalGrams += 1;
            if (normalizedOriginal.includes(seg.slice(i, i + 3))) hitGrams += 1;
        }
    }
    return totalGrams > 0 && hitGrams / totalGrams >= QUOTE_TRIGRAM_MIN_OVERLAP;
}

function expandGroupIndices(groupText: string): number[] {
    const indices: number[] = [];
    const rangeMatch = /^m(0\d{3,6}|[1-9]\d{4,6})\s*(?:–|—|-|~|\.\.)\s*m(0\d{3,6}|[1-9]\d{4,6})$/.exec(groupText);
    if (rangeMatch) {
        const lo = Math.min(Number(rangeMatch[1]), Number(rangeMatch[2]));
        const hi = Math.max(Number(rangeMatch[1]), Number(rangeMatch[2]));
        if (hi - lo <= MAX_RANGE_EXPANSION) {
            for (let n = lo; n <= hi; n++) indices.push(n);
            return indices;
        }
        return [lo, hi];
    }
    MESSAGE_REF_CITATION_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = MESSAGE_REF_CITATION_RE.exec(groupText)) !== null) {
        indices.push(Number(m[1]));
    }
    return indices;
}

function hasDirectUserAttribution(summary: string, gStart: number, gEnd: number): boolean {
    const before = clauseBefore(summary, gStart, 32);
    const after = clauseAfter(summary, gEnd, 28);
    if (DIRECT_USER_BEFORE_ZH_RE.test(before) || DIRECT_USER_BEFORE_EN_RE.test(before)) return true;
    if (USER_SUBJECT_BEFORE_ZH_RE.test(before) && USER_ACTION_AFTER_ZH_RE.test(after)) return true;
    if (DIRECT_USER_AFTER_ZH_RE.test(after) || DIRECT_USER_AFTER_EN_RE.test(after)) return true;
    return false;
}

interface QuoteSpan {
    readonly start: number;
    readonly end: number;
    readonly segments: readonly string[];
}

function collectQuoteSpans(summary: string): QuoteSpan[] {
    const codeRanges: Array<{ start: number; end: number }> = [];
    CODE_SPAN_RE.lastIndex = 0;
    let cm: RegExpExecArray | null;
    while ((cm = CODE_SPAN_RE.exec(summary)) !== null) {
        codeRanges.push({ start: cm.index, end: cm.index + cm[0].length });
    }
    const insideCode = (pos: number): boolean => codeRanges.some((r) => pos >= r.start && pos < r.end);
    const spans: QuoteSpan[] = [];
    QUOTE_SPANS_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = QUOTE_SPANS_RE.exec(summary)) !== null) {
        if (insideCode(m.index)) continue;
        const inner = m[1] ?? m[2] ?? m[3] ?? m[4] ?? "";
        const segments = extractVerifiableQuoteSegments(inner);
        if (segments) spans.push({ start: m.index, end: m.index + m[0].length, segments });
    }
    return spans;
}

function findAdjacentQuote(summary: string, gStart: number, gEnd: number, quotes: readonly QuoteSpan[]): QuoteSpan | undefined {
    let best: QuoteSpan | undefined;
    let bestDist = Infinity;
    for (const q of quotes) {
        if (q.start >= gEnd) {
            const gap = q.start - gEnd;
            if (gap <= MAX_QUOTE_AFTER_GAP && !CLAUSE_BREAK_RE.test(summary.slice(gEnd, q.start)) && gap < bestDist) {
                best = q;
                bestDist = gap;
            }
        } else if (gStart >= q.end) {
            const gap = gStart - q.end;
            if (gap <= MAX_QUOTE_BEFORE_GAP && !CLAUSE_BREAK_RE.test(summary.slice(q.end, gStart)) && gap < bestDist) {
                best = q;
                bestDist = gap;
            }
        }
    }
    return best;
}

interface NeutralizeCitationsOptions {
    readonly maxChars?: number;
    readonly messageFacts?: ReadonlyMap<number, CitationMessageFact>;
    readonly maxAllocatedRef?: number;
}

interface NeutralizeCitationsResult {
    readonly summary: string;
    readonly neutralized: readonly string[];
}

function findInvalidRefSpans(
    summary: string,
    groundedRefs: ReadonlySet<number> | null,
    options: NeutralizeCitationsOptions,
): Array<{ start: number; end: number; raw: string }> {
    const invalid: Array<{ start: number; end: number; raw: string }> = [];
    const facts = options.messageFacts;
    const maxAllocated = options.maxAllocatedRef;
    const quotes = facts && facts.size > 0 ? collectQuoteSpans(summary) : [];

    MESSAGE_REF_GROUP_RE.lastIndex = 0;
    let groupMatch: RegExpExecArray | null;
    while ((groupMatch = MESSAGE_REF_GROUP_RE.exec(summary)) !== null) {
        const groupText = groupMatch[0];
        const gStart = groupMatch.index;
        const gEnd = gStart + groupText.length;
        const beforeShort = clauseBefore(summary, gStart, 16);
        const afterShort = clauseAfter(summary, gEnd, 16);
        const isChronological = CHRONOLOGICAL_AFTER_RE.test(afterShort) || CHRONOLOGICAL_BEFORE_RE.test(beforeShort);

        const groupIndices = expandGroupIndices(groupText);
        let groupFactViolation = false;
        if (facts && facts.size > 0 && !isChronological && groupIndices.length > 0) {
            const resolvedFacts = groupIndices.map((n) => facts.get(n));
            const allHaveFacts = resolvedFacts.every((f): f is CitationMessageFact => f !== undefined);

            if (allHaveFacts && hasDirectUserAttribution(summary, gStart, gEnd)) {
                const allHaveRoles = resolvedFacts.every((f) => typeof f.role === "string" && f.role.length > 0);
                if (allHaveRoles && !resolvedFacts.some((f) => f.role === "user")) {
                    groupFactViolation = true;
                }
            }

            if (!groupFactViolation && allHaveFacts && quotes.length > 0) {
                const adjacentQuote = findAdjacentQuote(summary, gStart, gEnd, quotes);
                if (adjacentQuote) {
                    const normalizedTexts = resolvedFacts.map(normalizedFactText);
                    if (normalizedTexts.every((t): t is string => t !== undefined)) {
                        const anyMatches = normalizedTexts.some((t) => quoteSupportedByText(adjacentQuote.segments, t));
                        if (!anyMatches) groupFactViolation = true;
                    }
                }
            }
        }

        MESSAGE_REF_CITATION_RE.lastIndex = 0;
        let refMatch: RegExpExecArray | null;
        while ((refMatch = MESSAGE_REF_CITATION_RE.exec(groupText)) !== null) {
            const index = Number(refMatch[1]);
            const ungrounded = groundedRefs !== null && !groundedRefs.has(index);
            const aboveFrontier = maxAllocated !== undefined && maxAllocated > 0 && index > maxAllocated;
            if (groupFactViolation || ungrounded || aboveFrontier) {
                invalid.push({
                    start: gStart + refMatch.index,
                    end: gStart + refMatch.index + refMatch[0].length,
                    raw: refMatch[0],
                });
            }
        }
    }
    return invalid;
}

function applySpanReplacements(
    summary: string,
    spans: readonly { start: number; end: number; raw: string }[],
    placeholder: string,
): string {
    let out = "";
    let cursor = 0;
    for (const span of spans) {
        out += summary.slice(cursor, span.start) + placeholder;
        cursor = span.end;
    }
    return out + summary.slice(cursor);
}

/** Verify an existing/reference summary against original message facts (role,
 *  quoted original text, and session ref frontier) without requiring an
 *  input-grounding set. Used to stop stale wrong citations in `otherSummaries`
 *  or child blocks (e.g. `b32 -> b34`) from poisoning newly generated summaries. */
export function verifyMessageRefCitations(
    summary: string,
    messageFacts: ReadonlyMap<number, CitationMessageFact>,
    maxAllocatedRef?: number,
): NeutralizeCitationsResult {
    const spans = findInvalidRefSpans(summary, null, { messageFacts, maxAllocatedRef });
    if (spans.length === 0) return { summary, neutralized: [] };
    const neutralized: string[] = [];
    const seen = new Set<string>();
    for (const span of spans) {
        if (!seen.has(span.raw)) { seen.add(span.raw); neutralized.push(span.raw); }
    }
    return { summary: applySpanReplacements(summary, spans, UNVERIFIED_REF_PLACEHOLDER), neutralized };
}

/** Replace message-ref citations in a generated summary that either never
 *  appeared in authoritative summarizer inputs, exceed the session ref frontier,
 *  misattribute a user quote/authorization to a non-user message, or fail to
 *  match an adjacent verbatim quote in the cited original message. Never pushes
 *  an already-fitting summary over `maxChars`. */
export function neutralizeUngroundedMessageRefs(
    summary: string,
    groundedRefs: ReadonlySet<number>,
    options: NeutralizeCitationsOptions = {},
): NeutralizeCitationsResult {
    const spans = findInvalidRefSpans(summary, groundedRefs, options);
    if (spans.length === 0) return { summary, neutralized: [] };
    const neutralized: string[] = [];
    const seen = new Set<string>();
    for (const span of spans) {
        if (!seen.has(span.raw)) { seen.add(span.raw); neutralized.push(span.raw); }
    }
    let cleaned = applySpanReplacements(summary, spans, UNVERIFIED_REF_PLACEHOLDER);
    const cap = options.maxChars;
    if (cap !== undefined && Number.isFinite(cap) && cap > 0 && cleaned.length > cap) {
        cleaned = applySpanReplacements(summary, spans, COMPACT_UNVERIFIED_REF_PLACEHOLDER);
        if (cleaned.length > cap) cleaned = applySpanReplacements(summary, spans, SHORT_UNVERIFIED_REF_PLACEHOLDER);
    }
    return { summary: cleaned, neutralized };
}

/** Describe why an external-summary batch yielded no usable summary (#2484):
 *  the batch stop reason plus each candidate attempt's outcome, so an operator
 *  can tell timeout / error / cancelled / invalid-plan apart instead of seeing
 *  one opaque "failed or exceeded their budget" line. #3635: invalid_summary
 *  attempts carry their rejection reason in parentheses. */
export function describeExternalSummaryFailure(batch: ExternalSummaryBatchResult): string {
    if (batch.status === "failed") return `batch=failed reason=${batch.reason}`;
    // #2678: multi-range batches report the FIRST failing range, not slot 0 —
    // slot 0 may have succeeded while a later range is what stopped the batch.
    // Single-range callers see identical output (find falls back to results[0]).
    const result = batch.results.find((entry) => entry.status !== "success") ?? batch.results[0];
    if (!result) return `batch=${batch.status} (no result)`;
    const parts = [`batch=${batch.status}`, `result=${result.status}`];
    if (result.status === "failed") parts.push(`reason=${result.reason}`);
    parts.push(`attempts=[${result.attempts.map((a) => `t${a.targetIndex}:${a.outcome}${a.detail ? ` (${a.detail})` : ""}`).join(",")}]`);
    return parts.join(" ");
}

function validBudget(budget: SummaryBudget): boolean {
    return positiveInteger(budget.totalTimeoutMs) && budget.totalTimeoutMs <= 2_147_483_647
        && positiveInteger(budget.targetTimeoutMs) && budget.targetTimeoutMs <= 2_147_483_647
        && positiveInteger(budget.maxSummaryBytes);
}

function awaitAttempt(operation: Promise<AttemptResult>, signal: AbortSignal): Promise<AttemptResult> {
    return new Promise((resolve) => {
        const abort = () => resolve({ kind: "aborted" });
        signal.addEventListener("abort", abort, { once: true });
        if (signal.aborted) abort();
        void operation.then((result) => {
            signal.removeEventListener("abort", abort);
            resolve(result);
        });
    });
}

/** Internal execution boundary only: adapters resolve wire/auth; callers own fold commits. */
export class ExternalSummaryExecutor {
    private active = 0;
    private readonly waiting: Waiter[] = [];
    private readonly baseConcurrency: number;
    private readonly scopedConcurrency = new Map<number, number>();

    constructor(concurrency: number) {
        if (!positiveInteger(concurrency)) throw new TypeError("Summary concurrency must be a positive integer");
        this.baseConcurrency = concurrency;
    }

    async execute(work: SummaryWork, candidates: readonly SummaryCandidate[], budget: SummaryBudget, signal?: AbortSignal, options?: SummaryPoolOptions): Promise<ExternalSummaryResult> {
        const endScope = this.beginScope(options);
        try {
            return await this.executeUntil(work, candidates, budget, signal);
        } finally {
            endScope();
        }
    }

    async executeBatch(work: readonly SummaryWork[], candidates: readonly SummaryCandidate[], budget: SummaryBudget, signal?: AbortSignal, options?: SummaryPoolOptions): Promise<ExternalSummaryBatchResult> {
        const endScope = this.beginScope(options);
        try {
            return await this.executeBatchInner(work, candidates, budget, signal);
        } finally {
            endScope();
        }
    }

    private async executeBatchInner(work: readonly SummaryWork[], candidates: readonly SummaryCandidate[], budget: SummaryBudget, signal?: AbortSignal): Promise<ExternalSummaryBatchResult> {
        const results: ExternalSummaryResult[] = [];
        if (signal?.aborted) return { status: "cancelled", results };
        const limits = { ...budget };
        if (!validBudget(limits) || work.length === 0 || candidates.length === 0
            || candidates.some((target) => typeof target.summarize !== "function")) {
            return { status: "failed", reason: "invalid_plan", results: [] };
        }
        const deadline = performance.now() + limits.totalTimeoutMs;
        const requests = work.map((request) => Object.freeze({ ...request }));
        const targets = [...candidates];
        // #2667: workers pull indices from a shared counter so one batch runs up
        // to `concurrency` ranges in flight instead of strictly serial — the pool
        // now bounds a batch as well as cross-session calls. Results stay
        // index-aligned with `work`: the reported list is the DENSE PREFIX of
        // settled ranges (downstream zips entries with their work slot), and an
        // early stop (caller abort / any range hitting the shared deadline)
        // leaves unstarted ranges unreported rather than fabricated.
        const completed = new Map<number, ExternalSummaryResult>();
        let itemDeadline = false;
        const stop = new AbortController();
        const onAbort = (): void => stop.abort();
        signal?.addEventListener("abort", onAbort, { once: true });
        const nextIndex = { value: 0 };
        const worker = async (): Promise<void> => {
            for (;;) {
                if (stop.signal.aborted) return;
                const index = nextIndex.value++;
                if (index >= requests.length) return;
                const result = await this.executeUntil(requests[index], targets, limits, signal, deadline);
                completed.set(index, result);
                if (result.status === "deadline") itemDeadline = true;
                if (result.status === "cancelled" || result.status === "deadline") {
                    stop.abort();
                    return;
                }
            }
        };
        try {
            await Promise.all(Array.from({ length: Math.min(this.effectiveConcurrency(), requests.length) }, () => worker()));
        } finally {
            signal?.removeEventListener("abort", onAbort);
        }
        let prefix = 0;
        while (prefix < requests.length && completed.has(prefix)) prefix += 1;
        const settled: ExternalSummaryResult[] = [];
        for (let index = 0; index < prefix; index++) {
            const value = completed.get(index);
            if (value !== undefined) settled.push(value);
        }
        if (signal?.aborted) return { status: "cancelled", results: settled };
        if (itemDeadline || performance.now() >= deadline) return { status: "deadline", results: settled };
        return { status: "finished", results: settled };
    }

    private async executeUntil(work: SummaryWork, candidates: readonly SummaryCandidate[], budget: SummaryBudget, signal?: AbortSignal, sharedDeadline?: number): Promise<ExternalSummaryResult> {
        const attempts: SummaryAttempt[] = [];
        if (signal?.aborted) return { status: "cancelled", attempts };
        const limits = { ...budget };
        if (!validBudget(limits) || !work.content.trim() || !work.instructions.trim()
            || candidates.length === 0 || candidates.some((target) => typeof target.summarize !== "function")) {
            return { status: "failed", reason: "invalid_plan", attempts };
        }
        const targets = [...candidates];
        const request = Object.freeze({ ...work });
        const startedAt = performance.now();
        const overall = new AbortController();
        // #4833: every internal abort carries its provenance so the candidate's
        // warn line names the real source instead of guessing client-abort.
        const cancel = (source: Exclude<SummaryAbortSource, "target-timeout">): void => { if (!overall.signal.aborted) overall.abort(new SummaryAbortError(source)); };
        const onCallerAbort = (): void => cancel("cancelled");
        signal?.addEventListener("abort", onCallerAbort, { once: true });
        if (signal?.aborted) cancel("cancelled");
        const deadline = Math.min(performance.now() + limits.totalTimeoutMs, sharedDeadline ?? Infinity);
        const totalTimer = setTimeout(() => cancel("batch-deadline"), Math.max(0, deadline - performance.now()));
        const stopped = (): ExternalSummaryResult => ({ status: signal?.aborted ? "cancelled" : "deadline", attempts });
        try {
            for (const [targetIndex, target] of targets.entries()) {
                const release = await this.acquire(overall.signal);
                if (performance.now() >= deadline) cancel("batch-deadline");
                if (!release || overall.signal.aborted) {
                    release?.();
                    return stopped();
                }
                const attempt = new AbortController();
                // Propagate the overall abort WITH its reason so the candidate
                // sees which budget (or whose cancellation) killed this attempt.
                const propagateOverall = (): void => { attempt.abort(overall.signal.reason); };
                overall.signal.addEventListener("abort", propagateOverall, { once: true });
                if (overall.signal.aborted) propagateOverall();
                const targetDeadline = performance.now() + limits.targetTimeoutMs;
                const expireTarget = (): void => { attempt.abort(new SummaryAbortError("target-timeout")); };
                const timer = setTimeout(expireTarget, limits.targetTimeoutMs);
                // Retain the permit until the actual call settles, even if it ignores cancellation.
                const operation: Promise<AttemptResult> = Promise.resolve().then(() => {
                    if (performance.now() >= deadline) cancel("batch-deadline");
                    if (performance.now() >= targetDeadline) expireTarget();
                    if (attempt.signal.aborted) throw new Error("Summary attempt cancelled before dispatch");
                    return target.summarize(request, attempt.signal);
                }).then<AttemptResult, AttemptResult>((value) => ({ kind: "value", value }), () => ({ kind: "error" })).finally(release);
                let result: AttemptResult;
                try {
                    result = await awaitAttempt(operation, attempt.signal);
                } finally {
                    clearTimeout(timer);
                    overall.signal.removeEventListener("abort", propagateOverall);
                }
                if (performance.now() >= deadline) cancel("batch-deadline");
                if (overall.signal.aborted) {
                    if (signal?.aborted) attempts.push({ targetIndex, outcome: "cancelled", detail: "caller abort" });
                    else attempts.push({ targetIndex, outcome: "timeout", detail: timeoutDetail(summaryAbortSource(overall.signal) ?? "batch-deadline", limits, performance.now() - startedAt) });
                    return stopped();
                }
                if (performance.now() >= targetDeadline) expireTarget();
                if (result.kind === "aborted" || attempt.signal.aborted) {
                    attempts.push({ targetIndex, outcome: "timeout", detail: timeoutDetail(summaryAbortSource(attempt.signal) ?? "target-timeout", limits, performance.now() - startedAt) });
                } else if (result.kind === "error") {
                    attempts.push({ targetIndex, outcome: "error" });
                } else if (typeof result.value !== "string" || !result.value.trim()
                    || Buffer.byteLength(result.value, "utf8") > limits.maxSummaryBytes
                    || result.value.trim().length < (request.minSummaryChars ?? 1)
                    || ((request.maxSummaryChars ?? 0) > 0 && result.value.trim().length > request.maxSummaryChars!)) {
                    attempts.push({ targetIndex, outcome: "invalid_summary", detail: invalidSummaryDetail(result.value, limits, request) });
                } else {
                    attempts.push({ targetIndex, outcome: "success" });
                    return { status: "success", summary: result.value, targetIndex, attempts };
                }
            }
            return { status: "failed", reason: "exhausted", attempts };
        } finally {
            clearTimeout(totalTimer);
            signal?.removeEventListener("abort", onCallerAbort);
        }
    }

    /** [#2657] Register one batch's pool-size demand for its lifetime; the
     *  returned release restores the previous sizing. No-op when the batch
     *  requests no size (the base pool applies). */
    private beginScope(options?: SummaryPoolOptions): () => void {
        const requested = options?.concurrency;
        if (requested === undefined) return () => {};
        if (!positiveInteger(requested)) throw new TypeError("Summary concurrency must be a positive integer");
        const live = this.scopedConcurrency.get(requested) ?? 0;
        if (live === 0) {
            this.scopedConcurrency.set(requested, 1);
            this.drain();
        } else {
            this.scopedConcurrency.set(requested, live + 1);
        }
        let released = false;
        return () => {
            if (released) return;
            released = true;
            const remaining = (this.scopedConcurrency.get(requested) ?? 1) - 1;
            if (remaining > 0) this.scopedConcurrency.set(requested, remaining);
            else this.scopedConcurrency.delete(requested);
        };
    }

    private effectiveConcurrency(): number {
        let cap = this.baseConcurrency;
        for (const requested of this.scopedConcurrency.keys()) if (requested > cap) cap = requested;
        return cap;
    }

    private acquire(signal: AbortSignal): Promise<Release | undefined> {
        if (signal.aborted) return Promise.resolve(undefined);
        return new Promise((resolve) => {
            const waiter: Waiter = {
                signal, resolve,
                abort: () => {
                    const index = this.waiting.indexOf(waiter);
                    if (index >= 0) this.waiting.splice(index, 1);
                    signal.removeEventListener("abort", waiter.abort);
                    resolve(undefined);
                    this.drain();
                },
            };
            this.waiting.push(waiter);
            signal.addEventListener("abort", waiter.abort, { once: true });
            this.drain();
        });
    }

    private drain(): void {
        while (this.active < this.effectiveConcurrency() && this.waiting.length > 0) {
            const waiter = this.waiting.shift();
            if (!waiter) break;
            waiter.signal.removeEventListener("abort", waiter.abort);
            if (waiter.signal.aborted) {
                waiter.resolve(undefined);
                continue;
            }
            this.active++;
            let released = false;
            waiter.resolve(() => {
                if (released) return;
                released = true;
                this.active--;
                this.drain();
            });
        }
    }
}
