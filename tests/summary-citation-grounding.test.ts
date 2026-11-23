import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtempSync } from "node:fs";
import http from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { BLOCKED_REF, createContentStore, createCore, defaultConfig, defaultPrompts, storeOriginal, type Config, type CoreMessage } from "acp-kernel";
import {
    UNVERIFIED_REF_PLACEHOLDER,
    buildCitationMessageIndex,
    collectGroundedMessageRefs,
    highestAllocatedMessageRef,
    neutralizeUngroundedMessageRefs,
    verifyMessageRefCitations,
} from "../src/external-summary.ts";
import { applyConfiguredCompression } from "../src/external-summary-compress.ts";
import { SummaryCredentialStore } from "../src/external-summary-credentials.ts";
import { preflightCompress } from "../src/preflight.ts";
import { createSession } from "../src/session.ts";
import type { RewriteCtx } from "../src/stream.ts";
import { rmrf } from "./tmp-rm.ts";

process.env.NODE_ENV = "test";

type Handler = (req: http.IncomingMessage, res: http.ServerResponse, body: Record<string, unknown>) => void;
type SummarySettings = Record<string, unknown>;

async function withMockSummaryServer(
    run: (base: string, externalSummary: SummarySettings) => Promise<void>,
    handler: Handler,
): Promise<void> {
    const root = mkdtempSync(join(tmpdir(), "bili-citation-grounding-"));
    const configPath = join(root, "config.json");
    const previous = process.env.BILI_CONFIG_FILE;
    process.env.BILI_CONFIG_FILE = configPath;
    const server = http.createServer((req, res) => {
        const chunks: Buffer[] = [];
        req.on("data", (chunk: Buffer) => chunks.push(chunk));
        req.on("end", () => handler(req, res, JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>));
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const base = `http://127.0.0.1:${(server.address() as import("node:net").AddressInfo).port}`;
    const externalSummary: SummarySettings = {
        enabled: true,
        targets: [{ name: "primary", protocol: "responses", url: `${base}/primary`, model: "summary-primary", credentialRef: "secret:primary" }],
    };
    new SummaryCredentialStore().set("primary", "test-primary-key");
    try {
        await run(base, externalSummary);
    } finally {
        server.closeAllConnections();
        await new Promise<void>((resolve) => server.close(() => resolve()));
        if (previous === undefined) delete process.env.BILI_CONFIG_FILE;
        else process.env.BILI_CONFIG_FILE = previous;
        rmrf(root);
    }
}

test("neutralizeUngroundedMessageRefs keeps grounded refs, strips fabricated ones, and ignores identifiers", () => {
    const grounded = collectGroundedMessageRefs(["[m01635] user authorization", "[m01636] plan", "[m01672] done"]);
    const input = [
        "目标：见 m01635 授权与 m01636–m01672 执行记录（等价写法 m1635）。",
        "错号：原文见 m01538 附近，且泄露了示例 m00132 与四位补零 m0538。",
        "非编号标识符保留：foo_m01538, m01538a, m2026, m1024。",
    ].join("\n");

    const result = neutralizeUngroundedMessageRefs(input, grounded);
    assert.deepEqual(result.neutralized, ["m01538", "m00132", "m0538"]);
    assert.ok(result.summary.includes("m01635"));
    assert.ok(result.summary.includes("m01636–m01672"));
    assert.ok(result.summary.includes("m1635"));
    assert.ok(result.summary.includes(`原文见 ${UNVERIFIED_REF_PLACEHOLDER} 附近`));
    assert.equal(result.summary.includes("m01538 附近"), false);
    assert.equal(result.summary.includes("m00132"), false);
    assert.equal(result.summary.includes("m0538"), false);
    assert.ok(result.summary.includes("foo_m01538, m01538a, m2026, m1024"));
});

test("neutralizeUngroundedMessageRefs never pushes an already-fitting summary over maxChars", () => {
    const grounded = collectGroundedMessageRefs(["[m00001]"]);
    const base = `See m01538 ${"x".repeat(89)}`; // 100 chars total
    assert.equal(base.length, 100);

    const exactFit = neutralizeUngroundedMessageRefs(base, grounded, { maxChars: 100 });
    assert.deepEqual(exactFit.neutralized, ["m01538"]);
    assert.ok(exactFit.summary.length <= 100, `expected <= 100 chars, got ${exactFit.summary.length}`);
    assert.equal(exactFit.summary.includes("m01538"), false);
    assert.ok(exactFit.summary.includes("[ref?]"));

    const shortRef = `See m0538 ${"x".repeat(90)}`; // 100 chars total, 5-char ref
    const tightFit = neutralizeUngroundedMessageRefs(shortRef, grounded, { maxChars: 100 });
    assert.deepEqual(tightFit.neutralized, ["m0538"]);
    assert.ok(tightFit.summary.length <= 100);
    assert.ok(tightFit.summary.includes("[?]"));
});

test("external compress passes refs in currentContext and block spans in otherSummaries, and neutralizes ungrounded refs", async () => {
    let capturedReference: Record<string, unknown> | undefined;
    let capturedSystem = "";
    const logs: string[] = [];

    await withMockSummaryServer(async (_base, externalSummary) => {
        const core = createCore();
        const config = defaultConfig(400_000) as Config & { externalSummary?: unknown };
        config.externalSummary = externalSummary;
        config.preserveRecentMessages = 0;
        config.preserveRecentTokens = 0;
        config.compress.minCompressRange = 100;

        const session = createSession(`citation-compress-${Math.random()}`);
        const filler = "Detailed verification output for host cutover and backup checks. ".repeat(20);
        const messages: CoreMessage[] = [
            { id: "old-work", role: "assistant", contentType: "text", text: `Earlier step. ${filler}` },
            { id: "user-auth", role: "user", contentType: "text", text: "你推送到远端，并更新当前正在运行的到最新版本吧" },
            { id: "cutover-a", role: "assistant", contentType: "text", text: `Cutover step A. ${filler}` },
            { id: "cutover-b", role: "assistant", contentType: "text", text: `Cutover step B. ${filler}` },
            { id: "blocked-user", role: "user", contentType: "text", text: "Blocked ref user note." },
        ];
        session.state = core.processTurn({ messages, state: session.state, config, tokenCount: 20_000, renderTags: "text-only" }).state;
        // Seed an active sibling block (like b31) before compressing the target range (like b32).
        const seeded = core.applyCompression({
            messages,
            state: session.state,
            config,
            ranges: [{
                startRef: session.state.messageRefs.byRaw["old-work"],
                endRef: session.state.messageRefs.byRaw["old-work"],
                summary: "Historical step completed before cutover authorization; exact verification path src/cutover.ts:12 recorded.",
                topic: "pre-cutover",
            }],
        });
        session.state = seeded.state;
        // Simulate a message whose ref is BLOCKED — must not be advertised as ref: "BLOCKED".
        session.state.messageRefs.byRaw["blocked-user"] = BLOCKED_REF;

        const authRef = session.state.messageRefs.byRaw["user-auth"];
        const startRef = session.state.messageRefs.byRaw["cutover-a"];
        const endRef = session.state.messageRefs.byRaw["cutover-b"];
        const ctx: RewriteCtx = { core, config, session, messages, log: (m) => logs.push(m) };

        const result = await applyConfiguredCompression({
            content: [{
                startId: startRef,
                endId: endRef,
                // Non-authoritative hint carrying a bogus ref (m09999) — must NOT ground m09999.
                summary: "Hint mentioning ungrounded m09999.",
            }],
        }, ctx);

        assert.equal(result.outcome, "applied", result.text);
        assert.ok(capturedSystem.includes("never guess, approximate, or invent a message ref"));

        const currentContext = capturedReference?.currentContext as Array<{ ref?: string; role: string; text: string }>;
        const authEntry = currentContext.find((entry) => entry.text.includes("你推送到远端"));
        assert.deepEqual(authEntry, { ref: authRef, role: "user", text: "你推送到远端，并更新当前正在运行的到最新版本吧" });
        const blockedEntry = currentContext.find((entry) => entry.text.includes("Blocked ref user note."));
        assert.ok(blockedEntry && !("ref" in blockedEntry), "BLOCKED refs must not be emitted in currentContext");

        const otherSummaries = capturedReference?.otherSummaries as Array<{ blockId: string; range?: string; topic?: string; summary: string }>;
        assert.equal(otherSummaries.length, 1);
        assert.equal(otherSummaries[0].blockId, "b1");
        assert.equal(otherSummaries[0].range, `${session.state.messageRefs.byRaw["old-work"]}-${session.state.messageRefs.byRaw["old-work"]}`);

        const created = session.state.blocks.find((block) => block.blockId === "b2");
        assert.ok(created);
        assert.ok(created.summary.includes(authRef), "grounded currentContext ref stays verbatim");
        assert.ok(created.summary.includes(startRef), "grounded covered ref stays verbatim");
        assert.equal(created.summary.includes("m01538"), false, "hallucinated ref must be neutralized");
        assert.equal(created.summary.includes("m09999"), false, "ref only present in non-authoritative hint must be neutralized");
        assert.ok(created.summary.includes(UNVERIFIED_REF_PLACEHOLDER));
        assert.ok(logs.some((line) => line.includes("neutralized 2 unverified message ref(s)") && line.includes("m01538") && line.includes("m09999")), logs.join("\n"));
    }, (_req, res, body) => {
        capturedSystem = String(body.instructions ?? "");
        const rawInput = (body.input as Array<{ content: string }>)[0].content;
        const parsed = JSON.parse(rawInput) as { content: string; reference: string };
        capturedReference = JSON.parse(parsed.reference) as Record<string, unknown>;
        const currentContext = capturedReference.currentContext as Array<{ ref?: string; text: string }>;
        const authRef = currentContext.find((entry) => entry.text.includes("你推送到远端"))?.ref ?? "MISSING";
        const firstCoveredRef = /\[(m\d+)\]/.exec(parsed.content)?.[1] ?? "MISSING";
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({
            status: "completed",
            output_text: `用户授权见 ${authRef}，执行见 ${firstCoveredRef}；同时混入幻觉编号 m01538 与提示词未验证编号 m09999。历史结论与校验记录完整保留，未改变任何生产状态。`,
        }));
    });
});

test("preflight serial & batch folds label messages with refs and neutralize fabricated refs", async () => {
    const sentContents: string[] = [];
    const logs: string[] = [];

    await withMockSummaryServer(async (base, externalSummary) => {
        const core = createCore();
        const config = defaultConfig(400_000) as Config & { externalSummary?: unknown };
        config.externalSummary = externalSummary;
        config.modelContextLimit = 1_000;
        config.preserveRecentMessages = 0;
        config.preserveRecentTokens = 0;
        config.compress.minCompressRange = 100;

        const session = createSession(`citation-preflight-${Math.random()}`);
        const filler = "Preflight diagnostic payload covering build, test, and migration checks. ".repeat(35);
        const messages: CoreMessage[] = [
            { id: "msg-a", role: "user", contentType: "text", text: `First task detail. ${filler}` },
            { id: "msg-b", role: "assistant", contentType: "text", text: `First result detail. ${filler}` },
            { id: "msg-tail", role: "user", contentType: "text", text: "Current active question." },
        ];
        session.state = core.processTurn({ messages, state: session.state, config, tokenCount: 10_000, renderTags: "text-only" }).state;
        session.stats.lastInputTokens = 10_000;
        const refA = session.state.messageRefs.byRaw["msg-a"];
        const refB = session.state.messageRefs.byRaw["msg-b"];

        const result = await preflightCompress({
            core,
            config,
            session,
            prompts: defaultPrompts,
            protocol: "responses",
            url: `${base}/unused-main`,
            headers: {},
            model: "main-model",
            log: (_level, message) => logs.push(message),
        }, messages);

        assert.ok(result.compressedRanges > 0, JSON.stringify(result));
        assert.ok(sentContents.some((c) => c.includes(`[${refA} user]`) && c.includes(`[${refB} assistant]`)), sentContents.join("\n---\n"));
        const block = session.state.blocks[0];
        assert.ok(block);
        assert.ok(block.summary.includes(refA), "grounded preflight ref survives");
        assert.equal(block.summary.includes("m01538"), false, "hallucinated preflight ref is neutralized");
        assert.ok(block.summary.includes(UNVERIFIED_REF_PLACEHOLDER));
        assert.ok(logs.some((line) => line.includes("neutralized 1 unverified message ref(s)") && line.includes("m01538")), logs.join("\n"));
    }, (_req, res, body) => {
        const rawInput = (body.input as Array<{ content: string }>)[0].content;
        const parsed = JSON.parse(rawInput) as { content: string };
        sentContents.push(parsed.content);
        const firstRef = /\[(m\d+)\s/.exec(parsed.content)?.[1] ?? "MISSING";
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({
            status: "completed",
            output_text: `Historical task from ${firstRef} preserved with key paths src/preflight.ts:2710 and bogus citation m01538 stripped before commit.`,
        }));
    });
});

test("preflight batch fold labels every message with its ref and neutralizes ungrounded refs", async () => {
    const sentContents: string[] = [];
    const logs: string[] = [];

    await withMockSummaryServer(async (base, externalSummary) => {
        const core = createCore();
        const config = defaultConfig(3_500) as Config & { externalSummary?: unknown };
        config.externalSummary = externalSummary;
        config.preserveRecentMessages = 1;
        config.preserveRecentTokens = 0;
        config.compress.minCompressRange = 100;

        const session = createSession(`citation-preflight-batch-${Math.random()}`);
        const filler = "Batch diagnostic output verifying module layout and cache stability. ".repeat(35);
        // Alternating user/assistant + a covered middle block creates ≥2 disjoint viable ranges so batch preflight triggers.
        const messages: CoreMessage[] = [];
        for (let i = 0; i < 6; i++) {
            messages.push({ id: `u${i}`, role: "user", contentType: "text", text: `User turn ${i}: check module ${i}. ${filler}` });
            messages.push({ id: `a${i}`, role: "assistant", contentType: "text", text: `Assistant turn ${i}: checked module ${i}. ${filler}` });
        }
        messages.push({ id: "tail", role: "user", contentType: "text", text: "Latest active user question." });
        session.state = core.processTurn({ messages, state: session.state, config: { ...config, modelContextLimit: 1_000_000 }, tokenCount: 1_000_000, renderTags: "text-only" }).state;
        // Seed an already-covered middle block to split uncovered history into two disjoint ranges (forcing ≥2 nominations in batch preflight).
        const midRef = session.state.messageRefs.byRaw["u3"];
        session.state = core.applyCompression({
            messages,
            state: session.state,
            config: { ...config, modelContextLimit: 1_000_000, preserveRecentMessages: 0 },
            ranges: [{ startRef: midRef, endRef: midRef, summary: "Midpoint block completed cleanly with verified path src/mid.ts:10 and all checks passing." }],
        }).state;
        session.stats.lastInputTokens = 12_000;

        const result = await preflightCompress({
            core,
            config,
            session,
            prompts: defaultPrompts,
            protocol: "responses",
            url: `${base}/unused-main`,
            headers: {},
            model: "main-model",
            log: (_level, message) => logs.push(message),
        }, messages);

        assert.ok(result.compressedRanges >= 2, `expected batch fold of >=2 ranges, got ${JSON.stringify(result)}`);
        assert.ok(logs.some((line) => /\[external-summary\] completed \(preflight\) [2-9]\d*\/[2-9]\d* range\(s\); calls=/.test(line)), `expected batch preflight lane, got:\n${logs.join("\n")}`);
        assert.ok(sentContents.length >= 2);
        assert.ok(sentContents.every((c) => /\[m\d{5}\s+(?:user|assistant)\]/.test(c)), sentContents.join("\n---\n"));
        const newBlocks = session.state.blocks.filter((block) => block.blockId !== "b1");
        assert.ok(newBlocks.length >= 2);
        for (const block of newBlocks) {
            assert.equal(block.summary.includes("m07777"), false, `fabricated ref survived in ${block.blockId}`);
            assert.ok(block.summary.includes(UNVERIFIED_REF_PLACEHOLDER), `expected placeholder in ${block.blockId}`);
        }
        assert.ok(logs.some((line) => line.includes("neutralized 1 unverified message ref(s)") && line.includes("m07777")), logs.join("\n"));
    }, (_req, res, body) => {
        const rawInput = (body.input as Array<{ content: string }>)[0].content;
        const parsed = JSON.parse(rawInput) as { content: string };
        sentContents.push(parsed.content);
        const firstRef = /\[(m\d+)\s/.exec(parsed.content)?.[1] ?? "MISSING";
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({
            status: "completed",
            output_text: `Batch range anchored at ${firstRef} kept decisions intact while wrongly citing m07777 outside the input.`,
        }));
    });
});

test("preflight tier distillation includes block spans and inter-block message refs, and neutralizes fabricated refs", async () => {
    let distillContent = "";
    const logs: string[] = [];

    await withMockSummaryServer(async (base, externalSummary) => {
        const core = createCore();
        const config = defaultConfig(12_000) as Config & { externalSummary?: unknown };
        config.externalSummary = externalSummary;
        config.preserveRecentMessages = 1;
        config.preserveRecentTokens = 0;
        config.compress.minCompressRange = 100;

        const text = (size: number, label: string) => `${label} alpha beta gamma delta epsilon `.repeat(Math.ceil(size / 30)).slice(0, size);
        const messages: CoreMessage[] = [];
        for (let i = 0; i < 6; i++) {
            messages.push({ id: `u${i}`, role: "user", contentType: "text", text: text(18_000, `USER${i}`) });
            messages.push({ id: `a${i}`, role: "assistant", contentType: "text", text: text(120, `INTERBLOCK${i}`) });
        }
        messages.push({ id: "tail", role: "user", contentType: "text", text: text(200, "tail") });
        const session = createSession(`citation-distill-${Math.random()}`);
        session.state = core.processTurn({ messages, state: session.state, config: { ...config, modelContextLimit: 12_000_000 }, tokenCount: 12_000_000, renderTags: "text-only" }).state;
        for (let i = 0; i < 6; i++) {
            const ref = session.state.messageRefs.byRaw[`u${i}`]!;
            session.state = core.applyCompression({
                messages,
                state: session.state,
                config: { ...config, preserveRecentMessages: 0 },
                ranges: [{ startRef: ref, endRef: ref, summary: text(18_000, `BLOCK${i}`), topic: `phase-${i}` }],
            }).state;
        }
        session.stats.lastInputTokens = 48_000;

        const interBlockRef = session.state.messageRefs.byRaw["a0"]!;
        const block0Ref = session.state.messageRefs.byRaw["u0"]!;
        await preflightCompress({
            core,
            config,
            session,
            prompts: defaultPrompts,
            protocol: "responses",
            url: `${base}/unused-main`,
            headers: {},
            model: "main-model",
            log: (_level, message) => logs.push(message),
        }, messages);

        assert.ok(distillContent.includes(`[summarized b1 (${block0Ref}-${block0Ref}): phase-0]`), distillContent.slice(0, 300));
        assert.ok(distillContent.includes(`[${interBlockRef} assistant]`), distillContent.slice(0, 2000));
        const t2 = session.state.blocks.find((block) => block.tier === 2);
        assert.ok(t2, "expected tier-2 distilled block");
        assert.ok(t2.summary.includes(interBlockRef), "grounded inter-block ref stays verbatim");
        assert.equal(t2.summary.includes("m08888"), false, "fabricated ref in tier distillation must be neutralized");
        assert.ok(t2.summary.includes(UNVERIFIED_REF_PLACEHOLDER));
        assert.ok(logs.some((line) => line.includes("neutralized 1 unverified message ref(s)") && line.includes("m08888")), logs.join("\n"));
    }, (_req, res, body) => {
        const rawInput = (body.input as Array<{ content: string }>)[0].content;
        const parsed = JSON.parse(rawInput) as { content: string };
        const isDistill = parsed.content.includes("[summarized");
        if (isDistill && !distillContent) distillContent = parsed.content;
        const interRef = /\[(m\d+)\s+assistant\]/.exec(parsed.content)?.[1] ?? "m00001";
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({
            status: "completed",
            output_text: isDistill
                ? `Source: b1+b2. Distilled decisions anchored at ${interRef} with bogus citation m08888 removed.`
                : "Compact T1 summary preserving key paths and verified constraints.",
        }));
    });
});

test("counterexample #1: real grounded tool ref misattributed as user authorization or mismatched quote is neutralized", async () => {
    const logs: string[] = [];

    await withMockSummaryServer(async (_base, externalSummary) => {
        const core = createCore();
        const config = defaultConfig(400_000) as Config & { externalSummary?: unknown };
        config.externalSummary = externalSummary;
        config.preserveRecentMessages = 0;
        config.preserveRecentTokens = 0;
        config.compress.minCompressRange = 100;

        const session = createSession(`citation-role-mismatch-${Math.random()}`);
        const filler = "Tool verification output showing exit code 0 and healthy endpoints. ".repeat(20);
        const messages: CoreMessage[] = [
            { id: "tool-res", role: "tool", contentType: "tool-result", toolName: "bash", text: `exit 0; ${filler}` },
            { id: "user-auth", role: "user", contentType: "text", text: "你推送到远端，并更新当前正在运行的到最新版本吧" },
            { id: "work-done", role: "assistant", contentType: "text", text: `Executed verification after tool output. ${filler}` },
        ];
        session.state = core.processTurn({ messages, state: session.state, config, tokenCount: 20_000, renderTags: "text-only" }).state;

        const toolRef = session.state.messageRefs.byRaw["tool-res"];
        const authRef = session.state.messageRefs.byRaw["user-auth"];
        const endRef = session.state.messageRefs.byRaw["work-done"];
        const ctx: RewriteCtx = { core, config, session, messages, log: (m) => logs.push(m) };

        const result = await applyConfiguredCompression({
            content: [{ startId: toolRef, endId: endRef }],
        }, ctx);

        assert.equal(result.outcome, "applied", result.text);
        const block = session.state.blocks[0];
        assert.ok(block);
        // Tool ref misattributed as user authorization (both unquoted and quoted) must be neutralized even though toolRef IS in content.
        assert.equal(block.summary.includes(`用户授权见 ${toolRef}`), false, block.summary);
        assert.equal(block.summary.includes(`User authorized (${toolRef})`), false, block.summary);
        assert.equal(block.summary.includes(`原文见 ${toolRef} 附近「你推送到远端`), false, block.summary);
        // True user ref stays verbatim.
        assert.ok(block.summary.includes(`用户确认（${authRef}）`), block.summary);
        assert.ok(block.summary.includes(`原文见 ${authRef}「你推送到远端……最新版本吧」`), block.summary);
        // Legitimate non-attribution mention of toolRef stays verbatim.
        assert.ok(block.summary.includes(`用户要求检查 ${toolRef} 的工具输出`), block.summary);
        assert.ok(logs.some((line) => line.includes("neutralized 1 unverified message ref(s)") && line.includes(toolRef)), logs.join("\n"));
    }, (_req, res, body) => {
        const rawInput = (body.input as Array<{ content: string }>)[0].content;
        const parsed = JSON.parse(rawInput) as { content: string };
        const refs = [...parsed.content.matchAll(/\[(m\d+)\]/g)].map((m) => m[1]);
        const toolRef = refs[0]!;
        const authRef = refs[1]!;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({
            status: "completed",
            output_text: [
                `错引1：用户授权见 ${toolRef}。`,
                `错引2：User authorized (${toolRef}) for deployment.`,
                `错引3：原文见 ${toolRef} 附近「你推送到远端，并更新当前正在运行的到最新版本吧」。`,
                `正引1：用户确认（${authRef}），原文见 ${authRef}「你推送到远端……最新版本吧」。`,
                `正引2：用户要求检查 ${toolRef} 的工具输出，结果 exit 0。`,
            ].join("\n"),
        }));
    });
});

test("counterexample #2: stale wrong citation in prior block summary (b32 -> b34) is scrubbed before reference and blocked from repeating", async () => {
    let sentOtherSummaries: Array<{ blockId: string; summary: string }> = [];

    await withMockSummaryServer(async (_base, externalSummary) => {
        const core = createCore();
        const config = defaultConfig(400_000) as Config & { externalSummary?: unknown };
        config.externalSummary = externalSummary;
        config.preserveRecentMessages = 0;
        config.preserveRecentTokens = 0;
        config.compress.minCompressRange = 100;

        const session = createSession(`citation-old-summary-${Math.random()}`);
        const filler = "Cutover verification output and health checks passed cleanly. ".repeat(20);
        const messages: CoreMessage[] = [
            { id: "old-tool", role: "tool", contentType: "tool-result", toolName: "bash", text: `git status clean; ${filler}` },
            { id: "user-auth", role: "user", contentType: "text", text: "你推送到远端，并更新当前正在运行的到最新版本吧" },
            { id: "b32-work", role: "assistant", contentType: "text", text: `Prepared cutover backup. ${filler}` },
            { id: "b34-work", role: "assistant", contentType: "text", text: `Completed push and cutover restart. ${filler}` },
        ];
        session.state = core.processTurn({ messages, state: session.state, config, tokenCount: 20_000, renderTags: "text-only" }).state;

        const wrongToolRef = session.state.messageRefs.byRaw["old-tool"];
        const b32Ref = session.state.messageRefs.byRaw["b32-work"];
        const b34Ref = session.state.messageRefs.byRaw["b34-work"];

        // Seed a legacy poisoned summary (exact b32 shape: cites tool ref for user authorization).
        session.state = core.applyCompression({
            messages,
            state: session.state,
            config,
            ranges: [{
                startRef: b32Ref,
                endRef: b32Ref,
                summary: `切换前取证完成（原文见 ${wrongToolRef} 附近「你推送到远端，并更新当前正在运行的到最新版本吧」）。回退材料与备份路径均已记录。`,
                topic: "b32-poisoned",
            }],
        }).state;

        const ctx: RewriteCtx = { core, config, session, messages, log: () => {} };
        const result = await applyConfiguredCompression({
            content: [{ startId: b34Ref, endId: b34Ref }],
        }, ctx);

        assert.equal(result.outcome, "applied", result.text);
        // 1. Read-only reference handed to the summarizer must already scrub the poisoned ref from b1.
        assert.equal(sentOtherSummaries.length, 1);
        assert.equal(sentOtherSummaries[0].summary.includes(wrongToolRef), false, sentOtherSummaries[0].summary);
        assert.ok(sentOtherSummaries[0].summary.includes(UNVERIFIED_REF_PLACEHOLDER));
        // 2. Even if the summarizer still echoes wrongToolRef, it is neutralized in the new block.
        const b2 = session.state.blocks.find((b) => b.blockId === "b2");
        assert.ok(b2);
        assert.equal(b2.summary.includes(wrongToolRef), false, b2.summary);
        assert.ok(b2.summary.includes(UNVERIFIED_REF_PLACEHOLDER));
    }, (_req, res, body) => {
        const rawInput = (body.input as Array<{ content: string }>)[0].content;
        const parsed = JSON.parse(rawInput) as { reference: string };
        const refObj = JSON.parse(parsed.reference) as { otherSummaries: Array<{ blockId: string; summary: string }> };
        sentOtherSummaries = refObj.otherSummaries;
        // Simulate a stubborn model still repeating the old wrong ref m00001.
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({
            status: "completed",
            output_text: "TASK AS OF THIS BLOCK（历史）：用户先前授权（m00001 附近原文「你推送到远端，并更新当前正在运行的到最新版本吧」），现已完成切换与推送。",
        }));
    });
});

test("paired citations, chronological anchors, non-attribution mentions, minor quote edits, code spans, and contentStore fallback produce zero false positives", () => {
    const messages: CoreMessage[] = [
        { id: "u4", role: "user", contentType: "text", text: "你分析评估下当前项目，看看有什么可以优化的" },
        { id: "u5", role: "user", contentType: "text", text: "你按这个方案执行吧，先做全部 12 票" },
        { id: "a9", role: "assistant", contentType: "text", text: "交付 5 波 12 票方案概览并请用户拍板 2 个决策点后开工" },
        { id: "a888", role: "assistant", contentType: "reasoning", text: "**Preserving existing stats API**\n**Designing draft preview flow**\n**Checking draft parent scope**" },
        { id: "t1056", role: "tool", contentType: "tool-result", toolName: "exec", text: "{\"chunk_id\":\"652f7f\",\"output\":\"\"}" },
        { id: "a1383", role: "assistant", contentType: "text", text: "计划 §4 闸门：上线动作须先取得用户同意；第 2 步核验宿主" },
        { id: "t1745", role: "tool", contentType: "tool-result", toolName: "bash", text: "正式宿主的管理面、阅读面、时间轴、关系网与快照网取数通过。" },
    ];
    const byRaw = { u4: "m00004", u5: "m00005", a9: "m00009", a888: "m00888", t1056: "m01056", a1383: "m01383", t1745: "m01745" };
    // Simulate an older folded tool message only surviving in contentStore (not in live `messages`).
    let store = createContentStore();
    store = storeOriginal(store, {
        ref: "m01538",
        rawId: "old-tool-1538",
        text: "remote: Repository not found.",
        kind: "shell output",
        toolName: "bash",
        tokens: 10,
        head: "remote: Repository not found.",
    });

    const facts = buildCitationMessageIndex(messages, byRaw, store);
    const maxRef = highestAllocatedMessageRef({
        m00004: "u4", m00005: "u5", m00009: "a9", m00888: "a888",
        m01056: "t1056", m01383: "a1383", m01745: "t1745", m01538: "old-tool-1538",
    }, facts);

    const safeSummary = [
        "用户先在 m00004 要求「分析评估当前项目，看看有什么可以优化的」。",
        "Open objectives: 完成全部 12 票（m00005/m00009 用户已授权「你按这个方案执行吧」）。",
        "时间锚点：用户此前在（m00009 之后）授权「你按这个方案执行吧」。",
        "### 待用户拍板（m00009 结尾）",
        "m00888 标题「Preserving existing stats API / Designing draft preview flow / Checking draft parent scope」。",
        "`{\"cell_id\":\"300\",\"yield_time_ms\":1000}` → m01056 返回空 `output`（chunk `652f7f`）。",
        "计划 §4 闸门：上线动作须先取得用户同意；m01383 核验宿主，经用户同意后重启（m01383 第 2 步）。",
        "输出（m01745，exit 0）：「正式宿主的管理面、阅读面、时间轴、关系网与快照网取数通过。」",
    ].join("\n");

    const safeCheck = verifyMessageRefCitations(safeSummary, facts, maxRef);
    assert.deepEqual(safeCheck.neutralized, []);
    assert.equal(safeCheck.summary, safeSummary);

    // Meanwhile, contentStore-only tool ref m01538 misattributed as user authorization IS caught.
    const badFromStore = "用户授权见 m01538（原文「你推送到远端，并更新当前正在运行的到最新版本吧」），以及越界号 m99999。";
    const badCheck = verifyMessageRefCitations(badFromStore, facts, maxRef);
    assert.deepEqual(badCheck.neutralized, ["m01538", "m99999"]);
    assert.equal(badCheck.summary.includes("m01538"), false);
    assert.equal(badCheck.summary.includes("m99999"), false);
});

