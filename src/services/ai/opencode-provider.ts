import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { generateText, Output, tool } from "ai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import type { ZodType } from "zod";
import { log } from "../logger.js";

type OAuthAuth = { type: "oauth"; refresh: string; access: string; expires: number };
type ApiAuth = { type: "api"; key: string };
type Auth = OAuthAuth | ApiAuth;

// --- State (set from plugin init in index.ts, Task 4) ---
let _statePath: string | null = null;
let _connectedProviders: string[] = [];

export function setStatePath(path: string): void {
  _statePath = path;
}

export function getStatePath(): string {
  if (!_statePath) {
    throw new Error("opencode state path not initialized. Plugin may not be fully started.");
  }
  return _statePath;
}

export function setConnectedProviders(providers: string[]): void {
  _connectedProviders = providers;
}

export function isProviderConnected(providerName: string): boolean {
  return _connectedProviders.includes(providerName);
}

// --- Auth ---
function findAuthJsonPath(statePath: string): string | undefined {
  const localDir = dirname(statePath);
  const baseDir = dirname(localDir);
  const candidates = [
    join(statePath, "auth.json"),
    join(localDir, "share", "opencode", "auth.json"),
    join(baseDir, "share", "opencode", "auth.json"),
    join(statePath.replace("/state/", "/share/"), "auth.json"),
    join(statePath.replace("\\state\\", "\\share\\"), "auth.json"),
  ];
  return candidates.find(existsSync);
}

export function readOpencodeAuth(statePath: string, providerName: string): Auth {
  const authPath = findAuthJsonPath(statePath);
  let raw: string | undefined;
  if (authPath) {
    try {
      raw = readFileSync(authPath, "utf-8");
    } catch {}
  }
  if (!raw || !authPath) {
    throw new Error(
      `opencode auth.json not found at ${authPath ?? statePath}. Is opencode authenticated?`
    );
  }
  let parsed: Record<string, Auth>;
  try {
    parsed = JSON.parse(raw) as Record<string, Auth>;
  } catch {
    throw new Error(`Failed to read opencode auth.json: invalid JSON`);
  }
  const auth = parsed[providerName];
  if (!auth) {
    const connected = Object.keys(parsed).join(", ") || "none";
    throw new Error(
      `Provider '${providerName}' not found in opencode auth.json. Connected providers: ${connected}`
    );
  }
  return auth;
}

// --- OAuth Fetch ---
const OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token";
const OAUTH_REQUIRED_BETAS = ["oauth-2025-04-20", "interleaved-thinking-2025-05-14"];
const MCP_TOOL_PREFIX = "mcp_";

export function createOAuthFetch(
  statePath: string,
  providerName: string
): (input: string | Request | URL, init?: RequestInit) => Promise<Response> {
  return async (input: string | Request | URL, init?: RequestInit): Promise<Response> => {
    let auth = readOpencodeAuth(statePath, providerName) as OAuthAuth;

    // Refresh token if expired
    if (!auth.access || auth.expires < Date.now()) {
      const refreshResponse = await fetch(OAUTH_TOKEN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grant_type: "refresh_token",
          refresh_token: auth.refresh,
          client_id: OAUTH_CLIENT_ID,
        }),
      });
      if (!refreshResponse.ok) {
        throw new Error(`OAuth token refresh failed: ${refreshResponse.status}`);
      }
      const json = (await refreshResponse.json()) as {
        access_token: string;
        refresh_token: string;
        expires_in: number;
      };
      auth = {
        type: "oauth",
        refresh: json.refresh_token,
        access: json.access_token,
        expires: Date.now() + json.expires_in * 1000,
      };

      const authPath = findAuthJsonPath(statePath);
      if (authPath) {
        try {
          const allAuth = JSON.parse(readFileSync(authPath, "utf-8")) as Record<string, Auth>;
          allAuth[providerName] = auth;
          writeFileSync(authPath, JSON.stringify(allAuth));
        } catch {}
      }
    }

    // Build headers
    const requestInit = init ?? {};
    const requestHeaders = new Headers();
    if (input instanceof Request) {
      input.headers.forEach((value, key) => requestHeaders.set(key, value));
    }
    if (requestInit.headers) {
      if (requestInit.headers instanceof Headers) {
        requestInit.headers.forEach((value, key) => requestHeaders.set(key, value));
      } else if (Array.isArray(requestInit.headers)) {
        for (const pair of requestInit.headers) {
          const [key, value] = pair as [string, string];
          if (typeof value !== "undefined") requestHeaders.set(key, value);
        }
      } else {
        for (const [key, value] of Object.entries(requestInit.headers as Record<string, string>)) {
          if (typeof value !== "undefined") requestHeaders.set(key, String(value));
        }
      }
    }

    // Merge beta headers
    const incomingBeta = requestHeaders.get("anthropic-beta") ?? "";
    const incomingBetas = incomingBeta
      .split(",")
      .map((b) => b.trim())
      .filter(Boolean);
    const mergedBetas = [...new Set([...OAUTH_REQUIRED_BETAS, ...incomingBetas])].join(",");

    requestHeaders.set("authorization", `Bearer ${auth.access}`);
    requestHeaders.set("anthropic-beta", mergedBetas);
    requestHeaders.set("user-agent", "claude-cli/2.1.2 (external, cli)");
    requestHeaders.delete("x-api-key");

    // Prefix tool names in request body
    let body = requestInit.body;
    if (body && typeof body === "string") {
      try {
        const parsed = JSON.parse(body) as Record<string, unknown>;
        if (parsed.tools && Array.isArray(parsed.tools)) {
          parsed.tools = (parsed.tools as Array<Record<string, unknown>>).map((tool) => ({
            ...tool,
            name: tool.name ? `${MCP_TOOL_PREFIX}${tool.name as string}` : tool.name,
          }));
        }
        if (parsed.messages && Array.isArray(parsed.messages)) {
          parsed.messages = (parsed.messages as Array<Record<string, unknown>>).map((msg) => {
            if (msg.content && Array.isArray(msg.content)) {
              msg.content = (msg.content as Array<Record<string, unknown>>).map((block) => {
                if (block.type === "tool_use" && block.name) {
                  return { ...block, name: `${MCP_TOOL_PREFIX}${block.name as string}` };
                }
                return block;
              });
            }
            return msg;
          });
        }
        body = JSON.stringify(parsed);
      } catch {}
    }

    // Modify URL: add ?beta=true to /v1/messages
    let requestInput: string | Request | URL = input;
    try {
      let requestUrl: URL | null = null;
      if (typeof input === "string" || input instanceof URL) {
        requestUrl = new URL(input.toString());
      } else if (input instanceof Request) {
        requestUrl = new URL(input.url);
      }
      if (requestUrl?.pathname === "/v1/messages" && !requestUrl.searchParams.has("beta")) {
        requestUrl.searchParams.set("beta", "true");
        requestInput =
          input instanceof Request ? new Request(requestUrl.toString(), input) : requestUrl;
      }
    } catch {}

    const response = await fetch(requestInput, { ...requestInit, body, headers: requestHeaders });

    // Strip mcp_ prefix from tool names in streaming response
    if (response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async pull(controller) {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            return;
          }
          let text = decoder.decode(value, { stream: true });
          text = text.replace(/"name"\s*:\s*"mcp_([^"]+)"/g, '"name": "$1"');
          controller.enqueue(encoder.encode(text));
        },
      });
      return new Response(stream, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    return response;
  };
}

// --- GitHub Copilot Fetch ---
// Protocol verified from anomalyco/opencode source
// (packages/opencode/src/plugin/github-copilot/copilot.ts):
//   - The OAuth `refresh` field IS the GitHub OAuth token (gho_*); used directly.
//   - Bearer it against https://api.githubcopilot.com (OpenAI-compatible).
//   - Required headers: Authorization, User-Agent, Openai-Intent, x-initiator.
const COPILOT_BASE_URL = "https://api.githubcopilot.com";
const COPILOT_USER_AGENT = "opencode-mem-plugin";

function createCopilotFetch(
  statePath: string
): (input: string | Request | URL, init?: RequestInit) => Promise<Response> {
  return async (input: string | Request | URL, init?: RequestInit): Promise<Response> => {
    const auth = readOpencodeAuth(statePath, "github-copilot") as OAuthAuth;

    const requestInit = init ?? {};
    const headers = new Headers();
    if (input instanceof Request) {
      input.headers.forEach((v, k) => headers.set(k, v));
    }
    if (requestInit.headers) {
      if (requestInit.headers instanceof Headers) {
        requestInit.headers.forEach((v, k) => headers.set(k, v));
      } else if (Array.isArray(requestInit.headers)) {
        for (const [k, v] of requestInit.headers as [string, string][]) {
          if (typeof v !== "undefined") headers.set(k, v);
        }
      } else {
        for (const [k, v] of Object.entries(requestInit.headers as Record<string, string>)) {
          if (typeof v !== "undefined") headers.set(k, String(v));
        }
      }
    }

    headers.set("Authorization", `Bearer ${auth.refresh}`);
    headers.set("User-Agent", COPILOT_USER_AGENT);
    headers.set("Openai-Intent", "conversation-edits");
    headers.set("x-initiator", "agent");
    // ai-sdk-openai may inject these; copilot endpoint rejects them
    headers.delete("x-api-key");
    headers.delete("openai-organization");

    const response = await fetch(input, { ...requestInit, headers });

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      return response;
    }

    try {
      const data = await response.json();
      if (data && typeof data === "object" && Array.isArray((data as any).choices)) {
        (data as any).choices = (data as any).choices.map((choice: any, index: number) => ({
          index,
          ...choice,
        }));
        return new Response(JSON.stringify(data), {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }
    } catch {
      return response;
    }

    return response;
  };
}

// --- Provider ---
export function createOpencodeAIProvider(providerName: string, auth: Auth, statePath?: string) {
  if (providerName === "anthropic") {
    if (auth.type === "oauth") {
      if (!statePath) throw new Error("statePath is required for OAuth authentication");
      return createAnthropic({
        apiKey: "",
        fetch: createOAuthFetch(statePath, providerName) as unknown as typeof globalThis.fetch,
      });
    }
    return createAnthropic({ apiKey: auth.key });
  }
  if (providerName === "openai") {
    if (auth.type === "oauth") {
      throw new Error("OpenAI does not support OAuth authentication. Use an API key instead.");
    }
    return createOpenAI({ apiKey: auth.key });
  }
  if (providerName === "github-copilot") {
    if (auth.type !== "oauth") {
      throw new Error("github-copilot requires OAuth authentication via opencode auth login.");
    }
    if (!statePath) throw new Error("statePath is required for github-copilot");
    return createOpenAI({
      apiKey: "",
      baseURL: COPILOT_BASE_URL,
      fetch: createCopilotFetch(statePath) as unknown as typeof globalThis.fetch,
    });
  }
  throw new Error(
    `Unsupported opencode provider: '${providerName}'. Supported providers: anthropic, openai, github-copilot`
  );
}

// --- Structured Output ---
export async function generateStructuredOutput<T>(options: {
  providerName: string;
  modelId: string;
  statePath: string;
  systemPrompt: string;
  userPrompt: string;
  schema: ZodType<T>;
  temperature?: number;
}): Promise<T> {
  const auth = readOpencodeAuth(options.statePath, options.providerName);
  const provider = createOpencodeAIProvider(options.providerName, auth, options.statePath);

  // github-copilot: use plain chat completions + manual JSON parsing.
  // Output.object / Responses API are not supported by the Copilot endpoint.
  // Also strip any "use the X tool" instructions since we don't pass tools.
  if (options.providerName === "github-copilot") {
    const model = (provider as ReturnType<typeof createOpenAI>).chat(options.modelId);
    // Remove tool-call instructions from system prompt; replace with direct JSON instruction.
    const cleanedSystem = options.systemPrompt
      .replace(/Use the \S+ tool to [^\n.]+[.\n]?/gi, "")
      .trim();

    try {
      const toolResult = await generateText({
        model,
        system:
          cleanedSystem +
          "\n\nReturn the final structured result by calling the submit_structured_output tool.",
        prompt: options.userPrompt,
        temperature: options.temperature ?? 0.3,
        tools: {
          submit_structured_output: tool({
            description: "Submit the final structured output",
            inputSchema: options.schema as any,
          }),
        },
      });

      log("Copilot tool-call response", {
        modelId: options.modelId,
        toolCalls: toolResult.toolCalls,
      });

      if (toolResult.toolCalls.length > 0) {
        const firstToolCall = toolResult.toolCalls[0] as { input?: unknown };
        const toolParseResult = options.schema.safeParse(firstToolCall.input);
        if (toolParseResult.success) {
          return toolParseResult.data as T;
        }
      }
    } catch (error) {
      log("Copilot tool-call path failed, falling back to JSON", {
        modelId: options.modelId,
        error: String(error),
      });
    }

    const result = await generateText({
      model,
      system:
        cleanedSystem +
        "\n\nYou MUST respond with a single valid JSON object only. No markdown fences, no explanation, no tool calls.",
      prompt: options.userPrompt,
      temperature: options.temperature ?? 0.3,
    });
    const text = result.text.trim();
    log("Copilot raw text response", {
      modelId: options.modelId,
      preview: text.slice(0, 400),
    });
    // Strip markdown code fences if present
    const json = text
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```\s*$/, "")
      .trim();
    let parsed: unknown;
    try {
      parsed = JSON.parse(json);
    } catch (e) {
      throw new Error(`Failed to parse JSON from copilot response: ${text.slice(0, 200)}`);
    }
    // Coerce missing array fields to [] so zod doesn't reject a mostly-valid response
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const shape = (options.schema as any)._def?.shape;
      const shapeObj = typeof shape === "function" ? shape() : shape;
      if (shapeObj) {
        for (const [key, fieldDef] of Object.entries(shapeObj)) {
          const defType = (fieldDef as any)?._def?.type ?? (fieldDef as any)?._def?.typeName;
          if (defType === "array" || defType === "ZodArray") {
            if ((parsed as any)[key] === undefined) {
              (parsed as any)[key] = [];
            }
          }
        }
      }
    }
    log("Copilot parsed JSON response", {
      modelId: options.modelId,
      parsed,
    });
    const parseResult = options.schema.safeParse(parsed);
    if (!parseResult.success) {
      throw new Error(
        `Copilot JSON schema mismatch. Raw: ${JSON.stringify(parsed).slice(0, 400)}\nErrors: ${JSON.stringify(parseResult.error.issues)}`
      );
    }
    return parseResult.data as T;
  }

  // anthropic / openai: use ai-sdk Output.object for structured output
  const result = await generateText({
    model: provider(options.modelId),
    system: options.systemPrompt,
    prompt: options.userPrompt,
    output: Output.object({ schema: options.schema }),
    temperature: options.temperature ?? 0.3,
  });
  return result.output as T;
}
