import type { WSInbound, WSOutbound } from "./protocol";
import { normalizeInbound } from "./normalize";

export type WSStatus = "connecting" | "open" | "closed";

export interface WSClientOpts {
  url?: string;
  onMessage: (msg: WSInbound) => void;
  onStatus: (status: WSStatus) => void;
  /** ms; capped at 5000 */
  initialBackoff?: number;
  /** if true, do not actually try to connect — used by tests */
  disabled?: boolean;
}

export class WSClient {
  private socket: WebSocket | null = null;
  private url: string;
  private opts: WSClientOpts;
  private backoff = 500;
  private destroyed = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(opts: WSClientOpts) {
    this.url = opts.url ?? "ws://localhost:8765";
    this.opts = opts;
    if (!opts.disabled) this.connect();
  }

  private connect() {
    if (this.destroyed) return;
    this.opts.onStatus("connecting");
    try {
      this.socket = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.socket.onopen = () => {
      this.backoff = 500;
      this.opts.onStatus("open");
    };
    this.socket.onmessage = (ev) => {
      try {
        const data = typeof ev.data === "string" ? ev.data : "";
        if (!data) return;
        // Bridge may send multiple line-delimited JSONs in one frame
        for (const line of data.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          const raw = JSON.parse(trimmed);
          const normalized = normalizeInbound(raw);
          if (normalized) this.opts.onMessage(normalized);
        }
      } catch {
        // ignore parse errors
      }
    };
    this.socket.onerror = () => {
      // onclose will fire next
    };
    this.socket.onclose = () => {
      this.opts.onStatus("closed");
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect() {
    if (this.destroyed) return;
    if (this.reconnectTimer) return;
    const delay = Math.min(this.backoff, 5000);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.backoff = Math.min(this.backoff * 2, 5000);
      this.connect();
    }, delay);
  }

  send(msg: WSOutbound) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(msg));
    }
  }

  close() {
    this.destroyed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.socket?.close();
  }
}
