import type { Frame, Mode } from '../state/store';

const WS_BASE = import.meta.env.VITE_WS_BASE ?? 'ws://127.0.0.1:8000';

export interface WsStats {
  status: 'disconnected' | 'connecting' | 'connected';
  latencyMs: number;
}

export class TunnelWsClient {
  private socket: WebSocket | null = null;
  private stats: WsStats = { status: 'disconnected', latencyMs: 0 };
  private onFrame: ((frame: Frame) => void) | null = null;
  private onStats: ((stats: WsStats) => void) | null = null;
  private pingSentAt = 0;

  setFrameHandler(handler: (frame: Frame) => void): void {
    this.onFrame = handler;
  }

  setStatsHandler(handler: (stats: WsStats) => void): void {
    this.onStats = handler;
    this.onStats(this.stats);
  }

  connect(mode: Mode, scenarioId: string, sessionId?: string): void {
    this.disconnect();
    this.updateStats({ status: 'connecting' });
    const endpoint =
      mode === 'playback'
        ? `${WS_BASE}/ws/playback?scenario_id=${encodeURIComponent(scenarioId)}${sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : ''}`
        : `${WS_BASE}/ws/live`;
    this.socket = new WebSocket(endpoint);

    this.socket.onopen = () => {
      this.pingSentAt = performance.now();
      this.updateStats({ status: 'connected', latencyMs: 0 });
    };

    this.socket.onmessage = (event) => {
      const now = performance.now();
      if (this.pingSentAt > 0) {
        this.updateStats({ latencyMs: Math.max(0, now - this.pingSentAt) });
      }
      this.pingSentAt = now;

      const parsed = JSON.parse(event.data) as Frame;
      this.onFrame?.(parsed);
    };

    this.socket.onclose = () => {
      this.updateStats({ status: 'disconnected' });
      this.socket = null;
    };

    this.socket.onerror = () => {
      this.updateStats({ status: 'disconnected' });
    };
  }

  sendControl(payload: Record<string, unknown>): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    this.socket.send(JSON.stringify(payload));
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.updateStats({ status: 'disconnected' });
  }

  private updateStats(patch: Partial<WsStats>): void {
    this.stats = { ...this.stats, ...patch };
    this.onStats?.(this.stats);
  }
}
