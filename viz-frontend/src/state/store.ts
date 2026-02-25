export type Mode = 'playback' | 'live';

export interface VehicleState {
  id: string;
  tube: number;
  lane: number;
  x: number;
  v: number;
  a: number;
  type: 'car' | 'truck' | 'emergency';
}

export interface EventState {
  id: string;
  type: 'incident' | 'queue' | 'closure';
  tube: number;
  lane: number;
  x0: number;
  x1: number;
  severity?: number;
  active: boolean;
}

export interface Frame {
  schema: string;
  scenario_id: string;
  mode: Mode;
  t: number;
  dt: number;
  vehicles: VehicleState[];
  events: EventState[];
}

export interface ScenarioSummary {
  id: string;
  name: string;
  duration: number;
  dt: number;
  tubes: number;
  lanes: number;
  tags: string[];
}

export interface LayerVisibility {
  vehicles: boolean;
  events: boolean;
  laneMarkings: boolean;
  debugHud: boolean;
}

export type Selection =
  | { kind: 'vehicle'; data: VehicleState }
  | { kind: 'event'; data: EventState }
  | null;
