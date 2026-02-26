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
  type: 'incident' | 'queue' | 'closure' | 'fire' | 'smoke';
  tube: number;
  lane: number;
  x0: number;
  x1: number;
  severity?: number;
  active: boolean;
}

export interface ActuatorState {
  id: string;
  type: 'fan' | 'vms';
  zone: number;
  state: 'off' | 'on' | 'warning';
  value: number;
}

export interface TimebaseState {
  t: number;
  dt: number;
  paused: boolean;
  speed_factor: number;
}

export interface Frame {
  schema: string;
  scenario_id: string;
  mode: Mode;
  t: number;
  dt: number;
  timebase: TimebaseState;
  vehicles: VehicleState[];
  events: EventState[];
  actuators: ActuatorState[];
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
