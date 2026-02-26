export const TUNNEL_LENGTH_METERS = 1500;
export const TUBE_SEPARATION = 10;
export const LANE_WIDTH = 3.2;

export function laneCenterY(tube: number, lane: number): number {
  const tubeOffset = tube === 1 ? -TUBE_SEPARATION / 2 : TUBE_SEPARATION / 2;
  const laneOffset = lane === 1 ? LANE_WIDTH * 0.5 : -LANE_WIDTH * 0.5;
  return tubeOffset + laneOffset;
}

export function xToWorld(x: number): number {
  return x;
}
