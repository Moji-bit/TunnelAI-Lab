import * as THREE from 'three';
import { LANE_WIDTH, TUNNEL_LENGTH_METERS, TUBE_SEPARATION, laneCenterY } from './mapping';

export interface TunnelParts {
  group: THREE.Group;
  laneMarkings: THREE.Group;
}

export function createTunnelMesh(): TunnelParts {
  const group = new THREE.Group();
  const laneMarkings = new THREE.Group();

  const roadMaterial = new THREE.MeshStandardMaterial({ color: 0x2f3645, roughness: 0.95, metalness: 0.05 });
  const wallMaterial = new THREE.MeshStandardMaterial({ color: 0x3f4f72, roughness: 0.8, metalness: 0.1 });
  const markMaterial = new THREE.MeshBasicMaterial({ color: 0xe5edf8 });

  for (const tube of [1, 2]) {
    const road = new THREE.Mesh(new THREE.PlaneGeometry(TUNNEL_LENGTH_METERS, 2 * LANE_WIDTH + 0.5), roadMaterial);
    road.rotation.x = -Math.PI / 2;
    road.position.set(TUNNEL_LENGTH_METERS / 2, tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2, 0);
    group.add(road);

    for (const lane of [1, 2]) {
      const laneCenter = laneCenterY(tube, lane);
      const marker = new THREE.Mesh(new THREE.PlaneGeometry(TUNNEL_LENGTH_METERS, 0.04), markMaterial);
      marker.rotation.x = -Math.PI / 2;
      marker.position.set(TUNNEL_LENGTH_METERS / 2, laneCenter, 0.01);
      laneMarkings.add(marker);
    }

    const leftWall = new THREE.Mesh(new THREE.BoxGeometry(TUNNEL_LENGTH_METERS, 0.08, 0.35), wallMaterial);
    leftWall.position.set(
      TUNNEL_LENGTH_METERS / 2,
      (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) + (LANE_WIDTH + 0.25),
      0.175,
    );

    const rightWall = leftWall.clone();
    rightWall.position.y =
      (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) - (LANE_WIDTH + 0.25);
    group.add(leftWall, rightWall);
  }

  group.add(laneMarkings);
  return { group, laneMarkings };
}
