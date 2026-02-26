import * as THREE from 'three';
import { LANE_WIDTH, TUNNEL_LENGTH_METERS, TUBE_SEPARATION, laneCenterY } from './mapping';

export interface TunnelParts {
  group: THREE.Group;
  shellGroup: THREE.Group;
  equipmentGroup: THREE.Group;
}

export function createTunnelMesh(): TunnelParts {
  const group = new THREE.Group();
  const shellGroup = new THREE.Group();
  const equipmentGroup = new THREE.Group();

  const tunnelRadius = LANE_WIDTH * 2.35;
  const shellThickness = 0.7;

  const shellMaterial = new THREE.MeshStandardMaterial({
    color: 0x54698f,
    roughness: 0.85,
    metalness: 0.05,
    transparent: true,
    opacity: 0.26,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const liningMaterial = new THREE.MeshStandardMaterial({
    color: 0x263452,
    roughness: 0.92,
    metalness: 0.03,
    transparent: true,
    opacity: 0.2,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const roadMaterial = new THREE.MeshStandardMaterial({ color: 0x2f3645, roughness: 0.95, metalness: 0.05 });
  const markMaterial = new THREE.MeshBasicMaterial({ color: 0xe5edf8 });

  const nicheMaterial = new THREE.MeshStandardMaterial({ color: 0x344a6a, roughness: 0.8, metalness: 0.05 });
  const cameraMaterial = new THREE.MeshStandardMaterial({ color: 0xd7e0f0, roughness: 0.25, metalness: 0.55 });
  const fanMaterial = new THREE.MeshStandardMaterial({ color: 0x7f8ea8, roughness: 0.5, metalness: 0.5 });
  const hydrantMaterial = new THREE.MeshStandardMaterial({ color: 0xe53f3f, roughness: 0.45, metalness: 0.15 });
  const crossPassageMaterial = new THREE.MeshStandardMaterial({ color: 0x6c7691, roughness: 0.7, metalness: 0.08 });
  const doorMaterial = new THREE.MeshStandardMaterial({ color: 0xaeb6c7, roughness: 0.5, metalness: 0.35 });

  for (const tube of [1, 2]) {
    const tubeCenterX = tube === 1 ? -TUBE_SEPARATION / 2 : TUBE_SEPARATION / 2;

    const outerShell = new THREE.Mesh(
      new THREE.CylinderGeometry(tunnelRadius + shellThickness, tunnelRadius + shellThickness, TUNNEL_LENGTH_METERS, 48),
      shellMaterial,
    );
    outerShell.rotation.x = Math.PI / 2;
    outerShell.position.set(tubeCenterX, tunnelRadius - 0.2, TUNNEL_LENGTH_METERS / 2);
    shellGroup.add(outerShell);

    const innerLining = new THREE.Mesh(
      new THREE.CylinderGeometry(tunnelRadius, tunnelRadius, TUNNEL_LENGTH_METERS, 48),
      liningMaterial,
    );
    innerLining.rotation.x = Math.PI / 2;
    innerLining.position.copy(outerShell.position);
    shellGroup.add(innerLining);

    const road = new THREE.Mesh(new THREE.PlaneGeometry(2 * LANE_WIDTH + 3.2, TUNNEL_LENGTH_METERS), roadMaterial);
    road.rotation.x = -Math.PI / 2;
    road.position.set(tubeCenterX, 0.01, TUNNEL_LENGTH_METERS / 2);
    equipmentGroup.add(road);

    for (const lane of [1, 2]) {
      const laneCenterX = laneCenterY(tube, lane);
      const marker = new THREE.Mesh(new THREE.PlaneGeometry(0.16, TUNNEL_LENGTH_METERS), markMaterial);
      marker.rotation.x = -Math.PI / 2;
      marker.position.set(laneCenterX, 0.03, TUNNEL_LENGTH_METERS / 2);
      equipmentGroup.add(marker);
    }

    for (let z = 60; z < TUNNEL_LENGTH_METERS; z += 80) {
      const niche = new THREE.Mesh(new THREE.BoxGeometry(2.4, 2.1, 2.2), nicheMaterial);
      niche.position.set(tubeCenterX + LANE_WIDTH * 1.45, 1.5, z);
      equipmentGroup.add(niche);

      const hydrant = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.45, 1.2), hydrantMaterial);
      hydrant.position.set(niche.position.x, 1.0, z);
      equipmentGroup.add(hydrant);
    }

    for (let z = 90; z < TUNNEL_LENGTH_METERS; z += 120) {
      const camera = new THREE.Mesh(new THREE.SphereGeometry(0.35, 12, 12), cameraMaterial);
      camera.position.set(tubeCenterX + LANE_WIDTH * 1.05, tunnelRadius * 1.25, z);
      equipmentGroup.add(camera);
    }

    for (let z = 110; z < TUNNEL_LENGTH_METERS; z += 160) {
      const fan = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 3.8, 12), fanMaterial);
      fan.rotation.z = Math.PI / 2;
      fan.position.set(tubeCenterX, tunnelRadius * 1.45, z);
      equipmentGroup.add(fan);
    }
  }

  for (let z = 180; z < TUNNEL_LENGTH_METERS; z += 300) {
    const passage = new THREE.Mesh(new THREE.BoxGeometry(TUBE_SEPARATION - 2.6, 3.2, 7.0), crossPassageMaterial);
    passage.position.set(0, 1.8, z);
    equipmentGroup.add(passage);

    const doorA = new THREE.Mesh(new THREE.BoxGeometry(0.2, 2.2, 1.2), doorMaterial);
    doorA.position.set(-TUBE_SEPARATION / 2 + 1.3, 1.4, z);
    const doorB = doorA.clone();
    doorB.position.x = TUBE_SEPARATION / 2 - 1.3;
    equipmentGroup.add(doorA, doorB);
  }

  group.add(shellGroup, equipmentGroup);
  return { group, shellGroup, equipmentGroup };
}
