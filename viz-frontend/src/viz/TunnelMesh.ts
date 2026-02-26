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
  const ventilationMaterial = new THREE.MeshStandardMaterial({ color: 0x7f8ea8, roughness: 0.5, metalness: 0.5 });
  const cameraMaterial = new THREE.MeshStandardMaterial({ color: 0xcfd8e8, roughness: 0.25, metalness: 0.6 });
  const extinguisherMaterial = new THREE.MeshStandardMaterial({ color: 0xe53f3f, roughness: 0.45, metalness: 0.15 });
  const nicheMaterial = new THREE.MeshStandardMaterial({ color: 0x2d405f, roughness: 0.8, metalness: 0.05 });
  const crossPassageMaterial = new THREE.MeshStandardMaterial({ color: 0x69758d, roughness: 0.7, metalness: 0.1 });

  for (const tube of [1, 2]) {
    const road = new THREE.Mesh(new THREE.PlaneGeometry(TUNNEL_LENGTH_METERS, 2 * LANE_WIDTH + 3.2), roadMaterial);
    road.rotation.x = -Math.PI / 2;
    road.position.set(TUNNEL_LENGTH_METERS / 2, tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2, 0);
    group.add(road);

    for (const lane of [1, 2]) {
      const laneCenter = laneCenterY(tube, lane);
      const marker = new THREE.Mesh(new THREE.PlaneGeometry(TUNNEL_LENGTH_METERS, 0.16), markMaterial);
      marker.rotation.x = -Math.PI / 2;
      marker.position.set(TUNNEL_LENGTH_METERS / 2, laneCenter, 0.01);
      laneMarkings.add(marker);
    }

    const leftWall = new THREE.Mesh(new THREE.BoxGeometry(TUNNEL_LENGTH_METERS, 0.7, 4.2), wallMaterial);
    leftWall.position.set(
      TUNNEL_LENGTH_METERS / 2,
      (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) + (LANE_WIDTH + 1.2),
      2.1,
    );

    const rightWall = leftWall.clone();
    rightWall.position.y =
      (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) - (LANE_WIDTH + 1.2);
    group.add(leftWall, rightWall);

    for (let x = 70; x < TUNNEL_LENGTH_METERS; x += 120) {
      const ventilation = new THREE.Mesh(new THREE.BoxGeometry(12, 2.2, 1.1), ventilationMaterial);
      ventilation.position.set(x, tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2, 6.2);
      group.add(ventilation);

      const camera = new THREE.Mesh(new THREE.SphereGeometry(0.8, 16, 16), cameraMaterial);
      camera.position.set(x + 20, (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) + 5.3, 3.6);
      group.add(camera);

      const extinguisher = new THREE.Mesh(new THREE.BoxGeometry(1, 0.9, 3.2), extinguisherMaterial);
      extinguisher.position.set(x + 35, (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) - 5.7, 1.7);
      group.add(extinguisher);

      const emergencyNiche = new THREE.Mesh(new THREE.BoxGeometry(5.2, 2.4, 4.2), nicheMaterial);
      emergencyNiche.position.set(x + 46, (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) + 6.1, 2.1);
      group.add(emergencyNiche);
    }
  }

  for (let x = 150; x < TUNNEL_LENGTH_METERS; x += 260) {
    const crossPassage = new THREE.Mesh(new THREE.BoxGeometry(8, TUBE_SEPARATION - 2.8, 5.5), crossPassageMaterial);
    crossPassage.position.set(x, 0, 2.7);
    group.add(crossPassage);
  }

  group.add(laneMarkings);
  return { group, laneMarkings };
}
