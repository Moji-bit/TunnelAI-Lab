import * as THREE from 'three';
import type { EventState } from '../state/store';
import { laneCenterY, xToWorld } from './mapping';

export class EventsLayer {
  readonly group = new THREE.Group();
  private eventMeshes = new Map<string, THREE.Object3D>();
  private objectToEventId = new Map<string, string>();

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  update(events: EventState[]): void {
    const seen = new Set<string>();

    events.forEach((event) => {
      seen.add(event.id);
      const existing = this.eventMeshes.get(event.id);
      if (existing) {
        this.applyTransform(existing, event);
        existing.visible = event.active;
        return;
      }
      const mesh = this.createMesh(event);
      this.group.add(mesh);
      this.eventMeshes.set(event.id, mesh);
      this.objectToEventId.set(mesh.uuid, event.id);
      mesh.traverse((child: THREE.Object3D) => this.objectToEventId.set(child.uuid, event.id));
    });

    for (const [id, mesh] of this.eventMeshes) {
      if (!seen.has(id)) {
        this.group.remove(mesh);
        this.eventMeshes.delete(id);
        this.objectToEventId.delete(mesh.uuid);
        mesh.traverse((child: THREE.Object3D) => this.objectToEventId.delete(child.uuid));
      }
    }
  }

  resolveEventId(object: THREE.Object3D): string | null {
    return this.objectToEventId.get(object.uuid) ?? null;
  }

  private createMesh(event: EventState): THREE.Object3D {
    if (event.type === 'incident') {
      const incidentGroup = new THREE.Group();
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(0.35, 0.8, 16),
        new THREE.MeshBasicMaterial({ color: 0xffcf40, transparent: true, opacity: 0.9 }),
      );
      cone.rotation.x = Math.PI;
      cone.position.z = 0.7;
      incidentGroup.add(cone);
      this.applyTransform(incidentGroup, event);
      return incidentGroup;
    }

    if (event.type === 'fire') {
      const fire = new THREE.Mesh(
        new THREE.SphereGeometry(1.2, 20, 20),
        new THREE.MeshBasicMaterial({ color: 0xff5a1f, transparent: true, opacity: 0.8 }),
      );
      this.applyTransform(fire, event);
      fire.position.z = 1.2;
      return fire;
    }

    if (event.type === 'smoke') {
      const width = Math.max(4, event.x1 - event.x0);
      const smoke = new THREE.Mesh(
        new THREE.PlaneGeometry(width, 3),
        new THREE.MeshBasicMaterial({ color: 0x9197a3, transparent: true, opacity: 0.45 }),
      );
      smoke.rotation.y = Math.PI / 2;
      this.applyTransform(smoke, event);
      smoke.position.z = 2.2;
      return smoke;
    }

    const width = Math.max(2, event.x1 - event.x0);
    const color = event.type === 'queue' ? 0xff642e : 0x111111;
    const zone = new THREE.Mesh(
      new THREE.PlaneGeometry(width, 0.9),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: event.type === 'queue' ? 0.45 : 0.65 }),
    );
    zone.rotation.x = -Math.PI / 2;
    this.applyTransform(zone, event);
    return zone;
  }

  private applyTransform(object: THREE.Object3D, event: EventState): void {
    const centerX = (event.x0 + event.x1) / 2;
    object.position.set(xToWorld(centerX), laneCenterY(event.tube, event.lane), 0.05);
  }
}
