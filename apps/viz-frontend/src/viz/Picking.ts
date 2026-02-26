import * as THREE from 'three';

export function pointerToNdc(event: PointerEvent, element: HTMLElement): THREE.Vector2 {
  const rect = element.getBoundingClientRect();
  return new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
}
