# burv

Bevy demo: walk around a procedurally generated voxel terrain rendered via octree-driven ray marching, with a small particle-based water simulation rendered in the same ray marcher.

## Run

```bash
cargo run --release
```

Controls:
- `W/A/S/D` move
- Mouse look (click to capture mouse)
- `Space` / `Left Shift` up/down
- `Esc` release mouse

## Notes

- Terrain is generated on the CPU into a voxel grid and compacted into an octree.
- Rendering is a full-screen quad with a WGSL fragment shader that ray marches the octree (empty-space skipping) and a simple water SDF built from particles.
