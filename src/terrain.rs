#![allow(dead_code)]

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

pub const WORLD_SIZE: u32 = 128;
pub const WORLD_MIN: Vec3 = Vec3::new(-(WORLD_SIZE as f32) * 0.5, 0.0, -(WORLD_SIZE as f32) * 0.5);

pub const NODE_KIND_INTERNAL: u32 = 0;
pub const NODE_KIND_EMPTY: u32 = 1;
pub const NODE_KIND_SOLID: u32 = 2;

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct GpuNode {
    pub children0: UVec4,
    pub children1: UVec4,
    pub kind: u32,
    pub _pad: UVec3,
}

impl GpuNode {
    fn leaf(kind: u32) -> Self {
        Self {
            children0: UVec4::ZERO,
            children1: UVec4::ZERO,
            kind,
            _pad: UVec3::ZERO,
        }
    }

    fn internal(children: [u32; 8]) -> Self {
        Self {
            children0: UVec4::new(children[0], children[1], children[2], children[3]),
            children1: UVec4::new(children[4], children[5], children[6], children[7]),
            kind: NODE_KIND_INTERNAL,
            _pad: UVec3::ZERO,
        }
    }
}

#[derive(Resource, Clone)]
pub struct Terrain {
    pub world_min: Vec3,
    pub world_size: f32,
    pub max_depth: u32,
    pub root_index: u32,
    pub nodes: Vec<GpuNode>,
}

impl Terrain {
    pub fn generate() -> Self {
        let size = WORLD_SIZE as usize;
        let mut voxels = vec![0u8; size * size * size];

        for z in 0..size {
            for x in 0..size {
                let wx = WORLD_MIN.x + x as f32;
                let wz = WORLD_MIN.z + z as f32;
                let h = terrain_height(wx, wz)
                    .clamp(0.0, (WORLD_SIZE - 1) as f32)
                    .floor() as i32;
                for y in 0..size {
                    let solid = (y as i32) <= h;
                    voxels[voxel_index(x as u32, y as u32, z as u32)] = solid as u8;
                }
            }
        }

        let mut nodes = Vec::<GpuNode>::new();
        let root_index = build_node(&mut nodes, &voxels, UVec3::ZERO, WORLD_SIZE);
        let max_depth = WORLD_SIZE.trailing_zeros();

        Self {
            world_min: WORLD_MIN,
            world_size: WORLD_SIZE as f32,
            max_depth,
            root_index,
            nodes,
        }
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        terrain_height(x, z)
    }
}

fn voxel_index(x: u32, y: u32, z: u32) -> usize {
    let s = WORLD_SIZE as usize;
    x as usize + s * (y as usize + s * z as usize)
}

fn build_node(nodes: &mut Vec<GpuNode>, voxels: &[u8], min: UVec3, size: u32) -> u32 {
    if size == 1 {
        let solid = voxels[voxel_index(min.x, min.y, min.z)] != 0;
        let kind = if solid {
            NODE_KIND_SOLID
        } else {
            NODE_KIND_EMPTY
        };
        let idx = nodes.len() as u32;
        nodes.push(GpuNode::leaf(kind));
        return idx;
    }

    let half = size / 2;
    let start_len = nodes.len();

    let mut children = [0u32; 8];
    for child in 0..8 {
        let offset = UVec3::new(
            ((child & 1) as u32) * half,
            (((child >> 1) & 1) as u32) * half,
            (((child >> 2) & 1) as u32) * half,
        );
        children[child] = build_node(nodes, voxels, min + offset, half);
    }

    let c0 = nodes[children[0] as usize].kind;
    let can_collapse = c0 != NODE_KIND_INTERNAL
        && children.iter().all(|&ci| {
            nodes[ci as usize].kind == c0 && nodes[ci as usize].kind != NODE_KIND_INTERNAL
        });

    if can_collapse {
        nodes.truncate(start_len);
        let idx = nodes.len() as u32;
        nodes.push(GpuNode::leaf(c0));
        return idx;
    }

    let idx = nodes.len() as u32;
    nodes.push(GpuNode::internal(children));
    idx
}

fn terrain_height(x: f32, z: f32) -> f32 {
    // A blocky heightfield built from cheap value-noise FBM, with a small basin near the origin.
    let n = fbm2(x, z);
    let mut h = 18.0 + (n * 2.0 - 1.0) * 14.0;

    // Carve a shallow bowl for the water to pool in.
    let basin_r = Vec2::new(x, z).length();
    let basin = (1.0 - (basin_r / 22.0)).clamp(0.0, 1.0);
    h -= basin * 6.0;

    h.clamp(1.0, (WORLD_SIZE - 2) as f32).floor()
}

fn fbm2(x: f32, z: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 0.05;
    let mut norm = 0.0;
    for i in 0..5 {
        sum += value_noise2(x * freq, z * freq, 1337u32.wrapping_add(i as u32 * 1013)) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

fn value_noise2(x: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let xf = x - x0 as f32;
    let zf = z - z0 as f32;

    let u = smoothstep(xf);
    let v = smoothstep(zf);

    let v00 = rand2(x0, z0, seed);
    let v10 = rand2(x0 + 1, z0, seed);
    let v01 = rand2(x0, z0 + 1, seed);
    let v11 = rand2(x0 + 1, z0 + 1, seed);

    let a = lerp(v00, v10, u);
    let b = lerp(v01, v11, u);
    lerp(a, b, v)
}

fn rand2(ix: i32, iz: i32, seed: u32) -> f32 {
    let mut h = seed;
    h ^= (ix as u32).wrapping_mul(0x9e3779b1);
    h ^= (iz as u32).wrapping_mul(0x85ebca6b);
    h = hash_u32(h);
    (h as f32) / (u32::MAX as f32)
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
