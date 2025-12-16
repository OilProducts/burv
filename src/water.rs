use bevy::prelude::*;

use crate::terrain::Terrain;

pub const MAX_WATER_PARTICLES: usize = 128;

#[derive(Resource, Clone)]
pub struct WaterParticles {
    pub radius: f32,
    pub rest_distance: f32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    accumulator: f32,
}

impl WaterParticles {
    pub fn new(terrain: &Terrain) -> Self {
        let count = 96usize;
        let radius = 0.55;
        let rest_distance = radius * 1.85;

        let basin_center = Vec3::new(0.0, terrain.height_at(0.0, 0.0) + 14.0, 0.0);
        let bounds_min = Vec3::new(-18.0, 0.0, -18.0);
        let bounds_max = Vec3::new(18.0, 48.0, 18.0);

        let mut rng = 0x1234_5678u32;
        let mut positions = Vec::with_capacity(count);
        let mut velocities = Vec::with_capacity(count);

        for _ in 0..count {
            let x = lerp(-6.0, 6.0, next_f32(&mut rng));
            let z = lerp(-6.0, 6.0, next_f32(&mut rng));
            let y = lerp(0.0, 8.0, next_f32(&mut rng));
            positions.push(basin_center + Vec3::new(x, y, z));
            velocities.push(Vec3::ZERO);
        }

        Self {
            radius,
            rest_distance,
            bounds_min,
            bounds_max,
            positions,
            velocities,
            accumulator: 0.0,
        }
    }

    pub fn count(&self) -> u32 {
        self.positions.len().min(MAX_WATER_PARTICLES) as u32
    }

    pub fn packed_positions(&self) -> [Vec4; MAX_WATER_PARTICLES] {
        let mut out = [Vec4::splat(1.0e9); MAX_WATER_PARTICLES];
        for (i, p) in self
            .positions
            .iter()
            .copied()
            .take(MAX_WATER_PARTICLES)
            .enumerate()
        {
            out[i] = p.extend(0.0);
        }
        out
    }
}

pub fn simulate_water_system(
    time: Res<Time>,
    terrain: Res<Terrain>,
    mut water: ResMut<WaterParticles>,
) {
    let dt = time.delta_seconds().min(0.05);
    water.accumulator += dt;

    let fixed_dt = 1.0 / 60.0;
    while water.accumulator >= fixed_dt {
        step_water(fixed_dt, &terrain, &mut water);
        water.accumulator -= fixed_dt;
    }
}

fn step_water(dt: f32, terrain: &Terrain, water: &mut WaterParticles) {
    let g = Vec3::new(0.0, -18.0, 0.0);
    let damping = 0.996;
    let bounce = 0.15;

    let mut prev = water.positions.clone();

    for (p, v) in water.positions.iter_mut().zip(water.velocities.iter_mut()) {
        *v += g * dt;
        *v *= damping;
        *p += *v * dt;
    }

    // A couple of cheap PBD spacing iterations (O(n^2), but n is small here).
    for _ in 0..2 {
        for i in 0..water.positions.len() {
            for j in (i + 1)..water.positions.len() {
                let a = water.positions[i];
                let b = water.positions[j];
                let d = b - a;
                let dist = d.length();
                if dist > 0.0001 && dist < water.rest_distance {
                    let push = (water.rest_distance - dist) * 0.5;
                    let dir = d / dist;
                    water.positions[i] -= dir * push;
                    water.positions[j] += dir * push;
                }
            }
        }
    }

    // Bounds + terrain collisions.
    for (p, v) in water.positions.iter_mut().zip(water.velocities.iter_mut()) {
        // Keep water in a small simulation box (so the ray marcher can early-out cheaply).
        for axis in 0..3 {
            let min = water.bounds_min[axis];
            let max = water.bounds_max[axis];
            if p[axis] < min {
                p[axis] = min;
                v[axis] = v[axis].abs() * bounce;
            } else if p[axis] > max {
                p[axis] = max;
                v[axis] = -v[axis].abs() * bounce;
            }
        }

        let ground = terrain.height_at(p.x, p.z);
        let floor_y = ground + water.radius;
        if p.y < floor_y {
            p.y = floor_y;
            v.y = v.y.abs() * bounce;
        }
    }

    // Update velocities from corrected positions.
    for ((p, p_prev), v) in water
        .positions
        .iter()
        .zip(prev.drain(..))
        .zip(water.velocities.iter_mut())
    {
        *v = (*p - p_prev) / dt;
    }
}

fn next_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let bits = (*state >> 8) | 0x3f80_0000;
    f32::from_bits(bits) - 1.0
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
