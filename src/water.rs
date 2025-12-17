use bevy::prelude::*;

use crate::terrain::Terrain;

pub const MAX_WATER_PARTICLES: usize = 128;

#[derive(Resource, Clone)]
pub struct WaterParticles {
    pub radius: f32,
    pub kernel_radius: f32,
    pub rest_density: f32,
    pub sim_bounds_min: Vec3,
    pub sim_bounds_max: Vec3,
    pub render_bounds_min: Vec3,
    pub render_bounds_max: Vec3,
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    prev_positions: Vec<Vec3>,
    scratch_densities: Vec<f32>,
    scratch_lambdas: Vec<f32>,
    scratch_vecs: Vec<Vec3>,
    accumulator: f32,
}

impl WaterParticles {
    pub fn new(terrain: &Terrain) -> Self {
        let count = 128usize;
        let radius = 0.40;
        let kernel_radius = radius * 3.5;

        let sim_bounds_min = terrain.world_min;
        let sim_bounds_max = terrain.world_min + Vec3::splat(terrain.world_size);

        let mut rng = 0x1234_5678u32;
        let mut positions = Vec::with_capacity(count);
        let mut velocities = Vec::with_capacity(count);

        for _ in 0..count {
            let x = lerp(-6.0, 6.0, next_f32(&mut rng));
            let z = lerp(-6.0, 6.0, next_f32(&mut rng));
            // Terrain voxels are solid for y <= height, so the surface is at height + 1.
            let base_y = terrain.height_at(x, z) + 1.0 + radius;
            let y = base_y + lerp(0.0, 6.0, next_f32(&mut rng));
            positions.push(Vec3::new(x, y, z));
            velocities.push(Vec3::ZERO);
        }

        let rest_density = estimate_rest_density(&positions, kernel_radius).max(1.0e-3);

        let render_bounds_min = sim_bounds_min;
        let render_bounds_max = sim_bounds_max;

        Self {
            radius,
            kernel_radius,
            rest_density,
            sim_bounds_min,
            sim_bounds_max,
            render_bounds_min,
            render_bounds_max,
            positions,
            velocities,
            prev_positions: vec![Vec3::ZERO; count],
            scratch_densities: vec![0.0; count],
            scratch_lambdas: vec![0.0; count],
            scratch_vecs: vec![Vec3::ZERO; count],
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
    let particle_count = water.positions.len();
    if particle_count == 0 {
        return;
    }
    resize_scratch(water, particle_count);

    let g = Vec3::new(0.0, -18.0, 0.0);
    let damping = 0.99;
    let bounce = 0.02;
    let friction = 0.18;
    let solver_iters = 4;
    let pbf_relaxation = 0.15;
    let max_position_correction = water.radius * 0.35;
    let max_speed = 20.0;

    water.prev_positions.clone_from(&water.positions);

    for (p, v) in water.positions.iter_mut().zip(water.velocities.iter_mut()) {
        *v += g * dt;
        *v *= damping;
        *p += *v * dt;
    }

    // Position Based Fluids (PBF) density constraint solver (O(n^2), n is small here).
    let h = water.kernel_radius;
    let inv_rest_density = 1.0 / water.rest_density.max(1.0e-6);
    let w_q = kernel_w(0.3 * h, h).max(1.0e-6);
    for _ in 0..solver_iters {
        // Densities.
        for i in 0..particle_count {
            let mut rho = 0.0;
            let pi = water.positions[i];
            for pj in water.positions.iter().copied() {
                rho += kernel_w((pi - pj).length(), h);
            }
            water.scratch_densities[i] = rho;
        }

        // Lambdas.
        for i in 0..particle_count {
            let pi = water.positions[i];
            let c = water.scratch_densities[i] * inv_rest_density - 1.0;

            let mut grad_i = Vec3::ZERO;
            let mut sum_grad2 = 0.0;
            for j in 0..particle_count {
                if i == j {
                    continue;
                }
                let grad_w = kernel_grad(pi - water.positions[j], h);
                grad_i += grad_w;
                sum_grad2 += grad_w.length_squared();
            }
            let denom = (grad_i.length_squared() + sum_grad2) * inv_rest_density * inv_rest_density + 1.0e-5;
            water.scratch_lambdas[i] = -c / denom;
        }

        // Position deltas.
        for i in 0..particle_count {
            let pi = water.positions[i];
            let mut dp = Vec3::ZERO;
            for j in 0..particle_count {
                if i == j {
                    continue;
                }
                let rij = pi - water.positions[j];
                let r = rij.length();
                if r >= h || r <= 1.0e-6 {
                    continue;
                }

                // Small tensile correction to reduce clumping.
                let w = kernel_w(r, h);
                let s_corr = -0.001 * (w / w_q).powi(4);

                dp += (water.scratch_lambdas[i] + water.scratch_lambdas[j] + s_corr)
                    * kernel_grad(rij, h);
            }
            let mut delta = dp * inv_rest_density * pbf_relaxation;
            let len = delta.length();
            if len > max_position_correction && len > 1.0e-6 {
                delta *= max_position_correction / len;
            }
            water.scratch_vecs[i] = delta;
        }

        for i in 0..particle_count {
            water.positions[i] += water.scratch_vecs[i];
        }

        enforce_collisions(terrain, water);
    }

    enforce_collisions(terrain, water);
    update_render_bounds(water);

    // Update velocities from corrected positions.
    for i in 0..particle_count {
        let mut v = (water.positions[i] - water.prev_positions[i]) / dt;
        let speed = v.length();
        if speed > max_speed && speed > 1.0e-6 {
            v *= max_speed / speed;
        }
        water.velocities[i] = v;
    }

    // Viscosity (XSPH).
    let viscosity = 0.20;
    for i in 0..particle_count {
        let pi = water.positions[i];
        let vi = water.velocities[i];
        let mut dv = Vec3::ZERO;
        for j in 0..particle_count {
            if i == j {
                continue;
            }
            let r = (pi - water.positions[j]).length();
            if r >= h {
                continue;
            }
            dv += (water.velocities[j] - vi) * kernel_w(r, h);
        }
        water.scratch_vecs[i] = dv;
    }
    for i in 0..particle_count {
        water.velocities[i] += water.scratch_vecs[i] * viscosity;
    }
    for v in water.velocities.iter_mut() {
        let speed = v.length();
        if speed > max_speed && speed > 1.0e-6 {
            *v *= max_speed / speed;
        }
    }

    // Boundary response (keeps things from looking like bouncy balls).
    for i in 0..particle_count {
        let p = water.positions[i];
        let v = &mut water.velocities[i];

        for axis in 0..3 {
            let min = water.sim_bounds_min[axis];
            let max = water.sim_bounds_max[axis];
            if p[axis] <= min + 1.0e-4 && v[axis] < 0.0 {
                v[axis] = -v[axis] * bounce;
            } else if p[axis] >= max - 1.0e-4 && v[axis] > 0.0 {
                v[axis] = -v[axis] * bounce;
            }
        }

        let floor_y = terrain.height_at(p.x, p.z) + 1.0 + water.radius;
        if p.y <= floor_y + 1.0e-3 {
            let n = terrain_height_normal(terrain, p.x, p.z);
            let vn = v.dot(n);
            if vn < 0.0 {
                *v -= vn * n;
            }

            let vt = *v - v.dot(n) * n;
            *v -= vt * friction;
        }
    }

    for v in water.velocities.iter_mut() {
        let speed = v.length();
        if speed > max_speed && speed > 1.0e-6 {
            *v *= max_speed / speed;
        }
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

fn resize_scratch(water: &mut WaterParticles, count: usize) {
    if water.prev_positions.len() != count {
        water.prev_positions.resize(count, Vec3::ZERO);
    }
    if water.scratch_densities.len() != count {
        water.scratch_densities.resize(count, 0.0);
    }
    if water.scratch_lambdas.len() != count {
        water.scratch_lambdas.resize(count, 0.0);
    }
    if water.scratch_vecs.len() != count {
        water.scratch_vecs.resize(count, Vec3::ZERO);
    }
}

fn enforce_collisions(terrain: &Terrain, water: &mut WaterParticles) {
    let sim_min = water.sim_bounds_min;
    let sim_max = water.sim_bounds_max;
    let radius = water.radius;
    for p in water.positions.iter_mut() {
        clamp_to_sim_bounds(sim_min, sim_max, p);
        push_out_of_terrain(terrain, radius, sim_max.y, p);
        clamp_to_sim_bounds(sim_min, sim_max, p);
        push_out_of_terrain(terrain, radius, sim_max.y, p);
    }
}

fn update_render_bounds(water: &mut WaterParticles) {
    // Tight bounds keep the ray marcher from evaluating water far away from the particles.
    if water.positions.is_empty() {
        return;
    }

    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    for &p in water.positions.iter() {
        bmin = bmin.min(p);
        bmax = bmax.max(p);
    }

    let pad = water.radius * 3.0;
    water.render_bounds_min = bmin - Vec3::splat(pad);
    water.render_bounds_max = bmax + Vec3::splat(pad);
}

fn clamp_to_sim_bounds(sim_min: Vec3, sim_max: Vec3, p: &mut Vec3) {
    for axis in 0..3 {
        let min = sim_min[axis];
        let max = sim_max[axis] - 1.0e-4;
        p[axis] = p[axis].clamp(min, max);
    }
}

fn push_out_of_terrain(terrain: &Terrain, radius: f32, sim_max_y: f32, p: &mut Vec3) {
    let floor_y = (terrain.height_at(p.x, p.z) + 1.0 + radius).min(sim_max_y - 1.0e-4);
    if p.y >= floor_y {
        return;
    }

    let penetration_y = floor_y - p.y;
    let n = terrain_height_normal(terrain, p.x, p.z);
    let denom = n.y.max(0.2);
    *p += n * (penetration_y / denom);
}

fn terrain_height_normal(terrain: &Terrain, x: f32, z: f32) -> Vec3 {
    let e = 1.0;
    let hx1 = terrain.height_at(x + e, z);
    let hx0 = terrain.height_at(x - e, z);
    let hz1 = terrain.height_at(x, z + e);
    let hz0 = terrain.height_at(x, z - e);

    let dx = hx1 - hx0;
    let dz = hz1 - hz0;
    let n = Vec3::new(-dx, 2.0 * e, -dz);
    if n.length_squared() < 1.0e-8 {
        return Vec3::Y;
    }
    n.normalize()
}

fn kernel_w(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let x = 1.0 - (r / h);
    x * x * x
}

fn kernel_grad(rij: Vec3, h: f32) -> Vec3 {
    let r = rij.length();
    if r <= 1.0e-6 || r >= h {
        return Vec3::ZERO;
    }
    let x = 1.0 - (r / h);
    // d/dr (1 - r/h)^3 = -3(1 - r/h)^2 * (1/h)
    let coeff = -3.0 * x * x / (h * r);
    rij * coeff
}

fn estimate_rest_density(positions: &[Vec3], h: f32) -> f32 {
    if positions.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for &pi in positions.iter() {
        let mut rho = 0.0;
        for &pj in positions.iter() {
            rho += kernel_w((pi - pj).length(), h);
        }
        sum += rho;
    }
    sum / (positions.len() as f32)
}
