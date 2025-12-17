#import bevy_sprite::mesh2d_vertex_output::VertexOutput

const NODE_KIND_INTERNAL: u32 = 0u;
const NODE_KIND_EMPTY: u32 = 1u;
const NODE_KIND_SOLID: u32 = 2u;

const MAX_STEPS: u32 = 160u;
const MAX_DIST: f32 = 240.0;
const SURFACE_EPS: f32 = 0.012;

const MAX_SHADOW_STEPS: u32 = 96u;
const SHADOW_BIAS: f32 = 0.08;

const FOG_DENSITY_TERRAIN: f32 = 0.003;
const FOG_DENSITY_WATER: f32 = 0.004;

const MAX_WATER_PARTICLES: u32 = 128u;

struct Params {
    cam_pos_time: vec4<f32>,
    cam_right_tan: vec4<f32>,
    cam_up_aspect: vec4<f32>,
    cam_forward_pad: vec4<f32>,
    world_min_size: vec4<f32>,
    root_depth_watercount_pad: vec4<u32>,
    water_bounds_min_radius: vec4<f32>,
    water_bounds_max_pad: vec4<f32>,
    resolution_pad: vec4<f32>,
    water_particles: array<vec4<f32>, 128>,
};

struct Node {
    children0: vec4<u32>,
    children1: vec4<u32>,
    kind: u32,
    _pad: vec3<u32>,
};

@group(2) @binding(0) var<uniform> params: Params;
@group(2) @binding(1) var<storage, read> nodes: array<Node>;

struct Leaf {
    kind: u32,
    min: vec3<f32>,
    size: f32,
};

fn child_index(n: Node, child: u32) -> u32 {
    if child < 4u {
        return n.children0[i32(child)];
    }
    return n.children1[i32(child - 4u)];
}

fn in_bounds(p: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    return all(p >= bmin) && all(p < bmax);
}

fn query_leaf(p: vec3<f32>) -> Leaf {
    var node_index: u32 = params.root_depth_watercount_pad.x;
    var node_min: vec3<f32> = params.world_min_size.xyz;
    var size: f32 = params.world_min_size.w;

    for (var d: u32 = 0u; d < params.root_depth_watercount_pad.y; d = d + 1u) {
        let n = nodes[node_index];
        if n.kind != NODE_KIND_INTERNAL {
            return Leaf(n.kind, node_min, size);
        }

        let half = size * 0.5;
        let mid = node_min + vec3<f32>(half);
        var child: u32 = 0u;

        if p.x >= mid.x {
            child = child | 1u;
            node_min.x += half;
        }
        if p.y >= mid.y {
            child = child | 2u;
            node_min.y += half;
        }
        if p.z >= mid.z {
            child = child | 4u;
            node_min.z += half;
        }

        size = half;
        node_index = child_index(n, child);
    }

    let n = nodes[node_index];
    return Leaf(n.kind, node_min, size);
}

fn ray_exit_distance(p: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> f32 {
    var t_exit = 1.0e9;
    if rd.x > 0.0 {
        t_exit = min(t_exit, (bmax.x - p.x) / rd.x);
    } else if rd.x < 0.0 {
        t_exit = min(t_exit, (bmin.x - p.x) / rd.x);
    }
    if rd.y > 0.0 {
        t_exit = min(t_exit, (bmax.y - p.y) / rd.y);
    } else if rd.y < 0.0 {
        t_exit = min(t_exit, (bmin.y - p.y) / rd.y);
    }
    if rd.z > 0.0 {
        t_exit = min(t_exit, (bmax.z - p.z) / rd.z);
    } else if rd.z < 0.0 {
        t_exit = min(t_exit, (bmin.z - p.z) / rd.z);
    }
    return max(t_exit, 0.0);
}

fn terrain_shadow(p: vec3<f32>, n: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let world_min = params.world_min_size.xyz;
    let world_max = world_min + vec3<f32>(params.world_min_size.w);

    var pos = p + n * SHADOW_BIAS;
    if !in_bounds(pos, world_min, world_max) {
        return 1.0;
    }

    for (var step: u32 = 0u; step < MAX_SHADOW_STEPS; step = step + 1u) {
        if !in_bounds(pos, world_min, world_max) {
            break;
        }

        let leaf = query_leaf(pos);
        if leaf.kind == NODE_KIND_SOLID {
            return 0.0;
        }

        let exit_d = ray_exit_distance(pos, light_dir, leaf.min, leaf.min + vec3<f32>(leaf.size));
        pos = pos + light_dir * max(exit_d, SURFACE_EPS);
    }

    return 1.0;
}

fn water_sdf(p: vec3<f32>) -> f32 {
    let bmin = params.water_bounds_min_radius.xyz;
    let bmax = params.water_bounds_max_pad.xyz;
    if !in_bounds(p, bmin, bmax) {
        return 1.0e9;
    }

    let r = params.water_bounds_min_radius.w;
    let count = params.root_depth_watercount_pad.z;
    var d = 1.0e9;
    for (var i: u32 = 0u; i < MAX_WATER_PARTICLES; i = i + 1u) {
        if i >= count {
            break;
        }
        let c = params.water_particles[i].xyz;
        d = min(d, length(p - c) - r);
    }
    return d;
}

fn terrain_occ(p: vec3<f32>) -> f32 {
    let bmin = params.world_min_size.xyz;
    let bmax = bmin + vec3<f32>(params.world_min_size.w);
    if !in_bounds(p, bmin, bmax) {
        return 0.0;
    }
    let leaf = query_leaf(p);
    return select(0.0, 1.0, leaf.kind == NODE_KIND_SOLID);
}

fn terrain_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.25;
    // `terrain_occ` is 1.0 inside solid and 0.0 outside, so its gradient points inward.
    // Negate it to get an outward-facing surface normal.
    let nx = terrain_occ(p - vec3<f32>(e, 0.0, 0.0)) - terrain_occ(p + vec3<f32>(e, 0.0, 0.0));
    let ny = terrain_occ(p - vec3<f32>(0.0, e, 0.0)) - terrain_occ(p + vec3<f32>(0.0, e, 0.0));
    let nz = terrain_occ(p - vec3<f32>(0.0, 0.0, e)) - terrain_occ(p + vec3<f32>(0.0, 0.0, e));
    let n = vec3<f32>(nx, ny, nz);
    if length(n) < 0.001 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(n);
}

fn water_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.08;
    let dx = water_sdf(p + vec3<f32>(e, 0.0, 0.0)) - water_sdf(p - vec3<f32>(e, 0.0, 0.0));
    let dy = water_sdf(p + vec3<f32>(0.0, e, 0.0)) - water_sdf(p - vec3<f32>(0.0, e, 0.0));
    let dz = water_sdf(p + vec3<f32>(0.0, 0.0, e)) - water_sdf(p - vec3<f32>(0.0, 0.0, e));
    let n = vec3<f32>(dx, dy, dz);
    if length(n) < 0.001 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(n);
}

fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
    let a = vec3<f32>(0.62, 0.78, 0.98);
    let b = vec3<f32>(0.08, 0.12, 0.20);
    return mix(b, a, t);
}

fn terrain_color(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let h = p.y;
    let slope = 1.0 - clamp(n.y, 0.0, 1.0);

    let grass = vec3<f32>(0.18, 0.46, 0.17);
    let dirt = vec3<f32>(0.35, 0.26, 0.18);
    let rock = vec3<f32>(0.40, 0.40, 0.43);
    let snow = vec3<f32>(0.92, 0.94, 0.97);

    var base = mix(grass, dirt, smoothstep(0.4, 0.85, slope));
    base = mix(base, rock, smoothstep(0.15, 0.35, (h - 20.0) / 16.0));
    base = mix(base, snow, smoothstep(0.0, 1.0, (h - 44.0) / 12.0));
    return base;
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    let ro = params.cam_pos_time.xyz;
    let right = params.cam_right_tan.xyz;
    let tan_half_fov = params.cam_right_tan.w;
    let up = params.cam_up_aspect.xyz;
    let aspect = params.cam_up_aspect.w;
    let forward = params.cam_forward_pad.xyz;

    var rd = normalize(forward + ndc.x * right * tan_half_fov * aspect + ndc.y * up * tan_half_fov);

    let world_min = params.world_min_size.xyz;
    let world_max = world_min + vec3<f32>(params.world_min_size.w);

    var t = 0.0;
    var t_prev = 0.0;
    var hit_kind: u32 = 0u; // 1 terrain, 2 water
    var hit_pos = vec3<f32>(0.0);

    for (var step: u32 = 0u; step < MAX_STEPS; step = step + 1u) {
        if t > MAX_DIST {
            break;
        }
        let p = ro + rd * t;

        let d_water = water_sdf(p);
        if d_water < SURFACE_EPS {
            hit_kind = 2u;
            hit_pos = p;
            break;
        }

        if in_bounds(p, world_min, world_max) {
            let leaf = query_leaf(p);
            if leaf.kind == NODE_KIND_SOLID {
                // Binary search back to the boundary for a cleaner surface hit.
                var a = t_prev;
                var b = t;
                for (var i: u32 = 0u; i < 10u; i = i + 1u) {
                    let m = 0.5 * (a + b);
                    if query_leaf(ro + rd * m).kind == NODE_KIND_SOLID {
                        b = m;
                    } else {
                        a = m;
                    }
                }
                // Use the "outside" point so normal estimation + shadow rays don't start inside solid.
                t = a;
                hit_kind = 1u;
                hit_pos = ro + rd * t;
                break;
            }

            let exit_d = ray_exit_distance(p, rd, leaf.min, leaf.min + vec3<f32>(leaf.size));
            let step_len = min(exit_d, d_water);
            t_prev = t;
            t += max(step_len, SURFACE_EPS);
        } else {
            // Outside the voxel world: only march water (it's bounded, so this will usually just escape).
            t_prev = t;
            t += max(d_water, 1.0);
        }
    }

    let sun_dir = normalize(vec3<f32>(0.65, 0.72, 0.25));
    var col = sky(rd);

    if hit_kind == 1u {
        let n = terrain_normal(hit_pos);
        let ndotl = dot(n, sun_dir);
        var shadow = 1.0;
        if ndotl > 0.0 {
            shadow = terrain_shadow(hit_pos, n, sun_dir);
        }
        let diff = max(ndotl, 0.0) * shadow;
        let base = terrain_color(hit_pos, n);
        let ambient = 0.22;
        let lit = base * (ambient + diff * 0.95);
        let fog = 1.0 - exp(-t * FOG_DENSITY_TERRAIN);
        col = mix(lit, col, fog);
    } else if hit_kind == 2u {
        let n = water_normal(hit_pos);
        let ndotl = dot(n, sun_dir);
        var shadow = 1.0;
        if ndotl > 0.0 {
            shadow = terrain_shadow(hit_pos, n, sun_dir);
        }
        let diff = max(ndotl, 0.0) * shadow;
        let refl = sky(reflect(rd, n));
        let water_col = vec3<f32>(0.04, 0.22, 0.33);
        let fres = pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
        let spec = pow(max(dot(reflect(-sun_dir, n), -rd), 0.0), 96.0) * shadow;
        let lit = mix(water_col * (0.30 + 0.70 * diff), refl, fres) + spec * 0.7;
        let fog = 1.0 - exp(-t * FOG_DENSITY_WATER);
        col = mix(lit, col, fog);
    }

    return vec4<f32>(col, 1.0);
}
