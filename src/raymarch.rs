#![allow(dead_code)]

use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderRef, ShaderType};
use bevy::sprite::Material2d;

use crate::fps_camera::FpsCamera;
use crate::terrain::{GpuNode, Terrain};
use crate::water::{MAX_WATER_PARTICLES, WaterParticles};

#[derive(Resource, Clone)]
pub struct RaymarchMaterialHandle(pub Handle<RaymarchMaterial>);

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct RaymarchMaterial {
    #[uniform(0)]
    pub params: RaymarchParams,
    #[storage(1, read_only)]
    pub nodes: Vec<GpuNode>,
}

impl Material2d for RaymarchMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/raymarch_voxel.wgsl".into()
    }
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct RaymarchParams {
    pub cam_pos_time: Vec4,    // xyz: camera position, w: time (seconds)
    pub cam_right_tan: Vec4,   // xyz: camera right, w: tan(fov_y/2)
    pub cam_up_aspect: Vec4,   // xyz: camera up, w: aspect
    pub cam_forward_pad: Vec4, // xyz: camera forward, w: unused
    pub world_min_size: Vec4,  // xyz: world min, w: world size
    pub root_depth_watercount_pad: UVec4, // x: root index, y: max depth, z: water count, w: unused
    pub water_bounds_min_radius: Vec4, // xyz: min, w: radius
    pub water_bounds_max_pad: Vec4, // xyz: max, w: unused
    pub resolution_pad: Vec4,  // xy: resolution, zw: unused
    pub water_particles: [Vec4; MAX_WATER_PARTICLES],
}

impl RaymarchParams {
    pub fn from_world(
        terrain: &Terrain,
        water: &WaterParticles,
        camera: &FpsCamera,
        window: &Window,
    ) -> Self {
        let (right, up, forward) = camera.basis();
        let tan_half_fov = (camera.fov_y_radians * 0.5).tan();
        let aspect = window.resolution.width() / window.resolution.height().max(1.0);

        Self {
            cam_pos_time: camera.position.extend(0.0),
            cam_right_tan: right.extend(tan_half_fov),
            cam_up_aspect: up.extend(aspect),
            cam_forward_pad: forward.extend(0.0),
            world_min_size: terrain.world_min.extend(terrain.world_size),
            root_depth_watercount_pad: UVec4::new(
                terrain.root_index,
                terrain.max_depth,
                water.count(),
                0,
            ),
            water_bounds_min_radius: water.render_bounds_min.extend(water.radius),
            water_bounds_max_pad: water.render_bounds_max.extend(0.0),
            resolution_pad: Vec4::new(
                window.resolution.width(),
                window.resolution.height(),
                0.0,
                0.0,
            ),
            water_particles: water.packed_positions(),
        }
    }

    pub fn set_time(&mut self, t: f32) {
        self.cam_pos_time.w = t;
    }

    pub fn set_camera(&mut self, camera: &FpsCamera, window: &Window) {
        let (right, up, forward) = camera.basis();
        let tan_half_fov = (camera.fov_y_radians * 0.5).tan();
        let aspect = window.resolution.width() / window.resolution.height().max(1.0);
        self.cam_pos_time.x = camera.position.x;
        self.cam_pos_time.y = camera.position.y;
        self.cam_pos_time.z = camera.position.z;
        self.cam_right_tan = right.extend(tan_half_fov);
        self.cam_up_aspect = up.extend(aspect);
        self.cam_forward_pad = forward.extend(0.0);
        self.resolution_pad = Vec4::new(
            window.resolution.width(),
            window.resolution.height(),
            0.0,
            0.0,
        );
    }

    pub fn set_water(&mut self, water: &WaterParticles) {
        self.root_depth_watercount_pad.z = water.count();
        self.water_bounds_min_radius = water.render_bounds_min.extend(water.radius);
        self.water_bounds_max_pad = water.render_bounds_max.extend(0.0);
        self.water_particles = water.packed_positions();
    }
}

pub fn sync_raymarch_material_system(
    time: Res<Time>,
    terrain: Res<Terrain>,
    water: Res<WaterParticles>,
    camera: Res<FpsCamera>,
    handle: Res<RaymarchMaterialHandle>,
    mut materials: ResMut<Assets<RaymarchMaterial>>,
    primary_window: Query<&Window, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(window) = primary_window.get_single() else {
        return;
    };
    let Some(material) = materials.get_mut(&handle.0) else {
        return;
    };

    material.params.set_time(time.elapsed_seconds());
    material.params.set_camera(&camera, window);
    material.params.set_water(&water);

    // Keep world uniforms in sync (in case you regenerate later).
    material.params.world_min_size = terrain.world_min.extend(terrain.world_size);
    material.params.root_depth_watercount_pad.x = terrain.root_index;
    material.params.root_depth_watercount_pad.y = terrain.max_depth;
}
