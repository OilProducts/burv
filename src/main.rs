mod fps_camera;
mod raymarch;
mod terrain;
mod water;

use bevy::prelude::*;
use bevy::sprite::{Material2dPlugin, MaterialMesh2dBundle};
use bevy::window::PrimaryWindow;

use crate::fps_camera::FpsCamera;
use crate::raymarch::{RaymarchMaterial, RaymarchMaterialHandle, RaymarchParams};
use crate::terrain::Terrain;
use crate::water::WaterParticles;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "burv — voxel raymarch demo".to_string(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(Material2dPlugin::<RaymarchMaterial>::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                fps_camera::fps_camera_input_system,
                water::simulate_water_system,
                raymarch::sync_raymarch_material_system,
                fullscreen_quad_resize_system,
            )
                .chain(),
        )
        .run();
}

#[derive(Component)]
struct FullscreenQuad;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<RaymarchMaterial>>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
) {
    let window = primary_window.single();

    let terrain = Terrain::generate();
    let water = WaterParticles::new(&terrain);
    let camera = FpsCamera::new(&terrain);

    let mut params = RaymarchParams::from_world(&terrain, &water, &camera, window);
    params.set_time(0.0);

    let material_handle = materials.add(RaymarchMaterial {
        params,
        nodes: terrain.nodes.clone(),
    });

    commands.insert_resource(terrain);
    commands.insert_resource(water);
    commands.insert_resource(camera);
    commands.insert_resource(RaymarchMaterialHandle(material_handle.clone()));

    commands.spawn(Camera2dBundle::default());

    let quad_mesh = meshes.add(Mesh::from(Rectangle::new(1.0, 1.0)));
    commands.spawn((
        FullscreenQuad,
        MaterialMesh2dBundle {
            mesh: quad_mesh.into(),
            material: material_handle,
            transform: Transform::from_scale(Vec3::new(
                window.resolution.width(),
                window.resolution.height(),
                1.0,
            )),
            ..default()
        },
    ));
}

fn fullscreen_quad_resize_system(
    mut evr: EventReader<bevy::window::WindowResized>,
    mut q: Query<&mut Transform, With<FullscreenQuad>>,
) {
    let Ok(mut transform) = q.get_single_mut() else {
        return;
    };

    for e in evr.read() {
        // Keep the quad scaled to match the window in "pixel" world units.
        transform.scale.x = e.width;
        transform.scale.y = e.height;
    }
}
