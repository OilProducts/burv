use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use crate::terrain::Terrain;

#[derive(Resource, Debug, Clone)]
pub struct FpsCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_radians: f32,
    pub move_speed: f32,
    pub mouse_sensitivity: f32,
    pub mouse_captured: bool,
}

impl FpsCamera {
    pub fn new(terrain: &Terrain) -> Self {
        let start_xz = Vec2::ZERO;
        let start_y = terrain.height_at(start_xz.x, start_xz.y) + 6.0;
        Self {
            position: Vec3::new(start_xz.x, start_y, start_xz.y),
            yaw: 0.0,
            pitch: -0.2,
            fov_y_radians: 70.0_f32.to_radians(),
            move_speed: 14.0,
            mouse_sensitivity: 0.0022,
            mouse_captured: false,
        }
    }

    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        // Bevy-style: +Y up. We'll treat "forward" as looking into -Z when yaw=pitch=0.
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();

        let forward = Vec3::new(sy * cp, sp, -cy * cp).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        (right, up, forward)
    }
}

pub fn fps_camera_input_system(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut camera: ResMut<FpsCamera>,
    mut primary_window: Query<&mut Window, With<PrimaryWindow>>,
) {
    let Ok(mut window) = primary_window.get_single_mut() else {
        return;
    };

    if keyboard.just_pressed(KeyCode::Escape) {
        camera.mouse_captured = false;
        window.cursor.grab_mode = CursorGrabMode::None;
        window.cursor.visible = true;
    }
    if mouse_buttons.just_pressed(MouseButton::Left) {
        camera.mouse_captured = true;
        window.cursor.grab_mode = CursorGrabMode::Locked;
        window.cursor.visible = false;
    }

    if camera.mouse_captured {
        let mut delta = Vec2::ZERO;
        for ev in mouse_motion.read() {
            delta += ev.delta;
        }

        camera.yaw -= delta.x * camera.mouse_sensitivity;
        camera.pitch -= delta.y * camera.mouse_sensitivity;
        camera.pitch = camera.pitch.clamp(-1.54, 1.54);
    } else {
        mouse_motion.clear();
    }

    let (right, _up, forward) = camera.basis();
    let mut wish = Vec3::ZERO;

    if keyboard.pressed(KeyCode::KeyW) {
        wish += forward;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        wish -= forward;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        wish += right;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        wish -= right;
    }
    if keyboard.pressed(KeyCode::Space) {
        wish += Vec3::Y;
    }
    if keyboard.pressed(KeyCode::ShiftLeft) {
        wish -= Vec3::Y;
    }

    if wish.length_squared() > 0.0 {
        let dt = time.delta_seconds();
        let speed = camera.move_speed;
        camera.position += wish.normalize() * speed * dt;
    }
}
