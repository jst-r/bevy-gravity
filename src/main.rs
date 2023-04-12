//! Shows how to iterate over combinations of query results.

use std::f32::consts::PI;

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    pbr::AmbientLight,
    prelude::*,
};
use rand::{thread_rng, Rng};

const DELTA_TIME: f32 = 0.01;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(AmbientLight {
            brightness: 0.03,
            ..default()
        })
        .add_startup_system(generate_bodies)
        .insert_resource(FixedTime::new_from_secs(DELTA_TIME))
        .add_systems((interact_bodies, integrate).in_schedule(CoreSchedule::FixedUpdate))
        .insert_resource(ClearColor(Color::BLACK))
        .run();
}

const GRAVITY_CONSTANT: f32 = 0.001;
const NUM_BODIES: usize = 3000;

#[derive(Component, Default)]
struct Mass(f32);
#[derive(Component, Default)]
struct Acceleration(Vec3);
#[derive(Component, Default)]
struct LastPos(Vec3);
#[derive(Component)]
struct Star;

#[derive(Bundle, Default)]
struct BodyBundle {
    pbr: PbrBundle,
    mass: Mass,
    last_pos: LastPos,
    acceleration: Acceleration,
}

fn generate_bodies(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(
        Mesh::try_from(shape::Icosphere {
            radius: 1.0,
            subdivisions: 3,
        })
        .unwrap(),
    );

    let color_range = 0.9..1.0;
    let vel_range = -0.5..0.5;

    let mut rng = thread_rng();
    for _ in 0..NUM_BODIES {
        let radius: f32 = rng.gen_range(0.1..0.7);
        let mass_value = radius.powi(3) * 10.;

        let position = Vec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        )
        .normalize()
            * (Vec3 {
                x: 1.,
                y: 0.1,
                z: 1.,
            })
            * rng.gen_range(0.2f32..1.0).cbrt()
            * 100.;

        commands.spawn(BodyBundle {
            pbr: PbrBundle {
                transform: Transform {
                    translation: position,
                    scale: Vec3::splat(radius),
                    ..default()
                },
                mesh: mesh.clone(),
                material: materials.add(
                    StandardMaterial {
                        base_color: Color::rgb(
                            rng.gen_range(color_range.clone()),
                            rng.gen_range(color_range.clone()),
                            rng.gen_range(color_range.clone()),
                        ),
                        emissive: Color::WHITE,
                        ..default()
                    }
                    .into(),
                ),
                ..default()
            },
            mass: Mass(mass_value),
            acceleration: Acceleration(Vec3::ZERO),
            last_pos: LastPos(
                position
                    - Vec3::new(
                        rng.gen_range(vel_range.clone()),
                        rng.gen_range(vel_range.clone()),
                        rng.gen_range(vel_range.clone()),
                    ) * DELTA_TIME,
            ),
        });
    }

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 100.0, -100.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

fn interact_bodies(
    mut query: Query<(&GlobalTransform, &mut Acceleration)>,
    other: Query<(&Mass, &GlobalTransform)>,
) {
    query.par_iter_mut().for_each_mut(|(transform1, mut acc1)| {
        for (Mass(m2), transform2) in other.iter() {
            let delta = transform2.translation() - transform1.translation();
            let distance_sq: f32 = delta.length_squared();

            if distance_sq < 0.5 {
                continue;
            }

            let f = GRAVITY_CONSTANT / distance_sq;
            let force_unit_mass = delta * f;
            acc1.0 += force_unit_mass * *m2;
        }
    });
}

fn integrate(mut query: Query<(&mut Acceleration, &mut Transform, &mut LastPos)>) {
    let dt_sq = DELTA_TIME * DELTA_TIME;
    for (mut acceleration, mut transform, mut last_pos) in &mut query {
        // verlet integration
        // x(t+dt) = 2x(t) - x(t-dt) + a(t)dt^2 + O(dt^4)

        let new_pos = transform.translation * 2.0 - last_pos.0 + acceleration.0 * dt_sq;
        acceleration.0 = Vec3::ZERO;
        last_pos.0 = transform.translation;
        transform.translation = new_pos;
    }
}
