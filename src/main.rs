use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::query::{BatchingStrategy, QuerySingleError},
    pbr::{
        wireframe::{Wireframe, WireframePlugin},
        AmbientLight,
    },
    prelude::*,
    render::{
        settings::{WgpuFeatures, WgpuSettings},
        RenderPlugin,
    },
};
use rand::{prelude::Distribution, thread_rng, Rng};

mod camera;
use camera::{pan_orbit_camera, spawn_camera};
use rand_distr::Normal;

const DELTA_TIME: f32 = 1. / 10.;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(RenderPlugin {
            wgpu_settings: WgpuSettings {
                features: WgpuFeatures::POLYGON_MODE_LINE,
                ..default()
            },
        }))
        .add_plugin(WireframePlugin)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(AmbientLight {
            brightness: 0.03,
            ..default()
        })
        .add_startup_systems((generate_bodies, spawn_camera))
        .add_system(pan_orbit_camera)
        .insert_resource(FixedTime::new_from_secs(DELTA_TIME))
        .add_systems((build_tree, interact_bodies_tree, integrate))
        // .add_systems((draw_boxes,))
        .insert_resource(ClearColor(Color::BLACK))
        .run();
}

const GRAVITY_CONSTANT: f32 = 0.01;
const NUM_BODIES: usize = 7000;

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
            subdivisions: 2,
        })
        .unwrap(),
    );

    let color_range = 0.9..1.0;
    let vel_range = -0.5..0.5;
    let distribution = Normal::new(0f32, 1f32).unwrap();

    let mut rng = thread_rng();
    for _ in 0..NUM_BODIES {
        let radius: f32 = rng.gen_range(0.1..0.7);
        let mass_value = radius.powi(3) * 10.;

        let position = Vec3::new(
            distribution.sample(&mut rng),
            distribution.sample(&mut rng),
            distribution.sample(&mut rng),
        ) * (Vec3 {
            x: 1.3,
            y: 0.3,
            z: 1.0,
        }) * rng.gen_range(0f32..1.0).sqrt()
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
                    + Vec3::cross(Vec3::Y, position).normalize()
                        * DELTA_TIME
                        * position.length().cbrt()
                        * rng.gen_range(0.5..1.1),
            ),
        });
    }
}

fn interact_bodies(
    mut query: Query<(&GlobalTransform, &mut Acceleration)>,
    other: Query<(&Mass, &GlobalTransform)>,
) {
    query.par_iter_mut().for_each_mut(|(transform1, mut acc1)| {
        for (Mass(m2), transform2) in other.iter() {
            let delta = transform2.translation() - transform1.translation();
            let distance_sq: f32 = delta.length_squared();

            if distance_sq < 0.1 {
                continue;
            }

            let f = GRAVITY_CONSTANT / distance_sq;
            let force_unit_mass = delta * f;
            acc1.0 += force_unit_mass * *m2;
        }
    });
}

fn interact_bodies_tree(
    mut query: Query<(&GlobalTransform, &mut Acceleration)>,
    tree: Query<&Tree>,
) {
    query
        .par_iter_mut()
        .batching_strategy(BatchingStrategy::new().min_batch_size(128))
        .for_each_mut(|(transform1, mut acc1)| {
            let tree = match tree.get_single() {
                Ok(tree) => tree,
                Err(_) => return,
            };
            let mut queue = vec![tree];

            while let Some(node) = queue.pop() {
                match node {
                    Tree::Node {
                        neg,
                        pos,
                        center_of_mass,
                        mass,
                        normal,
                        cutoff,
                        bbox,
                    } => {
                        let delta = (*center_of_mass - transform1.translation()).length();
                        let sum_of_sides = bbox.max_x - bbox.min_x + bbox.max_y - bbox.min_y
                            + bbox.max_z
                            - bbox.min_z;

                        if delta / sum_of_sides < 5. {
                            queue.push(&neg);
                            queue.push(&pos);
                            continue;
                        }

                        let delta = *center_of_mass - transform1.translation();
                        let distance_sq: f32 = delta.length_squared();

                        if distance_sq < 0.1 {
                            continue;
                        }

                        let f = GRAVITY_CONSTANT / distance_sq;
                        let force_unit_mass = delta * f;
                        acc1.0 += force_unit_mass * *mass;
                    }
                    Tree::Leaf(bodies) => {
                        for Body(translation2, m2) in bodies {
                            let delta = *translation2 - transform1.translation();
                            let distance_sq: f32 = delta.length_squared();

                            if distance_sq < 0.1 {
                                continue;
                            }

                            let f = GRAVITY_CONSTANT / distance_sq;
                            let force_unit_mass = delta * f;
                            acc1.0 += force_unit_mass * *m2;
                        }
                    }
                }
            }
        });
}

#[derive(Debug, Clone, Copy)]
struct Body(Vec3, f32);

#[derive(Debug, Component)]
enum Tree {
    Node {
        neg: Box<Tree>,
        pos: Box<Tree>,
        center_of_mass: Vec3,
        mass: f32,
        normal: Vec3,
        cutoff: f32,
        bbox: shape::Box,
    },
    Leaf(Vec<Body>),
}

fn build_tree(
    mut commands: Commands,
    mut tree: Query<&mut Tree>,
    bodies: Query<(&GlobalTransform, &Mass)>,
) {
    let bodies: Vec<Body> = bodies
        .iter()
        .map(|(trasform, mass)| Body(trasform.translation(), mass.0))
        .collect();

    let translations = &bodies.iter().map(|b| b.0).collect::<Vec<_>>();

    let root_box = shape::Box {
        min_x: translations
            .iter()
            .map(|t| t.x)
            .reduce(f32::min)
            .unwrap_or(0.0),
        max_x: translations
            .iter()
            .map(|t| t.x)
            .reduce(f32::max)
            .unwrap_or(0.0),
        min_y: translations
            .iter()
            .map(|t| t.y)
            .reduce(f32::min)
            .unwrap_or(0.0),
        max_y: translations
            .iter()
            .map(|t| t.y)
            .reduce(f32::max)
            .unwrap_or(0.0),
        min_z: translations
            .iter()
            .map(|t| t.z)
            .reduce(f32::min)
            .unwrap_or(0.0),
        max_z: translations
            .iter()
            .map(|t| t.z)
            .reduce(f32::max)
            .unwrap_or(0.0),
    };

    let res = build_tree_impl(bodies, 0, root_box);

    match tree.get_single_mut() {
        Ok(mut prev) => *prev = res,
        Err(QuerySingleError::MultipleEntities(_)) => panic!(),
        Err(QuerySingleError::NoEntities(_)) => {
            commands.spawn(res);
        }
    };
}

fn build_tree_impl(mut bodies: Vec<Body>, depth: usize, bbox: shape::Box) -> Tree {
    if bodies.len() <= 32 {
        return Tree::Leaf(bodies);
    }

    let normal = Vec3::AXES[depth % 3];

    let mid = bodies.len() / 2;

    let (neg, eq, pos) = bodies.select_nth_unstable_by(mid, |Body(a, _), Body(b, _)| {
        match a.dot(normal).partial_cmp(&b.dot(normal)) {
            Some(o) => o,
            None => {
                dbg!((a, b));
                panic!()
            }
        }
    });

    let neg = neg.to_vec();
    let pos = [pos, &mut [*eq]].concat();
    let cutoff = eq.0.dot(normal);

    let mass = bodies.iter().fold(0.0, |a, b| a + b.1);
    let center_of_mass = bodies.iter().fold(Vec3::ZERO, |a, b| a + b.0 * b.1) / mass;

    let (neg_bbox, pos_bbox) = match depth % 3 {
        0 => (
            shape::Box {
                max_x: cutoff,
                ..bbox
            },
            shape::Box {
                min_x: cutoff,
                ..bbox
            },
        ),
        1 => (
            shape::Box {
                max_y: cutoff,
                ..bbox
            },
            shape::Box {
                min_y: cutoff,
                ..bbox
            },
        ),
        2 => (
            shape::Box {
                max_z: cutoff,
                ..bbox
            },
            shape::Box {
                min_z: cutoff,
                ..bbox
            },
        ),
        _ => panic!(),
    };

    Tree::Node {
        neg: Box::new(build_tree_impl(neg, depth + 1, neg_bbox)),
        pos: Box::new(build_tree_impl(pos, depth + 1, pos_bbox)),
        center_of_mass,
        mass,
        normal,
        cutoff,
        bbox,
    }
}

fn draw_boxes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    tree: Query<&Tree>,
    boxes: Query<Entity, With<Wireframe>>,
) {
    for entity in boxes.iter() {
        commands.entity(entity).despawn();
    }

    let material = materials.add(StandardMaterial {
        unlit: true,
        alpha_mode: AlphaMode::Mask(2.),
        ..default()
    });

    let tree = match tree.get_single() {
        Ok(tree) => tree,
        Err(_) => return (),
    };

    let mut queue = vec![tree];

    while let Some(node) = queue.pop() {
        match node {
            Tree::Leaf(_) => {}
            Tree::Node { bbox, neg, pos, .. } => {
                // queue.push(neg);
                queue.push(pos);
                let mesh = meshes.add((*bbox).into());
                commands.spawn((
                    PbrBundle {
                        mesh,
                        material: material.clone(),
                        ..default()
                    },
                    // This enables wireframe drawing on this entity
                    Wireframe,
                ));
            }
        }
    }
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
