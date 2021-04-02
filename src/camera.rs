use glace::Vec3;
use winit::event::ElementState;

pub struct InputMap {
    mouse1: ElementState,
    mouse_delta: (f32, f32),
}

impl InputMap {
    pub fn new() -> Self {
        InputMap {
            mouse1: ElementState::Released,
            mouse_delta: (0.0, 0.0),
        }
    }

    pub fn update_mouse1(&mut self, state: ElementState) {
        self.mouse1 = state;
    }

    pub fn update_mouse_motion(&mut self, (dx, dy): (f64, f64)) {
        if self.mouse1 == ElementState::Pressed {
            self.mouse_delta.0 += dx as f32;
            self.mouse_delta.1 += dy as f32;
        }
    }

    pub fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }

    pub fn reset_delta(&mut self) {
        self.mouse_delta = (0.0, 0.0);
    }
}

#[derive(Debug, Clone)]
pub struct Camera {
    rotation_speed: f32,
    position: Vec3<f32>,
    phi: f32,
    theta: f32,
}

impl Camera {
    pub fn new(position: Vec3<f32>, phi: f32, theta: f32) -> Self {
        Camera {
            position,
            rotation_speed: 1.0,
            phi,
            theta,
        }
    }

    pub fn update(&mut self, input: &InputMap) {
        let rotation = input.mouse_delta();

        self.phi -= rotation.0 * self.rotation_speed * 0.01;
        self.theta += rotation.1 * self.rotation_speed * 0.01;
    }

    pub fn position(&self) -> Vec3<f32> {
        self.position
    }

    pub fn view_dir(&self) -> Vec3<f32> {
        let (sp, cp) = self.phi.sin_cos();
        let (st, ct) = self.theta.sin_cos();

        Vec3 {
            x: ct * sp,
            y: st,
            z: ct * cp,
        }
    }
}
