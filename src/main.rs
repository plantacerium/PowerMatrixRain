use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    keyboard::{Key, NamedKey},
};
use wgpu::util::DeviceExt;
use rand::Rng;
use glam::{Mat4, Vec3};
use serde::Deserialize;
use std::fs;
use ab_glyph::{FontRef, Font, PxScale, ScaleFont};

const NUM_STREAMS: usize = 150; // Much lower density for clearer desktop visibility
const GLYPH_HEIGHT: u32 = 40; // Larger glyphs
const BASE_SPEED: f32 = 12.0;

#[derive(Deserialize)]
struct WordList {
    words: Vec<String>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    pos: [f32; 3],
    opacity: f32,
    glyph_index: u32,
    is_head: u32,
}

struct Stream {
    x: f32,
    y: f32,
    z: f32,
    speed: f32,
    len: usize,
    chars: Vec<u32>,
    direction: f32,
    phase: f32,
}

// --- SHADER SOURCE ---
const TEXTURE_SHADER: &str = r#"
struct CameraUniform { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

struct VertexOutput {
@builtin(position) clip_position: vec4<f32>,
@location(0) tex_coords: vec2<f32>,
@location(1) opacity: f32,
@location(2) is_head: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) in_idx: u32, @location(5) pos: vec3<f32>, @location(6) opacity: f32, @location(7) glyph: u32, @location(8) is_head: u32) -> VertexOutput {
let w = 24.0; let h = 40.0; // Larger quads
var uv = vec2(0.0); var off = vec2(0.0);
if (in_idx == 0u) { off = vec2(0., 0.); uv = vec2(0., 0.); }
else if (in_idx == 1u) { off = vec2(0., -h); uv = vec2(0., 1.); }
else if (in_idx == 2u) { off = vec2(w, 0.); uv = vec2(1., 0.); }
else if (in_idx == 3u) { off = vec2(w, 0.); uv = vec2(1., 0.); }
else if (in_idx == 4u) { off = vec2(0., -h); uv = vec2(0., 1.); }
else if (in_idx == 5u) { off = vec2(w, -h); uv = vec2(1., 1.); }

let uv_final = (uv / 16.0) + vec2(f32(glyph % 16u) / 16.0, f32(glyph / 16u) / 16.0);
var out: VertexOutput;
out.clip_position = camera.view_proj * vec4(pos.x + off.x, pos.y + off.y, pos.z, 1.0);
out.tex_coords = uv_final;
out.opacity = opacity;
out.is_head = f32(is_head);
return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
let sample = textureSample(t_diffuse, s_diffuse, in.tex_coords);
if (sample.r < 0.2) { discard; }

// Solar Plexus Yellow Palette
var color = vec3(1.0, 0.84, 0.0); // Gold base
if (in.is_head > 0.5) { 
    color = vec3(1.0, 1.0, 0.9); // Bright white-yellow spark
} else {
    // Gradient from Gold to Deep Amber/Brown based on opacity
    color = mix(vec3(0.4, 0.2, 0.0), vec3(1.0, 0.84, 0.0), in.opacity);
}

// Glow calculation (fake distance field from texture alpha)
let glow = sample.r * sample.r * 1.5; 
let alpha_final = sample.r * in.opacity;

return vec4(color * glow, alpha_final);
}
"#;

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    instance_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    bind_group_camera: wgpu::BindGroup,
    bind_group_texture: wgpu::BindGroup,
    streams: Vec<Stream>,
    instances: Vec<InstanceRaw>,
    start_time: std::time::Instant,
    speed_multiplier: f32,
}

impl State {
    async fn new(window: &winit::window::Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(window).unwrap()) }.unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { compatible_surface: Some(&surface), ..Default::default() }).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();

        let caps = surface.get_capabilities(&adapter);
        
        // FIX: Find a supported alpha mode, preferring PreMultiplied
        let alpha_mode = caps.alpha_modes.iter()
            .find(|&m| *m == wgpu::CompositeAlphaMode::PreMultiplied)
            .cloned()
            .or_else(|| caps.alpha_modes.iter().find(|&m| *m == wgpu::CompositeAlphaMode::PostMultiplied).cloned())
            .unwrap_or(caps.alpha_modes[0]);
        
        println!("Selected Alpha Mode: {:?} (Available: {:?})", alpha_mode, caps.alpha_modes);
            
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode, // Dynamically selected
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let toml_str = fs::read_to_string("words.toml").unwrap_or_else(|_| "words = []".to_string());
        let decoded: WordList = toml::from_str(&toml_str).unwrap_or(WordList { words: vec![] });
        let words = if decoded.words.is_empty() { vec!["MATRIX".to_string()] } else { decoded.words };

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera"), size: std::mem::size_of::<CameraUniform>() as u64,
                                                 usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (NUM_STREAMS * 60 * 24) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                                                   label: None, mapped_at_creation: false
        });

        // Texture generation with Font
        let mut tex_data = vec![0u8; 256 * 256 * 4];
        
        // Use font-kit to find a monospace font
        let font_data = {
            use font_kit::source::SystemSource;
            use font_kit::family_name::FamilyName;
            use font_kit::properties::Properties;
            use font_kit::handle::Handle;

            let source = SystemSource::new();
            let handle = source.select_best_match(&[FamilyName::Monospace], &Properties::new())
                .unwrap_or_else(|_| {
                    println!("Failed to find monospace font, trying SansSerif");
                    source.select_best_match(&[FamilyName::SansSerif], &Properties::new()).unwrap()
                });

            match handle {
                Handle::Path { path, .. } => fs::read(path).ok(),
                Handle::Memory { bytes, .. } => Some(bytes.to_vec()),
            }
        };

        if let Some(data) = font_data {
             if let Ok(font) = FontRef::try_from_slice(&data) {
                let scale = PxScale::from(24.0); // Increase font size for texture
                let scaled_font = font.as_scaled(scale);
                
                for i in 0..256 {
                    // Map 0..256 to glyphs.
                    let ch = if i >= 32 && i < 127 {
                        std::char::from_u32(i as u32).unwrap()
                    } else {
                        // Map 0-31 and 127+ to extended ASCII or randomness
                         std::char::from_u32(((i % 94) + 33) as u32).unwrap()
                    };

                    let glyph = scaled_font.scaled_glyph(ch);
                    if let Some(outlined) = scaled_font.outline_glyph(glyph) {
                        let _bounds = outlined.px_bounds();
                        let grid_x = (i % 16) * 16;
                        let grid_y = (i / 16) * 16;
                        
                        // Center in 16x16 cell (Note: 24px font on 16px cell might overflow, we might need to clamp or reduce scale if artifacts appear. 
                        // Actually, 256x256 texture with 16x16 cells means 16px max size. 
                        // Reverting scale to 14.0 to fit safely, but we render QUADS larger.)
                        let idx_scale = PxScale::from(14.0); 
                        let scaled_font_fit = font.as_scaled(idx_scale);
                         let glyph_fit = scaled_font_fit.scaled_glyph(ch);
                         if let Some(outlined_fit) = scaled_font_fit.outline_glyph(glyph_fit) {
                             let bounds = outlined_fit.px_bounds();
                              let width = bounds.width() as i32;
                            let height = bounds.height() as i32;
                            let off_x = (16 - width) / 2;
                            let off_y = (16 - height) / 2;

                            outlined_fit.draw(|x, y, v| {
                                let px = grid_x + x + off_x as u32;
                                let py = grid_y + y + off_y as u32; // basic centering
                                if px < 256 && py < 256 {
                                    let idx = (py as usize * 256 + px as usize) * 4;
                                    let val = (v * 255.0) as u8;
                                    tex_data[idx] = val;
                                    tex_data[idx+1] = val;
                                    tex_data[idx+2] = val;
                                    tex_data[idx+3] = val;
                                }
                            });
                         }
                    }
                }
             } else {
                 println!("Failed to parse font");
             }
        } else {
             println!("Failed to load font file");
        }

        let tex = device.create_texture_with_data(&queue, &wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::TEXTURE_BINDING,
                label: None, view_formats: &[],
        }, wgpu::util::TextureDataOrder::LayerMajor, &tex_data);

        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let cam_l = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }], label: None });
        let tex_l = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering), count: None }
        ], label: None });

        let bind_group_camera = device.create_bind_group(&wgpu::BindGroupDescriptor { layout: &cam_l, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }], label: None });
        let bind_group_texture = device.create_bind_group(&wgpu::BindGroupDescriptor { layout: &tex_l, entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tex_view) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) }], label: None });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { bind_group_layouts: &[&cam_l, &tex_l], ..Default::default() })),
                                                            vertex: wgpu::VertexState { module: &device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(TEXTURE_SHADER.into()) }), entry_point: "vs_main", buffers: &[wgpu::VertexBufferLayout { array_stride: 24, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![5 => Float32x3, 6 => Float32, 7 => Uint32, 8 => Uint32] }] },
                                                            fragment: Some(wgpu::FragmentState { module: &device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(TEXTURE_SHADER.into()) }), entry_point: "fs_main", targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
                                                            primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None,
        });

        let mut streams = Vec::new();
        let mut rng = rand::thread_rng();
        for i in 0..NUM_STREAMS {
            let w_str = &words[i % words.len()];
            streams.push(Stream {
                x: (i as f32 / NUM_STREAMS as f32) * (size.width as f32 * 2.5) - size.width as f32,
                         y: rng.gen_range(-1000.0..1000.0),
                         z: rng.gen_range(-400.0..50.0),
                         speed: rng.gen_range(0.8..1.2) * BASE_SPEED,
                         len: w_str.chars().count(),
                         chars: w_str.chars().map(|c| c as u32 % 256).collect(),
                         direction: if rng.gen_bool(0.1) { -1.0 } else { 1.0 },
                         phase: rng.gen_range(0.0..std::f32::consts::PI * 2.0),
            });
        }

        Self {
            surface, device, queue, config, size, render_pipeline,
            instance_buffer, camera_buffer, bind_group_camera, bind_group_texture,
            streams, instances: Vec::new(), start_time: std::time::Instant::now(),
            speed_multiplier: 1.0,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            for (i, s) in self.streams.iter_mut().enumerate() {
                s.x = (i as f32 / NUM_STREAMS as f32) * (new_size.width as f32 * 2.5) - new_size.width as f32;
            }
        }
    }
    
    pub fn adjust_speed(&mut self, delta: f32) {
        self.speed_multiplier = (self.speed_multiplier + delta).max(0.1).min(10.0);
        println!("Speed Multiplier: {:.1}x", self.speed_multiplier);
    }

    pub fn update(&mut self) {
        let mut rng = rand::thread_rng();
        let elapsed = self.start_time.elapsed().as_secs_f32();
        self.instances.clear();

        for s in &mut self.streams {
            let rhythm = 1.0 + 0.3 * (elapsed * 0.5 + s.phase).sin();
            s.y -= s.speed * rhythm * s.direction * self.speed_multiplier;

            let bound_y = self.size.height as f32 * 1.5;
            if s.y < -bound_y { s.y = bound_y; }
            if s.y > bound_y { s.y = -bound_y; }

            for i in 0..s.len {
                let pos_y = if s.direction > 0.0 { s.y + (i as f32 * GLYPH_HEIGHT as f32) } else { s.y - (i as f32 * GLYPH_HEIGHT as f32) };
                self.instances.push(InstanceRaw {
                    pos: [s.x, pos_y, s.z],
                    opacity: 1.0 - (i as f32 / s.len as f32),
                                    glyph_index: if rng.gen_bool(0.005) { rng.gen_range(0..256) } else { s.chars[i] },
                                    is_head: if i == 0 { 1 } else { 0 },
                });
            }
        }

        let proj = Mat4::perspective_rh(45.0f32.to_radians(), self.size.width as f32 / self.size.height as f32, 0.1, 4000.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 1000.0), Vec3::ZERO, Vec3::Y);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[CameraUniform { view_proj: (proj * view).to_cols_array_2d() }]));
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instances));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }), store: wgpu::StoreOp::Store }
                })], ..Default::default()
            });
            rp.set_pipeline(&self.render_pipeline);
            rp.set_bind_group(0, &self.bind_group_camera, &[]);
            rp.set_bind_group(1, &self.bind_group_texture, &[]);
            rp.set_vertex_buffer(0, self.instance_buffer.slice(..));
            rp.draw(0..6, 0..self.instances.len() as u32);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        Ok(())
    }
}

fn main() {
    let el = EventLoop::new().unwrap();
    let window = WindowBuilder::new().with_title("Matrix").with_transparent(true).with_decorations(false).build(&el).unwrap();
    let mut state = pollster::block_on(State::new(&window));
    
    el.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(s) => state.resize(s),
                WindowEvent::KeyboardInput { event: KeyEvent { state: ElementState::Pressed, logical_key, .. }, .. } => {
                    match logical_key.as_ref() {
                        Key::Named(NamedKey::ArrowUp) => state.adjust_speed(0.1),
                        Key::Named(NamedKey::ArrowDown) => state.adjust_speed(-0.1),
                        Key::Named(NamedKey::Escape) => target.exit(),
                        Key::Named(NamedKey::F11) => {
                             // Fix for Linux Transparency: "Fullscreen" often disables compositing (opaque background).
                             // "Maximized" with no decorations keeps transparency active.
                             let is_maximized = window.is_maximized();
                             window.set_maximized(!is_maximized);
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            Event::AboutToWait => { state.update(); state.render().unwrap(); }
            _ => ()
        }
    }).unwrap();
}
