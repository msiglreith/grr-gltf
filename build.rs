use spirv_builder::{MemoryModel, SpirvBuilder};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    SpirvBuilder::new("shader")
        .spirv_version(1, 0)
        .memory_model(MemoryModel::GLSL450)
        .build()?;
    Ok(())
}
