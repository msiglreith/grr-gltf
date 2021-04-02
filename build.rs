use spirv_builder::{MemoryModel, SpirvBuilder};

fn main() -> anyhow::Result<()> {
    let result = SpirvBuilder::new("shader")
        .spirv_version(1, 0)
        .memory_model(MemoryModel::GLSL450)
        .print_metadata(false)
        .build_multimodule()?;
    let directory = result
        .values()
        .next()
        .and_then(|path| path.parent())
        .unwrap();
    println!("cargo:rustc-env=spv={}", directory.to_str().unwrap());
    Ok(())
}
