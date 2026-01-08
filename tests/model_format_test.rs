use candle_core::{Device, Result};
use std::path::PathBuf;
use wick::measure_layer_loss;

fn get_test_data_path(filename: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("data").join(filename)
}

#[test]
#[ignore] // Ignore by default - these tests require large model files
fn test_load_model() -> Result<()> {
    let bert_path = get_test_data_path("bert-base-chinese.safetensors");
    let gpt2_path = get_test_data_path("gpt2.safetensors");

    let bert = candle_core::safetensors::load(&bert_path, &Device::Cpu)?;
    assert!(!bert.is_empty());

    let gpt2 = candle_core::safetensors::load(&gpt2_path, &Device::Cpu)?;
    assert!(!gpt2.is_empty());

    for (key, value) in bert {
        println!("[Bert] Key: {}, Value: {:?}", key, value);
    }

    for (key, value) in gpt2 {
        println!("[GPT2] Key: {}, Value: {:?}", key, value);
    }

    Ok(())
}

#[test]
#[ignore] // Ignore by default - these tests require large model files
fn test_quantize_loss() -> Result<()> {
    let gpt2_path = get_test_data_path("gpt2.safetensors");

    let loss_map = measure_layer_loss(&gpt2_path, &Device::Cpu);
    for (k, v) in loss_map? {
        println!("[{}] NRMSE for {} : {}", "GPT2", k, v);
    }
    Ok(())
}
