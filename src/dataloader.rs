use crate::quantize::{QuantizedTensors, dequantize_int8, quantize_int8};
use candle_core::{Device, Result, Tensor, safetensors};
use std::collections::HashMap;
use std::path::Path;

fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<HashMap<String, Tensor>> {
    let model_weight_map = candle_core::safetensors::load(path, device)?;
    Ok(model_weight_map)
}

pub fn quantize_model(
    model_weight_map: &HashMap<String, Tensor>,
) -> Result<HashMap<String, QuantizedTensors>> {
    let mut quantized_map: HashMap<String, QuantizedTensors> = HashMap::new();
    for (key, tensor) in model_weight_map {
        quantized_map.insert(key.to_string(), quantize_int8(&tensor));
    }
    Ok(quantized_map)
}

/// measure the quantized precision loss with Normalized Root Mean Squared Error (NRMSE)
pub fn measure_layer_loss<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, f32>> {
    let model_weight_map = load(path, device)?;
    let quantized_map = quantize_model(&model_weight_map)?;
    let mut dequantized_map: HashMap<String, Tensor> = HashMap::new();
    for (key, quantized_tensor) in quantized_map {
        dequantized_map.insert(key, dequantize_int8(quantized_tensor).unwrap());
    }
    let mut nrmse_map: HashMap<String, f32> = HashMap::new();

    for (key, _) in &dequantized_map {
        let original_tensor = model_weight_map.get(key).unwrap();
        let dequantized_tensor = dequantized_map.get(key).unwrap();
        let sse: f32 = original_tensor
            .sub(&dequantized_tensor)?
            .sqr()?
            .sum_all()?
            .to_scalar()?;
        let mse = sse / original_tensor.elem_count() as f32;
        let rmse = mse.sqrt();
        let nrmse = rmse / get_std(&original_tensor)?;
        nrmse_map.insert(key.to_string(), nrmse);
    }
    Ok(nrmse_map)
}

fn get_std(tensor: &Tensor) -> Result<f32> {
    let mean = tensor.mean_all()?.to_scalar::<f32>()?;

    let tensor = tensor.to_dtype(candle_core::DType::F32)?;
    let mean_tensor = Tensor::zeros_like(&tensor)?
        .affine(0.0, mean.into())?
        .to_dtype(candle_core::DType::F32)?;

    let std = tensor
        .sub(&mean_tensor)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?;

    Ok(std)
}

mod tests {
    use candle_core::MetalStorage;

    use super::*;

    #[test]
    fn test_load_model() -> Result<()> {
        let bert = candle_core::safetensors::load(
            "/Users/liangzicheng/Documents/GitHub/wick/data/bert-base-chinese.safetensors",
            &Device::Cpu,
        )?;
        assert!(!bert.is_empty());

        let gpt2 = load(
            "/Users/liangzicheng/Documents/GitHub/wick/data/gpt2.safetensors",
            &Device::Cpu,
        )?;
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
    fn test_quantize_loss() -> Result<()> {
        let loss_map = measure_layer_loss(
            "/Users/liangzicheng/Documents/GitHub/wick/data/gpt2.safetensors",
            &Device::Cpu,
        );
        for (k, v) in loss_map? {
            println!("[{}] NRMSE for {} : {}", "gpt2", k, v);
        }
        Ok(())
    }
}
