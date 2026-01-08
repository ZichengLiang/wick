use crate::quantize::{quantize_int8, QuantizedTensor};
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<HashMap<String, Tensor>> {
    let model_weight_map = candle_core::safetensors::load(path, device)?;
    Ok(model_weight_map)
}

pub fn quantize_model(
    model_weight_map: &HashMap<String, Tensor>,
) -> Result<HashMap<String, QuantizedTensor>> {
    let mut quantized_map: HashMap<String, QuantizedTensor> = HashMap::new();
    for (key, tensor) in model_weight_map {
        quantized_map.insert(key.to_string(), quantize_int8(&tensor)?);
    }
    Ok(quantized_map)
}

/// measure the quantized precision loss with NRMSE(std)
pub fn measure_layer_loss<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, f32>> {
    let model_weight_map = load(path, device)?;
    let quantized_map = quantize_model(&model_weight_map)?;
    let mut dequantized_map: HashMap<String, Tensor> = HashMap::new();
    for (key, quantized_tensor) in quantized_map {
        dequantized_map.insert(key, quantized_tensor.dequantize().unwrap());
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
