use candle_core::{DType, Result, Tensor};

#[derive(Debug, Clone)]
pub enum QuantizedTensor {
    Sym(SymQuantizedTensor),
    Asym(AsymQuantizedTensor),
}

#[derive(Debug, Clone)]
pub struct SymQuantizedTensor {
    pub tensor: Tensor, // U8
    pub absmax: f32,
}

#[derive(Debug, Clone)]
pub struct AsymQuantizedTensor {
    pub tensor: Tensor, // U8
    pub zero_point: u8,
    pub scale: f32,
}

impl QuantizedTensor {
    pub fn dequantize(&self) -> Result<Tensor> {
        match self {
            QuantizedTensor::Sym(q) => {
                // Formula: (q - 128) * (1 / 128) * absmax
                // Simplified: (absmax/128) * q - absmax
                let mul = q.absmax / 128.0;
                let add = -q.absmax;
                // Return restored tensor
                q.tensor
                    .to_dtype(DType::F32)?
                    .affine(mul.into(), add.into())
            }
            QuantizedTensor::Asym(q) => {
                // Formula: (q - zero_point) * scale
                // Simplified: q * 1/scale - zero_point * scale
                // Scale = range / 255
                let mul= 1.0 / q.scale;
                let add = q.zero_point as f32 * -mul;
                // Return restored tensor
                q.tensor
                    .to_dtype(DType::F32)?
                    .affine(mul.into(), add.into())
            }
        }
    }
}

fn get_min_max(tensor: &Tensor) -> Result<(f32, f32)> {
    // Reason: when calling min_all, max_all Candle flattens the tensor under the hood
    // to avoid flatten twice, we do this...
    let flattened = tensor.flatten_all()?;
    let min = flattened.min(0)?.to_scalar::<f32>()?;
    let max = flattened.max(0)?.to_scalar::<f32>()?;
    Ok((min, max))
}

pub fn quantize_int8(tensor: &Tensor) -> Result<QuantizedTensor> {
    let f32_tensor = tensor.to_dtype(DType::F32)?;
    let (min, max) = get_min_max(&f32_tensor)?;
    let non_negative = min >= 0.0;
    let absmax = max.abs().max(min.abs());
    let all_zero = absmax == 0.0;
    if all_zero {
        return Ok(QuantizedTensor::Sym(SymQuantizedTensor {
            tensor: f32_tensor,
            absmax: 0.0,
        }));
    } else if non_negative {
        // Asymmetric
        let scale = 255.0 / (max - min);
        let zero_point = (-min * scale).round();

        // quantized = scale * tensor + bias
        let quantized = f32_tensor
            .affine(scale.into(), zero_point.into())?
            .clamp(u8::MIN as f32, u8::MAX as f32)?
            .round()?
            .to_dtype(DType::U8);

        Ok(QuantizedTensor::Asym(AsymQuantizedTensor {
            tensor: quantized?,
            zero_point: zero_point as u8,
            scale,
        }))
    } else {
        // Symmetric
        let scale = 127.0 / absmax;
        let bias = 128.0;

        let quantized = f32_tensor
            .affine(scale as f64, bias as f64)?
            .clamp(u8::MIN as f32, u8::MAX as f32)? // SAFETY: clamp before cast
            .round()? // PRECISION: round nearest
            .to_dtype(DType::U8)?;

        Ok(QuantizedTensor::Sym(SymQuantizedTensor {
            tensor: quantized,
            absmax,
        }))
    }
}
