use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
/// Wick - Model Weight Quantization Library

#[pymodule]
mod wick {
    use super::*;
    use candle_core::{DType, Result, Tensor};

    pub(crate) struct SymQuantizedTensor {
        tensor: Tensor, // U8
        absmax: f32,
    }

    impl SymQuantizedTensor {
        fn restore_to_f32(&self) -> Result<Tensor> {
            // map from [0, 255] to [-1, 1]
            let f32_normalized = self
                .tensor
                .to_dtype(DType::F32)?
                .affine(2.0 / 255.0, -1.0)?;

            // use absmax to restore original value
            f32_normalized.affine(self.absmax.into(), 0.0)
        }
    }

    pub(crate) fn candle_err(e: candle_core::Error) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }

    /// Find the absolute max in a given tensor
    /// Returns a 0-dim candle::Tensor containing
    /// the absolute max value in original precision
    /// Example:
    /// [[-9.0, -1.0], [0.0, 2.0], [6.5, 8.0]] -> [9.0]
    pub(crate) fn absmax(tensor: &Tensor) -> Result<f32> {
        let tensor_f32 = tensor.to_dtype(DType::F32)?;
        let abs_max = tensor_f32.abs()?.max_all()?.to_scalar::<f32>();
        abs_max
    }

    /// Return the tensor normalized by a norm
    /// Any element in the result tensor range in [-1.0, 1.0]
    ///
    /// Example:
    /// [[-9.0, -1.0], [0.0, 2.0], [6.5, 8.0]]
    /// -> norm = 9.0
    /// -> [[-1.0, -0.1111], [0.0, 0.2222], [0.7222, 0.8889]]
    pub(crate) fn normalize(tensor: &Tensor, norm: f32) -> Result<Tensor> {
        let normalized = tensor.affine((1.0 / norm).into(), 0.0);
        normalized
    }

    /// set 0b10000000 (128) as the zero point and map any value from [-1, 1] to [0, 255]
    pub(crate) fn symmetric_u8_quantize(normalized_tensor: Tensor) -> Result<Tensor> {
        normalized_tensor.affine(127.0, 128.0)?.to_dtype(DType::U8)
    }

    /// Accept tensor, quantize it into unsigned int8 (u8) type tensor
    /// It computes a mapping [tensor.min, tensor.max] -> [0, 255]
    /// Return quantized Tensor<U8>
    pub(crate) fn quantize_int8(tensor: &Tensor) -> Result<SymQuantizedTensor> {
        let absmax = absmax(&tensor)?;
        let normalized_tensor = normalize(&tensor, absmax)?.to_dtype(DType::F32)?;
        //todo!("detect waste ratio to choose asym/sym strategy dynamically");

        // return the symmetrically quantized tensor as a struct
        Ok(SymQuantizedTensor {
            tensor: symmetric_u8_quantize(normalized_tensor)?,
            absmax,
        })
    }

    pub(crate) fn dequantize_int8(quantized_tensor: &SymQuantizedTensor) -> Result<Tensor> {
        quantized_tensor.restore_to_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::wick::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_absmax() {
        // Test with positive and negative values
        let tensor = Tensor::new(&[[-9.0f32, -1.0], [0.0, 2.0], [6.5, 8.0]], &Device::Cpu).unwrap();
        let result = absmax(&tensor).unwrap();
        assert_eq!(result, 9.0);

        // Test with all negative values
        let tensor = Tensor::new(&[-1.0f32, -5.0, -3.0], &Device::Cpu).unwrap();
        let result = absmax(&tensor).unwrap();
        assert_eq!(result, 5.0);

        // Test with all positive values
        let tensor = Tensor::new(&[1.0f32, 5.0, 3.0], &Device::Cpu).unwrap();
        let result = absmax(&tensor).unwrap();
        assert_eq!(result, 5.0);

        // Test with single element
        let tensor = Tensor::new(&[-42.0f32], &Device::Cpu).unwrap();
        let result = absmax(&tensor).unwrap();
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_normalize() {
        let tensor = Tensor::new(&[[-9.0f32, -1.0], [0.0, 2.0], [6.5, 8.0]], &Device::Cpu).unwrap();
        let norm = 9.0;
        let result = normalize(&tensor, norm).unwrap();

        // Expected: [[-1.0, -0.1111], [0.0, 0.2222], [0.7222, 0.8889]]
        let expected = vec![
            vec![-1.0, -1.0 / 9.0],
            vec![0.0, 2.0 / 9.0],
            vec![6.5 / 9.0, 8.0 / 9.0],
        ];

        let result_vec = result.to_vec2::<f32>().unwrap();
        for (i, row) in result_vec.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    (val - expected[i][j]).abs() < 1e-4,
                    "Mismatch at [{}, {}]: got {}, expected {}",
                    i,
                    j,
                    val,
                    expected[i][j]
                );
            }
        }

        // Test edge case: normalize by 1.0 (identity for range [-1, 1])
        let tensor = Tensor::new(&[0.5f32, -0.5, 1.0, -1.0], &Device::Cpu).unwrap();
        let result = normalize(&tensor, 1.0).unwrap();
        let result_vec = result.to_vec1::<f32>().unwrap();
        let expected = vec![0.5, -0.5, 1.0, -1.0];
        for (i, &val) in result_vec.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_symmetric_u8_quantize() {
        // Test zero point mapping: 0.0 -> 128
        let tensor = Tensor::new(&[0.0f32], &Device::Cpu).unwrap();
        let result = symmetric_u8_quantize(tensor).unwrap();
        assert_eq!(result.to_vec1::<u8>().unwrap()[0], 128);

        // Test boundary values: -1.0 -> 1, 1.0 -> 255
        let tensor = Tensor::new(&[-1.0f32, 1.0], &Device::Cpu).unwrap();
        let result = symmetric_u8_quantize(tensor).unwrap();
        let result_vec = result.to_vec1::<u8>().unwrap();
        assert_eq!(result_vec[0], 1); // -1.0 * 127 + 128 = 1
        assert_eq!(result_vec[1], 255); // 1.0 * 127 + 128 = 255

        // Test intermediate values
        let tensor = Tensor::new(&[-0.5f32, 0.5], &Device::Cpu).unwrap();
        let result = symmetric_u8_quantize(tensor).unwrap();
        let result_vec = result.to_vec1::<u8>().unwrap();
        // -0.5 * 127 + 128 = 64.5 -> 64 (truncated)
        // 0.5 * 127 + 128 = 191.5 -> 191 (truncated)
        assert_eq!(result_vec[0], 64);
        assert_eq!(result_vec[1], 191);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // This test should verify that quantizing then dequantizing a tensor
        // produces values close to the original (within quantization error bounds)
        fn ss_loss(original_tensor: &Tensor, std: &f32, description: &String) {
            let quantized_tensor = quantize_int8(&original_tensor).unwrap();
            let dequantized_tensor = dequantize_int8(&quantized_tensor).unwrap();
            let sse = original_tensor
                .sub(&dequantized_tensor).unwrap()
                .sqr().unwrap()
                .sum_all().unwrap()
                .to_scalar::<f32>().unwrap();

            // Mean squared error - normalizes by count
            let mse = sse / original_tensor.elem_count() as f32;
            // Root mean squared error - back to original units
            let rmse = mse.sqrt();
            // Normalized RMSE - dimensionless, comparable across scales
            // Values closer to 0 are better; 1.0 would mean your error is as large as the data's spread
            let nrmse = rmse / std;

            // SNR in decibels - higher is better
            // A well-quantized signal might have SNR of 40-60 dB
            let signal_power = original_tensor.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
            let noise_power = mse;  // MSE is essentially the noise power
            let snr_db = 10.0 * (signal_power / noise_power).log10();
            println!("[INFO] SSE: {sse:.4}, MSE: {mse:.4}, NRMSE: {nrmse:.4}, SNR_dB: {snr_db:.0}dB for {description}");
            assert!(sse > 0.0);
        }

        macro_rules! test_normdist_loss {
            ($mean:expr, $std:expr, $loss_fn:ident) => {
                let mean: f64 = $mean;
                let std: f64 = $std;
                let std_32 = std as f32;
                let tensor = Tensor::randn(mean, std, (10, 10), &Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let msg = format!(
                    "a tensor with Z~({}, {}) (mean, standard deviation)\n",
                    mean, std
                );
                $loss_fn(&tensor, &std_32, &msg);
            };
        }

        test_normdist_loss!(0.0, 0.3333, ss_loss);
        test_normdist_loss!(1.0, 0.3333, ss_loss);
        test_normdist_loss!(-1.0, 0.3333, ss_loss);
        test_normdist_loss!(0.0, 10.0, ss_loss);
        test_normdist_loss!(0.0, 100.0, ss_loss);
        test_normdist_loss!(0.0, 1000.0, ss_loss);
        test_normdist_loss!(1000.0, 10.0, ss_loss);
        // Interesting finding: 
        // I use very extreme data here, but the relative loss is still negligable,
        // AS LONG AS the original weights follows normal distribution
    }
}
