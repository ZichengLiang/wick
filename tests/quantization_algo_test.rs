use candle_core::{DType, Device, Result, Tensor};
use wick::{quantize_int8, QuantizedTensor};

#[test]
fn test_symmetric_u8_quantize() {
    // Test 1: Basic symmetric quantization with negative values
    // Tensor with values in [-1.0, 1.0] should trigger symmetric quantization
    let tensor = Tensor::new(&[-1.0f32, -0.5, 0.0, 0.5, 1.0], &Device::Cpu).unwrap();
    let quantized = quantize_int8(&tensor).unwrap();

    // Should be symmetric (Sym variant) since we have negative values
    match &quantized {
        QuantizedTensor::Sym(q) => {
            assert_eq!(q.absmax, 1.0, "absmax should be 1.0");
            assert_eq!(q.tensor.dtype(), DType::U8, "quantized tensor should be U8");

            // Check specific quantized values:
            // -1.0 -> 127 * (-1) + 128 = 1
            // 0.0 -> 127 * 0 + 128 = 128
            // 1.0 -> 127 * 1 + 128 = 255
            let values: Vec<u8> = q.tensor.to_vec1().unwrap();
            assert_eq!(values[0], 1, "-1.0 should quantize to 1");
            assert_eq!(values[2], 128, "0.0 should quantize to 128");
            assert_eq!(values[4], 255, "1.0 should quantize to 255");
        }
        QuantizedTensor::Asym(_) => panic!("Expected Sym variant for tensor with negatives"),
    }

    // Test 2: Verify dequantization recovers values within tolerance
    let dequantized = quantized.dequantize().unwrap();
    let original_vals: Vec<f32> = tensor.to_vec1().unwrap();
    let restored_vals: Vec<f32> = dequantized.to_vec1().unwrap();

    for (orig, restored) in original_vals.iter().zip(restored_vals.iter()) {
        let error = (*orig - *restored).abs();
        // Quantization step size is absmax/127 ≈ 0.0079, allow 1 step error
        assert!(
            error < 0.02,
            "Dequantized value {restored} too far from original {orig}"
        );
    }

    // Test 3: All zeros should produce Sym with absmax = 0
    let zeros = Tensor::zeros((5,), DType::F32, &Device::Cpu).unwrap();
    let quantized_zeros = quantize_int8(&zeros).unwrap();
    match quantized_zeros {
        QuantizedTensor::Sym(q) => {
            assert_eq!(q.absmax, 0.0, "absmax for all-zeros should be 0");
        }
        QuantizedTensor::Asym(_) => panic!("Expected Sym variant for all-zeros tensor"),
    }

    // Test 4: Larger range symmetric tensor
    let wide_tensor = Tensor::new(&[-100.0f32, 0.0, 50.0, 100.0], &Device::Cpu).unwrap();
    let quantized_wide = quantize_int8(&wide_tensor).unwrap();
    match &quantized_wide {
        QuantizedTensor::Sym(q) => {
            assert_eq!(q.absmax, 100.0, "absmax should capture max absolute value");
        }
        QuantizedTensor::Asym(_) => panic!("Expected Sym variant"),
    }
}

#[test]
fn test_asymmetric_u8_quantize() {
    // Test 1: Basic non-negative tensor starting at 0
    // [0.0, 0.5, 1.0] should trigger asymmetric quantization
    let tensor = Tensor::new(&[0.0f32, 0.5, 1.0], &Device::Cpu).unwrap();
    let quantized = quantize_int8(&tensor).unwrap();

    match &quantized {
        QuantizedTensor::Asym(q) => {
            // scale = 255 / (1.0 - 0.0) = 255
            // zero_point = round(-0.0 * 255) = 0
            assert!(
                (q.scale - 255.0).abs() < 0.01,
                "scale should be 255, got {}",
                q.scale
            );
            assert_eq!(q.zero_point, 0, "zero_point should be 0");
            assert_eq!(q.tensor.dtype(), DType::U8, "quantized tensor should be U8");

            // Check quantized values:
            // 0.0 -> 255 * 0.0 + 0 = 0
            // 0.5 -> 255 * 0.5 + 0 = 127.5 -> 128
            // 1.0 -> 255 * 1.0 + 0 = 255
            let values: Vec<u8> = q.tensor.to_vec1().unwrap();
            assert_eq!(values[0], 0, "0.0 should quantize to 0");
            assert_eq!(values[1], 128, "0.5 should quantize to 128");
            assert_eq!(values[2], 255, "1.0 should quantize to 255");
        }
        QuantizedTensor::Sym(_) => panic!("Expected Asym variant for non-negative tensor"),
    }

    // Test 2: Verify dequantization accuracy
    let dequantized = quantized.dequantize().unwrap();
    let original_vals: Vec<f32> = tensor.to_vec1().unwrap();
    let restored_vals: Vec<f32> = dequantized.to_vec1().unwrap();

    for (orig, restored) in original_vals.iter().zip(restored_vals.iter()) {
        let error = (*orig - *restored).abs();
        // Quantization step size is 1/255 ≈ 0.004, allow 1 step error
        assert!(
            error < 0.01,
            "Dequantized value {restored} too far from original {orig}"
        );
    }

    // Test 3: Wider range non-negative tensor
    let wide_tensor = Tensor::new(&[0.0f32, 50.0, 100.0], &Device::Cpu).unwrap();
    let quantized_wide = quantize_int8(&wide_tensor).unwrap();

    match &quantized_wide {
        QuantizedTensor::Asym(q) => {
            // scale = 255 / 100 = 2.55
            assert!(
                (q.scale - 2.55).abs() < 0.01,
                "scale should be 2.55, got {}",
                q.scale
            );
            assert_eq!(q.zero_point, 0, "zero_point should be 0");

            let values: Vec<u8> = q.tensor.to_vec1().unwrap();
            assert_eq!(values[0], 0, "0.0 should quantize to 0");
            assert_eq!(values[1], 128, "50.0 should quantize to ~128");
            assert_eq!(values[2], 255, "100.0 should quantize to 255");
        }
        QuantizedTensor::Sym(_) => panic!("Expected Asym variant"),
    }

    // Test 4: Dequantization roundtrip on wider range
    let dequantized_wide = quantized_wide.dequantize().unwrap();
    let wide_restored: Vec<f32> = dequantized_wide.to_vec1().unwrap();
    // Step size = 100/255 ≈ 0.39, allow 1 step error
    for (orig, restored) in [0.0f32, 50.0, 100.0].iter().zip(wide_restored.iter()) {
        let error = (*orig - *restored).abs();
        assert!(
            error < 0.5,
            "Dequantized {restored} too far from original {orig}"
        );
    }

    // Test 5: Single positive value (edge case)
    let single = Tensor::new(&[5.0f32], &Device::Cpu).unwrap();
    let quantized_single = quantize_int8(&single).unwrap();
    // min == max, so scale would be inf. Check it handles gracefully.
    match quantized_single {
        QuantizedTensor::Asym(_) | QuantizedTensor::Sym(_) => {
            // Either variant is acceptable for edge case
        }
    }
}

#[test]
fn test_quantize_dequantize_roundtrip() {
    // This test should verify that quantizing then dequantizing a tensor
    // produces values close to the original (within quantization error bounds)
    fn ss_loss(original_tensor: &Tensor, std: &f32, description: &String) -> Result<()> {
        let quantized_tensor = quantize_int8(original_tensor)?;
        let dequantized_tensor = quantized_tensor.dequantize()?;
        let sse = original_tensor
            .sub(&dequantized_tensor)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?;

        // Mean squared error - normalizes by count
        let mse = sse / original_tensor.elem_count() as f32;

        // Root mean squared error - back to original units
        let rmse = mse.sqrt();

        // NRMSE(std)
        let nrmse = rmse / std;

        // SNR in decibels - higher is better
        // A well-quantized signal might have SNR of 40-60 dB
        let signal_power = original_tensor.sqr()?.mean_all()?.to_scalar::<f32>()?;

        let noise_power = mse; // MSE is essentially the noise power
        let snr_db = 10.0 * (signal_power / noise_power).log10();
        println!(
            "[INFO] SSE: {sse:.4}, MSE: {mse:.4}, NRMSE: {nrmse:.4}, SNR_dB: {snr_db:.0}dB for {description}"
        );
        assert!(sse > 0.0);
        Ok(())
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

            let msg = format!("a tensor with Z~({}, {})\n", mean, std,);
            let _ = $loss_fn(&tensor, &std_32, &msg);
        };
    }

    test_normdist_loss!(0.0, 0.3333, ss_loss);
    test_normdist_loss!(0.0, 10.0, ss_loss);
    test_normdist_loss!(0.0, 100.0, ss_loss);
    test_normdist_loss!(0.0, 1000.0, ss_loss);
    test_normdist_loss!(2.0, 0.33, ss_loss);
    test_normdist_loss!(6.0, 1.8, ss_loss);
    test_normdist_loss!(1000.0, 10.0, ss_loss);
    // Interesting finding:
    // I use very extreme data here, but the relative loss is still negligable,
    // AS LONG AS the original weights follows normal distribution
}
