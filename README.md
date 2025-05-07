# CT Playground

A repository for experimenting with Computed Tomography (CT) reconstruction techniques, focusing on the comparison between different implementations of the Radon transform.

## Fourier-Based CT Reconstruction

The main focus of this repository is exploring different approaches to CT reconstruction, particularly comparing the Fourier Slice Theorem-based implementation with traditional methods.

### tigre_vs_ft.ipynb

This notebook implements and compares three different methods for computing the Radon transform (sinogram generation) for CT reconstruction:

For detailed analysis of how 1D projections align at different angles, refer to the `scikit_vs_our.ipynb` notebook (cell 4), which demonstrates excellent alignment between our Fourier method and scikit-image's radon transform across multiple projection angles. The visualizations confirm that both implementations produce nearly identical projection profiles, validating the mathematical correctness of our Fourier Slice implementation.

1. **Fourier Slice Theorem (FFT-based)**: A custom implementation that uses the Fourier Slice Theorem to compute projections in the frequency domain using PyTorch.
2. **scikit-image Radon**: Uses scikit-image's radon transform implementation.
3. **TIGRE GPU-accelerated**: Employs the TIGRE toolkit's GPU-accelerated projection algorithm.

#### Key Features

- Custom Shepp-Logan phantom generator that allows flexible resolution settings
- PyTorch-based implementation of the Fourier Slice Theorem for fast Radon transform
- GPU acceleration for both the Fourier method and TIGRE implementation
- Direct visual and running time comparison of sinograms from all three methods

#### Results

The Fourier method shows alignment with both scikit-image and TIGRE versions in terms of sinogram similarity. 

#### Performance

Performance benchmarks from the notebook using a T4 GPU on a custom Shepp-Logan phantom (1024×1024 resolution):

```
Time taken for FFT Radon transform (device: cuda): 0.0592 seconds
Time taken for skimage Radon transform: 5.8786 seconds
Time taken for TIGRE Radon transform: 0.1634 seconds
```

The Fourier Slice Theorem implementation achieves the fastest performance, outperforming even the GPU-accelerated TIGRE implementation, while maintaining comparable accuracy up to some artifacts possibly coming from interpolation on freq domain.

#### Implementation Details

- A custom Shepp-Logan phantom generator function is implemented to automatically set resolution as needed
- The FFT-based method uses PyTorch's grid_sample for better interpolation in the frequency domain
- All experiments were conducted on a T4 GPU with a 1024×1024 resolution phantom

## Repository Structure

- `baby-ct/` - Neural network and classical approaches to CT reconstruction
- `reverse-fourier-slice/` - Comparison of real-space CT projection with Fourier slice theorem
- `TIGRE/` - TIGRE toolkit for GPU-accelerated CT reconstruction (i did some changes on visualization scripts fo non-gui instances)

## Dependencies

- PyTorch
- NumPy
- scikit-image
- TIGRE
- Matplotlib