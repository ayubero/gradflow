extern "C" __global__ void conv2d_kernel(
    const float* x_padded,
    const float* w,
    const float* bias,
    float* y,
    int b, int c_out, int h_out, int w_out,
    int c_in, int k_h, int k_w,
    int stride, int x_padded_stride_h, int x_padded_stride_w,
    int w_stride_oc, int w_stride_ic, int w_stride_h, int w_stride_w,
    int y_stride_b, int y_stride_oc, int y_stride_h, int y_stride_w
) {
    int oc = blockIdx.x; // Output channel
    int i = blockIdx.y; // Output height
    int j = blockIdx.z; // Output width

    // Compute b_idx from thread/block indices or use a 1D grid and map indices manually
    int b_idx = gridDim.z * blockIdx.y + blockIdx.z; // Flatten 3D grid to include batch

    if (oc < c_out && i < h_out && j < w_out && b_idx < b) {
        float val = 0.0f;
        for (int ic = 0; ic < c_in; ic++) {
            for (int kh = 0; kh < k_h; kh++) {
                for (int kw = 0; kw < k_w; kw++) {
                    int h_start = i * stride + kh;
                    int w_start = j * stride + kw;
                    int x_idx = b_idx * x_padded_stride_h * x_padded_stride_w * c_in +
                                ic * x_padded_stride_h * x_padded_stride_w +
                                h_start * x_padded_stride_w +
                                w_start;
                    int w_idx = oc * w_stride_oc +
                                ic * w_stride_ic +
                                kh * w_stride_h +
                                kw * w_stride_w;
                    val += x_padded[x_idx] * w[w_idx];
                }
            }
        }
        float bias_val = (bias != nullptr) ? bias[oc] : 0.0f;
        int y_idx = b_idx * y_stride_b +
                    oc * y_stride_oc +
                    i * y_stride_h +
                    j * y_stride_w;
        y[y_idx] = val + bias_val;
    }
}
