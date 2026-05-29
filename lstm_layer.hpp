#ifndef LSTM_LAYER_HPP
#define LSTM_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

// =====================================================================
// HIGH-PERFORMANCE CORE COMPUTATION ENGINE
// =====================================================================
inline void _lstm_forward_compiled_core(
    int B, int I, int H,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& x,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& h_prev,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& c_prev,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& W_f,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& W_i,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& W_c,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& W_o,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U_f,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U_i,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U_c,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U_o,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& b_f,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& b_i,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& b_c,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& b_o,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& f_gate,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& i_gate,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& c_tilde,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& o_gate,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& h,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& c
) {
    // Concurrently distribute calculation sequence across available CPU threads
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int h_idx = 0; h_idx < H; ++h_idx) {
            // Initialize linear evaluations with gate bias thresholds
            float f_linear = b_f(0, h_idx);
            float i_linear = b_i(0, h_idx);
            float c_linear = b_c(0, h_idx);
            float o_linear = b_o(0, h_idx);

            // Accumulate input transformations
            for (int i_idx = 0; i_idx < I; ++i_idx) {
                float x_val = x(b, i_idx);
                f_linear += W_f(h_idx, i_idx) * x_val;
                i_linear += W_i(h_idx, i_idx) * x_val;
                c_linear += W_c(h_idx, i_idx) * x_val;
                o_linear += W_o(h_idx, i_idx) * x_val;
            }

            // Accumulate recurrent hidden transitions
            for (int j = 0; j < H; ++j) {
                float h_val = h_prev(b, j);
                f_linear += U_f(h_idx, j) * h_val;
                i_linear += U_i(h_idx, j) * h_val;
                c_linear += U_c(h_idx, j) * h_val;
                o_linear += U_o(h_idx, j) * h_val;
            }

            // Compute activation functions (Sigmoids & Hyperbolic Tangent)
            float f_val = 1.0f / (1.0f + std::exp(-f_linear));
            float i_val = 1.0f / (1.0f + std::exp(-i_linear));
            float c_tilde_val = std::tanh(c_linear);
            float o_val = 1.0f / (1.0f + std::exp(-o_linear));

            // Store intermediate activations required for backward pass execution
            f_gate(b, h_idx) = f_val;
            i_gate(b, h_idx) = i_val;
            c_tilde(b, h_idx) = c_tilde_val;
            o_gate(b, h_idx) = o_val;

            // Element-wise state updates
            float c_curr = f_val * c_prev(b, h_idx) + i_val * c_tilde_val;
            c(b, h_idx) = c_curr;
            h(b, h_idx) = o_val * std::tanh(c_curr);
        }
    }
}

inline void _lstm_backward_compiled_core(
    int B, int I, int H,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& x,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& h_prev,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& c_prev,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& c,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& W_f,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& W_i,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& W_c,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& W_o,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U_f,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U_i,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U_c,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U_o,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& f_gate,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& i_gate,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& c_tilde,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& o_gate,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dh,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dc_next,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dx,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dh_prev,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dc_prev,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dW_f,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dW_i,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dW_c,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dW_o,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dU_f,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dU_i,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dU_c,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dU_o,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& db_f,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& db_i,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& db_c,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& db_o
) {
    // Thread-isolated gradient buffers initialized to avoid cross-talk data races during parallel loops
    #pragma omp parallel
    {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dW_f_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(H, I);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dW_i_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(H, I);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dW_c_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(H, I);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dW_o_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(H, I);

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dU_f_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(H, H);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dU_i_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(H, H);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dU_c_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(H, H);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dU_o_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(H, H);

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> db_f_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(1, H);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> db_i_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(1, H);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> db_c_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(1, H);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> db_o_local = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(1, H);

        Eigen::VectorXf df_b(H);
        Eigen::VectorXf di_b(H);
        Eigen::VectorXf dc_tilde_b(H);
        Eigen::VectorXf do_b(H);

        #pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            // Pass A: Evaluate linear gate pre-activation gradients
            for (int h_idx = 0; h_idx < H; ++h_idx) {
                float dh_val = dh(b, h_idx);
                float o_val = o_gate(b, h_idx);
                float c_curr = c(b, h_idx);
                float tanh_c = std::tanh(c_curr);

                float do_val = dh_val * tanh_c * o_val * (1.0f - o_val);
                float dc_val = dh_val * o_val * (1.0f - tanh_c * tanh_c) + dc_next(b, h_idx);

                dc_prev(b, h_idx) = dc_val * f_gate(b, h_idx);

                float df_val = dc_val * c_prev(b, h_idx) * f_gate(b, h_idx) * (1.0f - f_gate(b, h_idx));
                float di_val = dc_val * c_tilde(b, h_idx) * i_gate(b, h_idx) * (1.0f - i_gate(b, h_idx));
                float dc_tilde_val = dc_val * i_gate(b, h_idx) * (1.0f - c_tilde(b, h_idx) * c_tilde(b, h_idx));

                df_b[h_idx] = df_val;
                di_b[h_idx] = di_val;
                dc_tilde_b[h_idx] = dc_tilde_val;
                do_b[h_idx] = do_val;

                db_f_local(0, h_idx) += df_val;
                db_i_local(0, h_idx) += di_val;
                db_c_local(0, h_idx) += dc_tilde_val;
                db_o_local(0, h_idx) += do_val;
            }

            // Pass B: Accumulate input parameter updates and compute dx
            for (int i_idx = 0; i_idx < I; ++i_idx) {
                float dx_accum = 0.0f;
                for (int h_idx = 0; h_idx < H; ++h_idx) {
                    dx_accum += df_b[h_idx] * W_f(h_idx, i_idx)
                              + di_b[h_idx] * W_i(h_idx, i_idx)
                              + dc_tilde_b[h_idx] * W_c(h_idx, i_idx)
                              + do_b[h_idx] * W_o(h_idx, i_idx);

                    dW_f_local(h_idx, i_idx) += df_b[h_idx] * x(b, i_idx);
                    dW_i_local(h_idx, i_idx) += di_b[h_idx] * x(b, i_idx);
                    dW_c_local(h_idx, i_idx) += dc_tilde_b[h_idx] * x(b, i_idx);
                    dW_o_local(h_idx, i_idx) += do_b[h_idx] * x(b, i_idx);
                }
                dx(b, i_idx) = dx_accum;
            }

            // Pass C: Accumulate recurrent parameter updates and compute dh_prev
            for (int j = 0; j < H; ++j) {
                float dh_prev_accum = 0.0f;
                for (int h_idx = 0; h_idx < H; ++h_idx) {
                    dh_prev_accum += df_b[h_idx] * U_f(h_idx, j)
                                   + di_b[h_idx] * U_i(h_idx, j)
                                   + dc_tilde_b[h_idx] * U_c(h_idx, j)
                                   + do_b[h_idx] * U_o(h_idx, j);

                    dU_f_local(h_idx, j) += df_b[h_idx] * h_prev(b, j);
                    dU_i_local(h_idx, j) += di_b[h_idx] * h_prev(b, j);
                    dU_c_local(h_idx, j) += dc_tilde_b[h_idx] * h_prev(b, j);
                    dU_o_local(h_idx, j) += do_b[h_idx] * h_prev(b, j);
                }
                dh_prev(b, j) = dh_prev_accum;
            }
        }

        // Thread block critical entry map combining regional calculations into output arrays
        #pragma omp critical
        {
            dW_f += dW_f_local; dW_i += dW_i_local; dW_c += dW_c_local; dW_o += dW_o_local;
            dU_f += dU_f_local; dU_i += dU_i_local; dU_c += dU_c_local; dU_o += dU_o_local;
            db_f += db_f_local; db_i += db_i_local; db_c += db_c_local; db_o += db_o_local;
        }
    }
}

// =====================================================================
// WRAPPER DEEP LEARNING LAYER STRUCTURAL CLASS OBJECT
// =====================================================================
class LSTM_Layer {
public:
    int B; // Batch capacity layout dimensions
    int I; // Input features mapping structural width
    int H; // Hidden recurrent cell processing depth

    // Feedforward Connection Target Parameter Weights
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W_f;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W_i;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W_c;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W_o;

    // Recurrent Structural Loop Parameter Weights
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_f;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_i;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_c;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_o;

    // Structural Bias Layer Threshold Arrays
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b_f;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b_i;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b_c;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b_o;

    // Intermediate Feedforward Gate Trackers (Retained for Backpropagation steps)
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_gate;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> i_gate;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c_tilde;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> o_gate;

    // Computational State Tracking Arrays
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> h;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c;

    // Parameter Optimization Matrix Gradient Trackers
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dW_f;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dW_i;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dW_c;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dW_o;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dU_f;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dU_i;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dU_c;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dU_o;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> db_f;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> db_i;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> db_c;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> db_o;

    LSTM_Layer(int batch_size = 32, int input_dim = 64, int hidden_dim = 128) 
        : B(batch_size), I(input_dim), H(hidden_dim) {
        
        _allocate_structural_matrices();
        _initialize_deterministic_parameters();
        zero_gradients();
    }

    void zero_gradients() {
        dW_f.setZero(); dW_i.setZero(); dW_c.setZero(); dW_o.setZero();
        dU_f.setZero(); dU_i.setZero(); dU_c.setZero(); dU_o.setZero();
        db_f.setZero(); db_i.setZero(); db_c.setZero(); db_o.setZero();
    }

    void step_forward(
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& x,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& h_prev,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& c_prev
    ) {
        _lstm_forward_compiled_core(
            B, I, H,
            x, h_prev, c_prev,
            W_f, W_i, W_c, W_o,
            U_f, U_i, U_c, U_o,
            b_f, b_i, b_c, b_o,
            f_gate, i_gate, c_tilde, o_gate,
            h, c
        );
    }

    void step_backward(
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& x,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& h_prev,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& c_prev,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dh,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dc_next,
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dx,
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dh_prev,
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dc_prev
    ) {
        _lstm_backward_compiled_core(
            B, I, H,
            x, h_prev, c_prev, c,
            W_f, W_i, W_c, W_o,
            U_f, U_i, U_c, U_o,
            f_gate, i_gate, c_tilde, o_gate,
            dh, dc_next,
            dx, dh_prev, dc_prev,
            dW_f, dW_i, dW_c, dW_o,
            dU_f, dU_i, dU_c, dU_o,
            db_f, db_i, db_c, db_o
        );
    }

private:
    void _allocate_structural_matrices() {
        W_f.resize(H, I); W_i.resize(H, I); W_c.resize(H, I); W_o.resize(H, I);
        U_f.resize(H, H); U_i.resize(H, H); U_c.resize(H, H); U_o.resize(H, H);
        b_f.resize(1, H); b_i.resize(1, H); b_c.resize(1, H); b_o.resize(1, H);

        f_gate.resize(B, H); i_gate.resize(B, H); c_tilde.resize(B, H); o_gate.resize(B, H);
        h.resize(B, H); c.resize(B, H);

        dW_f.resize(H, I); dW_i.resize(H, I); dW_c.resize(H, I); dW_o.resize(H, I);
        dU_f.resize(H, H); dU_i.resize(H, H); dU_c.resize(H, H); dU_o.resize(H, H);
        db_f.resize(1, H); db_i.resize(1, H); db_c.resize(1, H); db_o.resize(1, H);
    }

    void _initialize_deterministic_parameters() {
        // Deterministic architectural configuration equivalent to biophysical baseline limits
        W_f.setConstant(0.01f); W_i.setConstant(0.02f); W_c.setConstant(0.01f); W_o.setConstant(0.02f);
        U_f.setConstant(0.05f); U_i.setConstant(0.05f); U_c.setConstant(0.05f); U_o.setConstant(0.05f);
        
        // Standard structural bias setting (forcing forget gate high initially to mimic stable memory persistence)
        b_f.setConstant(1.0f);
        b_i.setConstant(-0.5f);
        b_c.setZero();
        b_o.setConstant(-0.5f);

        h.setZero();
        c.setZero();
    }
};

#endif // LSTM_LAYER_HPP