#ifndef CORTICAL_PYRAMIDAL_NEURON_HPP
#define CORTICAL_PYRAMIDAL_NEURON_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

// =====================================================================
// HIGH-PERFORMANCE CORE COMPUTATION ENGINE
// =====================================================================
inline void _step_tick_compiled_core(
    float dt, int B, int N, int S_E, int S_I,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& v,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& m,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& h,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& n,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& exc_u,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& exc_R,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& exc_T,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& r_ampa,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& r_nmda,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& Ca,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& running_avg_Ca,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& g_max_ampa,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& g_baseline_ampa,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& inh_u,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& inh_R,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& inh_T,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& r_gaba,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& g_max_gaba,
    const Eigen::VectorXi& ampa_targets,
    const Eigen::VectorXi& gaba_targets,
    const Eigen::VectorXi& parent,
    const Eigen::VectorXf& A_axial_vector,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& presynaptic_spikes_E,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& presynaptic_spikes_I,
    float C_m, float g_Na, float g_K, float g_L, float E_Na, float E_K, float E_L
) {
    // Concurrently distribute simulation batch items across available CPU cores
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        // 1. Thread-local scratchpads allocated dynamically to prevent cross-talk and avoid heap overhead
        Eigen::VectorXf g_syn_tot_local = Eigen::VectorXf::Zero(N);
        Eigen::VectorXf g_syn_E_tot_local = Eigen::VectorXf::Zero(N);
        Eigen::VectorXf D_local = Eigen::VectorXf::Zero(N);
        Eigen::VectorXf RHS_local = Eigen::VectorXf::Zero(N);

        // =====================================================================
        // PASS 1 & 2: EXCITATORY TRANSMITTER KINETICS & STDP/CALCIUM
        // =====================================================================
        for (int s = 0; s < S_E; ++s) {
            int target = ampa_targets[s];
            float v_post = v(b, target);

            if (presynaptic_spikes_E(b, s)) {
                float u_next = exc_u(b, s) + 0.2f * (1.0f - exc_u(b, s));
                float released = u_next * exc_R(b, s);
                exc_R(b, s) -= released;
                exc_T(b, s) += 1.5f * released;
                exc_u(b, s) = u_next;
            } else {
                exc_u(b, s) = std::max(exc_u(b, s) - (exc_u(b, s) / 50.0f) * dt, 0.2f);
                exc_R(b, s) += ((1.0f - exc_R(b, s)) / 800.0f) * dt;
            }

            exc_T(b, s) += ((-exc_T(b, s)) / 1.0f) * dt;
            if (exc_T(b, s) < 1e-6f) {
                exc_T(b, s) = 0.0f;
            }

            r_ampa(b, s) += (1.1f * exc_T(b, s) * (1.0f - r_ampa(b, s)) - 0.19f * r_ampa(b, s)) * dt;
            r_nmda(b, s) += (0.072f * exc_T(b, s) * (1.0f - r_nmda(b, s)) - 0.0066f * r_nmda(b, s)) * dt;

            if (r_ampa(b, s) < 0.0f) r_ampa(b, s) = 0.0f;
            else if (r_ampa(b, s) > 1.0f) r_ampa(b, s) = 1.0f;
            
            if (r_nmda(b, s) < 0.0f) r_nmda(b, s) = 0.0f;
            else if (r_nmda(b, s) > 1.0f) r_nmda(b, s) = 1.0f;

            float exponent = -0.062f * v_post;
            float mg_block = (exponent > 700.0f) ? 0.0f : (1.0f / (1.0f + std::exp(exponent) * (1.2f / 3.57f)));

            float g_ampa = g_max_ampa(b, s) * r_ampa(b, s);
            float g_nmda = 0.4f * r_nmda(b, s) * mg_block;
            float g_tot_s = g_ampa + g_nmda;

            g_syn_tot_local[target] += g_tot_s;

            float I_ampa = g_ampa * (v_post - 0.0f);
            float I_nmda = g_nmda * (v_post - 0.0f);

            float dCa = (0.1f * std::max(0.0f, -(I_ampa + I_nmda)) - (Ca(b, s) / 50.0f)) * dt;
            Ca(b, s) = std::max(0.0f, Ca(b, s) + dCa);

            if (Ca(b, s) >= 1.2f) {
                g_max_ampa(b, s) += 0.01f * ((g_baseline_ampa(b, s) * 2.5f) - g_max_ampa(b, s)) * dt;
            } else if (Ca(b, s) >= 0.4f) {
                g_max_ampa(b, s) += 0.005f * ((g_baseline_ampa(b, s) * 0.2f) - g_max_ampa(b, s)) * dt;
            }
            if (g_max_ampa(b, s) < 0.0f) {
                g_max_ampa(b, s) = 0.0f;
            }

            running_avg_Ca(b, s) += ((Ca(b, s) - running_avg_Ca(b, s)) / 100000.0f) * dt;
            g_max_ampa(b, s) += (-0.0001f * (running_avg_Ca(b, s) - 0.1f) * dt) * g_max_ampa(b, s);
        }

        // =====================================================================
        // PASS 3: INHIBITORY TRANSMITTER KINETICS (GABA_A)
        // =====================================================================
        for (int s = 0; s < S_I; ++s) {
            int target = gaba_targets[s];

            if (presynaptic_spikes_I(b, s)) {
                float u_next = inh_u(b, s) + 0.5f * (1.0f - inh_u(b, s));
                float released = u_next * inh_R(b, s);
                inh_R(b, s) -= released;
                inh_T(b, s) += 1.5f * released;
                inh_u(b, s) = u_next;
            } else {
                inh_u(b, s) = std::max(inh_u(b, s) - (inh_u(b, s) / 10.0f) * dt, 0.5f);
                inh_R(b, s) += ((1.0f - inh_R(b, s)) / 200.0f) * dt;
            }

            inh_T(b, s) += ((-inh_T(b, s)) / 1.0f) * dt;
            if (inh_T(b, s) < 1e-6f) {
                inh_T(b, s) = 0.0f;
            }

            r_gaba(b, s) += (1.05f * inh_T(b, s) * (1.0f - r_gaba(b, s)) - 0.166f * r_gaba(b, s)) * dt;
            if (r_gaba(b, s) < 0.0f) r_gaba(b, s) = 0.0f;
            else if (r_gaba(b, s) > 1.0f) r_gaba(b, s) = 1.0f;

            float g_gaba_s = g_max_gaba(b, s) * r_gaba(b, s);
            g_syn_tot_local[target] += g_gaba_s;
            g_syn_E_tot_local[target] += g_gaba_s * -75.0f;
        }

        // =====================================================================
        // PASS 4 & 5: SPATIAL COEFFICIENTS ASSEMBLY & O(N) HINES SOLVER
        // =====================================================================
        for (int i = 0; i < N; ++i) {
            float g_na_curr = g_Na * (m(b, i) * m(b, i) * m(b, i)) * h(b, i);
            float g_k_curr = g_K * (n(b, i) * n(b, i) * n(b, i) * n(b, i));

            D_local[i] = (C_m / dt) + g_syn_tot_local[i] + g_na_curr + g_k_curr + g_L;
            RHS_local[i] = (C_m / dt) * v(b, i) + g_syn_E_tot_local[i] + (g_na_curr * E_Na) + (g_k_curr * E_K) + (g_L * E_L);
        }

        // Hines Matrix Assembly Sweep
        for (int i = 1; i < N; ++i) {
            float g_ax = -A_axial_vector[i];
            int p = parent[i];
            D_local[i] += g_ax;
            D_local[p] += g_ax;
        }

        // Backward Elimination Pass
        for (int i = N - 1; i > 0; --i) {
            int p = parent[i];
            float factor = A_axial_vector[i] / D_local[i];
            D_local[p] -= factor * A_axial_vector[i];
            RHS_local[p] -= factor * RHS_local[i];
        }

        // Resolve Somatic Voltage Root Node Reference Point
        v(b, 0) = RHS_local[0] / D_local[0];

        // Forward Substitution Pass
        for (int i = 1; i < N; ++i) {
            int p = parent[i];
            v(b, i) = (RHS_local[i] - A_axial_vector[i] * v(b, p)) / D_local[i];
        }

        // =====================================================================
        // PASS 6: INLINED HODGKIN-HUXLEY GATING KINETICS INTEGRATION (ETD1)
        // =====================================================================
        for (int i = 0; i < N; ++i) {
            float v_curr = v(b, i);

            // Inlined alpha_m / beta_m evaluation sequence
            float am;
            if (std::abs(v_curr + 40.0f) < 1e-6f) {
                am = 1.0f;
            } else {
                am = 0.1f * (v_curr + 40.0f) / (1.0f - std::exp(-(v_curr + 40.0f) / 10.0f));
            }
            float bm = 4.0f * std::exp(-(v_curr + 65.0f) / 18.0f);
            float tau_m = 1.0f / (am + bm);
            float m_inf = am * tau_m;
            m(b, i) = m_inf + (m(b, i) - m_inf) * std::exp(-dt / tau_m);

            // Inlined alpha_h / beta_h evaluation sequence
            float ah = 0.07f * std::exp(-(v_curr + 65.0f) / 20.0f);
            float bh = 1.0f / (1.0f + std::exp(-(v_curr + 35.0f) / 10.0f));
            float tau_h = 1.0f / (ah + bh);
            float h_inf = ah * tau_h;
            h(b, i) = h_inf + (h(b, i) - h_inf) * std::exp(-dt / tau_h);

            // Inlined alpha_n / beta_n evaluation sequence
            float an;
            if (std::abs(v_curr + 55.0f) < 1e-6f) {
                an = 0.1f;
            } else {
                an = 0.01f * (v_curr + 55.0f) / (1.0f - std::exp(-(v_curr + 55.0f) / 10.0f));
            }
            float bn = 0.125f * std::exp(-(v_curr + 65.0f) / 80.0f);
            float tau_n = 1.0f / (an + bn);
            float n_inf = an * tau_n;
            n(b, i) = n_inf + (n(b, i) - n_inf) * std::exp(-dt / tau_n);
        }
    }
}

// =====================================================================
// WRAPPER BIOPHYSICAL LAYER STRUCTURAL CLASS OBJECT
// =====================================================================
class Cortical_Pyramidal_Neuron_Layer {
public:
    int B;
    int N;
    int S_E;
    int S_I;

    // Hodgkin-Huxley parameters
    float C_m;
    float g_Na;
    float g_K;
    float g_L;
    float E_Na;
    float E_K;
    float E_L;

    // Structural Layout Topology Map vectors
    Eigen::VectorXi parent;
    Eigen::VectorXf g_a;
    Eigen::VectorXf A_axial_vector;
    Eigen::VectorXi ampa_targets;
    Eigen::VectorXi gaba_targets;

    // Cell Biophysical Status States Matrices
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> h;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> n;

    // Synaptic Plasticity Array Structs
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> exc_u;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> exc_R;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> exc_T;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> r_ampa;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> r_nmda;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ca;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> running_avg_Ca;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> g_baseline_ampa;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> g_max_ampa;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inh_u;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inh_R;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inh_T;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> r_gaba;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> g_baseline_gaba;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> g_max_gaba;

    Cortical_Pyramidal_Neuron_Layer(int batch_size = 32) : B(batch_size) {
        C_m = 1.0f;
        g_Na = 120.0f;
        g_K = 36.0f;
        g_L = 0.3f;
        E_Na = 50.0f;
        E_K = -77.0f;
        E_L = -54.387f;

        _build_morphology();

        v = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(B, N, -65.0f);

        float am = _alpha_m(-65.0f), bm = _beta_m(-65.0f);
        float ah = _alpha_h(-65.0f), bh = _beta_h(-65.0f);
        float an = _alpha_n(-65.0f), bn = _beta_n(-65.0f);

        m = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(B, N, am / (am + bm));
        h = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(B, N, ah / (ah + bh));
        n = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(B, N, an / (an + bn));

        S_E = 16000;
        S_I = 4000;

        _allocate_deterministic_synapses();
        _precompute_axial_matrix();
    }

    std::vector<uint8_t> step_tick(
        float dt, 
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& presynaptic_spikes_E,
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& presynaptic_spikes_I
    ) {
        _step_tick_compiled_core(
            dt, B, N, S_E, S_I,
            v, m, h, n,
            exc_u, exc_R, exc_T, r_ampa, r_nmda, Ca, running_avg_Ca, g_max_ampa, g_baseline_ampa,
            inh_u, inh_R, inh_T, r_gaba, g_max_gaba,
            ampa_targets, gaba_targets, parent, A_axial_vector,
            presynaptic_spikes_E, presynaptic_spikes_I,
            C_m, g_Na, g_K, g_L, E_Na, E_K, E_L
        );

        std::vector<uint8_t> responses(B);
        for (int b = 0; b < B; ++b) {
            responses[b] = (v(b, 0) > 0.0f) ? 1 : 0;
        }
        return responses;
    }

private:
    void _build_morphology() {
        std::vector<int> apical_parents(20);
        apical_parents[0] = -1;
        for (int i = 1; i < 20; ++i) {
            apical_parents[i] = i - 1;
        }

        int nxt_idx = 20;
        std::vector<int> trunk_attaches = {4, 8, 12};
        for (int trunk_attach : trunk_attaches) {
            int p = trunk_attach;
            for (int step = 0; step < 4; ++step) {
                if (nxt_idx >= (int)apical_parents.size()) apical_parents.resize(nxt_idx + 1);
                apical_parents[nxt_idx] = p;
                p = nxt_idx;
                nxt_idx++;
            }
        }

        std::vector<int> active_leaves = {19};
        for (int gen = 0; gen < 4; ++gen) {
            std::vector<int> new_leaves;
            for (int leaf : active_leaves) {
                for (int branch = 0; branch < 2; ++branch) {
                    int p = leaf;
                    for (int step = 0; step < 2; ++step) {
                        if (nxt_idx >= (int)apical_parents.size()) apical_parents.resize(nxt_idx + 1);
                        apical_parents[nxt_idx] = p;
                        p = nxt_idx;
                        nxt_idx++;
                    }
                    new_leaves.push_back(p);
                }
            }
            active_leaves = new_leaves;
        }
        int num_apical = nxt_idx;

        std::vector<int> basal_parents(4);
        basal_parents[0] = -1; basal_parents[1] = 0; basal_parents[2] = 1; basal_parents[3] = 2;
        nxt_idx = 4;
        active_leaves = {3};
        for (int gen = 0; gen < 3; ++gen) {
            std::vector<int> new_leaves;
            for (int leaf : active_leaves) {
                for (int branch = 0; branch < 2; ++branch) {
                    int p = leaf;
                    for (int c_step = 0; c_step < 3; ++c_step) {
                        if (nxt_idx >= (int)basal_parents.size()) basal_parents.resize(nxt_idx + 1);
                        basal_parents[nxt_idx] = p;
                        p = nxt_idx;
                        nxt_idx++;
                    }
                    new_leaves.push_back(p);
                }
            }
            active_leaves = new_leaves;
        }
        int num_basal = nxt_idx;

        N = 1 + num_apical + (6 * num_basal);
        parent = Eigen::VectorXi::Constant(N, -1);
        g_a = Eigen::VectorXf::Zero(N);

        int apical_offset = 1;
        for (size_t k = 0; k < apical_parents.size(); ++k) {
            int g_idx = apical_offset + k;
            parent[g_idx] = (apical_parents[k] == -1) ? 0 : (apical_offset + apical_parents[k]);
            g_a[g_idx] = 2.0f;
        }

        int basal_offset = 1 + num_apical;
        for (int b_id = 0; b_id < 6; ++b_id) {
            int start_idx = basal_offset + (b_id * num_basal);
            for (size_t k = 0; k < basal_parents.size(); ++k) {
                int g_idx = start_idx + k;
                parent[g_idx] = (basal_parents[k] == -1) ? 0 : (start_idx + basal_parents[k]);
                g_a[g_idx] = 2.5f;
            }
        }
    }

    void _allocate_deterministic_synapses() {
        int apical_offset = 1;
        int num_apical = 92;
        int basal_offset = 93;
        int num_basal_total = 276;

        ampa_targets.resize(S_E);
        for (int i = 0; i < 9600; ++i) ampa_targets[i] = basal_offset + (i % num_basal_total);
        for (int i = 0; i < 6400; ++i) ampa_targets[9600 + i] = apical_offset + (i % num_apical);

        gaba_targets.resize(S_I);
        for (int i = 0; i < 2400; ++i) gaba_targets[i] = basal_offset + (i % num_basal_total);
        for (int i = 0; i < 1600; ++i) gaba_targets[2400 + i] = apical_offset + (i % num_apical);

        exc_u = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(B, S_E, 0.2f);
        exc_R = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(B, S_E, 1.0f);
        exc_T = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(B, S_E);
        r_ampa = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(B, S_E);
        r_nmda = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(B, S_E);
        Ca = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(B, S_E);
        running_avg_Ca = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(B, S_E, 0.1f);

        g_baseline_ampa.resize(B, S_E);
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < 9600; ++i) g_baseline_ampa(b, i) = 0.005f;
            for (int i = 0; i < 6400; ++i) g_baseline_ampa(b, 9600 + i) = 1.2f;
        }
        g_max_ampa = g_baseline_ampa;

        inh_u = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(B, S_I, 0.5f);
        inh_R = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(B, S_I, 1.0f);
        inh_T = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(B, S_I);
        r_gaba = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(B, S_I);

        g_baseline_gaba.resize(B, S_I);
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < 2400; ++i) g_baseline_gaba(b, i) = 0.012f;
            for (int i = 0; i < 1600; ++i) g_baseline_gaba(b, 2400 + i) = 1.5f;
        }
        g_max_gaba = g_baseline_gaba;
    }

    void _precompute_axial_matrix() {
        A_axial_vector = Eigen::VectorXf::Zero(N);
        float g_sd = 1.6f;
        for (int i = 1; i < N; ++i) {
            int p = parent[i];
            A_axial_vector[i] = (p == 0) ? -g_sd : -g_a[i];
        }
    }

    inline float _alpha_m(float v) { return (std::abs(v + 40.0f) < 1e-6f) ? 1.0f : 0.1f * (v + 40.0f) / (1.0f - std::exp(-(v + 40.0f) / 10.0f)); }
    inline float _beta_m(float v)  { return 4.0f * std::exp(-(v + 65.0f) / 18.0f); }
    inline float _alpha_h(float v) { return 0.07f * std::exp(-(v + 65.0f) / 20.0f); }
    inline float _beta_h(float v)  { return 1.0f / (1.0f + std::exp(-(v + 35.0f) / 10.0f)); }
    inline float _alpha_n(float v) { return (std::abs(v + 55.0f) < 1e-6f) ? 0.1f : 0.01f * (v + 55.0f) / (1.0f - std::exp(-(v + 55.0f) / 10.0f)); }
    inline float _beta_n(float v)  { return 0.125f * std::exp(-(v + 65.0f) / 80.0f); }
};

#endif // CORTICAL_PYRAMIDAL_NEURON_HPP