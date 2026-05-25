#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <memory>
#include <random>
#include <array>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <charconv>
#include <stdexcept>
#include <immintrin.h>

#pragma GCC optimize("O3")
#pragma GCC target("avx2")
#pragma GCC target("fma")
#pragma GCC target("bmi2")
#pragma GCC target("bmi")

#define custom_max(a, b) ((a) > (b) ? (a) : (b))
#define custom_min(a, b) ((a) < (b) ? (a) : (b))

// =====================================================================
// 1. CONSTANTS & PARAMETERS (Standard Hodgkin-Huxley Values)
// =====================================================================
const float C_m = 1.0f;       // Membrane capacitance (uF/cm^2)
const float g_Na = 120.0f;    // Maximum Sodium (Na) conductance (mS/cm^2)
const float g_K = 36.0f;      // Maximum Potassium (K) conductance (mS/cm^2)
const float g_L = 0.3f;       // Leak conductance (mS/cm^2)

const float E_Na = 50.0f;     // Sodium reversal potential (mV)
const float E_K = -77.0f;     // Potassium reversal potential (mV)
const float E_L = -54.387f;   // Leak reversal potential (mV)

// Magic patterns containing bit-position configurations (0 to 63)
constexpr uint64_t msk_1 = 0xAAAAAAAAAAAAAAAAULL; // 0b10101010... (Bit 0 of index)
constexpr uint64_t msk_2 = 0xCCCCCCCCCCCCCCCCULL; // 0b11001100... (Bit 1 of index)
constexpr uint64_t msk_3 = 0xF0F0F0F0F0F0F0F0ULL; // 0b11110000... (Bit 2 of index)
constexpr uint64_t msk_4 = 0xFF00FF00FF00FF00ULL; // 0b11111111_00000000... (Bit 3 of index)
constexpr uint64_t msk_5 = 0xFFFF0000FFFF0000ULL; // (Bit 4 of index)
constexpr uint64_t msk_6 = 0xFFFFFFFF00000000ULL; // (Bit 5 of index)

// =====================================================================
// 2. FLAT STRUCTURE OF ARRAYS (SoA) NEURON ARCHITECTURE
// =====================================================================
struct NeuronSoA {
    int num_compartments;
    
    // Contiguous Core State Variable Tracks
    std::vector<float> v;
    std::vector<float> m;
    std::vector<float> h;
    std::vector<float> n;
    
    // Fast Structural Routing Index Lists
    std::vector<int> parent;
    std::vector<float> g_a; // Axial connection conductance to parent
    
    // Pre-allocated Contiguous Implicit Solver Memory Profiles
    std::vector<float> g_syn_tot;
    std::vector<float> g_syn_E_tot;
    std::vector<float> D;   // Hines Tree Matrix Diagonal Elements
    std::vector<float> A;   // Hines Tree Matrix Lower/Upper Off-Diagonal Elements
    std::vector<float> RHS; // Hines Tree Matrix Right-Hand Side Vector

    NeuronSoA(int size) : 
        num_compartments(size), v(size), m(size), h(size), n(size),
        parent(size, -1), g_a(size, 0.0f),
        g_syn_tot(size, 0.0f), g_syn_E_tot(size, 0.0f),
        D(size, 0.0f), A(size, 0.0f), RHS(size, 0.0f) {}
};

// =====================================================================
// 3. SEPARATED HOMOGENEOUS FLAT SYNAPSE GROUPS (No Branching / No Pointers)
// =====================================================================
struct AmpaNmdaSynapseGroupSoA {
    std::vector<int> target_comp;
    std::vector<float> u, R, T;
    std::vector<float> r_ampa, r_nmda, g_max_ampa, g_baseline;
    std::vector<float> Ca, running_avg_Ca;

    // Constant Synaptic Scaling Properties
    const float Mg2 = 1.2f;
    const float U = 0.2f;
    const float tau_rec = 800.0f;
    const float tau_fac = 50.0f;
    const float inverse_tau_rec = 1.0f / tau_rec;
    const float inverse_tau_fac = 1.0f / tau_fac;
    const float T_max = 1.5f;
    const float tau_clear = 1.0f;
    const float alpha_ampa = 1.1f;
    const float beta_ampa = 0.19f;
    const float alpha_nmda = 0.072f;
    const float beta_nmda = 0.0066f;
    const float g_max_nmda = 0.4f;
    const float Ca_factor = 0.1f;
    const float tau_Ca = 50.0f;
    const float theta_ltd = 0.4f;
    const float theta_ltp = 1.2f;
    const float eta_p = 0.01f;
    const float eta_d = 0.005f;
    const float tau_homeostasis = 100000.0f;
    const float scaling_rate = 0.0001f;
    const float E_rev = 0.0f;
};

struct GabaASynapseGroupSoA {
    std::vector<int> target_comp;
    std::vector<float> u, R, T;
    std::vector<float> r_gaba, g_max_gaba, g_baseline;

    // Constant Inhibitory Properties
    const float U = 0.5f;
    const float tau_rec = 200.0f;
    const float tau_fac = 10.0f;
    const float T_max = 1.5f;
    const float tau_clear = 1.0f;
    const float alpha_gaba = 1.05f;
    const float beta_gaba = 0.166f;
    const float E_rev = -75.0f;
};

// =====================================================================
// 4. GATING VARIABLE EMPIRICAL EQUATIONS
// =====================================================================
inline float alpha_m(float u) {
    if (std::abs(u + 40.0f) < 1e-6f) return 1.0f;
    return 0.1f * (u + 40.0f) / (1.0f - std::exp(-(u + 40.0f) / 10.0f));
}
inline float beta_m(float u) { return 4.0f * std::exp(-(u + 65.0f) / 18.0f); }
inline float alpha_h(float u) { return 0.07f * std::exp(-(u + 65.0f) / 20.0f); }
inline float beta_h(float u) { return 1.0f / (1.0f + std::exp(-(u + 35.0f) / 10.0f)); }
inline float alpha_n(float u) {
    if (std::abs(u + 55.0f) < 1e-6f) return 0.1f;
    return 0.01f * (u + 55.0f) / (1.0f - std::exp(-(u + 55.0f) / 10.0f));
}
inline float beta_n(float u) { return 0.125f * std::exp(-(u + 65.0f) / 80.0f); }

// =====================================================================
// 5. LEGACY TOPOLOGY BUILDERS
// =====================================================================
struct Topology {
    int num_compartments;
    std::map<int, int> parents;
    std::map<int, std::vector<int>> children;
    std::map<int, std::string> labels;
};

inline Topology build_biological_basal_topology() {
    std::map<int, int> parents; parents[0] = -1; parents[1] = 0; parents[2] = 1; parents[3] = 2;
    std::map<int, std::string> labels;
    labels[0] = "Basal Root"; labels[1] = "Basal Stem"; labels[2] = "Basal Stem"; labels[3] = "Bifurcation Point Gen 1";
    int nxt_idx = 4; std::vector<int> active_leaves = {3};
    for (int gen = 0; gen < 3; ++gen) {
        std::vector<int> new_leaves;
        for (int leaf : active_leaves) {
            for (int branch = 0; branch < 2; ++branch) {
                int p = leaf;
                for (int c_step = 0; c_step < 3; ++c_step) {
                    parents[nxt_idx] = p; labels[nxt_idx] = "Basal Gen " + std::to_string(gen + 1) + " Cable";
                    p = nxt_idx; nxt_idx++;
                }
                new_leaves.push_back(p);
            }
        }
        active_leaves = new_leaves;
    }
    std::map<int, std::vector<int>> children;
    for (int i = 0; i < nxt_idx; ++i) children[i] = std::vector<int>();
    for (const auto& pair : parents) { if (pair.second != -1) children[pair.second].push_back(pair.first); }
    int max_key = 0; for (const auto& pair : parents) { if (pair.first > max_key) max_key = pair.first; }
    labels[max_key] = "Basal Terminal Tip";
    return {nxt_idx, parents, children, labels};
}

inline Topology build_biological_apical_topology() {
    std::map<int, int> parents; parents[0] = -1; for (int i = 1; i < 20; ++i) parents[i] = i - 1;
    std::map<int, std::string> labels; for (int i = 0; i < 20; ++i) labels[i] = "Main Apical Trunk"; labels[0] = "Apical Root";
    int nxt_idx = 20; std::vector<int> trunk_attachments = {4, 8, 12};
    for (int trunk_attach : trunk_attachments) {
        int p = trunk_attach;
        for (int i = 0; i < 4; ++i) {
            parents[nxt_idx] = p; labels[nxt_idx] = "Horizontal Oblique (from Trunk " + std::to_string(trunk_attach) + ")";
            p = nxt_idx; nxt_idx++;
        }
    }
    labels[19] = "Main Apical Trunk Bifurcation Hub"; std::vector<int> active_leaves = {19};
    for (int gen = 0; gen < 4; ++gen) {
        std::vector<int> new_leaves;
        for (int leaf : active_leaves) {
            for (int branch = 0; branch < 2; ++branch) {
                int p = leaf;
                for (int step = 0; step < 2; ++step) {
                    parents[nxt_idx] = p; labels[nxt_idx] = "Apical Tuft Gen " + std::to_string(gen + 1) + " Surface Filament";
                    p = nxt_idx; nxt_idx++;
                }
                new_leaves.push_back(p);
            }
        }
        active_leaves = new_leaves;
    }
    std::map<int, std::vector<int>> children;
    for (int i = 0; i < nxt_idx; ++i) children[i] = std::vector<int>();
    for (const auto& pair : parents) { if (pair.second != -1) children[pair.second].push_back(pair.first); }
    for (int leaf : active_leaves) labels[leaf] = "Surface Layer 1 Tuft Extremity Tip";
    return {nxt_idx, parents, children, labels};
}

struct Xorshift64 {
    uint64_t state;
    using result_type = uint64_t;
    
    Xorshift64(uint64_t seed = 1) : state(seed) {}
    
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }
    
    uint64_t operator()() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return state = x;
    }
};

// =====================================================================
// 6. Cortical_Pyramidal_Neuron Refactored Class Architecture
// =====================================================================
class Cortical_Pyramidal_Neuron {
private:
    int flat_total_compartments;
    int N;
    NeuronSoA neuron;
    AmpaNmdaSynapseGroupSoA ampa_group;
    GabaASynapseGroupSoA gaba_group;
    bool was_spiking;

public:
    Cortical_Pyramidal_Neuron() : 
        flat_total_compartments(1 + build_biological_apical_topology().num_compartments + (6 * build_biological_basal_topology().num_compartments)),
        neuron(flat_total_compartments),
        was_spiking(false)
        {
        Xorshift64 gen{42ULL};
        auto next_float = [&gen]() {
            return static_cast<float>(gen()) / 18446744073709551615.0f;
        };

        auto apical_topology = build_biological_apical_topology();
        auto basal_topology = build_biological_basal_topology();
        N = flat_total_compartments;

        // Initialize state variables to standard resting state
        float V_rest = -65.0f;
        float m0 = alpha_m(V_rest) / (alpha_m(V_rest) + beta_m(V_rest));
        float h0 = alpha_h(V_rest) / (alpha_h(V_rest) + beta_h(V_rest));
        float n0 = alpha_n(V_rest) / (alpha_n(V_rest) + beta_n(V_rest));
        std::fill(neuron.v.begin(), neuron.v.end(), V_rest);
        std::fill(neuron.m.begin(), neuron.m.end(), m0);
        std::fill(neuron.h.begin(), neuron.h.end(), h0);
        std::fill(neuron.n.begin(), neuron.n.end(), n0);

        // Map apical structural paths
        int apical_offset = 1;
        for (int i = 0; i < apical_topology.num_compartments; ++i) {
            int g_idx = apical_offset + i;
            int p_orig = apical_topology.parents[i];
            neuron.parent[g_idx] = (p_orig == -1) ? 0 : (apical_offset + p_orig);
            neuron.g_a[g_idx] = 2.0f; 
        }

        // Map basal structural paths
        int basal_offset = 1 + apical_topology.num_compartments;
        for (int b_id = 0; b_id < 6; ++b_id) {
            int start_idx = basal_offset + (b_id * basal_topology.num_compartments);
            for (int i = 0; i < basal_topology.num_compartments; ++i) {
                int g_idx = start_idx + i;
                int p_orig = basal_topology.parents[i];
                neuron.parent[g_idx] = (p_orig == -1) ? 0 : (start_idx + p_orig);
                neuron.g_a[g_idx] = 2.5f;
            }
        }

        // Allocate and distribute exactly 20,000 unchangeable structural synapses
        const float g_sd = 1.6f;

        // Basal Dendrite Synapses (12,000 total)
        for (int b_id = 0; b_id < 6; ++b_id) {
            int start_idx = basal_offset + (b_id * basal_topology.num_compartments);
            for (int k = 0; k < 2000; ++k) {
                int g_idx = start_idx + (gen() % basal_topology.num_compartments);
                if (next_float() < 0.80f) {
                    ampa_group.target_comp.push_back(g_idx);
                    ampa_group.u.push_back(0.2f); ampa_group.R.push_back(1.0f); ampa_group.T.push_back(0.0f);
                    ampa_group.r_ampa.push_back(0.0f); ampa_group.r_nmda.push_back(0.0f);
                    ampa_group.g_max_ampa.push_back(0.005f); ampa_group.g_baseline.push_back(0.005f);
                    ampa_group.Ca.push_back(0.0f); ampa_group.running_avg_Ca.push_back(0.1f);
                } else {
                    gaba_group.target_comp.push_back(g_idx);
                    gaba_group.u.push_back(0.5f); gaba_group.R.push_back(1.0f); gaba_group.T.push_back(0.0f);
                    gaba_group.r_gaba.push_back(0.0f);
                    gaba_group.g_max_gaba.push_back(0.012f); gaba_group.g_baseline.push_back(0.012f);
                }
            }
        }

        // Apical Trunk and Tuft Synapses (8,000 total)
        for (int k = 0; k < 8000; ++k) {
            int g_idx = apical_offset + (gen() % apical_topology.num_compartments);
            if (next_float() < 0.80f) {
                ampa_group.target_comp.push_back(g_idx);
                ampa_group.u.push_back(0.2f); ampa_group.R.push_back(1.0f); ampa_group.T.push_back(0.0f);
                ampa_group.r_ampa.push_back(0.0f); ampa_group.r_nmda.push_back(0.0f);
                ampa_group.g_max_ampa.push_back(1.2f); ampa_group.g_baseline.push_back(1.2f);
                ampa_group.Ca.push_back(0.0f); ampa_group.running_avg_Ca.push_back(0.1f);
            } else {
                gaba_group.target_comp.push_back(g_idx);
                gaba_group.u.push_back(0.5f); gaba_group.R.push_back(1.0f); gaba_group.T.push_back(0.0f);
                gaba_group.r_gaba.push_back(0.0f);
                gaba_group.g_max_gaba.push_back(1.5f); gaba_group.g_baseline.push_back(1.5f);
            }
        }

        // Pre-calculate permanent Off-Diagonal Coupling matrix components
        float* const n_A = neuron.A.data();
        const int* const n_parent = neuron.parent.data();
        const float* const n_g_a = neuron.g_a.data();
        for (int i = 1; i < N; ++i) {
            n_A[i] = (n_parent[i] == 0) ? -g_sd : -n_g_a[i];
        }
        
    }

    inline uint64_t extract_8_indices(uint32_t mask) {
        // Cast to 64-bit to map up to 8 bytes of indices
        uint64_t v = mask; 

        // Extract raw bit configurations using BMI2
        uint64_t v1 = _pext_u64(msk_1, v);
        uint64_t v2 = _pext_u64(msk_2, v);
        uint64_t v3 = _pext_u64(msk_3, v);
        uint64_t v4 = _pext_u64(msk_4, v);
        uint64_t v5 = _pext_u64(msk_5, v);
        uint64_t v6 = _pext_u64(msk_6, v);

        // Interleave and re-assemble the bits into 8 separate bytes
        uint64_t indices = (v1 & 1)        | ((v2 & 1) << 8)  | 
                        ((v3 & 1) << 16) | ((v4 & 1) << 24) |
                        ((v5 & 1) << 32) | ((v6 & 1) << 40);
                        
        indices |= (((v1 >> 1) & 1) << 1)  | (((v2 >> 1) & 1) << 9)  | 
                (((v3 >> 1) & 1) << 17) | (((v4 >> 1) & 1) << 25) |
                (((v5 >> 1) & 1) << 33) | (((v6 >> 1) & 1) << 41);

        indices |= (((v1 >> 2) & 1) << 2)  | (((v2 >> 2) & 1) << 10) | 
                (((v3 >> 2) & 1) << 18) | (((v4 >> 2) & 1) << 26) |
                (((v5 >> 2) & 1) << 34) | (((v6 >> 2) & 42));

        indices |= (((v1 >> 3) & 1) << 3)  | (((v2 >> 3) & 1) << 11) | 
                (((v3 >> 3) & 1) << 19) | (((v4 >> 3) & 1) << 27) |
                (((v5 >> 3) & 1) << 35) | (((v6 >> 3) & 1) << 43);

        indices |= (((v1 >> 4) & 1) << 4)  | (((v2 >> 4) & 1) << 12) | 
                (((v3 >> 4) & 1) << 20) | (((v4 >> 4) & 1) << 28) |
                (((v5 >> 4) & 1) << 36) | (((v6 >> 4) & 1) << 44);

        indices |= (((v1 >> 5) & 1) << 5)  | (((v2 >> 5) & 1) << 13) | 
                (((v3 >> 5) & 1) << 21) | (((v4 >> 5) & 1) << 29) |
                (((v5 >> 5) & 1) << 37) | (((v6 >> 5) & 1) << 45);

        indices |= (((v1 >> 6) & 1) << 6)  | (((v2 >> 6) & 1) << 14) | 
                (((v3 >> 6) & 1) << 22) | (((v4 >> 6) & 1) << 30) |
                (((v5 >> 6) & 1) << 38) | (((v6 >> 6) & 1) << 46);

        indices |= (((v1 >> 7) & 1) << 7)  | (((v2 >> 7) & 1) << 15) | 
                (((v3 >> 7) & 1) << 23) | (((v4 >> 7) & 1) << 31) |
                (((v5 >> 7) & 1) << 39) | (((v6 >> 7) & 1) << 47);

        return indices;
    }

    void remove_elements_simd(std::vector<int>& main_vec, const std::vector<int>& to_remove) {
        // If to_remove is tiny, a simple lambda allows compiler auto-vectorizers 
        // to map this check directly to SIMD broadcast instructions (e.g., _mm256_cmpeq_epi32)
        auto it = std::remove_if(main_vec.begin(), main_vec.end(), 
            [&to_remove](int x) {
                // Unrolled execution enables SIMD vector lanes to check multiple elements at once
                for (int r : to_remove) {
                    if (x == r) return true;
                }
                return false;
            });

        main_vec.erase(it, main_vec.end());
    }

    /**
     * Executes a single evaluation tick of the structural neural matrix.
     * @param dt Time differential step size (ms).
     * @param presynaptic_spikes Byte vector mapping sequentially to internal AMPA
     * group entries followed by GABA entries.
     * @return True when a new somatic action potential edge occurs.
     */
    bool step_tick(float dt, const std::vector<uint8_t>& presynaptic_spikes) {
        // Cache performance pointers dynamically locally to assist optimization loops
        float* const n_v = neuron.v.data();
        float* const n_m = neuron.m.data();
        float* const n_h = neuron.h.data();
        float* const n_n = neuron.n.data();
        const int* const n_parent = neuron.parent.data();
        const float* const n_g_a = neuron.g_a.data();

        float* const n_g_syn_tot = neuron.g_syn_tot.data();
        float* const n_g_syn_E_tot = neuron.g_syn_E_tot.data();
        float* const n_D = neuron.D.data();
        float* const n_A = neuron.A.data();
        float* const n_RHS = neuron.RHS.data();

        const int* const ampa_target = ampa_group.target_comp.data();
        float* const ampa_u_ptr = ampa_group.u.data();
        float* const ampa_R_ptr = ampa_group.R.data();
        float* const ampa_T_ptr = ampa_group.T.data();
        float* const ampa_r_ampa_ptr = ampa_group.r_ampa.data();
        float* const ampa_r_nmda_ptr = ampa_group.r_nmda.data();
        float* const ampa_g_max_ptr = ampa_group.g_max_ampa.data();
        const float* const ampa_g_base_ptr = ampa_group.g_baseline.data();
        float* const ampa_Ca_ptr = ampa_group.Ca.data();
        float* const ampa_avg_Ca_ptr = ampa_group.running_avg_Ca.data();

        const int* const gaba_target = gaba_group.target_comp.data();
        float* const gaba_u_ptr = gaba_group.u.data();
        float* const gaba_R_ptr = gaba_group.R.data();
        float* const gaba_T_ptr = gaba_group.T.data();
        float* const gaba_r_gaba_ptr = gaba_group.r_gaba.data();
        float* const gaba_g_max_ptr = gaba_group.g_max_gaba.data();

        const uint8_t* const presynaptic_spikes_ptr = presynaptic_spikes.data();

        std::fill(neuron.g_syn_tot.begin(), neuron.g_syn_tot.end(), 0.0f);
        std::fill(neuron.g_syn_E_tot.begin(), neuron.g_syn_E_tot.end(), 0.0f);

        // --- Pass 1: Excitatory AMPA / NMDA Integration & STDP/Homeostasis ---
        size_t ampa_size = ampa_group.target_comp.size();
        __m256i zero_vec = _mm256_setzero_si256();
        std::vector<int> spike_vector;
        std::vector<float> spiked_flag(ampa_size, 0.0f);

        for (size_t base = 0; base < ampa_size; base += 32) {
            size_t rem = ampa_size - base;
            if (rem < 32) {
                // tail: scalar
                for (size_t j = 0; j < rem; ++j) {
                    if (presynaptic_spikes_ptr[base + j]) {
                        int gidx = static_cast<int>(base + j);
                        spike_vector.push_back(gidx);
                        spiked_flag[gidx] = 1.0f;
                    }
                }
            } else {
                __m256i spiked_mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(presynaptic_spikes_ptr + base));
                __m256i is_zero_mask = _mm256_cmpeq_epi8(spiked_mask, zero_vec);
                uint32_t nonzero_byte_bits = ~_mm256_movemask_epi8(is_zero_mask);

                while (nonzero_byte_bits) {
                    int bit_index = __builtin_ctz(nonzero_byte_bits); // 0..31
                    int global_idx = static_cast<int>(base + bit_index);
                    spike_vector.push_back(global_idx);
                    spiked_flag[global_idx] = 1.0f;
                    nonzero_byte_bits &= nonzero_byte_bits - 1;
                }
            }
        }

        // Apply updates for spiking synapses
        for (int idx : spike_vector) {
            ampa_u_ptr[idx] += ampa_group.U * (1.0f - ampa_u_ptr[idx]);
            float released = ampa_u_ptr[idx] * ampa_R_ptr[idx];
            ampa_R_ptr[idx] -= released; ampa_T_ptr[idx] += ampa_group.T_max * released;
        }

        // Apply updates for non-spiking synapses (single scan, O(n)) -> Single precision steps by 8
        __m256 v_c_fac           = _mm256_set1_ps(ampa_group.inverse_tau_fac * dt);
        __m256 v_c_rec           = _mm256_set1_ps(ampa_group.inverse_tau_rec * dt);
        __m256 v_one_minus_c_rec = _mm256_set1_ps(1.0f - ampa_group.inverse_tau_rec * dt);
        __m256 v_U               = _mm256_set1_ps(ampa_group.U);
        __m256 v_zero_si256      = _mm256_setzero_ps();

        size_t i = 0;
        // Process elements in blocks of 8
        size_t vectorized_size = ampa_size - (ampa_size % 8);
        for (; i < vectorized_size; i += 8) {
            __m256 bools_vec = _mm256_setr_ps(spiked_flag[i], spiked_flag[i + 1], spiked_flag[i + 2], spiked_flag[i + 3],
                                              spiked_flag[i + 4], spiked_flag[i + 5], spiked_flag[i + 6], spiked_flag[i + 7]);
            __m256 v_mask = _mm256_cmp_ps(bools_vec, v_zero_si256, _CMP_GT_OQ);

            __m256 v_u_old = _mm256_loadu_ps(&ampa_u_ptr[i]);
            __m256 v_R_old = _mm256_loadu_ps(&ampa_R_ptr[i]);

            __m256 v_u_new = _mm256_fnmadd_ps(v_u_old, v_c_fac, v_u_old);
        
            // Vectorized clamp: if (u < U) u = U;
            v_u_new = _mm256_max_ps(v_u_new, v_U);

            // Compute new R using FMA: (v_R_old * v_one_minus_c_rec) + v_c_rec
            __m256 v_R_new = _mm256_fmadd_ps(v_R_old, v_one_minus_c_rec, v_c_rec);

            // _mm256_blendv_ps selects from the second param if the mask's sign bit is 1.
            // If spiked_flag was true (1.0f), we pick v_old (simulating the 'continue').
            // If spiked_flag was false (0.0f), we pick the calculated v_new.
            __m256 v_u_final = _mm256_blendv_ps(v_u_new, v_u_old, v_mask);
            __m256 v_R_final = _mm256_blendv_ps(v_R_new, v_R_old, v_mask);

            // Store results back.
            _mm256_storeu_ps(&ampa_u_ptr[i], v_u_final);
            _mm256_storeu_ps(&ampa_R_ptr[i], v_R_final);
        }
        // Handle remaining elements if ampa_size is not a multiple of 8
        for (; i < ampa_size; ++i) {
            if (spiked_flag[i]) continue;
            ampa_u_ptr[i] += ((-ampa_u_ptr[i]) * ampa_group.inverse_tau_fac) * dt;
            if (ampa_u_ptr[i] < ampa_group.U) ampa_u_ptr[i] = ampa_group.U;
            ampa_R_ptr[i] += ((1.0f - ampa_R_ptr[i]) * ampa_group.inverse_tau_rec) * dt;
        }
        
        for (size_t i = 0; i < ampa_size; ++i) {
            int comp = ampa_target[i];
            float v_post = n_v[comp];

            ampa_T_ptr[i] += ((-ampa_T_ptr[i]) / ampa_group.tau_clear) * dt;
            if (ampa_T_ptr[i] < 1e-6f) ampa_T_ptr[i] = 0.0f;

            float dr_ampa = (ampa_group.alpha_ampa * ampa_T_ptr[i] * (1.0f - ampa_r_ampa_ptr[i]) - ampa_group.beta_ampa * ampa_r_ampa_ptr[i]) * dt;
            ampa_r_ampa_ptr[i] = custom_max(0.0f, custom_min(1.0f, ampa_r_ampa_ptr[i] + dr_ampa));

            float dr_nmda = (ampa_group.alpha_nmda * ampa_T_ptr[i] * (1.0f - ampa_r_nmda_ptr[i]) - ampa_group.beta_nmda * ampa_r_nmda_ptr[i]) * dt;
            ampa_r_nmda_ptr[i] = custom_max(0.0f, custom_min(1.0f, ampa_r_nmda_ptr[i] + dr_nmda));

            float g_ampa = ampa_g_max_ptr[i] * ampa_r_ampa_ptr[i];
            float exponent = -0.062f * v_post;
            float mg_block = (exponent > 700.0f) ? 0.0f : (1.0f / (1.0f + std::exp(exponent) * (ampa_group.Mg2 / 3.57f)));
            float g_nmda = ampa_group.g_max_nmda * ampa_r_nmda_ptr[i] * mg_block;

            float g_tot = g_ampa + g_nmda;
            n_g_syn_tot[comp] += g_tot;
            n_g_syn_E_tot[comp] += g_tot * ampa_group.E_rev;

            float I_ampa = g_ampa * (v_post - ampa_group.E_rev);
            float I_nmda = g_nmda * (v_post - ampa_group.E_rev);
            float dCa = (ampa_group.Ca_factor * custom_max(0.0f, -(I_ampa + I_nmda)) - (ampa_Ca_ptr[i] / ampa_group.tau_Ca)) * dt;
            ampa_Ca_ptr[i] = custom_max(0.0f, ampa_Ca_ptr[i] + dCa);

            if (ampa_Ca_ptr[i] >= ampa_group.theta_ltp) {
                ampa_g_max_ptr[i] += ampa_group.eta_p * ((ampa_g_base_ptr[i] * 2.5f) - ampa_g_max_ptr[i]) * dt;
            } else if (ampa_Ca_ptr[i] >= ampa_group.theta_ltd) {
                ampa_g_max_ptr[i] += ampa_group.eta_d * ((ampa_g_base_ptr[i] * 0.2f) - ampa_g_max_ptr[i]) * dt;
            }
            if (ampa_g_max_ptr[i] < 0.0f) ampa_g_max_ptr[i] = 0.0f;

            ampa_avg_Ca_ptr[i] += ((ampa_Ca_ptr[i] - ampa_avg_Ca_ptr[i]) / ampa_group.tau_homeostasis) * dt;
            ampa_g_max_ptr[i] += (-ampa_group.scaling_rate * (ampa_avg_Ca_ptr[i] - 0.1f) * dt) * ampa_g_max_ptr[i];
        }

        // --- Pass 2: Inhibitory GABA_A Evaluation Track ---
        size_t gaba_size = gaba_group.target_comp.size();
        for (size_t i = 0; i < gaba_size; ++i) {
            int comp = gaba_target[i];
            bool spiked = ((ampa_size + i) < presynaptic_spikes.size()) ? presynaptic_spikes[ampa_size + i] != 0 : false;

            if (spiked) {
                gaba_u_ptr[i] += gaba_group.U * (1.0f - gaba_u_ptr[i]);
                float released = gaba_u_ptr[i] * gaba_R_ptr[i];
                gaba_R_ptr[i] -= released; gaba_T_ptr[i] += gaba_group.T_max * released;
            } else {
                gaba_u_ptr[i] += ((-gaba_u_ptr[i]) / gaba_group.tau_fac) * dt;
                if (gaba_u_ptr[i] < gaba_group.U) gaba_u_ptr[i] = gaba_group.U;
                gaba_R_ptr[i] += ((1.0f - gaba_R_ptr[i]) / gaba_group.tau_rec) * dt;
            }
            gaba_T_ptr[i] += ((-gaba_T_ptr[i]) / gaba_group.tau_clear) * dt;
            if (gaba_T_ptr[i] < 1e-6f) gaba_T_ptr[i] = 0.0f;

            float dr_gaba = (gaba_group.alpha_gaba * gaba_T_ptr[i] * (1.0f - gaba_r_gaba_ptr[i]) - gaba_group.beta_gaba * gaba_r_gaba_ptr[i]) * dt;
            gaba_r_gaba_ptr[i] = custom_max(0.0f, custom_min(1.0f, gaba_r_gaba_ptr[i] + dr_gaba));

            float g_gaba = gaba_g_max_ptr[i] * gaba_r_gaba_ptr[i];
            n_g_syn_tot[comp] += g_gaba;
            n_g_syn_E_tot[comp] += g_gaba * gaba_group.E_rev;
        }

        // --- Pass 3: AVX2 SIMD Core Matrix Vector Initialization -> Single precision processing 8 elements per iteration ---
        __m256 c_m_over_dt_vec = _mm256_set1_ps(C_m / dt);
        __m256 g_Na_vec = _mm256_set1_ps(g_Na);
        __m256 g_K_vec = _mm256_set1_ps(g_K);
        __m256 g_L_vec = _mm256_set1_ps(g_L);
        __m256 E_Na_vec = _mm256_set1_ps(E_Na);
        __m256 E_K_vec = _mm256_set1_ps(E_K);
        __m256 E_L_vec = _mm256_set1_ps(E_L);

        int i_vec = 0;
        for (; i_vec <= N - 8; i_vec += 8) {
            __m256 m_vec = _mm256_loadu_ps(&n_m[i_vec]);
            __m256 h_vec = _mm256_loadu_ps(&n_h[i_vec]);
            __m256 n_vec_val = _mm256_loadu_ps(&n_n[i_vec]);
            __m256 v_vec = _mm256_loadu_ps(&n_v[i_vec]);
            __m256 g_syn = _mm256_loadu_ps(&n_g_syn_tot[i_vec]);
            __m256 g_syn_E = _mm256_loadu_ps(&n_g_syn_E_tot[i_vec]);

            __m256 m2 = _mm256_mul_ps(m_vec, m_vec);
            __m256 m3 = _mm256_mul_ps(m2, m_vec);
            __m256 m3_h = _mm256_mul_ps(m3, h_vec);
            __m256 g_na_curr = _mm256_mul_ps(g_Na_vec, m3_h);

            __m256 n2 = _mm256_mul_ps(n_vec_val, n_vec_val);
            __m256 n4 = _mm256_mul_ps(n2, n2);
            __m256 g_k_curr = _mm256_mul_ps(g_K_vec, n4);

            __m256 d_val = _mm256_add_ps(c_m_over_dt_vec, g_syn);
            d_val = _mm256_add_ps(d_val, g_na_curr);
            d_val = _mm256_add_ps(d_val, g_k_curr);
            d_val = _mm256_add_ps(d_val, g_L_vec);
            _mm256_storeu_ps(&n_D[i_vec], d_val);

            __m256 rhs_val = _mm256_mul_ps(c_m_over_dt_vec, v_vec);
            rhs_val = _mm256_add_ps(rhs_val, g_syn_E);
            rhs_val = _mm256_fmadd_ps(g_na_curr, E_Na_vec, rhs_val);
            rhs_val = _mm256_fmadd_ps(g_k_curr, E_K_vec, rhs_val);
            rhs_val = _mm256_fmadd_ps(g_L_vec, E_L_vec, rhs_val);
            _mm256_storeu_ps(&n_RHS[i_vec], rhs_val);
        }
        
        for (; i_vec < N; ++i_vec) {
            float m_val = n_m[i_vec];
            float h_val = n_h[i_vec];
            float n_val = n_n[i_vec];
            float g_na_curr = g_Na * (m_val * m_val * m_val) * h_val;
            float g_k_curr = g_K * (n_val * n_val * n_val * n_val);

            n_D[i_vec] = (C_m / dt) + n_g_syn_tot[i_vec] + g_na_curr + g_k_curr + g_L;
            n_RHS[i_vec] = (C_m / dt) * n_v[i_vec] + n_g_syn_E_tot[i_vec] + g_na_curr * E_Na + g_k_curr * E_K + g_L * E_L;
        }

        // --- Pass 4: Topology Assembly Sweep ---
        for (int i = 1; i < N; ++i) {
            float g_ax = -n_A[i]; 
            int p = n_parent[i];
            n_D[i] += g_ax;
            n_D[p] += g_ax;
        }

        // --- Pass 5: Hines Tree Linear Solver (Eliminate and Substitution) ---
        for (int i = N - 1; i >= 1; --i) {
            int p = n_parent[i];
            float factor = n_A[i] / n_D[i];
            n_D[p] -= factor * n_A[i];
            n_RHS[p] -= factor * n_RHS[i];
        }

        n_v[0] = n_RHS[0] / n_D[0];

        for (int i = 1; i < N; ++i) {
            int p = n_parent[i];
            n_v[i] = (n_RHS[i] - n_A[i] * n_v[p]) / n_D[i];
        }

        // --- Pass 6: Gating Kinematics Updates via ETD1 ---
        for (int i = 0; i < N; ++i) {
            float v_curr = n_v[i];

            float am = alpha_m(v_curr);
            float bm = beta_m(v_curr);
            float tau_m = 1.0f / (am + bm);
            float m_inf = am * tau_m;
            n_m[i] = m_inf + (n_m[i] - m_inf) * std::exp(-dt / tau_m);

            float ah = alpha_h(v_curr);
            float bh = beta_h(v_curr);
            float tau_h = 1.0f / (ah + bh);
            float h_inf = ah * tau_h;
            n_h[i] = h_inf + (n_h[i] - h_inf) * std::exp(-dt / tau_h);

            float an = alpha_n(v_curr);
            float bn = beta_n(v_curr);
            float tau_n = 1.0f / (an + bn);
            float n_inf = an * tau_n;
            n_n[i] = n_inf + (n_n[i] - n_inf) * std::exp(-dt / tau_n);
        }

        // Perform threshold checking (> 0.0 mV) to capture Action Potential edges
        bool is_currently_spiking = (n_v[0] > 0.0f);

        return is_currently_spiking;
    }

    // Accessors to ensure correct flat synchronization structure sizes
    size_t get_total_synapses() const { return ampa_group.target_comp.size() + gaba_group.target_comp.size(); }
    size_t get_ampa_count() const { return ampa_group.target_comp.size(); }
    size_t get_gaba_count() const { return gaba_group.target_comp.size(); }
};

// =====================================================================
// 7. HELPER FUNCTIONS AND UTILITIES
// =====================================================================

/**
 * Saves the multi-compartment neuron simulation data into an HDF5 file using
 * an efficient sparse schema for inputs and a dense array for somatic outputs.
 */
void save_simulation_to_raw_bin(const std::string& filename,
                               const std::vector<uint32_t>& spike_steps,
                               const std::vector<uint16_t>& spike_synapse_ids,
                               const std::vector<uint32_t>& soma_spikes) 
{
    // Open file in binary write mode
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Helper lambda to write a vector's raw memory buffer
    auto write_vector = [&out](const auto& vec) {
        uint64_t size = vec.size();
        // 1. Write the size of the vector first (so the parser knows how much to read)
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        // 2. Write the raw continuous block of data
        if (size > 0) {
            out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(vec[0]));
        }
    };

    // Sequential flat dump of all fields
    write_vector(spike_steps);
    write_vector(spike_synapse_ids);
    write_vector(soma_spikes);

    out.close();
    std::cout << "Successfully saved raw binary data to " << filename << std::endl;
}