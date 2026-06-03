#!/usr/bin/env python3
"""
Corrected 2-position experiment with 10000 random bits on yellow_shuffled.
"""
import sys, os
import numpy as np
import pandas as pd
import types

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHT_CONDITION = "yellow_shuffled"
all_probes = np.arange(5, 366, 5, dtype=float)

# ============================================================================
# 1. Load test module with UTF-8 BOM handling
# ============================================================================
module_name = "test_3_simple_runtime_random"
module_path = os.path.join(project_root, "test-3", "test_3_simple.py")
with open(module_path, "r", encoding="utf-8-sig") as f:
    source = f.read().lstrip("\ufeff")

test = types.ModuleType(module_name)
test.__file__ = module_path
test.__package__ = ""
sys.modules[module_name] = test
exec(compile(source, module_path, "exec"), test.__dict__)

print("Test module loaded.")


# ============================================================================
# 2. Pre-compute fingerprints
# ============================================================================
print("Pre-computing fingerprints...")
all_models = {}
for pos in range(1, 29):
    csv_file = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{pos}.csv")
    mat = pd.read_csv(csv_file).values.astype(float)
    model = test.extract_fingerprint(all_probes, mat, force_positive_first=True)
    all_models[pos] = model
print("Done.\n")


# ============================================================================
# 3. Evaluate with 10000 random bits
# ============================================================================
def evaluate_with_random_bits(pos_a, pos_b, probes, hue_mapping, num_bits=10000, seed=42):
    """Evaluate BER using random bit sequences."""
    rng = np.random.RandomState(seed)
    
    probes_array = np.asarray(probes, dtype=float)
    probe_indices = [int((p/5)-1) for p in probes]
    
    # Load models
    csv_a = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{pos_a}.csv")
    csv_b = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{pos_b}.csv")
    
    mat_a = pd.read_csv(csv_a).values[probe_indices].astype(float)
    mat_b = pd.read_csv(csv_b).values[probe_indices].astype(float)
    
    model_a = test.extract_fingerprint(probes_array, mat_a, force_positive_first=True)
    model_b = test.extract_fingerprint(probes_array, mat_b, force_positive_first=True)
    models = test.align_model_directions([model_a, model_b])
    codes = [m.code for m in models]
    probe_to_row = test.build_probe_to_row(probes_array)
    
    # Generate random bits
    bits_a = rng.choice([-1, 1], size=num_bits)
    bits_b = rng.choice([-1, 1], size=num_bits)
    
    # Legal decoding
    err_legal = [0, 0]
    for t in range(num_bits):
        bits_pm = np.array([bits_a[t], bits_b[t]])
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        
        for idx, model in enumerate(models):
            Y_obs = test.observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            dec = test.decode_local_block(Y_obs, model.w, model.code)
            if dec.bit_hat_pm != bits_pm[idx]:
                err_legal[idx] += 1
    
    ber_legal = [e / num_bits for e in err_legal]
    secure_legal = [min(b, 1-b) for b in ber_legal]
    
    # Illegal decoding
    illegal_positions = [p for p in range(1, 29) if p not in (pos_a, pos_b)]
    min_secure_illegal = float('inf')
    worst_pos = -1
    worst_legal_idx = -1
    
    for pos_il in illegal_positions:
        csv_il = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{pos_il}.csv")
        mat_il = pd.read_csv(csv_il).values[probe_indices].astype(float)
        model_il = test.extract_fingerprint(probes_array, mat_il, force_positive_first=True)
        
        # Align to first legal model
        corr = np.mean(model_il.code * models[0].code)
        if corr < 0:
            model_il.w = -model_il.w
            model_il.z = -model_il.z
            model_il.code = -model_il.code
        
        err_illegal = [0, 0]
        for t in range(num_bits):
            bits_pm = np.array([bits_a[t], bits_b[t]])
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            
            Y_obs = test.observe_block_from_measured_matrix(hue_seq, model_il.Y, probe_to_row)
            dec = test.decode_local_block(Y_obs, model_il.w, model_il.code)
            
            for idx in range(2):
                if dec.bit_hat_pm != bits_pm[idx]:
                    err_illegal[idx] += 1
        
        ber_il = [e / num_bits for e in err_illegal]
        secure_il = [min(b, 1-b) for b in ber_il]
        
        for idx in range(2):
            if secure_il[idx] < min_secure_illegal:
                min_secure_illegal = secure_il[idx]
                worst_pos = pos_il
                worst_legal_idx = idx
    
    return {
        'legal_ber': ber_legal,
        'legal_secure': secure_legal,
        'min_illegal_secure': min_secure_illegal,
        'worst_illegal_pos': worst_pos,
        'worst_legal_idx': worst_legal_idx,
    }


# ============================================================================
# 4. Test
# ============================================================================
safe_combos = [(5, 13), (5, 16), (5, 23)]

print("="*70)
print("2-Position System with 10000 Random Bits")
print(f"Dataset: {LIGHT_CONDITION}")
print("Target: legal BER ≈ 0, min illegal BER > 0.3")
print("="*70)

best_result = None
best_min_illegal = -1

for pos_a, pos_b in safe_combos:
    print(f"\nTesting ({pos_a}, {pos_b})...")
    
    probe_sets = [
        np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float),
        np.array([5, 45, 90, 135, 180, 225, 270, 315], dtype=float),
        np.array([10, 50, 100, 150, 200, 250, 300, 350], dtype=float),
    ]
    
    for probes in probe_sets:
        probe_indices = [int((p/5)-1) for p in probes]
        
        # Build hue mapping
        model_a = all_models[pos_a]
        model_b = all_models[pos_b]
        code_a = model_a.code[probe_indices]
        code_b = model_b.code[probe_indices]
        
        symbols = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        hue_mapping = {}
        for sym in symbols:
            s_a, s_b = sym
            scores = s_a * code_a + s_b * code_b
            best_idx = int(np.argmax(scores))
            hue_mapping[sym] = int(probes[best_idx])
        
        # Evaluate
        result = evaluate_with_random_bits(pos_a, pos_b, probes, hue_mapping, num_bits=10000, seed=42)
        
        print(f"  Probes: {list(probes)}")
        print(f"  Legal secure BER: {result['legal_secure']}")
        print(f"  Min illegal secure BER: {result['min_illegal_secure']:.4f}")
        print(f"  Worst illegal pos: {result['worst_illegal_pos']} (attacking legal idx {result['worst_legal_idx']})")
        
        if result['min_illegal_secure'] > best_min_illegal:
            best_min_illegal = result['min_illegal_secure']
            best_result = result
            best_result['positions'] = (pos_a, pos_b)
            best_result['probes'] = probes

print("\n" + "="*70)
print("BEST RESULT:")
if best_result:
    print(f"  Positions: {best_result['positions']}")
    print(f"  Probes: {list(best_result['probes'])}")
    print(f"  Legal secure BER: {best_result['legal_secure']}")
    print(f"  Min illegal secure BER: {best_result['min_illegal_secure']:.4f}")
print("="*70)
print("\nDone.")
