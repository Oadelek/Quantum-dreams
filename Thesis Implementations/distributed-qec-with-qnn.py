import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, pauli_error, QuantumError
from qiskit.quantum_info import Statevector, Pauli
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.visualization import plot_state_city, plot_histogram
from qiskit.circuit import Parameter
import networkx as nx
import random
import matplotlib.pyplot as plt
import logging
import time

# Configure logging (as before)
logging.basicConfig(filename='decoder_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# --- Configuration Parameters ---
class Config:
    LATTICE_SIZE = 7
    NUM_DATA_QUBITS = 25
    NUM_ANCILLA_QUBITS = 24
    QPUS_GRID_SIZE = 4 # Size of each QPU grid (e.g., 4x4)
    NUM_QPUS = 4
    ERROR_RATES = [0.02, 0.024, 0.018, 0.022] # Example error rates per QPU - Increased rates slightly
    READOUT_ERROR_PROB = 0.05 # Probability of readout error
    SHOTS = 1024 # Number of shots for simulation
    EVAL_SHOTS = 4096 # More shots for evaluation to get smoother results
    SPSA_MAX_ITER = 30 # Increased iterations for better training
    VQC_REPS = 2 # Number of repetitions in VQC layers (increased for complexity)
    NUM_SYNDROME_SAMPLES = 50 # Increased training data size
    NUM_EVALUATION_SAMPLES = 100 # Samples for evaluation
    CENTRALIZED_VQC_QUBITS = 8 # Number of qubits for centralized VQC
    GLOBAL_VQC_QUBITS = 8 # Number of qubits for global VQC
    LOCAL_VQC_QUBITS = 8 # Number of qubits for local VQC
    PAULI_OBSERVABLE = 'Z' # Observable for cost function, 'Z' or 'X' or 'Y' or 'I'
    TRAINING_COST_SHOTS = 1024 # Shots for cost function evaluation during training
    EVALUATION_COST_SHOTS = 4096 # Shots for cost function evaluation during final evaluation


config = Config() # Instantiate configuration

# --- Initialize the Surface Code Lattice (unchanged but using config) ---
def initialize_lattice(lattice_size=config.LATTICE_SIZE):
    """Initialize a surface code lattice with data and ancilla qubits."""
    n_rows, n_cols = lattice_size, lattice_size
    data_qubits = [f"D{i}" for i in range(1, config.NUM_DATA_QUBITS + 1)] # Using config
    ancilla_qubits = [f"A{i}" for i in range(1, config.NUM_ANCILLA_QUBITS + 1)] # Using config

    lattice = {}
    data_idx, ancilla_idx = 0, 0
    for i in range(n_rows):
        for j in range(n_cols):
            if (i + j) % 2 == 0:
                if data_idx < len(data_qubits): # Added check to prevent index out of range
                    lattice[(i, j)] = data_qubits[data_idx]
                    data_idx += 1
            else:
                if ancilla_idx < len(ancilla_qubits): # Added check to prevent index out of range
                    lattice[(i, j)] = ancilla_qubits[ancilla_idx]
                    ancilla_idx += 1

    assigned_data_qubits = [q for q in lattice.values() if q.startswith("D")]
    assigned_ancilla_qubits = [q for q in lattice.values() if q.startswith("A")]
    data_indices = {q: i for i, q in enumerate(assigned_data_qubits)}
    ancilla_indices = {q: i + len(assigned_data_qubits) for i, q in enumerate(assigned_ancilla_qubits)}

    qpus = []
    qpu_grid_size = config.QPUS_GRID_SIZE # Using config
    for qpu_row_start in range(0, lattice_size, qpu_grid_size):
        for qpu_col_start in range(0, lattice_size, qpu_grid_size):
            qpu_qubits = [
                lattice[(i, j)]
                for i in range(qpu_row_start, min(qpu_row_start + qpu_grid_size, lattice_size))
                for j in range(qpu_col_start, min(qpu_col_start + qpu_grid_size, lattice_size))
                if (i, j) in lattice
            ]
            qpus.append(qpu_qubits)
    qpus = qpus[:config.NUM_QPUS] # Ensure only NUM_QPUS are used

    return lattice, qpus, assigned_data_qubits, assigned_ancilla_qubits, data_indices, ancilla_indices


def compute_cost(bound_circuit, backend, shots):
    """This is to compute the expectation value of Pauli-Z on the first qubit."""
    from qiskit import transpile
    transpiled_circuit = transpile(bound_circuit, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    counts = job.result().get_counts()

    exp_val = 0
    for bitstring, count in counts.items():
        qubit_0_state = int(bitstring[-1])  # Last qubit in bitstring
        value = 1 if qubit_0_state == 0 else -1
        exp_val += value * count
    exp_val /= shots
    return -exp_val  # Negative for minimization



# --- Create Stabilizer Circuit (Simplified X Stabilizer for Data Qubits) ---
def create_stabilizer_circuit_x_basis(data_qubits, data_indices):
    """Simplified stabilizer circuit measuring X stabilizers around data qubits."""
    num_stabilizers = len(data_qubits)
    stabilizer_circuit = QuantumCircuit(len(data_qubits) + num_stabilizers, num_stabilizers) # Data + ancilla qubits, classical for syndromes
    data_q_bits = list(range(len(data_qubits)))
    ancilla_q_bits = list(range(len(data_qubits), len(data_qubits) + num_stabilizers))

    for idx, data_q in enumerate(data_qubits):
        if idx % 5 == 0 and idx > 0 and idx < len(data_qubits):
            stabilizer_circuit.h(ancilla_q_bits[idx]) # Prepare ancilla in + state
            stabilizer_circuit.cx(data_q_bits[idx-1], ancilla_q_bits[idx]) # Entangle with nearby data qubits
            stabilizer_circuit.cx(data_q_bits[idx], ancilla_q_bits[idx])
            stabilizer_circuit.h(ancilla_q_bits[idx]) # Change back to Z basis for measurement
            stabilizer_circuit.measure(ancilla_q_bits[idx], idx) # Measure ancilla to get syndrome bit

    return stabilizer_circuit

# --- Error Simulation with Pauli Noise ---
def simulate_errors_pauli(num_qubits, error_rates, qpu_assignment):
    """Simulate Pauli errors with variable rates per QPU and qubit assignment."""
    noise_model = NoiseModel()
    for qubit_idx in range(num_qubits):
        qpu_idx = qpu_assignment[qubit_idx]  # Get QPU index for this qubit
        error_rate = error_rates[qpu_idx]
        if error_rate > 0:
            # Calculate individual error probabilities
            prob_error_each = error_rate / 3.0  # Equal distribution among X, Y, Z
            prob_I = 1.0 - error_rate  # Probability of identity (no error)

            # Fix for normalization issue: explicitly normalize probabilities
            for op, channel in [('X', 'rx'), ('Y', 'ry'), ('Z', 'rz')]:
                # Create normalized Pauli error
                error = pauli_error([
                    (op, prob_error_each),
                    ('I', 1.0 - prob_error_each)  # Ensure exact normalization
                ])
                noise_model.add_quantum_error(error, [channel], [qubit_idx])

            # Add error after Hadamard
            h_error = pauli_error([
                ('X', prob_error_each),
                ('I', 1.0 - prob_error_each)  # Ensure exact normalization
            ])
            noise_model.add_quantum_error(h_error, ['h'], [qubit_idx])

    # Add readout error
    readout_error = ReadoutError([
        [1-config.READOUT_ERROR_PROB, config.READOUT_ERROR_PROB],
        [config.READOUT_ERROR_PROB, 1-config.READOUT_ERROR_PROB]
    ])
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model

# --- Generate Syndrome Data (Now using Stabilizer Circuit and Error Simulation) ---
def generate_syndrome_data(num_samples, data_qubits, data_indices, error_rates, qpus):
    """Generate syndrome data by simulating error injection and stabilizer measurement."""
    syndrome_labels_list = []
    syndrome_patterns_list = []

    for _ in range(num_samples):
        # 1. Stabilizer Circuit
        stabilizer_circuit = create_stabilizer_circuit_x_basis(data_qubits, data_indices)

        # 2. QPU assignment for noise simulation
        num_circuit_qubits = stabilizer_circuit.num_qubits
        qpu_assignment_data_ancilla = {}  # Map qubit index to QPU index
        all_circuit_qubits_names = data_qubits + ancilla_qubits  # Combined list of qubit names

        # Ensure enough qubit names for all circuit qubits
        if len(all_circuit_qubits_names) < num_circuit_qubits:
            all_circuit_qubits_names.extend([f'Q_{i}' for i in range(len(all_circuit_qubits_names), num_circuit_qubits)])

        for i in range(num_circuit_qubits):
            qubit_name = all_circuit_qubits_names[i]
            for qpu_idx, qpu_qs in enumerate(qpus):
                if qubit_name in qpu_qs:
                    qpu_assignment_data_ancilla[i] = qpu_idx
                    break
            else:
                qpu_assignment_data_ancilla[i] = 0

        # 3. Noise Model
        noise_model = simulate_errors_pauli(stabilizer_circuit.num_qubits, error_rates, qpu_assignment_data_ancilla)

        # 4. Simulation with Noise
        backend = AerSimulator(method='statevector')
        job = backend.run(stabilizer_circuit, shots=1, noise_model=noise_model)  # Run directly
        result = job.result()
        counts = result.get_counts(stabilizer_circuit)
        syndrome_pattern_str = list(counts.keys())[0]
        syndrome_pattern = [int(bit) for bit in syndrome_pattern_str[::-1]]
        syndrome_labels = syndrome_pattern

        syndrome_labels_list.append(syndrome_labels)
        syndrome_patterns_list.append(syndrome_pattern)

    return syndrome_patterns_list, syndrome_labels_list

# --- Local VQC for Any QPU (using config) ---
def create_local_vqc(qpu_qubits, syndrome, qpu_idx, error_rates):
    """Create a local VQC for a given QPU."""
    ancilla_qs = [q for q in qpu_qubits if q.startswith("A")]
    shared_qs = [q for q in qpu_qubits if q in ["D6", "D8", "D15", "D16", "D17", "D24"] and q.startswith("D")]

    n_qubits = min(len(ancilla_qs) + len(shared_qs), config.LOCAL_VQC_QUBITS) # Using config
    local_circuit = QuantumCircuit(n_qubits)
    cr = ClassicalRegister(n_qubits, name=f"c_qpu{qpu_idx+1}")
    local_circuit.add_register(cr)
    local_indices = {q: i for i, q in enumerate(ancilla_qs + shared_qs) if i < n_qubits}

    # Syndrome Encoding (simplified - encode on first qubits)
    for i in range(min(n_qubits, len(syndrome))):
        local_circuit.ry(np.pi * syndrome[i % len(syndrome)], i) # Encode syndrome bits

    # Variational parameters and layers (using config)
    theta = [Parameter(f'θ_local_{qpu_idx}_{i}_{j}') for i in range(n_qubits) for j in range(3 * config.VQC_REPS)] # Adjusted parameter count
    for layer in range(config.VQC_REPS): # Using config reps
        for i in range(n_qubits):
            local_circuit.rx(theta[i * 3 * config.VQC_REPS + layer * 3 + 0], i)
            local_circuit.ry(theta[i * 3 * config.VQC_REPS + layer * 3 + 1], i)
            local_circuit.rz(theta[i * 3 * config.VQC_REPS + layer * 3 + 2], i)
        for i in range(0, n_qubits - 1, 2): # Entangling layer
            local_circuit.cx(i, i + 1)

    for i in range(n_qubits):
        local_circuit.measure(i, cr[i])

    return local_circuit, len(theta)

# --- Global VQC for Central Aggregator (using config) ---
def create_global_vqc(qpu_outputs):
    """Create a global VQC."""
    n_qubits_global = config.GLOBAL_VQC_QUBITS # Using config
    global_circuit = QuantumCircuit(n_qubits_global)
    cr_global = ClassicalRegister(n_qubits_global, name="c_global")
    global_circuit.add_register(cr_global)

    # Encode QPU outputs (simplified - encode prediction and first state bit)
    for i, (qpu, data) in enumerate(qpu_outputs.items()):
        if i < config.NUM_QPUS and i * 2 < n_qubits_global: # Process up to NUM_QPUS and within qubit limit
            global_circuit.ry(np.pi * data["pred"], i * 2)
            if i * 2 + 1 < n_qubits_global:
                global_circuit.rx(np.pi * data["state"][0], i * 2 + 1)

    # Variational parameters and layers (using config)
    theta_global = [Parameter(f'θ_global_{i}_{j}') for i in range(n_qubits_global) for j in range(3 * config.VQC_REPS)] # Adjusted parameter count
    for layer in range(config.VQC_REPS): # Using config reps
        for i in range(n_qubits_global):
            global_circuit.rx(theta_global[i * 3 * config.VQC_REPS + layer * 3 + 0], i)
            global_circuit.ry(theta_global[i * 3 * config.VQC_REPS + layer * 3 + 1], i)
            global_circuit.rz(theta_global[i * 3 * config.VQC_REPS + layer * 3 + 2], i)
        for i in range(0, n_qubits_global - 1, 2): # Entangling layer
            global_circuit.cx(i, i + 1)

    for i in range(n_qubits_global):
        global_circuit.measure(i, cr_global[i])
    return global_circuit, len(theta_global)

# --- Centralized VQC (for comparison) ---
def create_centralized_vqc():
    """Create a centralized VQC for comparison."""
    circuit = EfficientSU2(config.CENTRALIZED_VQC_QUBITS, reps=config.VQC_REPS) # Using EfficientSU2 and config
    circuit.measure_all()
    return circuit, len(circuit.parameters)

# --- Cost Function (Expected value of Pauli-Z on first qubit) ---
def cost_function(params, circuit, backend, shots=config.TRAINING_COST_SHOTS, pauli_observable=config.PAULI_OBSERVABLE):
    """Cost function based on expectation value of Pauli-Z on the first qubit (qubit 0)."""
    # Bind parameters to the circuit
    param_dict = {param: params[i] for i, param in enumerate(circuit.parameters)}
    bound_circuit = circuit.assign_parameters(param_dict)

    # Transpile and run the circuit
    transpiled_circuit = transpile(bound_circuit, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    counts = job.result().get_counts()

    # Compute expectation value of Pauli-Z on qubit 0 from measurement counts
    exp_val = 0
    for bitstring, count in counts.items():
        # Qubit 0 is the last bit in the bitstring
        qubit_0_state = int(bitstring[-1])  # '0' -> 0, '1' -> 1
        value = 1 if qubit_0_state == 0 else -1  # +1 for '0', -1 for '1'
        exp_val += value * count
    exp_val /= shots  # Normalize by number of shots

    # Return negative expectation value (for minimization)
    return -exp_val

# --- Training Function (Generalized for Local, Global, Centralized) ---
def train_vqnn_architecture(architecture_type, circuit, initial_params, backend, cost_func):
    """Generic training function for different VQNN architectures."""
    optimizer = SPSA(maxiter=config.SPSA_MAX_ITER) # Using config
    try:
        result = optimizer.minimize(cost_func, initial_params)
        optimal_params = result.x
        logging.info(f"Training successful for {architecture_type} architecture.")
        return optimal_params, result.fun # Return optimal parameters and final cost
    except Exception as e:
        error_message = f"Error in {architecture_type} training: {str(e)}"
        logging.error(error_message)
        print(error_message)
        return initial_params, np.inf # Return initial parameters and infinite cost in case of error

# --- Evaluation Function ---
def evaluate_architecture(architecture_type, circuit_or_data, optimal_params, evaluation_syndromes, backend):
    """Evaluate the performance of a given architecture."""
    evaluation_costs = []
    predicted_logical_values = []

    for syndrome in evaluation_syndromes:
        bound_circuit = None
        try:
            if architecture_type == 'local':
                qpu_idx, qpu_qubits = circuit_or_data  
                local_circuit, _ = create_local_vqc(qpu_qubits, syndrome, qpu_idx, error_rates)
                param_dict = {local_circuit.parameters[i]: optimal_params[i] for i in range(len(optimal_params))}
                bound_circuit = local_circuit.assign_parameters(param_dict)
            elif architecture_type == 'global':
                qpu_outputs = syndrome  # syndrome is qpu_outputs for global case
                global_circuit, _ = create_global_vqc(qpu_outputs)
                param_dict = {global_circuit.parameters[i]: optimal_params[i] for i in range(len(optimal_params))}
                bound_circuit = global_circuit.assign_parameters(param_dict)
            elif architecture_type == 'centralized':
                circuit = circuit_or_data
                param_dict = {circuit.parameters[i]: optimal_params[i] for i in range(len(optimal_params))}
                bound_circuit = circuit.assign_parameters(param_dict)

            if bound_circuit is not None:
                evaluation_cost = compute_cost(bound_circuit, backend, config.EVALUATION_COST_SHOTS)
                evaluation_costs.append(evaluation_cost)

                # Predict logical value (example implementation)
                transpiled_eval_circuit = transpile(bound_circuit, backend)
                job_eval = backend.run(transpiled_eval_circuit, shots=config.EVAL_SHOTS)
                counts_eval = job_eval.result().get_counts()
                predicted_bitstring = max(counts_eval, key=counts_eval.get)
                predicted_logical_value = int(predicted_bitstring[0])  
                predicted_logical_values.append(predicted_logical_value)

        except Exception as e:
            error_message = f"Evaluation error for {architecture_type} with syndrome {syndrome}: {str(e)}"
            print(error_message)
            evaluation_costs.append(float('inf'))
            predicted_logical_values.append(None)

    avg_cost = np.mean(evaluation_costs) if evaluation_costs else float('inf')
    return avg_cost, predicted_logical_values

# --- Run Training and Evaluation for Architectures ---
def run_experiment(lattice, qpus, data_qubits, data_indices, ancilla_qubits, error_rates, num_syndrome_samples=config.NUM_SYNDROME_SAMPLES, num_eval_samples=config.NUM_EVALUATION_SAMPLES):

    backend = AerSimulator() # Default backend for training and eval

    # Dictionary to store execution times
    execution_times = {}

    # --- Syndrome Data Generation ---
    training_syndromes, training_labels = generate_syndrome_data(num_syndrome_samples, data_qubits, data_indices, error_rates, qpus)
    evaluation_syndromes, evaluation_labels = generate_syndrome_data(num_eval_samples, data_qubits, data_indices, error_rates, qpus)

    # --- Architectures ---
    architectures = {}

    # 1. Localized VQC Architecture (Independent Local VQCs - no Global VQC)
    start_time_local = time.time()  # Start timing
    optimal_params_local = []
    initial_params_local = []
    local_circuits_trained = [] # Store trained local circuits for evaluation
    initial_local_circuits = [] # Store initial local circuits for visualization
    for qpu_idx, qpu_qubits in enumerate(qpus):
        local_circuit, n_params_local = create_local_vqc(qpu_qubits, training_syndromes[0], qpu_idx, error_rates) # Use first syndrome for initial circuit creation - will be adapted in evaluation
        initial_local_circuits.append(local_circuit) # Store initial circuit
        initial_params = np.random.rand(n_params_local)
        initial_params_local.append(initial_params)

        # Define cost function that is *partially* dependent on syndrome (or independent for unsupervised)
        def local_cost_function_train(params): # Example: using first syndrome for simplicity
            local_circuit_instance, _ = create_local_vqc(qpu_qubits, training_syndromes[0], qpu_idx, error_rates) # Recreate circuit for each cost eval (inefficient, but clearer for example)
            return cost_function(params, local_circuit_instance, backend) # Use generic cost function

        optimal_param_qpu, _ = train_vqnn_architecture(f"Local_QPU{qpu_idx+1}", local_circuit, initial_params, backend, local_cost_function_train)
        optimal_params_local.append(optimal_param_qpu)
        local_circuits_trained.append((qpu_idx, qpu_qubits)) # Store QPU info for evaluation

    architectures['localized_vqnn'] = {'circuits': local_circuits_trained, 'params': optimal_params_local, 'type': 'local', 'initial_circuits': initial_local_circuits} # Store circuits as list of tuples and initial circuits
    execution_times['localized_vqnn'] = time.time() - start_time_local  # End timing for local

    # 2. Distributed VQC Architecture (Local + Global)
    start_time_distributed = time.time()  # Start timing
    global_circuit, n_params_global = create_global_vqc({}) # Initial global circuit - qpu outputs will be fed in evaluation
    initial_global_circuit = global_circuit # Store initial global circuit
    initial_params_global = np.random.rand(n_params_global)

    def global_cost_function_train(params):
        """Cost function for training the global VQC."""
        dummy_qpu_outputs = {f"QPU{i+1}": {"pred": 0.5, "state": [0, 0]} for i in range(config.NUM_QPUS)}
        global_circuit, _ = create_global_vqc(dummy_qpu_outputs)
        param_dict = {global_circuit.parameters[i]: params[i] for i in range(len(params))}
        bound_circuit = global_circuit.assign_parameters(param_dict)
        return compute_cost(bound_circuit, backend, config.TRAINING_COST_SHOTS)

    optimal_params_global, _ = train_vqnn_architecture("Global_VQC", global_circuit, initial_params_global, backend, global_cost_function_train)
    architectures['distributed_vqnn'] = {'circuits': global_circuit, 'params': optimal_params_global, 'type': 'global', 'initial_circuit': initial_global_circuit}
    execution_times['distributed_vqnn'] = time.time() - start_time_distributed  # End timing for distributed

    # 3. Centralized VQC Architecture
    start_time_centralized = time.time()  # Start timing
    centralized_circuit, n_params_centralized = create_centralized_vqc()
    initial_centralized_circuit = centralized_circuit # Store initial centralized circuit
    initial_params_centralized = np.random.rand(n_params_centralized)

    def centralized_cost_function_train(params):
        return cost_function(params, centralized_circuit, backend)

    optimal_params_centralized, _ = train_vqnn_architecture("Centralized_VQC", centralized_circuit, initial_params_centralized, backend, centralized_cost_function_train)
    architectures['centralized_vqnn'] = {'circuits': centralized_circuit, 'params': optimal_params_centralized, 'type': 'centralized', 'initial_circuit': initial_centralized_circuit}
    execution_times['centralized_vqnn'] = time.time() - start_time_centralized  # End timing for centralized

    # 4. Baseline: No Correction (Simulate Logical Error Rate without QNN)
    start_time_baseline = time.time()  # Start timing
    architectures['baseline_no_correction'] = {'circuits': None, 'params': None, 'type': 'baseline'} # placeholder


    # --- Evaluation of Architectures ---
    evaluation_results = {}

    # Evaluate Baseline (No Correction) - just error rate of syndrome generation process itself
    baseline_costs = []
    baseline_predicted_values = [] # Baseline has no prediction, so keep it empty
    for syndrome in evaluation_syndromes:
        # "Cost" for baseline can be a simple function of syndrome bits - example: sum of syndrome bits
        baseline_cost = sum(syndrome) # Higher syndrome sum = potentially more errors
        baseline_costs.append(baseline_cost)
        baseline_predicted_values.append(None) # No prediction for baseline

    avg_baseline_cost = np.mean(baseline_costs)
    evaluation_results['baseline_no_correction'] = {'avg_cost': avg_baseline_cost, 'predicted_values': baseline_predicted_values}
    execution_times['baseline_no_correction'] = time.time() - start_time_baseline  # End timing for baseline


    # Evaluate Localized VQNN
    start_time_local_eval = time.time()  # Start timing for evaluation
    local_vqnn_costs = []
    local_vqnn_predicted_values = []
    qpu_outputs_localized = {} # To store QPU outputs for localized VQNN
    qpu_states_localized = {}
    for syndrome in evaluation_syndromes:
        qpu_local_preds = {} # Simulate local QPU predictions based on local VQCs
        qpu_states = {}
        syndrome_costs_local = [] # Costs for each local VQC for this syndrome
        syndrome_predicted_values_local = [] # Predictions from each local VQC

        for qpu_idx, qpu_qubits_eval in enumerate(architectures['localized_vqnn']['circuits']): # qpu_qubits_eval is tuple (qpu_idx, qpu_qubits)
            local_circuit_eval, _ = create_local_vqc(qpu_qubits_eval[1], syndrome, qpu_qubits_eval[0], error_rates) # Adapt local circuit for eval syndrome
            avg_cost_local_qpu, predicted_values_local_qpu = evaluate_architecture('local', qpu_qubits_eval, architectures['localized_vqnn']['params'][qpu_idx], [syndrome], backend)
            qpu_local_preds[f"QPU{qpu_idx+1}"] = {"pred": -avg_cost_local_qpu, "state": syndrome[:2]} # Example: cost as prediction, syndrome as state
            qpu_states[f"QPU{qpu_idx+1}"] = syndrome[:2]
            syndrome_costs_local.append(avg_cost_local_qpu)
            syndrome_predicted_values_local.extend(predicted_values_local_qpu) # Aggregate predictions

        avg_cost_for_syndrome = np.mean(syndrome_costs_local) # Average cost across local VQCs for this syndrome
        local_vqnn_costs.append(avg_cost_for_syndrome)
        local_vqnn_predicted_values.extend(syndrome_predicted_values_local) # Aggregate predictions
        qpu_outputs_localized[str(syndrome)] = qpu_local_preds # Store QPU outputs for each syndrome
        qpu_states_localized[str(syndrome)] = qpu_states

    avg_local_vqnn_cost = np.mean(local_vqnn_costs) if local_vqnn_costs else np.inf
    evaluation_results['localized_vqnn'] = {'avg_cost': avg_local_vqnn_cost, 'predicted_values': local_vqnn_predicted_values, 'qpu_outputs': qpu_outputs_localized, 'qpu_states': qpu_states_localized} # Store QPU outputs for potential further use
    execution_times['localized_vqnn_eval'] = time.time() - start_time_local_eval

    # Evaluate Distributed VQNN (Local + Global)
    start_time_dist_eval = time.time()  # Start timing for evaluation
    distributed_vqnn_costs = []
    distributed_vqnn_predicted_values = []
    qpu_outputs_distributed = {} # Store QPU outputs for distributed VQNN
    qpu_states_distributed = {}

    for syndrome in evaluation_syndromes:
        qpu_local_preds_distributed = {} # Store local predictions for distributed VQNN
        qpu_states_distributed_syndrome = {} # Local states for this syndrome
        syndrome_costs_distributed = [] # Costs from global VQC for each syndrome
        syndrome_predicted_values_distributed = [] # Predictions from global VQC

        for qpu_idx, qpu_qubits_eval in enumerate(architectures['localized_vqnn']['circuits']): # Re-use local circuits from localized architecture for now - could train separate local VQCs if needed
            local_circuit_eval, _ = create_local_vqc(qpu_qubits_eval[1], syndrome, qpu_qubits_eval[0], error_rates)
            avg_cost_local_qpu, predicted_values_local_qpu = evaluate_architecture('local', qpu_qubits_eval, architectures['localized_vqnn']['params'][qpu_idx], [syndrome], backend)
            qpu_local_preds_distributed[f"QPU{qpu_idx+1}"] = {"pred": -avg_cost_local_qpu, "state": syndrome[:2]} # Example: cost as prediction, syndrome as state
            qpu_states_distributed_syndrome[f"QPU{qpu_idx+1}"] = syndrome[:2]

        qpu_outputs_distributed[str(syndrome)] = qpu_local_preds_distributed # Store QPU outputs for each syndrome
        qpu_states_distributed[str(syndrome)] = qpu_states_distributed_syndrome

        global_vqnn_cost, predicted_values_global_vqnn = evaluate_architecture('global', architectures['distributed_vqnn']['circuits'], architectures['distributed_vqnn']['params'], [qpu_outputs_distributed[str(syndrome)]], backend)
        distributed_vqnn_costs.append(global_vqnn_cost)
        distributed_vqnn_predicted_values.extend(predicted_values_global_vqnn) # Aggregate predictions

    avg_distributed_vqnn_cost = np.mean(distributed_vqnn_costs) if distributed_vqnn_costs else np.inf
    evaluation_results['distributed_vqnn'] = {'avg_cost': avg_distributed_vqnn_cost, 'predicted_values': distributed_vqnn_predicted_values, 'qpu_outputs': qpu_outputs_distributed, 'qpu_states': qpu_states_distributed}
    execution_times['distributed_vqnn_eval'] = time.time() - start_time_dist_eval

    # Evaluate Centralized VQNN
    start_time_central_eval = time.time()  # Start timing for evaluation
    centralized_vqnn_costs = []
    centralized_vqnn_predicted_values = []
    for syndrome in evaluation_syndromes:
        centralized_vqnn_cost, predicted_values_centralized_vqnn = evaluate_architecture('centralized', architectures['centralized_vqnn']['circuits'], architectures['centralized_vqnn']['params'], [syndrome], backend)
        centralized_vqnn_costs.append(centralized_vqnn_cost)
        centralized_vqnn_predicted_values.extend(predicted_values_centralized_vqnn)

    avg_centralized_vqnn_cost = np.mean(centralized_vqnn_costs) if centralized_vqnn_costs else np.inf
    evaluation_results['centralized_vqnn'] = {'avg_cost': avg_centralized_vqnn_cost, 'predicted_values': centralized_vqnn_predicted_values}
    execution_times['centralized_vqnn_eval'] = time.time() - start_time_central_eval

    return evaluation_results, architectures, execution_times # Return evaluation results and trained architectures for further analysis/plotting

# --- Visualization Functions (Improved) ---
def visualize_training_results(evaluation_results, evaluation_syndromes, execution_times):
    """Visualize and compare the evaluation costs and logical error rates for different architectures."""
    arch_names = list(evaluation_results.keys())
    avg_costs = [evaluation_results[name]['avg_cost'] for name in arch_names]

    # --- Bar Chart for Average Costs ---
    plt.figure(figsize=(12, 6))
    plt.bar(arch_names, avg_costs, color=['r', 'g', 'b', 'y']) # Example colors
    plt.ylabel('Average Cost (Lower is better - Expectation Value)')
    plt.title('Comparison of VQNN Architectures - Average Cost')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # --- Calculate and Visualize Logical Error Rates ---
    logical_error_rates = {}
    predicted_values_per_arch = {} # Store predictions for histogram

    for name in arch_names:
        if name != 'baseline_no_correction': # Baseline has no predictions
            predicted_values = evaluation_results[name]['predicted_values']
            predicted_values_per_arch[name] = predicted_values # Store for histogram
            errors = 0
            for i in range(len(evaluation_syndromes)): # assuming labels are the syndromes themselves for now
                if predicted_values[i] != evaluation_syndromes[i][0]: 
                    errors += 1
            error_rate = errors / len(evaluation_syndromes)
            logical_error_rates[name] = error_rate
        else:
            logical_error_rates[name] = np.nan # Baseline error rate is not defined by prediction
            predicted_values_per_arch[name] = [] # No predictions for baseline

    arch_names_error_rate = list(logical_error_rates.keys())
    error_rates_values = [logical_error_rates[name] for name in arch_names_error_rate]

    plt.figure(figsize=(12, 6))
    plt.bar(arch_names_error_rate, error_rates_values, color=['r', 'g', 'b', 'y'])
    plt.ylabel('Logical Error Rate (Lower is better)')
    plt.title('Comparison of VQNN Architectures - Logical Error Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # --- Bar Chart for Execution Times ---
    plt.figure(figsize=(12, 6))
    exec_times = [execution_times[name] for name in arch_names]
    plt.bar(arch_names, exec_times, color=['r', 'g', 'b', 'y'])
    plt.ylabel('Execution Time (seconds)')
    plt.title('Comparison of VQNN Architectures - Execution Time')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # --- Histograms of Predicted Values ---
    for name in arch_names:
        if name != 'baseline_no_correction':
            plt.figure(figsize=(8, 5))
            plt.hist(predicted_values_per_arch[name], bins=2, range=(-0.5, 1.5), rwidth=0.8, color='skyblue') # Histogram of 0s and 1s
            plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
            plt.ylabel('Frequency')
            plt.title(f'Predicted Logical Values - {name}')
            plt.tight_layout()
            plt.show()


    # --- Print Detailed Results in console/log ---
    print("\n--- Evaluation Results ---")
    logging.info("\n--- Evaluation Results ---")
    for name in arch_names:
        cost = evaluation_results[name]['avg_cost']
        error_rate_val = logical_error_rates[name] if name != 'baseline_no_correction' else "N/A"
        exec_time = execution_times[name]
        print(f"Architecture: {name}, Average Cost: {cost:.4f}, Logical Error Rate: {error_rate_val}, Execution Time: {exec_time:.4f} seconds")
        logging.info(f"Architecture: {name}, Average Cost: {cost:.4f}, Logical Error Rate: {error_rate_val}, Execution Time: {exec_time:.4f} seconds")

def visualize_lattice(lattice, optimal_params=None):
    """Visualize the surface code lattice with qubit assignments."""
    plt.figure(figsize=(10, 10))

    # Extract positions and types
    positions = list(lattice.keys())
    types = list(lattice.values())

    # Plot data qubits
    data_positions = [pos for pos, qubit in lattice.items() if qubit.startswith('D')]
    data_x = [pos[1] for pos in data_positions]
    data_y = [-pos[0] for pos in data_positions]  # Negative to flip the y-axis for better visualization
    plt.scatter(data_x, data_y, color='blue', s=100, label='Data Qubits')

    # Add data qubit labels
    for pos, qubit in lattice.items():
        if qubit.startswith('D'):
            plt.text(pos[1], -pos[0], qubit, fontsize=8, ha='center', va='center')

    # Plot ancilla qubits
    ancilla_positions = [pos for pos, qubit in lattice.items() if qubit.startswith('A')]
    ancilla_x = [pos[1] for pos in ancilla_positions]
    ancilla_y = [-pos[0] for pos in ancilla_positions]  # Negative to flip the y-axis
    plt.scatter(ancilla_x, ancilla_y, color='red', s=100, label='Ancilla Qubits')

    # Add ancilla qubit labels
    for pos, qubit in lattice.items():
        if qubit.startswith('A'):
            plt.text(pos[1], -pos[0], qubit, fontsize=8, ha='center', va='center')

    # Draw stabilizer interactions (connecting data and ancilla qubits)
    for i, j in ancilla_positions:
        # Connect to nearby data qubits (simplified - connect to adjacent positions)
        neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        for ni, nj in neighbors:
            if (ni, nj) in data_positions:
                plt.plot([j, nj], [-i, -ni], 'gray', linestyle='--', alpha=0.7)

    # If optimal parameters are provided, indicate trained status
    if optimal_params is not None:
        plt.title("Surface Code Lattice (After Training)")

        # Visualize parameter strength with color intensity
        if isinstance(optimal_params, list) and len(optimal_params) > 0:
            param_sum = sum(abs(p) for p in optimal_params[0]) if isinstance(optimal_params[0], np.ndarray) else 0
            param_text = f"Trained with {len(optimal_params)} parameter set(s)\nFirst set sum: {param_sum:.2f}"
            plt.figtext(0.5, 0.01, param_text, ha='center')
    else:
        plt.title("Surface Code Lattice (Before Training)")

    # Draw QPU boundaries
    qpu_size = config.QPUS_GRID_SIZE
    lattice_size = config.LATTICE_SIZE

    for i in range(0, lattice_size, qpu_size):
        for j in range(0, lattice_size, qpu_size):
            rect = plt.Rectangle((j-0.5, -i+0.5), qpu_size, -qpu_size,
                                 fill=False, edgecolor='green', linewidth=2, alpha=0.7)
            plt.gca().add_patch(rect)
            plt.text(j+qpu_size/2-0.5, -i-qpu_size/2+0.5,
                     f"QPU{int(i/qpu_size)*2 + int(j/qpu_size) + 1}",
                     color='green', fontsize=10, ha='center', va='center')

    plt.grid(True)
    plt.legend()
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.xlim(-1, lattice_size)
    plt.ylim(-lattice_size, 1)
    plt.tight_layout()
    plt.show()

def visualize_circuits(architectures):
    """Visualize the initial and trained circuits for each architecture."""
    for arch_name, arch_data in architectures.items():
        if arch_name == 'localized_vqnn':
            initial_circuits = arch_data.get('initial_circuits')
            trained_circuits_info = arch_data.get('circuits') # List of (qpu_idx, qpu_qubits)
            optimal_params = arch_data.get('params')

            if initial_circuits and trained_circuits_info and optimal_params: # Corrected if condition - removed optimal_params_local check
                print(f"\n--- {arch_name.upper()} ---")
                for i in range(len(initial_circuits)):
                    if optimal_params and len(optimal_params) > i and optimal_params[i] is not None: # Check if optimal_params is not empty and has params for this QPU
                        initial_circuit = initial_circuits[i]
                        trained_circuit_info = trained_circuits_info[i]
                        qpu_idx = trained_circuit_info[0]
                        qpu_qubits = trained_circuit_info[1]
                        local_circuit_trained, _ = create_local_vqc(qpu_qubits, [0]*config.NUM_ANCILLA_QUBITS, qpu_idx, config.ERROR_RATES) #dummy syndrome
                        param_dict_trained = {local_circuit_trained.parameters[j]: optimal_params[i][j] for j, param in enumerate(local_circuit_trained.parameters) if j < len(optimal_params[i])}
                        bound_circuit_trained = local_circuit_trained.assign_parameters(param_dict_trained)

                        print(f"\n--- Local VQC for QPU{qpu_idx+1} ---")
                        print("Before Training Circuit:")
                        plt.figure(figsize=(10,6))
                        initial_circuit.draw(output='mpl', fold=20, filename=f'{arch_name}_qpu{qpu_idx+1}_initial_circuit.png')
                        plt.show()
                        print("After Training Circuit:")
                        plt.figure(figsize=(10,6))
                        bound_circuit_trained.draw(output='mpl', fold=20, filename=f'{arch_name}_qpu{qpu_idx+1}_trained_circuit.png')
                        plt.show()


        elif arch_name in ['distributed_vqnn', 'centralized_vqnn']:
            initial_circuit = arch_data.get('initial_circuit')
            trained_circuit = arch_data.get('circuits')
            optimal_params = arch_data.get('params')

            if initial_circuit is not None and trained_circuit is not None and optimal_params is not None: # corrected if condition - removed optimal_params_local check (not needed here either, but keeping explicit None check for clarity)
                print(f"\n--- {arch_name.upper()} ---")
                print("Before Training Circuit:")
                plt.figure(figsize=(10,6))
                initial_circuit.draw(output='mpl', fold=20, filename=f'{arch_name}_initial_circuit.png')
                plt.show()

                if arch_name == 'distributed_vqnn':
                    dummy_qpu_outputs = {f"QPU{i+1}": {"pred": 0.5, "state": [0, 0]} for i in range(config.NUM_QPUS)}
                    global_circuit_trained, _ = create_global_vqc(dummy_qpu_outputs) # dummy input
                    param_dict_trained = {global_circuit_trained.parameters[i]: optimal_params[i] for i, param in enumerate(global_circuit_trained.parameters)}
                    bound_circuit_trained = global_circuit_trained.assign_parameters(param_dict_trained)
                    trained_circuit_to_draw = bound_circuit_trained
                elif arch_name == 'centralized_vqnn':
                    centralized_circuit_trained, _ = create_centralized_vqc()
                    param_dict_trained = {centralized_circuit_trained.parameters[i]: optimal_params[i] for i, param in enumerate(centralized_circuit_trained.parameters)}
                    bound_circuit_trained = centralized_circuit_trained.assign_parameters(param_dict_trained)
                    trained_circuit_to_draw = bound_circuit_trained

                print("After Training Circuit:")
                plt.figure(figsize=(10,6))
                trained_circuit_to_draw.draw(output='mpl', fold=20, filename=f'{arch_name}_trained_circuit.png')
                plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize lattice and other components
    lattice, qpus, data_qubits, ancilla_qubits, data_indices, ancilla_indices = initialize_lattice()
    error_rates = config.ERROR_RATES # use error rates from config


    # Run the experiment and get results
    evaluation_results, trained_architectures, execution_times = run_experiment(lattice, qpus, data_qubits, data_indices, ancilla_qubits, error_rates)

    # Generate evaluation syndromes again to pass to visualization for error rate calculation
    _, evaluation_syndromes_for_vis = generate_syndrome_data(config.NUM_EVALUATION_SAMPLES, data_qubits, data_indices, error_rates, qpus)

    visualize_lattice(lattice)
    visualize_circuits(trained_architectures) # Visualize circuits before and after training
    # Visualize the results
    visualize_training_results(evaluation_results, evaluation_syndromes_for_vis, execution_times)

    print("Experiment completed.")