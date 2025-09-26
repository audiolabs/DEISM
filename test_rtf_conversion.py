#!/usr/bin/env python3
"""
Test script for RTF to RIR conversion
Demonstrates the usage of the RTF conversion functions
"""

import numpy as np
import matplotlib.pyplot as plt
from deism.rtf_utils import (
    convert_RTF_to_RIR,
    generate_rtf_from_parameters,
    validate_rtf_inputs,
)


def test_rtf_conversion():
    """Test the RTF to RIR conversion functionality."""

    print("=== RTF to RIR Conversion Test ===\n")

    # Test parameters
    fstart, fstep, fend = 20, 2, 2000
    fs = 44100
    ir_length = 1.0

    # Generate frequency array
    frequencies = np.arange(fstart, fend + fstep, fstep)
    print(f"Frequency range: {fstart} Hz to {fend} Hz (step: {fstep} Hz)")
    print(f"Number of frequency points: {len(frequencies)}")

    # Generate RTF using different models
    models = ["exponential", "linear", "constant"]

    for model in models:
        print(f"\n--- Testing {model} model ---")

        # Generate RTF
        RTF = generate_rtf_from_parameters(frequencies, model_type=model)

        # Validate inputs
        try:
            validate_rtf_inputs(RTF, frequencies)
            print("✓ Input validation passed")
        except ValueError as e:
            print(f"✗ Input validation failed: {e}")
            continue

        # Convert to RIR
        rir = convert_RTF_to_RIR(RTF, frequencies, fs=fs, ir_length=ir_length)

        # Print results
        print(
            f"RTF magnitude range: {np.min(np.abs(RTF)):.3e} to {np.max(np.abs(RTF)):.3e}"
        )
        print(
            f"RIR magnitude range: {np.min(np.abs(rir)):.3e} to {np.max(np.abs(rir)):.3e}"
        )
        print(f"RIR length: {len(rir)} samples ({len(rir)/fs:.3f} seconds)")

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot RTF
        ax1.plot(frequencies, 20 * np.log10(np.abs(RTF)))
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude [dB]")
        ax1.set_title(f"RTF - {model} model")
        ax1.grid(True)

        # Plot RIR
        t = np.arange(len(rir)) / fs
        ax2.plot(t, 20 * np.log10(np.abs(rir)))
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Magnitude [dB]")
        ax2.set_title(f"RIR - {model} model")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"rtf_rir_{model}.png", dpi=150, bbox_inches="tight")
        plt.show()

        print(f"✓ Plot saved as 'rtf_rir_{model}.png'")


def test_custom_rtf():
    """Test with custom RTF data."""

    print("\n=== Custom RTF Test ===\n")

    # Create custom RTF (e.g., from your DEISM simulation)
    frequencies = np.linspace(20, 2000, 100)

    # Simulate a more realistic RTF with multiple resonances
    RTF = np.zeros_like(frequencies, dtype=complex)

    # Add multiple resonances
    resonance_freqs = [100, 200, 500, 1000, 1500]
    for f_res in resonance_freqs:
        # Resonance with Q factor
        Q_factor = 10
        omega_res = 2 * np.pi * f_res
        omega = 2 * np.pi * frequencies

        # Lorentzian resonance
        resonance = 1 / (1 + 1j * Q_factor * (omega / omega_res - omega_res / omega))
        RTF += resonance

    # Add some noise
    noise_level = 0.01
    RTF += noise_level * (
        np.random.randn(len(frequencies)) + 1j * np.random.randn(len(frequencies))
    )

    print(f"Custom RTF with {len(resonance_freqs)} resonances")
    print(f"Frequency range: {frequencies[0]:.1f} Hz to {frequencies[-1]:.1f} Hz")

    # Convert to RIR
    rir = convert_RTF_to_RIR(RTF, frequencies, fs=44100, ir_length=2.0)

    print(f"RIR length: {len(rir)} samples ({len(rir)/44100:.3f} seconds)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # RTF
    ax1.plot(frequencies, 20 * np.log10(np.abs(RTF)))
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Magnitude [dB]")
    ax1.set_title("Custom RTF with Multiple Resonances")
    ax1.grid(True)

    # RIR
    t = np.arange(len(rir)) / 44100
    ax2.plot(t, 20 * np.log10(np.abs(rir)))
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_title("Corresponding RIR")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("custom_rtf_rir.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✓ Custom RTF test completed")


if __name__ == "__main__":
    # Run tests
    test_rtf_conversion()
    test_custom_rtf()

    print("\n=== All tests completed ===")
