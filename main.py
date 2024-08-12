import os
import numpy as np
from scipy.io import wavfile
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, Toplevel, Text, Listbox, MULTIPLE
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Global variables to store frequency data, common spectrum, unique spectra, and comparison results
frequency_data = {}
common_spectrum = {}
unique_spectra = {}
comparison_results = {}

def parse_wav_files(file_paths):
    global frequency_data
    frequency_data = {}
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name.endswith('.wav'):
            # Read the wav file
            sample_rate, data = wavfile.read(file_path)
            
            # Perform Fourier Transform
            n = len(data)
            yf = np.fft.fft(data)
            xf = np.fft.fftfreq(n, 1 / sample_rate)
            
            # Extract magnitudes (An)
            magnitudes = np.abs(yf[:n // 2])
            frequencies = xf[:n // 2]
            
            # Round frequencies and magnitudes for easier matching
            rounded_frequencies = np.round(frequencies, -1)
            rounded_magnitudes = np.round(magnitudes, -1)
            
            # Store the results
            frequency_data[file_name] = {
                'frequencies': rounded_frequencies,
                'magnitudes': rounded_magnitudes
            }
    
    return frequency_data

def calculate_common_spectrum(frequency_data):
    global common_spectrum
    all_frequencies = []
    for data in frequency_data.values():
        all_frequencies.append(set(data['frequencies']))

    # Find intersection of all frequency sets to identify common frequencies
    common_frequencies = set.intersection(*all_frequencies)

    B = {}
    for freq in common_frequencies:
        magnitudes = []
        for data in frequency_data.values():
            idx = np.where(data['frequencies'] == freq)[0]
            if len(idx) > 0:
                magnitudes.append(data['magnitudes'][idx[0]])
        if magnitudes:
            B[freq] = np.round(np.mean(magnitudes), -1)  # Round the mean magnitude for consistency
    
    common_spectrum = B
    return common_spectrum

def calculate_unique_spectra(frequency_data, B):
    global unique_spectra
    unique_spectra = {}
    
    for file_name, data in frequency_data.items():
        Cn = {}
        for i, freq in enumerate(data['frequencies']):
            if freq in B:
                Cn[freq] = np.round(data['magnitudes'][i] - B[freq], -1)  # Ensure consistent rounding
            else:
                Cn[freq] = data['magnitudes'][i]
        
        unique_spectra[file_name] = Cn
    
    return unique_spectra

def compare_files():
    global comparison_results, frequency_data
    comparison_files = filedialog.askopenfilenames(filetypes=[("WAV files", "*.wav")])
    if comparison_files:
        comparison_results = {}  # Clear previous results
        all_results = []
        for dn_file in comparison_files:
            file_name = os.path.basename(dn_file)
            # Perform Fourier Transform on the new file
            sample_rate, data = wavfile.read(dn_file)
            n = len(data)
            yf = np.fft.fft(data)
            xf = np.fft.fftfreq(n, 1 / sample_rate)
            magnitudes_dn = np.abs(yf[:n // 2])
            frequencies_dn = xf[:n // 2]
            
            # Round frequencies and magnitudes for easier matching
            rounded_frequencies_dn = np.round(frequencies_dn, -1)
            rounded_magnitudes_dn = np.round(magnitudes_dn, -1)

            # Store the results for the comparison file
            frequency_data[file_name] = {
                'frequencies': rounded_frequencies_dn,
                'magnitudes': rounded_magnitudes_dn
            }

            # Subtract common spectrum B from Dn to get Dn-B
            dn_minus_b = {}
            for i, freq in enumerate(rounded_frequencies_dn):
                if freq in common_spectrum:
                    dn_minus_b[freq] = np.round(rounded_magnitudes_dn[i] - common_spectrum[freq], -1)

            # Compare Dn-B with the unique spectra Cn
            similarities = {}
            for sample_file_name, Cn in unique_spectra.items():
                matching_freqs = [freq for freq in dn_minus_b if freq in Cn]
                if matching_freqs:
                    dn_magnitudes = [dn_minus_b[freq] for freq in matching_freqs]
                    sample_magnitudes = [Cn[freq] for freq in matching_freqs]

                    if dn_magnitudes and sample_magnitudes:
                        similarity = np.dot(dn_magnitudes, sample_magnitudes) / (np.linalg.norm(dn_magnitudes) * np.linalg.norm(sample_magnitudes))
                        similarity = max(similarity, 0)  # Ensure non-negative similarity
                        similarities[sample_file_name] = similarity * 100  # Convert to percentage
                        update_log(f"Similarity of {file_name} with {sample_file_name}: {similarity * 100:.2f}%")

            # Store similarities for later plotting
            comparison_results[file_name] = similarities

            # Sort similarities and prepare the result text
            if similarities:
                sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
                result_text = f"Results for {file_name}:\n" + "\n".join([f"{file}: {similarity:.2f}%" for file, similarity in sorted_similarities])
                all_results.append(result_text)
            else:
                all_results.append(f"No matching frequencies found for comparison in {file_name}.")

        # Show results in a new window
        show_results_window("\n\n".join(all_results))

def show_results_window(results_text):
    result_window = Toplevel(root)
    result_window.title("Comparison Results")
    
    text_area = Text(result_window, wrap=tk.WORD, width=80, height=20)
    text_area.pack(expand=True, fill=tk.BOTH)
    
    text_area.insert(tk.END, results_text)
    text_area.configure(state='disabled')

def plot_original_spectra():
    if not frequency_data:
        messagebox.showwarning("Plotting Error", "No files loaded to plot.")
        return

    fig = go.Figure()

    # Plot original spectra
    for file_name, data in frequency_data.items():
        fig.add_trace(go.Scatter(
            x=data['frequencies'],
            y=data['magnitudes'],
            mode='lines',
            name=f"Original: {file_name}"
        ))

    fig.update_layout(
        title="Original Spectral Components of Audio Files",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        showlegend=True
    )

    fig.show()

def plot_selected_results():
    if not comparison_results:
        messagebox.showwarning("Plotting Error", "No comparison results available to plot.")
        return
    
    # Open a selection dialog for which files to plot
    select_window = Toplevel(root)
    select_window.title("Select Files to Plot")

    listbox = Listbox(select_window, selectmode=MULTIPLE)
    for dn_file in comparison_results.keys():
        listbox.insert(tk.END, dn_file)
    listbox.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    def plot_selected():
        selected_files = listbox.curselection()
        fig = go.Figure()

        for i in selected_files:
            dn_file_name = listbox.get(i)
            dn_file_name = listbox.get(i)  # Get the name of the selected file

            # Plot original spectra of the comparison file
            if dn_file_name in frequency_data:
                data_dn = frequency_data[dn_file_name]
                fig.add_trace(go.Scatter(
                    x=data_dn['frequencies'],
                    y=data_dn['magnitudes'],
                    mode='lines',
                    name=f"Original: {dn_file_name} Magnitudes"
                ))

            # Plot unique spectra for each sample file compared to the current file
            if dn_file_name in comparison_results:
                for sample_file_name in comparison_results[dn_file_name]:
                    if sample_file_name in unique_spectra:
                        fig.add_trace(go.Scatter(
                            x=list(unique_spectra[sample_file_name].keys()),
                            y=list(unique_spectra[sample_file_name].values()),
                            mode='lines',
                            name=f"Unique Spectrum Cn: {sample_file_name}",
                            line=dict(dash='dot')
                        ))

            # Optionally, you can uncomment the next lines if you want to include the common spectrum:
            # fig.add_trace(go.Scatter(
            #     x=list(common_spectrum.keys()),
            #     y=list(common_spectrum.values()),
            #     mode='lines',
            #     name="Common Spectrum (B)",
            #     line=dict(dash='dash', color='black', width=3)
            # ))

        fig.update_layout(
            title="Comparison Results: Frequency vs Magnitude",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            showlegend=True
        )

        fig.show()

    plot_button = tk.Button(select_window, text="Plot Selected", command=plot_selected)
    plot_button.pack(pady=10)

def update_log(message):
    log_text.insert(tk.END, message + "\n")
    log_text.see(tk.END)
    logging.info(message)

# Tkinter-based GUI

def clear_log():
    log_text.delete(1.0, tk.END)
    logging.info("Log cleared.")

# Tkinter-based GUI

def load_sample_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("WAV files", "*.wav")])
    if file_paths:
        frequency_data = parse_wav_files(file_paths)
        common_spectrum = calculate_common_spectrum(frequency_data)
        unique_spectra = calculate_unique_spectra(frequency_data, common_spectrum)
        update_log("Sample files loaded and analyzed.")

# Main GUI Application
root = tk.Tk()
root.title("Audio Analysis Application")

frame = tk.Frame(root)
frame.pack(pady=10)

load_samples_button = tk.Button(frame, text="Load Sample Files", command=load_sample_files)
load_samples_button.grid(row=0, column=0, padx=5, pady=5)

compare_button = tk.Button(frame, text="Compare Files", command=compare_files)
compare_button.grid(row=0, column=1, padx=5, pady=5)

plot_button = tk.Button(frame, text="Plot Original Spectra", command=plot_original_spectra)
plot_button.grid(row=0, column=2, padx=5, pady=5)

plot_results_button = tk.Button(frame, text="Plot Selected Results", command=plot_selected_results)
plot_results_button.grid(row=0, column=3, padx=5, pady=5)

clear_log_button = tk.Button(frame, text="Clear Log", command=clear_log)
clear_log_button.grid(row=0, column=4, padx=5, pady=5)

log_text = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
log_text.pack(pady=10)

root.mainloop()