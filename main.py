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
    comparison_files = filedialog.askopenfilenames(filetypes=[("WAV文件", "*.wav")])
    if comparison_files:
        comparison_results = {}  # Clear previous results
        all_results = []
        for dn_file in comparison_files:
            file_name = os.path.basename(dn_file)
            # Perform Fourier Transform on the comparison file (Dn)
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

            # Calculate the unique spectrum of the comparison file by subtracting the common spectrum B from Dn
            dn_unique_spectrum = {}
            for i, freq in enumerate(rounded_frequencies_dn):
                if freq in common_spectrum:
                    # Subtract the common spectrum from the original spectrum of Dn
                    dn_unique_spectrum[freq] = np.round(rounded_magnitudes_dn[i] - common_spectrum[freq], -1)
                else:
                    dn_unique_spectrum[freq] = rounded_magnitudes_dn[i]

            # Compare the unique spectrum of the comparison file (Dn - B) with the unique spectra of the sample files (Cn)
            similarities = {}
            for sample_file_name, Cn in unique_spectra.items():
                matching_freqs = [freq for freq in dn_unique_spectrum if freq in Cn]
                if matching_freqs:
                    dn_magnitudes = [dn_unique_spectrum[freq] for freq in matching_freqs]
                    sample_magnitudes = [Cn[freq] for freq in matching_freqs]

                    if dn_magnitudes and sample_magnitudes:
                        # Compute the similarity using dot product
                        similarity = np.dot(dn_magnitudes, sample_magnitudes) / (np.linalg.norm(dn_magnitudes) * np.linalg.norm(sample_magnitudes))
                        similarity = max(similarity, 0)  # Ensure non-negative similarity
                        similarities[sample_file_name] = similarity * 100  # Convert to percentage
                        update_log(f"{file_name} 与 {sample_file_name} 的相似度: {similarity * 100:.2f}%")

            # Store similarities for later plotting
            comparison_results[file_name] = similarities

            # Sort similarities and prepare the result text
            if similarities:
                sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
                result_text = f"{file_name} 的结果:\n" + "\n".join([f"{file}: {similarity:.2f}%" for file, similarity in sorted_similarities])
                all_results.append(result_text)
            else:
                all_results.append(f"{file_name} 中未找到匹配的频率用于比较。")

        # Show results in a new window
        show_results_window("\n\n".join(all_results))

def show_results_window(results_text):
    result_window = Toplevel(root)
    result_window.title("比较结果")
    
    text_area = Text(result_window, wrap=tk.WORD, width=80, height=20)
    text_area.pack(expand=True, fill=tk.BOTH)
    
    text_area.insert(tk.END, results_text)
    text_area.configure(state='disabled')

def plot_original_spectra():
    if not frequency_data:
        messagebox.showwarning("绘图错误", "未加载任何文件用于绘图。")
        return

    fig = go.Figure()

    # Plot original spectra
    for file_name, data in frequency_data.items():
        fig.add_trace(go.Scatter(
            x=data['frequencies'],
            y=data['magnitudes'],
            mode='lines',
            name=f"原始频谱: {file_name}"
        ))

    fig.update_layout(
        title="音频文件的原始频谱成分",
        xaxis_title="频率 (Hz)",
        yaxis_title="幅度",
        showlegend=True
    )

    fig.show()

def plot_selected_results():
    if not comparison_results:
        messagebox.showwarning("绘图错误", "没有可用于绘图的比较结果。")
        return
    
    fig = go.Figure()

    # Keep track of which sample files' unique spectra have already been plotted
    plotted_sample_files = set()

    # After plotting all comparison files, plot the unique spectra of the sample files
    for dn_file_name in comparison_results.keys():
        for sample_file_name in comparison_results[dn_file_name]:
            if sample_file_name not in plotted_sample_files and sample_file_name in unique_spectra:
                sample_spectrum_frequencies = list(unique_spectra[sample_file_name].keys())
                sample_spectrum_magnitudes = list(unique_spectra[sample_file_name].values())

                fig.add_trace(go.Scatter(
                    x=sample_spectrum_frequencies,
                    y=sample_spectrum_magnitudes,
                    mode='lines',
                    name=f"样本特征频谱 Cn: {sample_file_name}",
                    line=dict(dash='solid')
                ))

                # Mark this sample file as plotted
                plotted_sample_files.add(sample_file_name)

    # Plot the unique spectrum of the comparison files (Dn - B) first
    for dn_file_name in comparison_results.keys():
        if dn_file_name in frequency_data:
            compare_file_unique_spectrum = frequency_data[dn_file_name]['frequencies']
            compare_file_unique_magnitudes = frequency_data[dn_file_name]['magnitudes']

            fig.add_trace(go.Scatter(
                x=list(compare_file_unique_spectrum),
                y=list(compare_file_unique_magnitudes),
                mode='lines',
                name=f"对比文件特殊频谱: {dn_file_name}",
                line=dict(dash='dot')
            ))

    # Update the layout of the plot
    fig.update_layout(
        title="比较结果: 特征频谱 (频率 vs 幅度)",
        xaxis_title="频率 (Hz)",
        yaxis_title="幅度",
        showlegend=True
    )

    # Show the plot
    fig.show()


def update_log(message):
    log_text.insert(tk.END, message + "\n")
    log_text.see(tk.END)
    logging.info(message)

def clear_log():
    log_text.delete(1.0, tk.END)
    logging.info("日志已清除。")

# Tkinter-based GUI

def load_sample_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("WAV文件", "*.wav")])
    if file_paths:
        frequency_data = parse_wav_files(file_paths)
        common_spectrum = calculate_common_spectrum(frequency_data)
        unique_spectra = calculate_unique_spectra(frequency_data, common_spectrum)
        update_log("样本文件已加载并分析。")

# Main GUI Application
root = tk.Tk()
root.title("音频分析应用程序")

frame = tk.Frame(root)
frame.pack(pady=10)

load_samples_button = tk.Button(frame, text="加载样本文件", command=load_sample_files)
load_samples_button.grid(row=0, column=0, padx=5, pady=5)

compare_button = tk.Button(frame, text="选择比较文件", command=compare_files)
compare_button.grid(row=0, column=1, padx=5, pady=5)

plot_button = tk.Button(frame, text="绘制原始频谱", command=plot_original_spectra)
plot_button.grid(row=0, column=2, padx=5, pady=5)

plot_results_button = tk.Button(frame, text="绘制特征频谱", command=plot_selected_results)
plot_results_button.grid(row=0, column=3, padx=5, pady=5)

clear_log_button = tk.Button(frame, text="清除日志", command=clear_log)
clear_log_button.grid(row=0, column=4, padx=5, pady=5)

log_text = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
log_text.pack(pady=10)

root.mainloop()
