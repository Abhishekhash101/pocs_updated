import streamlit as st
import numpy as np
from scipy.signal import hilbert, butter, lfilter, resample
from PIL import Image
import soundfile as sf
import matplotlib.pyplot as plt
import io
import cv2  # OpenCV
import imageio.v3 as iio # ImageIO
from streamlit_drawable_canvas import st_canvas
from skimage.metrics import structural_similarity as ssim
import numexpr as ne

st.set_page_config(page_title="Intelligent Modulation Optimizer", layout="wide")

# ---------------- Helper Functions ----------------

def time_axis(N, fs):
    return np.arange(N) / fs

def awgn(signal, snr_db, seed=None):
    rng = np.random.default_rng(seed)
    sig_pow = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_pow = sig_pow / snr_linear
    noise = rng.normal(0, np.sqrt(noise_pow), size=signal.shape)
    return signal + noise

def envelope_detect(signal):
    analytic = hilbert(signal)
    env = np.abs(analytic)
    b, a = butter(4, 0.05)
    return lfilter(b, a, env)

def fm_demodulate(signal, fs):
    analytic = hilbert(signal)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    inst_freq -= np.mean(inst_freq)
    return np.concatenate(([inst_freq[0]], inst_freq))

def recover_image(signal, shape):
    """Recovers an image from a 1D signal array."""
    h, w = shape
    num_pixels = h * w
    
    if len(signal) < num_pixels:
        padded_signal = np.zeros(num_pixels)
        padded_signal[:len(signal)] = signal
    else:
        padded_signal = signal[:num_pixels]
        
    arr = np.clip(padded_signal, -1, 1)
    arr = ((arr + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(arr.reshape((h, w)))


def coherent_demodulate(signal, fc, fs, phase_offset_deg=0):
    t = time_axis(len(signal), fs)
    phase_offset_rad = np.deg2rad(phase_offset_deg)
    carrier = np.cos(2 * np.pi * fc * t + phase_offset_rad)
    demod = signal * carrier
    b, a = butter(4, 0.05)
    return lfilter(b, a, demod)

def array_to_audio_bytes(audio_array, sample_rate):
    audio_array = audio_array.astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format='WAV')
    return buf.getvalue()

def plot_spectrum(ax, signal, fs, title, carrier_freq=None):
    N = len(signal)
    if N == 0: return
    yf = np.fft.fft(signal * np.hanning(N))
    xf = np.fft.fftfreq(N, 1 / fs)
    ax.plot(np.fft.fftshift(xf), 20 * np.log10(np.abs(np.fft.fftshift(yf)) + 1e-9))
    ax.set_title(title); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude (dB)"); ax.grid(True)
    zoom_range = fs / 4
    if carrier_freq: ax.set_xlim(carrier_freq - zoom_range, carrier_freq + zoom_range)
    else: ax.set_xlim(-zoom_range, zoom_range)

def calculate_signal_metrics(original_signal, recovered_signal, original_2d_shape=None):
    """
    Calculates metrics. Assumes both signals are already the correct type 
    (e.g., both are baseband audio, or both are LPF'd video).
    """
    min_len = min(len(original_signal), len(recovered_signal))
    original = original_signal[:min_len].astype(np.float64)
    recovered = recovered_signal[:min_len].astype(np.float64)
    
    if np.std(original) < 1e-6 or np.std(recovered) < 1e-6: correlation = 0.0
    else: correlation = np.corrcoef(original, recovered)[0, 1]
    
    accuracy_percentage = correlation * 100
    mse = np.mean((original - recovered)**2)
    psnr = 20 * np.log10(2.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    ssim_score = None
    
    if original_2d_shape is not None and original.size > 1:
        try:
            h, w = original_2d_shape
            num_pixels = h * w
            
            if len(original) < num_pixels:
                orig_padded = np.zeros(num_pixels); orig_padded[:len(original)] = original
            else:
                orig_padded = original[:num_pixels]
            
            if len(recovered) < num_pixels:
                rec_padded = np.zeros(num_pixels); rec_padded[:len(recovered)] = recovered
            else:
                rec_padded = recovered[:num_pixels]

            original_2d = ((orig_padded.reshape(h, w) + 1) * 127.5).astype(np.uint8)
            recovered_2d = ((rec_padded.reshape(h, w) + 1) * 127.5).astype(np.uint8)
            
            ssim_score = ssim(original_2d, recovered_2d, data_range=255)
        except Exception as e:
            ssim_score = None
            
    return accuracy_percentage, mse, psnr, ssim_score

def estimate_bandwidth(signal, fs, threshold=0.995):
    N = len(signal)
    if N == 0: return 0
    yf = np.fft.fft(signal * np.hanning(N))
    psd = np.abs(yf[:N // 2])**2
    freqs = np.fft.fftfreq(N, 1 / fs)[:N // 2]
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    if total_power == 0: return 0
    try:
        idx = np.where(cumulative_power >= total_power * threshold)[0][0]
        return freqs[idx]
    except IndexError: return fs / 2

@st.cache_data
def load_audio(file, target_fs):
    data, fs_orig = sf.read(file)
    if data.ndim > 1: data = data.mean(axis=1)
    if fs_orig != target_fs: data = resample(data, int(len(data) * target_fs / fs_orig))
    if np.max(np.abs(data)) > 0: data /= np.max(np.abs(data))
    return data

@st.cache_data
def load_image(file):
    img = Image.open(file).convert('L')
    arr = np.array(img).astype(float)
    return (arr - 127.5) / 127.5, img.size # size is (w, h)

@st.cache_data
def load_and_process_video(_video_file, target_fs, resolution_wh, audio_subcarrier_fc, kf_audio=5.0):
    """
    Loads video, extracts audio, calculates per-frame detail, 
    and multiplexes them into a composite baseband signal.
    """
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(_video_file.getbuffer())
    
    # 2. Prepare Audio Stream
    try:
        with iio.imopen(temp_video_path, "r", plugin="ffmpeg") as vid_reader:
            audio_data = vid_reader.audio()
            audio_fs = vid_reader.audio_metadata()["sample_rate"]
        
        if audio_fs != target_fs:
            audio_data = resample(audio_data, int(len(audio_data) * target_fs / audio_fs))
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if np.max(np.abs(audio_data)) > 0:
            audio_data /= np.max(np.abs(audio_data))
            
    except Exception as e:
        st.warning(f"Could not read audio stream: {e}. Generating silence.")
        try:
            cap_for_duration = cv2.VideoCapture(temp_video_path)
            frame_count = int(cap_for_duration.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_for_duration = cap_for_duration.get(cv2.CAP_PROP_FPS)
            duration = (frame_count / fps_for_duration) if fps_for_duration > 0 else 1
            audio_data = np.zeros(int(duration * target_fs))
            cap_for_duration.release()
        except Exception:
             audio_data = np.zeros(int(1 * target_fs))

    m_audio_fm = fm_modulate(audio_data, fc=audio_subcarrier_fc, fs=target_fs, kf=kf_audio)
    
    # 3. Prepare Video Stream & Detail Analysis
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        cap.release()
        return None, None, None
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    num_samples_per_frame = int(target_fs / fps)
    
    if num_samples_per_frame < (resolution_wh[0] * resolution_wh[1]):
        num_samples_per_frame = resolution_wh[0] * resolution_wh[1]

    m_video_list = []
    detail_scores = []
    total_audio_samples = len(m_audio_fm)
    total_frames_needed = int(np.ceil(total_audio_samples / num_samples_per_frame))
    
    st.session_state.original_fps = fps
    
    for _ in range(total_frames_needed):
        ret, frame = cap.read()
        if not ret:
            m_video_list.append(np.zeros(num_samples_per_frame))
            detail_scores.append(0)
            continue
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, resolution_wh, interpolation=cv2.INTER_AREA)
        
        edges = cv2.Canny(resized_frame, 100, 200) 
        detail_score = np.mean(edges) / 255.0
        detail_scores.append(detail_score)
            
        flat_frame = (resized_frame.astype(float) - 127.5) / 127.5
        final_frame_data = np.zeros(num_samples_per_frame)
        len_to_copy = min(len(flat_frame.flatten()), num_samples_per_frame)
        final_frame_data[:len_to_copy] = flat_frame.flatten()[:len_to_copy]
        
        m_video_list.append(final_frame_data)
            
    m_video = np.concatenate(m_video_list)[:total_audio_samples]
    cap.release()
    
    m_composite = 0.7 * m_video + 0.3 * m_audio_fm
    
    return m_composite, (resolution_wh[1], resolution_wh[0]), np.array(detail_scores)

# ----------------- OPTIMIZER FUNCTIONS -----------------

### AM OPTIMIZER (Universal) ###
@st.cache_data
def find_best_am_depth(_msg_for_metrics):
    """Calculates max AM depth without overmodulation."""
    m_min = np.min(_msg_for_metrics)
    optimal_depth = 1.0
    if m_min < -1e-6: # Check if negative
        critical_depth = -1.0 / m_min
        optimal_depth = critical_depth * 0.95 # Apply 5% safety margin
    return np.clip(optimal_depth, 0.1, 2.0)

### FM OPTIMIZER (NOW UNIVERSAL) ###
@st.cache_data
def find_best_kf(_msg_for_metrics, _carrier, _fs, _target_snr, _input_type, _original_shape):
    """Grid search to find the best kf for standard FM."""
    st.info(f"Running FM optimization for {_target_snr} dB SNR...")
    
    kf_range = np.linspace(1.0, 15.0, 10) # Test 10 kf values
    best_metric = -1.0 # Changed from best_ssim
    best_kf = 0
    metric_name = "Metric"
    
    progress_bar = st.progress(0, text="Optimizing FM...")
    
    for i, kf_test in enumerate(kf_range):
        tx = fm_modulate(_msg_for_metrics, _carrier, _fs, kf=kf_test)
        tx_noisy = awgn(tx, _target_snr)
        rec = fm_demodulate(tx_noisy, _fs)
        if np.max(np.abs(rec)) > 0: rec /= np.max(np.abs(rec))

        rec_for_metrics = rec
        if _input_type == "Video": # Need to LPF the recovered video
            video_cutoff = st.session_state.get('video_cutoff_hz', st.session_state.audio_subcarrier * 0.9)
            b, a = butter(6, video_cutoff / (_fs / 2), btype='low')
            rec_for_metrics = lfilter(b, a, rec)
            if np.max(np.abs(rec_for_metrics)) > 0:
                rec_for_metrics /= np.max(np.abs(rec_for_metrics))

        # --- UPDATED: Universal Metric Logic ---
        acc, _, psnr, ssim_val = calculate_signal_metrics(_msg_for_metrics, rec_for_metrics, _original_shape)
        
        current_metric = 0.0
        if _input_type in ["Image", "Live Camera", "Video"]:
            current_metric = ssim_val if ssim_val is not None else 0.0
            metric_name = "SSIM"
        else: # For Audio, Draw, Equation
            current_metric = acc
            metric_name = "Accuracy"
            
        if current_metric > best_metric:
            best_metric = current_metric
            best_kf = kf_test
        # --- END UPDATED ---
        
        progress_bar.progress((i + 1) / len(kf_range), text=f"Best kf: {best_kf:.2f} -> {metric_name}: {best_metric:.4f}")

    progress_bar.empty()
    st.success(f"**FM Optimization Complete!**")
    return best_kf

### VSB OPTIMIZER (NOW UNIVERSAL) ###
@st.cache_data
def find_best_vsb_beta(_msg_for_metrics, _carrier, _fs, _target_snr, _input_type, _original_shape):
    """Grid search to find the best vsb_beta for VSB."""
    st.info(f"Running VSB optimization for {_target_snr} dB SNR...")
    
    beta_range = np.linspace(200, 2000, 10) # Test 10 beta values
    best_metric = -1.0 # Changed from best_ssim
    best_beta = 0
    metric_name = "Metric"
    
    progress_bar = st.progress(0, text="Optimizing VSB...")
    
    for i, beta_test in enumerate(beta_range):
        tx = vsb_modulate(_msg_for_metrics, _carrier, _fs, beta=beta_test)
        tx_noisy = awgn(tx, _target_snr)
        rec = coherent_demodulate(tx_noisy, _carrier, _fs, phase_offset_deg=0) # Assume 0 phase
        if np.max(np.abs(rec)) > 0: rec /= np.max(np.abs(rec))

        rec_for_metrics = rec
        if _input_type == "Video": # Need to LPF the recovered video
            video_cutoff = st.session_state.get('video_cutoff_hz', st.session_state.audio_subcarrier * 0.9)
            b, a = butter(6, video_cutoff / (_fs / 2), btype='low')
            rec_for_metrics = lfilter(b, a, rec)
            if np.max(np.abs(rec_for_metrics)) > 0:
                rec_for_metrics /= np.max(np.abs(rec_for_metrics))

        # --- UPDATED: Universal Metric Logic ---
        acc, _, psnr, ssim_val = calculate_signal_metrics(_msg_for_metrics, rec_for_metrics, _original_shape)

        current_metric = 0.0
        if _input_type in ["Image", "Live Camera", "Video"]:
            current_metric = ssim_val if ssim_val is not None else 0.0
            metric_name = "SSIM"
        else: # For Audio, Draw, Equation
            current_metric = acc
            metric_name = "Accuracy"

        if current_metric > best_metric:
            best_metric = current_metric
            best_beta = beta_test
        # --- END UPDATED ---
        
        progress_bar.progress((i + 1) / len(beta_range), text=f"Best Beta: {best_beta:.0f} Hz -> {metric_name}: {best_metric:.4f}")

    progress_bar.empty()
    st.success(f"**VSB Optimization Complete!**")
    return best_beta

### COSTAS LOOP (Universal) ###
@st.cache_data
def costas_loop_find_phase(_tx_noisy, _carrier, _fs):
    """Implements a Costas Loop to find the carrier phase."""
    st.info("Attempting carrier phase lock...")
    
    N = min(len(_tx_noisy), 50000)
    signal = _tx_noisy[:N]
    t = time_axis(N, _fs)
    
    b_lpf, a_lpf = butter(4, 0.05)
    
    alpha = 0.01; beta = 0.001
    loop_phase = 0.0; phase_error_int = 0.0
    phase_estimates = []
    
    # Pre-filter signals to simulate iterative LPF
    i_mixed_filtered = lfilter(b_lpf, a_lpf, signal * np.cos(2 * np.pi * _carrier * t))
    q_mixed_filtered = lfilter(b_lpf, a_lpf, signal * -np.sin(2 * np.pi * _carrier * t))

    # This is a block-based phase detector, much faster for simulation
    error_signal = i_mixed_filtered * q_mixed_filtered
    
    for i in range(N):
        phase_error_int += beta * error_signal[i]
        loop_phase = (loop_phase + alpha * error_signal[i] + phase_error_int) % (2 * np.pi)
        if i > N // 2: # Let the loop settle, then average
            phase_estimates.append(loop_phase)

    if not phase_estimates:
        return 0.0 # Lock failed

    final_phase_rad = np.mean(phase_estimates)
    final_phase_deg = np.rad2deg(final_phase_rad)
    
    st.success(f"**Phase Lock Acquired!**")
    # Test 0 and 180 degree offsets
    test_0 = coherent_demodulate(signal, _carrier, _fs, final_phase_deg)
    test_180 = coherent_demodulate(signal, _carrier, _fs, final_phase_deg + 180)
    
    # Pick the one with more power (correct lock)
    if np.mean(test_0**2) > np.mean(test_180**2):
        return final_phase_deg
    else:
        return final_phase_deg + 180.0


### ADAPTIVE FM OPTIMIZER (Video-Only) ###
@st.cache_data
def find_best_adaptive_parameters(_msg_flat_composite, _msg_for_metrics, _detail_scores, 
                                  _carrier, _fs, _original_fps, _target_snr, _original_shape):
    """Grid search for Adaptive FM."""
    st.info(f"Running Adaptive FM optimization for {_target_snr} dB SNR...")
    
    kf_min_range = np.linspace(1.0, 4.0, 4)
    kf_max_range = np.linspace(5.0, 15.0, 5)
    best_ssim = -1.0
    best_params = (0, 0)
    
    total_sims = len(kf_min_range) * len(kf_max_range)
    progress_bar = st.progress(0, text="Optimizing Adaptive FM...")
    sim_count = 0
    
    video_cutoff = st.session_state.get('video_cutoff_hz', st.session_state.audio_subcarrier * 0.9)
    b, a = butter(6, video_cutoff / (_fs / 2), btype='low')

    for kf_min_test in kf_min_range:
        for kf_max_test in kf_max_range:
            tx = adaptive_fm_modulate(_msg_flat_composite, _detail_scores, _carrier, _fs, 
                                      kf_min_test, kf_max_test, _original_fps)
            tx_noisy = awgn(tx, _target_snr)
            rec = fm_demodulate(tx_noisy, _fs)
            if np.max(np.abs(rec)) > 0: rec /= np.max(np.abs(rec))

            rec_video_signal = lfilter(b, a, rec)
            if np.max(np.abs(rec_video_signal)) > 0:
                rec_video_signal /= np.max(np.abs(rec_video_signal))
            
            _, _, _, ssim_val = calculate_signal_metrics(_msg_for_metrics, rec_video_signal, _original_shape)

            if ssim_val is not None and ssim_val > best_ssim:
                best_ssim = ssim_val
                best_params = (kf_min_test, kf_max_test)
            
            sim_count += 1
            progress_bar.progress(sim_count / total_sims, text=f"Best: {best_params} -> SSIM: {best_ssim:.4f}")

    progress_bar.empty()
    st.success(f"**Adaptive FM Optimization Complete!**")
    return best_params, best_ssim


# ---------------- Modulation Functions ----------------

def am_modulate(msg, fc, fs, depth=0.7):
    t = time_axis(len(msg), fs); return (1 + depth * msg) * np.cos(2 * np.pi * fc * t)

def dsb_sc_modulate(msg, fc, fs):
    t = time_axis(len(msg), fs); return msg * np.cos(2 * np.pi * fc * t)

def ssb_modulate(msg, fc, fs, side='usb'):
    t = time_axis(len(msg), fs); m_hat = np.imag(hilbert(msg)); wc = 2 * np.pi * fc
    if side == 'usb': return msg * np.cos(wc * t) - m_hat * np.sin(wc * t)
    else: return msg * np.cos(wc * t) + m_hat * np.sin(wc * t)

def fm_modulate(msg, fc, fs, kf=5.0):
    t = time_axis(len(msg), fs); dt = 1 / fs; integral = np.cumsum(msg) * dt
    return np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integral)

def adaptive_fm_modulate(msg, detail_scores, fc, fs, kf_min=2.0, kf_max=10.0, fps=25):
    """Modulates using FM with a dynamically changing kf."""
    t = time_axis(len(msg), fs)
    dt = 1 / fs
    num_samples_per_frame = int(fs / fps)
    mapped_kf_scores = kf_min + (detail_scores * (kf_max - kf_min))
    kf_array = np.repeat(mapped_kf_scores, num_samples_per_frame)
    kf_array = kf_array[:len(msg)]
    phase_integral = np.cumsum(kf_array * msg) * dt
    return np.cos(2 * np.pi * fc * t + 2 * np.pi * phase_integral)

def vsb_modulate(msg, fc, fs, beta=500):
    N = len(msg)
    t = time_axis(N, fs)
    dsb_signal = msg * np.cos(2 * np.pi * fc * t)
    dsb_fft = np.fft.fft(dsb_signal)
    xf = np.fft.fftfreq(N, 1 / fs)
    hv = np.zeros(N)
    hv[np.abs(xf) > fc + beta] = 1.0
    hv[np.abs(xf) < fc - beta] = 0.0
    pos_mask = (xf >= fc - beta) & (xf <= fc + beta)
    hv[pos_mask] = 0.5 + 0.5 * (xf[pos_mask] - fc) / beta
    neg_mask = (xf >= -fc - beta) & (xf <= -fc + beta)
    hv[neg_mask] = 0.5 - 0.5 * (xf[neg_mask] + fc) / beta
    vsb_fft = dsb_fft * hv
    vsb_signal = np.real(np.fft.ifft(vsb_fft))
    return vsb_signal

# ---------------- Streamlit UI ----------------

st.title("Intelligent Modulation Simulator & Optimizer")

# --- UI Sidebar for ALL controls ---
with st.sidebar:
    st.header("Controls")
    
    input_type = st.radio("1. Select Input Type", ["Audio", "Image", "Draw Signal", "Live Camera", "Equation", "Video"])
    st.session_state.input_type_used = input_type
    
    file_types = { "Audio": ["wav", "mp3"], "Image": ["jpg", "png"], "Video": ["mp4", "mov", "avi"], }
    upload_label = f"Upload {input_type} File"
    if input_type in file_types:
        uploaded_file = st.file_uploader(upload_label, type=file_types[input_type])
    else:
        uploaded_file = st.file_uploader("Upload File (if needed)")

    process_drawing_button = False
    if input_type == "Draw Signal":
        process_drawing_button = st.button("Process Drawing", key="process_drawing_sidebar")

    generate_sine_button = False
    generate_freeform_button = False
    if input_type == "Equation":
        eq_type = st.selectbox("Select Generator Type", ["Sine Wave Parameters", "Freeform Equation"])
        if eq_type == "Sine Wave Parameters":
            amplitude = st.slider("Amplitude (A)", 0.0, 1.0, 1.0, 0.05)
            frequency = st.number_input("Frequency (f) in Hz", 1, 10000, 100)
            phase = st.slider("Phase (ϕ) in Degrees", -180, 180, 0)
            duration_sine = st.number_input("Signal Duration (s)", 0.1, 10.0, 1.0, 0.1, key="sine_duration")
            generate_sine_button = st.button("Generate Sine Wave", key="generate_sine_sidebar")
        elif eq_type == "Freeform Equation":
            if 'equation_str' not in st.session_state: st.session_state.equation_str = "sin(2 * pi * 100 * t)"
            equation_str = st.text_input("Enter Equation `m(t)`:", st.session_state.equation_str)
            duration_ff = st.number_input("Signal Duration (s)", 0.1, 10.0, 1.0, 0.1, key="freeform_duration")
            generate_freeform_button = st.button("Generate from Equation", key="generate_freeform_sidebar")

    if input_type == "Video":
        st.subheader("Video Broadcast Settings")
        st.info("This will create a 'miniature TV channel'...")
        video_res_w = st.select_slider("Processing Width (pixels)", options=[32, 40, 48, 56, 64, 80, 96], value=64)
        video_res_h = st.select_slider("Processing Height (pixels)", options=[32, 40, 48, 56, 64, 80, 96], value=48)
        audio_sub_fc = st.number_input("Audio Subcarrier (Hz)", min_value=1000, value=15000, step=1000)
        st.session_state.video_res = (video_res_w, video_res_h)
        st.session_state.audio_subcarrier = audio_sub_fc

    st.header("Simulation Parameters")
    modulations = st.multiselect("2. Select Modulations", ['AM', 'DSB-SC', 'SSB', 'VSB', 'FM', 'Adaptive FM'])
    carrier = st.number_input("Carrier Frequency (Hz)", value=50000, min_value=1000, step=1000)
    fs = st.number_input("Sampling Frequency (Hz)", value=200000, min_value=10000, step=10000)
    st.session_state.fs_used = fs

    st.header("Channel Effects")
    snr_db = st.slider("Signal-to-Noise Ratio (SNR in dB)", -10, 40, 20, key="main_snr_slider")
    
    st.header("Modulation & Demodulation Settings")
    
    # AM Settings (Stateful)
    if 'AM' in modulations:
        st.subheader("AM Settings")
        am_depth_default = st.session_state.get('am_depth_optimized', 0.7)
        am_depth = st.slider("AM Modulation Depth", 0.1, 2.0, am_depth_default, 0.05, 
                           key="am_depth_slider", 
                           help="Values > 1.0 will cause overmodulation. Use the Auto-Optimize button in the AM tab.")
        st.session_state.am_depth_optimized = am_depth # Store current value

    # FM Settings (Stateful)
    kf_fm = 5.0
    if 'FM' in modulations:
        st.subheader("Standard FM Settings")
        kf_fm_default = st.session_state.get('kf_fm_optimized', 5.0)
        kf_fm = st.slider("FM Freq. Deviation (kf)", 1.0, 20.0, kf_fm_default, 0.5,
                          key="kf_fm_slider",
                          help="Use the Auto-Optimize button in the FM tab.")
        st.session_state.kf_fm_optimized = kf_fm

    # VSB Settings (Stateful)
    vsb_beta = 500
    if 'VSB' in modulations:
        st.subheader("VSB Settings")
        vsb_beta_default = st.session_state.get('vsb_beta_optimized', 500)
        vsb_beta = st.number_input("VSB Vestige Width (Hz)", value=vsb_beta_default, min_value=1, step=50, 
                                 key="vsb_beta_input",
                                 help="Use the Auto-Optimize button in the VSB tab.")
        st.session_state.vsb_beta_optimized = int(vsb_beta) # Ensure it's an int

    # SSB Settings
    ssb_side = 'usb'
    if 'SSB' in modulations:
        st.subheader("SSB Settings")
        ssb_side_choice = st.radio("SSB Sideband", ('USB', 'LSB'), key='ssb_side')
        ssb_side = ssb_side_choice.lower()

    # Adaptive FM Settings (Stateful)
    kf_min = 2.0
    kf_max = 10.0
    if 'Adaptive FM' in modulations:
        st.subheader("Adaptive FM Settings")
        st.info("Run the 'Optimizer' in the main panel to set these automatically.")
        kf_min_default = st.session_state.get('optimized_kf_min', 2.0)
        kf_max_default = st.session_state.get('optimized_kf_max', 10.0)
        kf_min = st.number_input("Min Freq. Deviation (kf_min)", 0.5, 5.0, kf_min_default, 0.5)
        kf_max = st.number_input("Max Freq. Deviation (kf_max)", 5.0, 20.0, kf_max_default, 0.5)

    # Coherent Receiver Settings (Stateful)
    phase_offsets = {}
    coherent_mods = [m for m in modulations if m in ['DSB-SC', 'SSB', 'VSB']]
    if coherent_mods:
        st.subheader("Coherent Receiver Settings")
        st.info("Use the Auto-Find button in the DSB-SC/SSB/VSB tabs.")
        for mod in coherent_mods:
            phase_default = st.session_state.get(f'phase_optimized_{mod}', 0.0)
            phase_offsets[mod] = st.slider(f"Phase Offset for {mod} (°)", -180.0, 180.0, phase_default, 0.5, 
                                           key=f"phase_slider_{mod}")
            st.session_state[f'phase_optimized_{mod}'] = phase_offsets[mod]

    st.divider()
    st.subheader("App Creators")
    st.markdown("""
    - Abhishek Kumar (24BIT0104)
    - Khushvendra Singh (24BIT0093)
    - Aditya Raj (24BIT0087)
    - Madhur Deshmukh (24BIT0113)
    - Manav mishra (24BIT0100)
    - Sarthak Agarwal (24BIT0116)
    """)

# --- Main Panel for Display and Processing ---
msg_flat = None
original_image_shape = None # Will be (h, w)
msg_for_metrics = None # The "golden" baseband signal (LPF'd for video)
msg_flat_composite = None # The full composite signal (video + audio)

# ... (Loading logic for Live Camera, Draw Signal, Equation) ...
if input_type == "Live Camera":
    camera_photo = st.camera_input("Take a picture")
    if camera_photo:
        try:
            msg, shape_wh = load_image(camera_photo)
            msg_flat = msg.flatten()
            original_image_shape = (shape_wh[1], shape_wh[0]) # (h,w)
            st.image(camera_photo, caption="Captured Image")
        except Exception as e:
            st.error(f"Error processing camera image: {e}")
            msg_flat = None
elif input_type == "Draw Signal":
    st.subheader("Draw Your Custom Waveform")
    canvas_result = st_canvas(stroke_width=5, stroke_color="#000000", background_color="#EEEEEE", height=200, width=700, drawing_mode="freedraw", key="canvas")
    if process_drawing_button:
        try:
            if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
                gray_image = 255 - canvas_result.image_data.astype(np.float32)[:, :, 0]
                height, width = gray_image.shape
                raw_signal = np.array([np.sum(gray_image[:, x] * np.arange(height)) / np.sum(gray_image[:, x]) if np.sum(gray_image[:, x]) > 0 else height / 2.0 for x in range(width)])
                raw_signal = height - raw_signal
                centered_signal = raw_signal - np.mean(raw_signal)
                max_abs = np.max(np.abs(centered_signal))
                if max_abs > 0: msg_flat = centered_signal / max_abs
                else: msg_flat = centered_signal
                st.session_state.drawn_signal = msg_flat
                st.success("Drawing processed!")
        except Exception as e:
            st.error(f"Error processing drawing: {e}")
    if 'drawn_signal' in st.session_state:
        msg_flat = st.session_state.drawn_signal
elif input_type == "Equation":
    st.subheader("Signal Defined by Equation")
    if generate_sine_button:
        N = int(duration_sine * fs); t = np.arange(N) / fs
        msg_flat = amplitude * np.sin(2 * np.pi * frequency * t + np.deg2rad(phase))
        st.session_state.equation_signal = msg_flat
    if generate_freeform_button:
        try:
            N = int(duration_ff * fs); t = np.arange(N) / fs
            context_dict = {'np': np, 't': t, 'pi': np.pi, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'sign': np.sign}
            generated_signal = ne.evaluate(equation_str.replace("np.", ""), local_dict=context_dict)
            if isinstance(generated_signal, np.ndarray):
                if np.max(np.abs(generated_signal)) > 0: msg_flat = generated_signal / np.max(np.abs(generated_signal))
                else: msg_flat = generated_signal
                st.session_state.equation_signal = msg_flat
            else: st.error("Equation did not produce a valid signal.")
        except Exception as e: st.error(f"Invalid equation or syntax: {e}")
    if 'equation_signal' in st.session_state:
        msg_flat = st.session_state.equation_signal
# ... (End of unchanged loading logic) ...

elif uploaded_file is not None:
    if input_type == "Audio":
        msg_flat = load_audio(uploaded_file, fs)
        st.subheader("Original Audio"); st.audio(array_to_audio_bytes(msg_flat, fs))
    elif input_type == "Image":
        msg, shape_wh = load_image(uploaded_file); msg_flat = msg.flatten()
        original_image_shape = (shape_wh[1], shape_wh[0]) # (h,w)
        st.subheader("Original Image"); st.image(recover_image(msg_flat, original_image_shape), use_container_width=True)
            
    elif input_type == "Video":
        try:
            with st.spinner(f"Processing video, analyzing content..."):
                msg_flat, original_image_shape, detail_scores = load_and_process_video(
                    uploaded_file, fs, st.session_state.video_res, st.session_state.audio_subcarrier
                )
            
            if msg_flat is not None:
                st.session_state.detail_scores = detail_scores
                st.session_state.original_shape = original_image_shape
                msg_flat_composite = msg_flat.copy()
                
                st.subheader("Original Composite Baseband Signal")
                fig_msg, ax_msg = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1]})
                ax_msg[0].plot(time_axis(len(msg_flat), fs), msg_flat); ax_msg[0].set_title("Composite Signal (Time Domain)"); ax_msg[0].grid(True)
                plot_spectrum(ax_msg[1], msg_flat, fs, "Composite Spectrum (Baseband)")
                ax_msg[1].axvline(st.session_state.audio_subcarrier, color='r', linestyle='--', label=f'Audio Subcarrier'); ax_msg[1].legend()
                plt.tight_layout(); st.pyplot(fig_msg)
                
                st.subheader("Per-Frame Detail Analysis (for Adaptive FM)")
                fig_detail, ax_detail = plt.subplots(figsize=(12, 3))
                ax_detail.plot(detail_scores); ax_detail.set_title("Frame Detail Score (Edge Detection)"); ax_detail.set_xlabel("Frame Number"); ax_detail.set_ylabel("Detail Score");
                st.pyplot(fig_detail)
                
                # --- Create the "golden" original video signal for metrics ---
                video_cutoff = st.session_state.audio_subcarrier * 0.9
                st.session_state.video_cutoff_hz = video_cutoff
                b, a = butter(6, video_cutoff / (fs / 2), btype='low')
                original_video_signal = lfilter(b, a, msg_flat)
                if np.max(np.abs(original_video_signal)) > 0:
                    original_video_signal /= np.max(np.abs(original_video_signal))
                msg_for_metrics = original_video_signal # This is the LPF'd original
                
            else: st.error("Failed to process video.")
        except Exception as e:
            st.error(f"Error processing video file: {e}"); st.exception(e)
            msg_flat = None

# --- Post-processing and Optimizer UI ---
if msg_flat is not None:
    if input_type != "Video": 
        msg_for_metrics = msg_flat.copy()
        st.subheader("Original Message Signal `m(t)`")
        fig_msg, ax_msg = plt.subplots(figsize=(12, 3))
        ax_msg.plot(time_axis(len(msg_for_metrics), fs), msg_for_metrics)
        ax_msg.set_title("Message Signal (Time Domain)"); ax_msg.set_xlabel("Time (s)"); ax_msg.grid(True)
        st.pyplot(fig_msg)
    else:
        # Video-only: Show Adaptive FM Optimizer
        with st.expander("🚀 Run Automated Parameter Optimizer (for Adaptive FM)"):
            st.info("Find the best 'Min/Max' deviation settings for a given channel noise level.")
            target_snr_opt = st.session_state.get('main_snr_slider', 20)
            st.markdown(f"**Target SNR:** `{target_snr_opt} dB` (from main Channel Effects slider)")

            if st.button("Find Best Adaptive FM Parameters"):
                (best_min, best_max), best_ssim = find_best_adaptive_parameters(
                    msg_flat_composite, msg_for_metrics, st.session_state.detail_scores,
                    carrier, fs, st.session_state.original_fps,
                    target_snr_opt, st.session_state.original_shape
                )
                
                st.metric("Predicted Best SSIM", f"{best_ssim:.4f}")
                col1, col2 = st.columns(2); col1.metric("Recommended kf_min", f"{best_min:.2f}"); col2.metric("Recommended kf_max", f"{best_max:.2f}")
                st.info("Parameters saved to sidebar. 'Adaptive FM' tab will now use these.")
                
                st.session_state.optimized_kf_min = best_min
                st.session_state.optimized_kf_max = best_max
    
    if 'AM' in modulations and np.any(am_depth * msg_for_metrics < -1):
        st.warning("**AM Overmodulation Alert:** The selected AM modulation depth is causing the term `(1 + depth * m(t))` to become negative. This will lead to phase reversal and distortion in the demodulated signal.")
    
    if input_type in ["Image", "Live Camera", "Video"]:
        bandwidth_hz = estimate_bandwidth(msg_flat if input_type != "Video" else msg_for_metrics, fs)
        if input_type == "Video": st.session_state.video_cutoff_hz = bandwidth_hz
        suggested_fc = int(np.ceil(bandwidth_hz * 5 / 1000) * 1000) if bandwidth_hz > 0 else 50000
        suggested_fs = int(np.ceil(2.5 * (suggested_fc + bandwidth_hz) / 1000) * 1000) if suggested_fc > 0 else 200000
        if input_type == "Video":
            st.info(f"**Suggestion for Video:** Est. Video Bandwidth: `{bandwidth_hz:,.0f} Hz`. Audio is at {st.session_state.audio_subcarrier} Hz. "
                    f"Ensure main carrier `fc` > `{suggested_fc:,} Hz` and `fs` > `{suggested_fs:,} Hz`.")
        else:
            st.info(f"**Suggestion for Image:** Est. Bandwidth: `{bandwidth_hz:,.0f} Hz`. Try fc approx `{suggested_fc:,} Hz` and fs `{suggested_fs:,} Hz`.")


# --- Main Processing & Display Loop ---
if msg_for_metrics is not None and modulations: # Use msg_for_metrics
    st.header("Modulated Signal Analysis and Recovery")
    mod_tabs = st.tabs(modulations)
    
    for i, mod in enumerate(modulations):
        with mod_tabs[i]:
            
            # --- Auto-Optimizer Buttons ---
            if mod == 'AM':
                if st.button("🚀 Auto-Optimize AM Depth", key="am_opt_btn"):
                    with st.spinner("Calculating optimal depth..."):
                        opt_depth = find_best_am_depth(msg_for_metrics)
                        st.session_state.am_depth_optimized = opt_depth
                        st.success(f"Optimal depth set to {opt_depth:.2f}. Re-running.")
                        st.rerun()
                st.divider()

            if mod == 'FM':
                if st.button("🚀 Auto-Optimize FM Deviation (kf)", key="fm_opt_btn"):
                    best_kf = find_best_kf(
                        msg_for_metrics, carrier, fs, snr_db, 
                        input_type, original_image_shape
                    )
                    st.session_state.kf_fm_optimized = best_kf
                    st.success(f"Optimal kf set to {best_kf:.2f}. Re-running.")
                    st.rerun()
                st.divider()

            if mod == 'VSB':
                if st.button("🚀 Auto-Optimize VSB Width (beta)", key="vsb_opt_btn"):
                    best_beta = find_best_vsb_beta(
                        msg_for_metrics, carrier, fs, snr_db,
                        input_type, original_image_shape
                    )
                    st.session_state.vsb_beta_optimized = int(best_beta)
                    st.success(f"Optimal beta set to {best_beta:.0f} Hz. Re-running.")
                    st.rerun()
                st.divider()
            
            # --- Main Simulation ---
            try:
                phase_offset_deg = phase_offsets.get(mod, 0)
                tx_noisy = None
                
                with st.spinner(f"Simulating {mod}..."):
                    # Modulation
                    if mod == 'AM': tx = am_modulate(msg_for_metrics, carrier, fs, depth=am_depth)
                    elif mod == 'DSB-SC': tx = dsb_sc_modulate(msg_for_metrics, carrier, fs)
                    elif mod == 'SSB': tx = ssb_modulate(msg_for_metrics, carrier, fs, side=ssb_side)
                    elif mod == 'VSB': tx = vsb_modulate(msg_for_metrics, carrier, fs, beta=vsb_beta)
                    elif mod == 'FM': 
                        # Use composite for video, baseband for all others
                        msg_to_mod = msg_flat_composite if input_type == 'Video' else msg_for_metrics
                        tx = fm_modulate(msg_to_mod, carrier, fs, kf=kf_fm)
                    elif mod == 'Adaptive FM':
                        if input_type != 'Video':
                            st.error("Adaptive FM only works with Video input.")
                            continue
                        tx = adaptive_fm_modulate(msg_flat_composite, st.session_state.detail_scores, 
                                                  carrier, fs, kf_min, kf_max, st.session_state.original_fps)
                    
                    if 'tx' not in locals(): continue
                    tx_noisy = awgn(tx, snr_db)
                    
                    # Demodulation
                    if mod == 'AM': rec = envelope_detect(tx_noisy)
                    elif mod in ['DSB-SC', 'SSB', 'VSB']: rec = coherent_demodulate(tx_noisy, carrier, fs, phase_offset_deg)
                    elif mod == 'FM' or mod == 'Adaptive FM': 
                        rec = fm_demodulate(tx_noisy, fs)
                
                if np.max(np.abs(rec)) > 0: rec /= np.max(np.abs(rec))
                
                st.subheader("Signal Plots")
                fig, axs = plt.subplots(2, 2, figsize=(14, 8))
                axs[0, 0].plot(time_axis(len(tx_noisy), fs), tx_noisy, color='red'); axs[0, 0].set_title("Noisy Modulated Signal (Time)")
                axs[0, 1].plot(time_axis(len(rec), fs), rec, color='green'); axs[0, 1].set_title("Recovered Signal (Time)")
                plot_spectrum(axs[1, 0], tx_noisy, fs, "Modulated Spectrum", carrier_freq=carrier)
                plot_spectrum(axs[1, 1], rec, fs, "Recovered Spectrum")
                plt.tight_layout(); st.pyplot(fig)
                
                # --- Costas Loop Button (must be *after* tx_noisy is created) ---
                if mod in ['DSB-SC', 'SSB', 'VSB']:
                    st.divider()
                    if st.button("🚀 Auto-Find Carrier Phase (Costas Loop)", key=f"costas_btn_{mod}"):
                        est_phase = costas_loop_find_phase(tx_noisy, carrier, fs)
                        st.session_state[f'phase_optimized_{mod}'] = est_phase
                        st.success(f"Phase lock estimated at {est_phase:.2f} degrees. Re-running.")
                        st.rerun()
                
                # --- Demultiplexing Logic for Video ---
                if input_type == "Video" and (mod == 'FM' or mod == 'Adaptive FM'):
                    st.subheader("Recovered Video and Audio")
                    with st.spinner("Demultiplexing video and audio..."):
                        video_cutoff = st.session_state.video_cutoff_hz
                        b, a = butter(6, video_cutoff / (fs / 2), btype='low')
                        rec_video_signal = lfilter(b, a, rec)
                        if np.max(np.abs(rec_video_signal)) > 0: rec_video_signal /= np.max(np.abs(rec_video_signal))
                        
                        audio_bw = (st.session_state.audio_subcarrier - video_cutoff) * 2
                        b_a, a_a = butter(6, [video_cutoff / (fs / 2), (st.session_state.audio_subcarrier + audio_bw) / (fs / 2)], btype='band')
                        rec_audio_fm = lfilter(b_a, a_a, rec)
                        
                        t = time_axis(len(rec_audio_fm), fs); lo = np.exp(-1j * 2 * np.pi * st.session_state.audio_subcarrier * t)
                        rec_audio_baseband = rec_audio_fm * lo
                        phase = np.unwrap(np.angle(rec_audio_baseband)); rec_audio = np.diff(phase) * fs / (2 * np.pi)
                        rec_audio = np.concatenate(([rec_audio[0]], rec_audio))
                        b_aud, a_aud = butter(4, video_cutoff / (fs/2), btype='low'); rec_audio = lfilter(b_aud, a_aud, rec_audio)
                        if np.max(np.abs(rec_audio)) > 0: rec_audio /= np.max(np.abs(rec_audio))

                    st.audio(array_to_audio_bytes(rec_audio, fs))
                    
                    with st.spinner("Reconstructing recovered video GIF..."):
                        fps = st.session_state.get('original_fps', 25)
                        num_samples_per_frame = int(fs / fps); num_frames = len(rec_video_signal) // num_samples_per_frame
                        h, w = original_image_shape; recovered_frames = []
                        for i in range(num_frames):
                            frame_data = rec_video_signal[i*num_samples_per_frame : (i+1)*num_samples_per_frame]
                            img = recover_image(frame_data, (h, w)) 
                            recovered_frames.append(np.array(img))
                        if recovered_frames:
                            iio.imwrite('recovered_video.gif', recovered_frames, fps=fps, loop=0)
                            st.image('recovered_video.gif', caption=f"{mod} Recovered Video", use_container_width=True)

                    st.subheader("Video Signal Quality Metrics")
                    accuracy, mse, psnr, ssim_val = calculate_signal_metrics(msg_for_metrics, rec_video_signal, original_image_shape)
                    # --- THIS IS THE CORRECTED BLOCK ---
                    cols = st.columns(4); cols[0].metric("Accuracy (Video)", f"{accuracy:.2f}%"); cols[1].metric("PSNR (Video, dB)", f"{psnr:.2f}")
                    cols[2].metric("MSE (Video)", f"{mse:.4f}", delta_color="inverse")
                    if ssim_val is not None: cols[3].metric("SSIM (Video)", f"{ssim_val:.4f}")
                    # --- END OF CORRECTION ---

                # --- Metrics for other signal types ---
                else:
                    st.subheader("Output & Quality Metrics")
                    rec_for_metrics = rec
                    if mod in ['DSB-SC', 'SSB', 'VSB', 'AM'] and input_type == "Video":
                        video_cutoff = st.session_state.video_cutoff_hz
                        b, a = butter(6, video_cutoff / (fs / 2), btype='low')
                        rec_for_metrics = lfilter(b, a, rec)
                        if np.max(np.abs(rec_for_metrics)) > 0:
                            rec_for_metrics /= np.max(np.abs(rec_for_metrics))
                        
                    accuracy, mse, psnr, ssim_val = calculate_signal_metrics(msg_for_metrics, rec_for_metrics, original_image_shape)
                    cols = st.columns(4); cols[0].metric("Accuracy", f"{accuracy:.2f}%"); cols[1].metric("PSNR (dB)", f"{psnr:.2f}")
                    cols[2].metric("MSE", f"{mse:.4f}", delta_color="inverse"); 
                    if ssim_val is not None: cols[3].metric("SSIM", f"{ssim_val:.4f}")
                    
                    if input_type == "Audio": 
                        st.audio(array_to_audio_bytes(rec_for_metrics, fs))
                    elif input_type in ["Image", "Live Camera"] and original_image_shape: 
                        st.image(recover_image(rec_for_metrics, original_image_shape), caption=f"{mod} Recovered Image", use_container_width=True)
                    elif input_type == "Video":
                        # This is for VSB/SSB/AM video recovery
                        with st.spinner("Reconstructing recovered video GIF..."):
                            fps = st.session_state.get('original_fps', 25)
                            num_samples_per_frame = int(fs / fps); num_frames = len(rec_for_metrics) // num_samples_per_frame
                            h, w = original_image_shape; recovered_frames = []
                            for i in range(num_frames):
                                frame_data = rec_for_metrics[i*num_samples_per_frame : (i+1)*num_samples_per_frame]
                                img = recover_image(frame_data, (h, w)); recovered_frames.append(np.array(img))
                            if recovered_frames:
                                iio.imwrite('recovered_video.gif', recovered_frames, fps=fps, loop=0)
                                st.image('recovered_video.gif', caption=f"{mod} Recovered Video", use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during the {mod} simulation.")
                st.exception(e)
else:
    st.info("**Welcome!** Please select an input type and simulation parameters from the sidebar to begin.")