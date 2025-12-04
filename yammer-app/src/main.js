// Yammer frontend - minimal for now
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';

document.addEventListener('DOMContentLoaded', async () => {
  console.log('Yammer app loaded');

  // Get UI elements
  const statusIndicator = document.querySelector('.status-indicator');
  const statusText = document.querySelector('.status-text');
  const transcriptText = document.querySelector('.transcript-text');
  const waveformCanvas = document.getElementById('waveform');
  const waveformCtx = waveformCanvas.getContext('2d');

  // State management
  let state = {
    status: 'idle', // idle, listening, processing, error
    transcript: ''
  };

  // Waveform visualization
  const WAVEFORM_WIDTH = 268;
  const WAVEFORM_HEIGHT = 40;
  const BAR_COUNT = 40;
  const BAR_WIDTH = Math.floor(WAVEFORM_WIDTH / BAR_COUNT);
  const BAR_GAP = 1;

  let audioSamples = new Array(BAR_COUNT).fill(0);

  function drawWaveform() {
    // Clear canvas
    waveformCtx.clearRect(0, 0, WAVEFORM_WIDTH, WAVEFORM_HEIGHT);

    // Draw bars
    for (let i = 0; i < BAR_COUNT; i++) {
      const amplitude = audioSamples[i];
      const barHeight = Math.max(2, amplitude * WAVEFORM_HEIGHT);
      const x = i * BAR_WIDTH;
      const y = (WAVEFORM_HEIGHT - barHeight) / 2;

      // Color gradient based on amplitude
      const intensity = Math.min(255, Math.floor(amplitude * 255 + 100));
      waveformCtx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.7)}, ${Math.floor(intensity * 0.3)})`;

      waveformCtx.fillRect(x, y, BAR_WIDTH - BAR_GAP, barHeight);
    }
  }

  function updateUI() {
    // Update status indicator
    statusIndicator.className = `status-indicator ${state.status}`;

    // Update status text
    const statusLabels = {
      idle: 'Ready',
      listening: 'Listening...',
      processing: 'Processing...',
      error: 'Error'
    };
    statusText.textContent = statusLabels[state.status] || 'Ready';

    // Update transcript
    if (state.transcript) {
      transcriptText.textContent = state.transcript;
      transcriptText.classList.remove('placeholder');
    } else {
      transcriptText.textContent = 'Press hotkey to start dictating...';
      transcriptText.classList.add('placeholder');
    }
  }

  // Listen for audio samples from Rust backend
  await listen('audio-samples', (event) => {
    const samples = event.payload;
    if (Array.isArray(samples) && samples.length > 0) {
      // Downsample to BAR_COUNT bars
      const step = Math.floor(samples.length / BAR_COUNT);
      for (let i = 0; i < BAR_COUNT; i++) {
        const idx = i * step;
        audioSamples[i] = Math.abs(samples[idx] || 0);
      }
      drawWaveform();
    }
  });

  // Animate waveform decay when no audio
  setInterval(() => {
    if (state.status !== 'listening') {
      // Decay waveform bars toward zero
      for (let i = 0; i < BAR_COUNT; i++) {
        audioSamples[i] *= 0.85;
      }
      drawWaveform();
    }
  }, 50); // 20 fps decay

  // Expose state setter for Tauri commands
  window.setYammerState = (newState) => {
    state = { ...state, ...newState };
    updateUI();
  };

  // Test function for waveform (remove in production)
  window.testWaveform = async () => {
    try {
      await invoke('simulate_audio');
      console.log('Simulated audio sent');
    } catch (e) {
      console.error('Failed to simulate audio:', e);
    }
  };

  // Initial render
  updateUI();
  drawWaveform(); // Draw initial empty waveform

  // Auto-test waveform on load (remove in production)
  setTimeout(() => {
    window.testWaveform();
    // Repeat every second for demo
    setInterval(() => window.testWaveform(), 1000);
  }, 1000);
});
