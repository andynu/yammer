// Yammer frontend - dictation UI
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
    status: 'idle', // idle, listening, processing, correcting, done, error
    transcript: '',
    isPartial: false,
    pipelineInitialized: false,
    isRunning: false
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
      correcting: 'Correcting...',
      done: 'Done',
      error: 'Error'
    };
    statusText.textContent = statusLabels[state.status] || 'Ready';

    // Update transcript
    if (state.transcript) {
      transcriptText.textContent = state.transcript;
      transcriptText.classList.remove('placeholder');

      // Mark as partial if still processing
      if (state.isPartial) {
        transcriptText.classList.add('partial');
      } else {
        transcriptText.classList.remove('partial');
      }
    } else {
      transcriptText.textContent = state.pipelineInitialized
        ? 'Press Ctrl+Alt+D to start dictating...'
        : 'Initializing...';
      transcriptText.classList.add('placeholder');
      transcriptText.classList.remove('partial');
    }
  }

  // Listen for audio samples from Rust backend
  await listen('audio-samples', (event) => {
    const samples = event.payload;
    if (Array.isArray(samples) && samples.length > 0) {
      // Downsample to BAR_COUNT bars if needed
      if (samples.length === BAR_COUNT) {
        audioSamples = samples;
      } else {
        const step = Math.floor(samples.length / BAR_COUNT);
        for (let i = 0; i < BAR_COUNT; i++) {
          const idx = i * step;
          audioSamples[i] = Math.abs(samples[idx] || 0);
        }
      }
      drawWaveform();
    }
  });

  // Listen for pipeline state changes from backend
  await listen('pipeline-state', (event) => {
    const newState = event.payload;
    console.log('Pipeline state:', newState);

    state.status = newState;
    state.isRunning = newState === 'listening' || newState === 'processing' || newState === 'correcting';

    // Clear transcript on new listening session
    if (newState === 'listening') {
      state.transcript = '';
      state.isPartial = false;
    }

    // Return to idle after showing done/error briefly
    if (newState === 'done' || newState === 'error') {
      setTimeout(() => {
        if (state.status === newState) {
          state.status = 'idle';
          updateUI();
        }
      }, 2000);
    }

    updateUI();
  });

  // Listen for transcript updates
  await listen('transcript', (event) => {
    const { text, isPartial } = event.payload;
    console.log('Transcript:', text, 'isPartial:', isPartial);

    state.transcript = text;
    state.isPartial = isPartial;
    updateUI();
  });

  // Listen for pipeline errors
  await listen('pipeline-error', (event) => {
    const error = event.payload;
    console.error('Pipeline error:', error);

    state.status = 'error';
    state.transcript = `Error: ${error}`;
    state.isPartial = false;
    updateUI();
  });

  // Listen for global hotkey dictation toggle
  await listen('dictation-toggle', async () => {
    console.log('Dictation toggle hotkey received');

    if (!state.pipelineInitialized) {
      console.log('Pipeline not initialized, initializing first...');
      try {
        await initializePipeline();
      } catch (e) {
        console.error('Failed to initialize pipeline:', e);
        state.status = 'error';
        state.transcript = `Initialization failed: ${e}`;
        updateUI();
        return;
      }
    }

    try {
      // Toggle dictation via backend
      const nowRunning = await invoke('toggle_dictation');
      console.log('Dictation toggled, now running:', nowRunning);
    } catch (e) {
      console.error('Toggle dictation error:', e);
      // If "already running", try stopping
      if (e.includes('already in progress')) {
        try {
          await invoke('stop_dictation');
        } catch (stopErr) {
          console.error('Stop dictation error:', stopErr);
        }
      }
    }
  });

  // Animate waveform decay when not listening
  setInterval(() => {
    if (state.status !== 'listening') {
      // Decay waveform bars toward zero
      for (let i = 0; i < BAR_COUNT; i++) {
        audioSamples[i] *= 0.85;
      }
      drawWaveform();
    }
  }, 50); // 20 fps decay

  // Initialize pipeline with models
  async function initializePipeline() {
    console.log('Checking models...');

    // Check if models exist
    const modelStatus = await invoke('check_models');
    console.log('Model status:', modelStatus);

    if (!modelStatus.whisper.exists) {
      throw new Error(`Whisper model not found at ${modelStatus.whisper.path}. Run: yammer download-models`);
    }

    // Initialize pipeline
    console.log('Initializing pipeline...');
    await invoke('initialize_pipeline', {
      whisperModel: null, // Use default
      llmModel: null, // Use default
      useCorrection: modelStatus.llm.exists // Enable if LLM model exists
    });

    state.pipelineInitialized = true;
    console.log('Pipeline initialized successfully');
    updateUI();
  }

  // Expose state setter for debugging
  window.setYammerState = (newState) => {
    state = { ...state, ...newState };
    updateUI();
  };

  // Test function for waveform (for debugging)
  window.testWaveform = async () => {
    try {
      await invoke('simulate_audio');
      console.log('Simulated audio sent');
    } catch (e) {
      console.error('Failed to simulate audio:', e);
    }
  };

  // Test function to cycle through states (for debugging)
  window.testStates = () => {
    const states = [
      { status: 'listening', transcript: '', isPartial: false },
      { status: 'processing', transcript: 'This is a test...', isPartial: true },
      { status: 'correcting', transcript: 'This is a test transcript', isPartial: false },
      { status: 'done', transcript: 'This is a test transcript.', isPartial: false },
      { status: 'idle', transcript: '', isPartial: false }
    ];

    let index = 0;
    const interval = setInterval(() => {
      window.setYammerState(states[index]);
      console.log('State:', states[index].status);
      index++;
      if (index >= states.length) {
        clearInterval(interval);
      }
    }, 2000);
  };

  // Initial render
  updateUI();
  drawWaveform();

  // Auto-initialize pipeline on load
  setTimeout(async () => {
    try {
      await initializePipeline();
    } catch (e) {
      console.error('Auto-initialization failed:', e);
      state.transcript = `Ready. ${e.message || e}`;
      updateUI();
    }
  }, 500);
});
