// Yammer frontend - dictation UI
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { availableMonitors, primaryMonitor } from '@tauri-apps/api/window';

document.addEventListener('DOMContentLoaded', async () => {
  console.log('Yammer app loaded');

  // Audio feedback using Web Audio API
  let audioContext = null;

  function getAudioContext() {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContext;
  }

  // Play a short beep tone
  function playTone(frequency, duration, volume = 0.3) {
    try {
      const ctx = getAudioContext();
      const oscillator = ctx.createOscillator();
      const gainNode = ctx.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(ctx.destination);

      oscillator.frequency.value = frequency;
      oscillator.type = 'sine';

      gainNode.gain.setValueAtTime(0, ctx.currentTime);
      gainNode.gain.linearRampToValueAtTime(volume, ctx.currentTime + 0.01);
      gainNode.gain.linearRampToValueAtTime(0, ctx.currentTime + duration);

      oscillator.start(ctx.currentTime);
      oscillator.stop(ctx.currentTime + duration);
    } catch (e) {
      console.warn('Audio feedback not available:', e);
    }
  }

  // Audio feedback for recording start (ascending tone)
  function playStartSound() {
    playTone(440, 0.08, 0.2);  // A4
    setTimeout(() => playTone(554, 0.08, 0.2), 80);  // C#5
    setTimeout(() => playTone(659, 0.12, 0.2), 160); // E5
  }

  // Audio feedback for recording stop/done (descending tone)
  function playStopSound() {
    playTone(659, 0.08, 0.2);  // E5
    setTimeout(() => playTone(554, 0.08, 0.2), 80);  // C#5
    setTimeout(() => playTone(440, 0.12, 0.2), 160); // A4
  }

  // Get UI elements
  const statusIndicator = document.querySelector('.status-indicator');
  const statusText = document.querySelector('.status-text');
  const recordButton = document.querySelector('.record-btn');
  const appContainer = document.querySelector('.app-container');
  const appWindow = getCurrentWindow();

  // Load and apply saved window position
  async function loadWindowPosition() {
    try {
      // Get primary monitor size for bounds checking
      const monitor = await primaryMonitor();
      if (!monitor) {
        console.log('No primary monitor detected');
        return;
      }

      const screenWidth = monitor.size.width;
      const screenHeight = monitor.size.height;
      const windowSize = await appWindow.innerSize();
      const windowWidth = windowSize.width;
      const windowHeight = windowSize.height;

      console.log(`Screen: ${screenWidth}x${screenHeight}, Window: ${windowWidth}x${windowHeight}`);

      // Get saved position (validated against current screen)
      const position = await invoke('get_saved_window_position', {
        screenWidth,
        screenHeight,
        windowWidth,
        windowHeight
      });

      if (position) {
        const [x, y] = position;
        console.log(`Restoring window position: (${x}, ${y})`);
        await appWindow.setPosition({ x, y, type: 'Physical' });
      }
    } catch (e) {
      console.error('Failed to load window position:', e);
    }
  }

  // Save current window position
  async function saveWindowPosition(x, y) {
    try {
      console.log(`Saving window position: (${x}, ${y})`);
      await invoke('save_window_position', { x, y });
    } catch (e) {
      console.error('Failed to save window position:', e);
    }
  }

  // Save position before app closes
  async function saveCurrentPosition() {
    try {
      const position = await appWindow.outerPosition();
      console.log(`Saving position before close: (${position.x}, ${position.y})`);
      await invoke('save_window_position', { x: position.x, y: position.y });
    } catch (e) {
      console.error('Failed to save position before close:', e);
    }
  }

  // Enable window dragging on the container
  appContainer.addEventListener('mousedown', (e) => {
    // Only drag on left mouse button - must be sync, not async
    // Don't drag if clicking on interactive elements
    if (e.button === 0 &&
        !e.target.closest('.close-btn') &&
        !e.target.closest('.transcript-area') &&
        !e.target.closest('.record-btn') &&
        !e.target.closest('.waveform-container')) {
      e.preventDefault();
      appWindow.startDragging();
    }
  });

  // Listen for window move events from Tauri (fires during/after drag)
  let saveTimeout = null;
  appWindow.onMoved(({ payload: position }) => {
    console.log(`Window moved to: (${position.x}, ${position.y})`);
    // Debounce saves - only save after position stabilizes
    if (saveTimeout) clearTimeout(saveTimeout);
    saveTimeout = setTimeout(() => {
      saveWindowPosition(position.x, position.y);
    }, 300);
  });

  // Close button handler - minimize to tray instead of quit
  const closeBtn = document.querySelector('.close-btn');
  closeBtn.addEventListener('mousedown', async (e) => {
    e.stopPropagation();
    e.preventDefault();
    console.log('Close button clicked - minimizing to tray');
    await saveCurrentPosition();
    await appWindow.hide();
  });

  // Escape key to minimize to tray
  document.addEventListener('keydown', async (e) => {
    if (e.key === 'Escape') {
      console.log('Escape pressed - minimizing to tray');
      await saveCurrentPosition();
      await appWindow.hide();
    }
  });

  const transcriptText = document.querySelector('.transcript-text');
  const transcriptArea = document.querySelector('.transcript-area');
  const waveformCanvas = document.getElementById('waveform');
  const waveformCtx = waveformCanvas.getContext('2d');

  // Click on transcript to copy to clipboard
  transcriptArea.addEventListener('click', async (e) => {
    e.stopPropagation(); // Don't trigger window drag
    console.log('Transcript area clicked');
    console.log('Current transcript:', state.transcript);
    console.log('Transcript trimmed:', state.transcript ? state.transcript.trim() : '(empty)');

    if (state.transcript && state.transcript.trim()) {
      console.log('Attempting to copy to clipboard...');
      try {
        await navigator.clipboard.writeText(state.transcript);
        console.log('SUCCESS: Copied to clipboard:', state.transcript);

        // Visual feedback - briefly highlight
        transcriptText.classList.add('copied');
        setTimeout(() => transcriptText.classList.remove('copied'), 300);
      } catch (err) {
        console.error('FAILED to copy to clipboard:', err);
        console.error('Error name:', err.name);
        console.error('Error message:', err.message);
      }
    } else {
      console.log('No transcript to copy (empty or whitespace only)');
    }
  });

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

    // Update record button - disabled until initialized, then shows recording state
    if (!state.pipelineInitialized) {
      recordButton.classList.add('disabled');
      recordButton.classList.remove('recording');
    } else if (state.isRunning) {
      recordButton.classList.remove('disabled');
      recordButton.classList.add('recording');
    } else {
      recordButton.classList.remove('disabled');
      recordButton.classList.remove('recording');
    }

    // Update status text
    const statusLabels = {
      idle: 'Ready',
      listening: 'Listening...',
      processing: 'Processing...',
      correcting: 'Correcting...',
      done: 'Done',
      error: 'Error'
    };
    // Show "Loading..." during initialization
    if (!state.pipelineInitialized && state.status === 'processing') {
      statusText.textContent = 'Loading...';
    } else {
      statusText.textContent = statusLabels[state.status] || 'Ready';
    }

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
        ? 'Click record or press Ctrl+Space'
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
  let previousState = 'idle';
  await listen('pipeline-state', (event) => {
    const newState = event.payload;
    console.log('Pipeline state:', newState);

    // Play audio feedback on state transitions
    if (newState === 'listening' && previousState !== 'listening') {
      // Started recording
      playStartSound();
    } else if (newState === 'done' && previousState !== 'done') {
      // Finished successfully
      playStopSound();
    }
    previousState = newState;

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

  // Toggle dictation (shared logic for hotkey and click)
  let lastToggleTime = 0;
  let isToggling = false;

  async function toggleDictation() {
    // Block if not initialized
    if (!state.pipelineInitialized) {
      console.log('Pipeline not initialized yet, ignoring toggle');
      return;
    }

    // Debounce: ignore toggles within 500ms of each other
    const now = Date.now();
    if (now - lastToggleTime < 500) {
      console.log('Toggle debounced (too fast)');
      return;
    }

    // Prevent concurrent toggle calls
    if (isToggling) {
      console.log('Toggle already in progress');
      return;
    }

    lastToggleTime = now;
    isToggling = true;

    console.log('Dictation toggle triggered');

    try {
      // Toggle dictation via backend
      const nowRunning = await invoke('toggle_dictation');
      console.log('Dictation toggled, now running:', nowRunning);
    } catch (e) {
      console.error('Toggle dictation error:', e);
      // If "already running", try stopping
      if (e.includes && e.includes('already in progress')) {
        try {
          await invoke('stop_dictation');
        } catch (stopErr) {
          console.error('Stop dictation error:', stopErr);
        }
      }
    } finally {
      isToggling = false;
    }
  }

  // Start dictation (only if not already running)
  async function startDictation() {
    // Block if not initialized
    if (!state.pipelineInitialized) {
      console.log('Pipeline not initialized yet, ignoring start');
      return;
    }

    // If already running, don't do anything
    if (state.isRunning) {
      console.log('Already recording, ignoring start');
      return;
    }

    console.log('Starting dictation (from hidden window)');

    try {
      await invoke('start_dictation');
      console.log('Dictation started');
    } catch (e) {
      console.error('Start dictation error:', e);
    }
  }

  // Listen for global hotkey events
  await listen('dictation-toggle', toggleDictation);
  await listen('dictation-start', startDictation);

  // Record button - clear toggle behavior
  const recordBtn = document.querySelector('.record-btn');

  recordBtn.addEventListener('click', async (e) => {
    e.stopPropagation(); // Don't trigger window drag
    console.log('Record button clicked');
    await toggleDictation();
  });

  // Prevent mousedown from triggering drag
  recordBtn.addEventListener('mousedown', (e) => {
    e.stopPropagation();
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

  // Load saved window position
  await loadWindowPosition();

  // Auto-initialize pipeline immediately on load
  console.log('Starting pipeline initialization...');
  state.transcript = 'Loading models...';
  state.status = 'processing';
  updateUI();

  try {
    await initializePipeline();
    state.status = 'idle';
    state.transcript = '';  // Clear loading message to show ready placeholder
    updateUI();
  } catch (e) {
    console.error('Pipeline initialization failed:', e);
    state.status = 'error';
    state.transcript = `Init failed: ${e.message || e}`;
    updateUI();
  }
});
