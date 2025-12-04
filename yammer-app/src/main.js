// Yammer frontend - minimal for now

document.addEventListener('DOMContentLoaded', () => {
  console.log('Yammer app loaded');

  // Get UI elements
  const statusIndicator = document.querySelector('.status-indicator');
  const statusText = document.querySelector('.status-text');
  const transcriptText = document.querySelector('.transcript-text');

  // State management
  let state = {
    status: 'idle', // idle, listening, processing, error
    transcript: ''
  };

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

  // Expose state setter for Tauri commands
  window.setYammerState = (newState) => {
    state = { ...state, ...newState };
    updateUI();
  };

  // Initial render
  updateUI();
});
