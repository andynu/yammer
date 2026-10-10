//! Per-dictation stage timing, logged once at the end of each dictation.

use std::time::{Duration, Instant};

/// Accumulates named stage durations for one dictation. Stages that never
/// ran (discarded recording, correction disabled) are simply absent.
#[derive(Debug)]
pub struct StageTimings {
    started: Instant,
    stages: Vec<(&'static str, Duration)>,
}

impl StageTimings {
    pub fn new(started: Instant) -> Self {
        Self {
            started,
            stages: Vec::new(),
        }
    }

    pub fn started(&self) -> Instant {
        self.started
    }

    pub fn record(&mut self, stage: &'static str, duration: Duration) {
        self.stages.push((stage, duration));
    }

    /// Record the span between two instants, saturating to zero if `to`
    /// precedes `from`.
    pub fn record_span(&mut self, stage: &'static str, from: Instant, to: Instant) {
        self.record(stage, to.saturating_duration_since(from));
    }

    /// `stage=Nms ...` in recording order, followed by the total since start.
    pub fn summary(&self, end: Instant) -> String {
        let mut parts: Vec<String> = self
            .stages
            .iter()
            .map(|(name, d)| format!("{}={}ms", name, d.as_millis()))
            .collect();
        parts.push(format!(
            "total={}ms",
            end.saturating_duration_since(self.started).as_millis()
        ));
        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_lists_stages_in_order_then_total() {
        let start = Instant::now();
        let mut t = StageTimings::new(start);
        t.record("keypress_to_listening", Duration::from_millis(120));
        t.record("correction", Duration::from_millis(850));
        assert_eq!(
            t.summary(start + Duration::from_millis(4321)),
            "keypress_to_listening=120ms correction=850ms total=4321ms"
        );
    }

    #[test]
    fn summary_with_no_stages_is_just_total() {
        let start = Instant::now();
        let t = StageTimings::new(start);
        assert_eq!(t.summary(start + Duration::from_millis(7)), "total=7ms");
    }

    #[test]
    fn record_span_measures_between_instants() {
        let start = Instant::now();
        let mut t = StageTimings::new(start);
        t.record_span(
            "speech_to_transcript",
            start + Duration::from_millis(100),
            start + Duration::from_millis(350),
        );
        assert_eq!(
            t.summary(start + Duration::from_millis(400)),
            "speech_to_transcript=250ms total=400ms"
        );
    }

    #[test]
    fn spans_running_backwards_saturate_to_zero() {
        let start = Instant::now();
        let mut t = StageTimings::new(start + Duration::from_millis(50));
        t.record_span("typing", start + Duration::from_millis(10), start);
        assert_eq!(t.summary(start), "typing=0ms total=0ms");
    }
}
