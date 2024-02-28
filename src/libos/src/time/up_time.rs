use super::do_clock_gettime;
use std::time::Duration;
use vdso_time::ClockId;

lazy_static! {
    static ref BOOT_TIME_STAMP: Duration = do_clock_gettime(ClockId::CLOCK_MONOTONIC_RAW)
        .unwrap()
        .as_duration();
    static ref BOOT_TIME_STAMP_SINCE_EPOCH: Duration = do_clock_gettime(ClockId::CLOCK_REALTIME)
        .unwrap()
        .as_duration();
}

pub fn init() {
    *BOOT_TIME_STAMP;
    *BOOT_TIME_STAMP_SINCE_EPOCH;
}

pub fn boot_time_since_epoch() -> Duration {
    *BOOT_TIME_STAMP_SINCE_EPOCH
}

pub fn get() -> Option<Duration> {
    do_clock_gettime(ClockId::CLOCK_MONOTONIC_RAW)
        .unwrap()
        .as_duration()
        .checked_sub(*BOOT_TIME_STAMP)
}
