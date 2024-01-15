use super::*;
use std::untrusted::fs;

pub struct MountsInfoNode;

impl MountsInfoNode {
    pub fn new() -> Arc<dyn INode> {
        Arc::new(File::new(Self))
    }
}

impl ProcINode for MountsInfoNode {
    fn generate_data_in_bytes(&self) -> vfs::Result<Vec<u8>> {
        Ok(MOUNTS.to_vec())
    }
}

lazy_static! {
    static ref MOUNTS: Vec<u8> = get_untrusted_mounts().unwrap();
}

fn get_untrusted_mounts() -> Result<Vec<u8>> {
    // let mounts_info = fs::read_to_string("/proc/mounts")?.into_bytes();
    // Ok(mounts_info)
    // Just return a proc mounts info
    Ok(format!("proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0").into_bytes())
}
