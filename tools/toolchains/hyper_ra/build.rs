use std::env;

fn main() {
    let sdk_dir = env::var("HYPER_SDK").unwrap_or_else(|_| "/opt/intel/sgxsdk".to_string());

    println!("cargo:rustc-link-search=native={}/sdk_libs", sdk_dir);
    println!("cargo:rustc-link-lib=dylib=sgx_uquote_verify_hyper");
}