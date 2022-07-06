use std::ffi::CString;
use std::mem::size_of;
use std::slice;
use std::ptr;
use libc::{open, ioctl, close, c_void, c_int, O_RDONLY};
// Defined in "occlum/deps/rust-sgx-sdk/sgx_types"
use sgx_types::*;

const SGXIOC_HYPER_GEN_RA_QUOTE: u64 = 0xc0807302;

cfg_if::cfg_if! {
    if #[cfg(target_env = "musl")] {
        const IOCTL_HYPER_GEN_RA_QUOTE: i32 = SGXIOC_HYPER_GEN_RA_QUOTE as i32;
    } else {
        const IOCTL_HYPER_GEN_RA_QUOTE: u64 = SGXIOC_HYPER_GEN_RA_QUOTE;
    }
}

#[repr(C)]
struct HyperGenQuoteArg {
    report_data: sgx_report_data_t,    // Input
    quote_type: sgx_quote_sign_type_t, // Input
    spid: sgx_spid_t,                  // Input
    nonce: sgx_quote_nonce_t,          // Input
    sigrl_ptr: *const u8,              // Input (optional)
    sigrl_len: u32,                    // Input (optional)
    quote_buf_len: u32,                // Input
    quote_buf: *mut u8,                // Output
}

// Defined in HyperEnclave SDK common/inc/sgx_quote_verify.h
const RESERVED_CERT_LEN: usize = 1024;
const RESERVED_LEN: usize = size_of::<CertInfo>() * 2 - size_of::<RawConfig>();
const SM2_SIG_SIZE: usize = 64;
const PUB_KEY_LEN: usize = (SM2_SIG_SIZE + 1) * 2;
const HASH_LEN: usize = 32;
const PCR_LEN: usize = HASH_LEN * 2;

#[repr(C)]
struct CertInfo {
    pem: [c_char; RESERVED_CERT_LEN],
    pem_len: u32
}

#[repr(C)]
struct RawConfig {
    tpm_ak_pub: [c_char; PUB_KEY_LEN],
    pcr_general: [c_char; PCR_LEN],
    pcr_5: [c_char; PCR_LEN],
    pcr_13: [c_char; PCR_LEN],
}

#[repr(C)]
struct CfcaCert {
    root_cert: CertInfo,
    second_level_cert: CertInfo,
}

#[repr(C)]
struct OtherCaCert {
    root_cert: CertInfo,
    reserved: [u8; RESERVED_LEN],
}

#[repr(C)]
struct NoCert {
    raw_config: RawConfig,
    reserved: CertInfo,
}

#[repr(C)]
union CertUnion {
    cfca_cert: CfcaCert,
    other_ca_cert: OtherCaCert,
    no_cert: NoCert,
}

#[repr(C)]
struct CertChain {
    cert_mode: u32,
    cert: CertUnion,
}

#[link(name = "sgx_uquote_verify_hyper")]
extern "C" {
    fn sgx_verify_quote(
        quote: *mut u8,
        quote_size: u32,
        user_data: *mut u8,
        user_data_size: u32,
        cert_chain: *const CertChain,
    ) -> c_int;
}

// Get quote as type sgx_quote_t
// pub struct sgx_quote_t {
//     pub version: uint16_t,                    /* 0   */
//     pub sign_type: uint16_t,                  /* 2   */
//     pub epid_group_id: sgx_epid_group_id_t,   /* 4   */
//     pub qe_svn: sgx_isv_svn_t,                /* 8   */
//     pub pce_svn: sgx_isv_svn_t,               /* 10  */
//     pub xeid: uint32_t,                       /* 12  */
//     pub basename: sgx_basename_t,             /* 16  */
//     pub report_body: sgx_report_body_t,       /* 48  */
//     pub signature_len: uint32_t,              /* 432 */
//     pub signature: [uint8_t; 0],              /* 436 */
// }
pub fn ra_generate(
    quote_buf: *mut u8,
    quote_buf_len: u32,
    user_data: *mut u8,
    user_data_size: u32) -> Result<i32, &'static str>
{
    if quote_buf.is_null() ||
        quote_buf_len == 0 ||
        user_data.is_null() ||
        user_data_size == 0 {
        return Err("Invlaid input parameters");
    }
    
    if user_data_size > SGX_REPORT_DATA_SIZE.try_into().unwrap() {
        return Err("usr data size is too big");
    }

    println!("Hyper RA Generate ...");

    let path =  CString::new("/dev/sgx").unwrap();
    let fd = unsafe { libc::open(path.as_ptr(), O_RDONLY) };
    if fd < 0 {
        return Err("Open /dev/sgx failed");
    }

    let mut report_data = sgx_report_data_t::default();
    let spid = sgx_spid_t::default();
    let nouce = sgx_quote_nonce_t::default();

    //fill in the report data array
    let data = unsafe { slice::from_raw_parts(user_data, user_data_size.try_into().unwrap()) };
    report_data.d.copy_from_slice(data);

    let quote_arg: HyperGenQuoteArg = HyperGenQuoteArg {
        report_data: report_data,
        quote_type: sgx_quote_sign_type_t::SGX_LINKABLE_SIGNATURE,
        spid: spid,
        nonce: nouce,
        sigrl_ptr: ptr::null_mut(),
        sigrl_len: 0,
        quote_buf_len: quote_buf_len,
        quote_buf: quote_buf,
    };

    let ret = unsafe { libc::ioctl(fd, IOCTL_HYPER_GEN_RA_QUOTE, &quote_arg) };
    unsafe { libc::close(fd) };
    if ret < 0 {
        return Err("IOCTRL IOCTL_HYPER_GEN_RA_QUOTE failed.");
    }

    // Do simple check
    let quote: *const sgx_quote_t = quote_buf as *const sgx_quote_t;
    
    let signature_len = unsafe { (*quote).signature_len };
    if signature_len == 0 {
        return Err("Invalid quote: zero-length signature.");
    }

    if report_data.d != unsafe { (*quote).report_body.report_data.d } {
        return Err("Invalid quote: wrong report data.");
    }
 
    println!("Hyper RA Generation successfully.");
    Ok( 0 )
}

pub fn ra_verify(
    ca_file: *const c_char,
    quote_buf: *mut u8,
    quote_buf_len: u32,
    user_data: *mut u8,
    user_data_size: u32) -> Result<i32, &'static str>
{
    Ok( 0 )
}