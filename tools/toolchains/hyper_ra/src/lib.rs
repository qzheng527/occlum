#![feature(untagged_unions)]

mod hyper_ra;

use std::os::raw::c_char;
pub use crate::hyper_ra::{ra_generate, ra_verify};

#[no_mangle]
pub extern "C" fn hyper_ra_generate(
    quote_buf: *mut u8,
    quote_buf_len: u32,
    user_data: *mut u8,
    user_data_size: u32) -> i32
{
    if quote_buf.is_null() ||
        quote_buf_len == 0 ||
        user_data.is_null() ||
        user_data_size == 0 {
        return -1
    }

    ra_generate(quote_buf, quote_buf_len, user_data, user_data_size).unwrap();

    0
}

#[no_mangle]
pub extern "C" fn hyper_ra_verify(
    ca_file: *const c_char,
    quote_buf: *mut u8,
    quote_buf_len: u32,
    user_data: *mut u8,
    user_data_size: u32) -> i32
{
    if ca_file.is_null() ||
        quote_buf.is_null() ||
        quote_buf_len == 0 ||
        user_data.is_null() ||
        user_data_size == 0 {
        return -1
    }

    ra_verify(ca_file, quote_buf, quote_buf_len, user_data, user_data_size).unwrap();

    0
}

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn it_works() {
//         let result = 2 + 2;
//         assert_eq!(result, 4);
//     }
// }
