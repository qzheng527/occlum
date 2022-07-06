extern crate hyper_ra;
use std::str;
use std::io::Result;
use hyper_ra::{ra_generate, ra_verify};

fn main() {
    let mut quote_buf: [u8; 2048] = [0; 2048];
    let mut user_data: [u8; 64] = [8; 64];

    ra_generate(
        quote_buf.as_mut_ptr(), 2048,
        user_data.as_mut_ptr(), 64
    ).unwrap();
}