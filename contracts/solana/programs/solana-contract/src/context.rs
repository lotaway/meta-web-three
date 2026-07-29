use anchor_lang::prelude::*;
use std::mem::size_of;

#[account]
pub struct Listing {
    pub seller: Pubkey,
    pub mint: Pubkey,
    pub payment_mint: Pubkey,
    pub price: u64,
    pub listed_amount: u64,
    pub status: u8,
    pub created_at: i64,
}

impl Listing {
    pub const LEN: usize = size_of::<Pubkey>() * 3 + size_of::<u64>() * 2 + size_of::<u8>() + size_of::<i64>() + 1;
}

#[account]
pub struct Activity {
    pub authority: Pubkey,
    pub start_time: i64,
    pub end_time: i64,
    pub entry_fee: u64,
    pub reward_pcts: [u16; 3],
    pub total_pool: u64,
    pub participant_count: u64,
    pub merkle_root: [u8; 32],
}

impl Activity {
    pub const LEN: usize = 8 + 32 + 8 + 8 + 8 + 6 + 8 + 8 + 32;
}

#[account]
pub struct CommissionGraph {
    pub upline: Pubkey,
    pub level: u32,
}

impl CommissionGraph {
    pub const LEN: usize = 8 + 32 + 4;
}

#[account]
pub struct Coupon {
    pub authority: Pubkey,
    pub mint: Pubkey,
    pub discount_amount: u64,
    pub max_uses: u64,
    pub total_redeemed: u64,
    pub merkle_root: [u8; 32],
    pub expiry: i64,
}

impl Coupon {
    pub const LEN: usize = 8 + 32 + 32 + 8 + 8 + 8 + 32 + 8;
}
