use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount, Mint};
use std::mem::size_of;

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init_if_needed,
        payer = signer,
        seeds=[seeds::TOKEN_MANAGER],
        bump,
        space = 8,
    )]
    token_manager: AccountInfo<'info>,

    #[account(
        init_if_needed,
        payer = signer,
        mint::decimals = 9,
        mint::authority = signer,
    )]
    token_mint_account: Account<'info, Mint>,

    #[account(
        init_if_needed,
        payer = signer,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        token::mint = token_mint_account,
        token::authority = token_manager,
    )]
    program_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    signer: Signer<'info>,

    system_program: Program<'info, System>,
    token_program: Program<'info, Token>,
    rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct CreateTokenAndNFT<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        init,
        payer = authority,
        seeds = [b"mint", authority.key().as_ref()],
        bump,
        mint::decimals = 0,
        mint::authority = authority,
        mint::freeze_authority = authority
    )]
    pub mint: Account<'info, Mint>,

    #[account(
        init,
        payer = authority,
        associated_token::mint = mint,
        associated_token::authority = authority
    )]
    pub token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    pub metadata: UncheckedAccount<'info>,

    pub token_metadata_program: UncheckedAccount<'info>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

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

#[derive(Accounts)]
#[instruction(price: u64, listed_amount: u64)]
pub struct ListGood<'info> {
    #[account(mut)]
    pub seller: Signer<'info>,

    #[account(
        init,
        payer = seller,
        seeds = [seeds::LISTING, seller.key().as_ref(), mint.key().as_ref()],
        bump,
        space = Listing::LEN,
    )]
    pub listing: Account<'info, Listing>,

    pub mint: Account<'info, Mint>,

    pub payment_mint: Account<'info, Mint>,

    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = seller,
    )]
    pub seller_token_account: Account<'info, TokenAccount>,

    #[account(
        init_if_needed,
        payer = seller,
        seeds = [seeds::LISTING_ESCROW, mint.key().as_ref()],
        bump,
        token::mint = mint,
        token::authority = listing,
    )]
    pub escrow_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct BuyGood<'info> {
    #[account(mut)]
    pub buyer: Signer<'info>,

    #[account(
        mut,
        seeds = [seeds::LISTING, listing.seller.key().as_ref(), listing.mint.key().as_ref()],
        bump,
        close = seller,
    )]
    pub listing: Account<'info, Listing>,

    #[account(mut)]
    pub seller: SystemAccount<'info>,

    pub mint: Account<'info, Mint>,

    pub payment_mint: Account<'info, Mint>,

    #[account(
        mut,
        associated_token::mint = payment_mint,
        associated_token::authority = buyer,
    )]
    pub buyer_payment_token_account: Account<'info, TokenAccount>,

    #[account(
        mut,
        associated_token::mint = payment_mint,
        associated_token::authority = seller,
    )]
    pub seller_payment_token_account: Account<'info, TokenAccount>,

    #[account(
        init_if_needed,
        payer = buyer,
        associated_token::mint = mint,
        associated_token::authority = buyer,
    )]
    pub buyer_receive_token_account: Account<'info, TokenAccount>,

    #[account(
        mut,
        seeds = [seeds::LISTING_ESCROW, mint.key().as_ref()],
        bump,
        token::mint = mint,
        token::authority = listing,
    )]
    pub escrow_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
#[instruction(name: String, symbol: String)]
pub struct CreateToken<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        init,
        payer = authority,
        seeds = [b"token", name.as_bytes(), authority.key().as_ref()],
        bump,
        mint::decimals = 9,
        mint::authority = authority,
        mint::freeze_authority = authority,
    )]
    pub mint: Account<'info, Mint>,

    #[account(
        init,
        payer = authority,
        associated_token::mint = mint,
        associated_token::authority = authority
    )]
    pub token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    pub metadata: UncheckedAccount<'info>,

    pub token_metadata_program: UncheckedAccount<'info>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
#[instruction(name: String, symbol: String)]
pub struct CreateSFT<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        init,
        payer = authority,
        seeds = [b"sft", name.as_bytes(), authority.key().as_ref()],
        bump,
        mint::decimals = 0,
        mint::authority = authority,
        mint::freeze_authority = authority
    )]
    pub mint: Account<'info, Mint>,

    #[account(
        init,
        payer = authority,
        associated_token::mint = mint,
        associated_token::authority = authority
    )]
    pub token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    pub metadata: UncheckedAccount<'info>,

    pub token_metadata_program: UncheckedAccount<'info>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct MintTokens<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        mut,
        mint::authority = authority,
    )]
    pub mint: Account<'info, Mint>,

    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = receiver,
    )]
    pub token_account: Account<'info, TokenAccount>,

    pub receiver: SystemAccount<'info>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct BurnTokens<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(mut)]
    pub mint: Account<'info, Mint>,

    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = authority,
    )]
    pub token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct DepositAccounts<'info> {
    #[account(
        mut,
        seeds=[seeds::TOKEN_MANAGER],
        bump,
    )]
    pub token_manager: AccountInfo<'info>,

    #[account(
        mut,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        token::mint = token_mint_account,
        token::authority = token_manager,
    )]
    pub program_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    pub sender_token_account: AccountInfo<'info>,

    pub token_mint_account: AccountInfo<'info>,

    #[account(mut)]
    pub signer: Signer<'info>,

    pub system_program: Program<'info, System>,
    pub token_program: Program<'info, Token>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct WithdrawAccounts<'info> {
    #[account(
        mut,
        seeds=[seeds::TOKEN_MANAGER],
        bump,
    )]
    pub token_manager: AccountInfo<'info>,

    #[account(
        mut,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        token::mint = token_mint_account,
        token::authority = token_manager,
    )]
    pub program_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    pub receiver_token_account: AccountInfo<'info>,

    pub token_mint_account: AccountInfo<'info>,

    #[account(mut)]
    pub signer: Signer<'info>,

    pub system_program: Program<'info, System>,
    pub token_program: Program<'info, Token>,
    pub rent: Sysvar<'info, Rent>,
}

#[account]
pub struct CommissionGraph {
    pub upline: Pubkey,
    pub level: u32,
}

impl CommissionGraph {
    pub const LEN: usize = 8 + 32 + 4;
}

#[derive(Accounts)]
pub struct SetUpline<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    /// CHECK: upline wallet address
    pub upline: AccountInfo<'info>,
    #[account(
        init,
        payer = signer,
        seeds = [seeds::COMMISSION, target.key().as_ref()],
        bump,
        space = CommissionGraph::LEN,
    )]
    pub commission_graph: Account<'info, CommissionGraph>,
    /// CHECK: target wallet address
    pub target: AccountInfo<'info>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct DistributeCommission<'info> {
    #[account(mut)]
    pub seller: Signer<'info>,
    #[account(
        mut,
        seeds = [seeds::COMMISSION, seller.key().as_ref()],
        bump,
    )]
    pub commission_graph: Account<'info, CommissionGraph>,
    #[account(
        mut,
        associated_token::mint = payment_mint,
        associated_token::authority = seller,
    )]
    pub seller_token_account: Account<'info, TokenAccount>,
    pub payment_mint: Account<'info, Mint>,
    pub token_program: Program<'info, Token>,
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

#[derive(Accounts)]
pub struct CreateCoupon<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    #[account(
        init,
        payer = authority,
        seeds = [seeds::COUPON, authority.key().as_ref()],
        bump,
        space = Coupon::LEN,
    )]
    pub coupon: Account<'info, Coupon>,
    #[account(
        init_if_needed,
        payer = authority,
        seeds = [seeds::COUPON, authority.key().as_ref(), b"pool"],
        bump,
        token::mint = mint,
        token::authority = coupon,
    )]
    pub pool_token_account: Account<'info, TokenAccount>,
    pub mint: Account<'info, Mint>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct RedeemCoupon<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    /// CHECK: coupon authority (used for PDA seed derivation)
    pub authority: AccountInfo<'info>,
    #[account(
        mut,
        seeds = [seeds::COUPON, authority.key().as_ref()],
        bump,
    )]
    pub coupon: Account<'info, Coupon>,
    #[account(
        mut,
        seeds = [seeds::COUPON, authority.key().as_ref(), b"pool"],
        bump,
        token::mint = mint,
        token::authority = coupon,
    )]
    pub pool_token_account: Account<'info, TokenAccount>,
    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = user,
    )]
    pub user_token_account: Account<'info, TokenAccount>,
    pub mint: Account<'info, Mint>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    pub system_program: Program<'info, System>,
}
