use anchor_lang::prelude::*;
use anchor_spl::token::{Transfer, transfer, MintTo, mint_to, Burn, burn};
use anchor_spl::associated_token::AssociatedToken;
use mpl_token_metadata::instruction::create_metadata_accounts_v3;
use solana_program::keccak::hashv;
pub mod seeds;
pub mod context;

declare_id!("EUDxXt8kG9o76MWGwyZCGUL1oPPnoNvmAprdZskjyBTh");

#[program]
pub mod solana_contract {

    use super::*;

    pub fn initialize(ctx: Context<context::Initialize>) -> Result<()> {
        msg!("solana_contract initialized from: {:?}", ctx.program_id);
        Ok(())
    }

    pub fn create_token_and_nft(
        ctx: Context<context::CreateTokenAndNFT>,
        name: String,
        symbol: String,
        uri: String,
    ) -> Result<()> {
        anchor_spl::token::initialize_mint(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            InitializeMint {
                mint: ctx.accounts.mint.to_account_info(),
                rent: ctx.accounts.rent.to_account_info(),
            },
        ), 0, &ctx.accounts.authority.key(), Some(&ctx.accounts.authority.key()))?;

        anchor_spl::token::mint_to(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            MintTo {
                mint: ctx.accounts.mint.to_account_info(),
                to: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), 1)?;

        let accounts = vec![
            ctx.accounts.metadata.to_account_info(),
            ctx.accounts.mint.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.token_metadata_program.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
            ctx.accounts.rent.to_account_info(),
        ];

        let ix = create_metadata_accounts_v3(
            ctx.accounts.token_metadata_program.key(),
            ctx.accounts.metadata.key(),
            ctx.accounts.mint.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            name,
            symbol,
            uri,
            None,
            1,
            true,
            false,
            None,
            None,
            None,
        );

        solana_program::program::invoke(&ix, &accounts)?;

        Ok(())
    }

    pub fn create_token(
        ctx: Context<context::CreateToken>,
        name: String,
        symbol: String,
        uri: String,
        supply: u64,
    ) -> Result<()> {
        anchor_spl::token::initialize_mint(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            InitializeMint {
                mint: ctx.accounts.mint.to_account_info(),
                rent: ctx.accounts.rent.to_account_info(),
            },
        ), 9, &ctx.accounts.authority.key(), Some(&ctx.accounts.authority.key()))?;

        anchor_spl::token::mint_to(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            MintTo {
                mint: ctx.accounts.mint.to_account_info(),
                to: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), supply)?;

        let accounts = vec![
            ctx.accounts.metadata.to_account_info(),
            ctx.accounts.mint.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.token_metadata_program.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
            ctx.accounts.rent.to_account_info(),
        ];

        let ix = create_metadata_accounts_v3(
            ctx.accounts.token_metadata_program.key(),
            ctx.accounts.metadata.key(),
            ctx.accounts.mint.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            name,
            symbol,
            uri,
            None,
            0,
            true,
            false,
            None,
            None,
            None,
        );

        solana_program::program::invoke(&ix, &accounts)?;

        Ok(())
    }

    pub fn create_sft(
        ctx: Context<context::CreateSFT>,
        name: String,
        symbol: String,
        uri: String,
        supply: u64,
    ) -> Result<()> {
        anchor_spl::token::initialize_mint(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            InitializeMint {
                mint: ctx.accounts.mint.to_account_info(),
                rent: ctx.accounts.rent.to_account_info(),
            },
        ), 0, &ctx.accounts.authority.key(), Some(&ctx.accounts.authority.key()))?;

        anchor_spl::token::mint_to(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            MintTo {
                mint: ctx.accounts.mint.to_account_info(),
                to: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), supply)?;

        let accounts = vec![
            ctx.accounts.metadata.to_account_info(),
            ctx.accounts.mint.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.token_metadata_program.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
            ctx.accounts.rent.to_account_info(),
        ];

        let ix = create_metadata_accounts_v3(
            ctx.accounts.token_metadata_program.key(),
            ctx.accounts.metadata.key(),
            ctx.accounts.mint.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            name,
            symbol,
            uri,
            None,
            0,
            true,
            true,
            None,
            None,
            None,
        );

        solana_program::program::invoke(&ix, &accounts)?;

        Ok(())
    }

    pub fn mint_to(
        ctx: Context<context::MintTokens>,
        amount: u64,
    ) -> Result<()> {
        mint_to(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            MintTo {
                mint: ctx.accounts.mint.to_account_info(),
                to: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), amount)?;
        Ok(())
    }

    pub fn burn_tokens(
        ctx: Context<context::BurnTokens>,
        amount: u64,
    ) -> Result<()> {
        burn(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Burn {
                mint: ctx.accounts.mint.to_account_info(),
                from: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), amount)?;
        Ok(())
    }

    pub fn deposit(ctx: Context<context::DepositAccounts>, amount: u64) -> Result<()> {
        let transaction = Transfer {
            from: ctx.accounts.sender_token_account.to_account_info(),
            to: ctx.accounts.program_token_account.to_account_info(),
            authority: ctx.accounts.signer.to_account_info(),
        };
        let cpi_ctx = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            transaction,
        );
        transfer(cpi_ctx, amount)?;
        Ok(())
    }

    pub fn withdraw(ctx: Context<context::WithdrawAccounts>, amount: u64) -> Result<()> {
        let transaction = Transfer {
            from: ctx.accounts.program_token_account.to_account_info(),
            to: ctx.accounts.receiver_token_account.to_account_info(),
            authority: ctx.accounts.token_manager.to_account_info(),
        };
        let bump: u8 = ctx.bumps.token_manager;
        let seeds = &[seeds::TOKEN_MANAGER, &[bump]];
        let signer = &[&seeds[..]];
        let cpi_ctx = CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            transaction,
            signer,
        );
        transfer(cpi_ctx, amount)?;
        Ok(())
    }

    pub fn list_good(
        ctx: Context<context::ListGood>,
        price: u64,
        listed_amount: u64,
    ) -> Result<()> {
        let listing = &mut ctx.accounts.listing;
        listing.seller = ctx.accounts.seller.key();
        listing.mint = ctx.accounts.mint.key();
        listing.payment_mint = ctx.accounts.payment_mint.key();
        listing.price = price;
        listing.listed_amount = listed_amount;
        listing.status = 0;
        listing.created_at = Clock::get()?.unix_timestamp;

        anchor_spl::token::transfer(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.seller_token_account.to_account_info(),
                to: ctx.accounts.escrow_token_account.to_account_info(),
                authority: ctx.accounts.seller.to_account_info(),
            },
        ), listed_amount)?;

        Ok(())
    }

    pub fn buy_good(
        ctx: Context<context::BuyGood>,
    ) -> Result<()> {
        let listing = &mut ctx.accounts.listing;
        require!(listing.status == 0, ErrorCode::ListingNotActive);

        anchor_spl::token::transfer(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.buyer_payment_token_account.to_account_info(),
                to: ctx.accounts.seller_payment_token_account.to_account_info(),
                authority: ctx.accounts.buyer.to_account_info(),
            },
        ), listing.price)?;

        let seeds = &[
            seeds::LISTING,
            listing.seller.as_ref(),
            listing.mint.as_ref(),
            &[ctx.bumps.listing],
        ];
        let signer = &[&seeds[..]];
        anchor_spl::token::transfer(CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.escrow_token_account.to_account_info(),
                to: ctx.accounts.buyer_receive_token_account.to_account_info(),
                authority: ctx.accounts.listing.to_account_info(),
            },
            signer,
        ), listing.listed_amount)?;

        listing.status = 1;

        Ok(())
    }

    pub fn delist_good(
        ctx: Context<context::DelistGood>,
    ) -> Result<()> {
        let listing = &mut ctx.accounts.listing;
        require!(listing.status == 0, ErrorCode::ListingNotActive);

        let seeds = &[
            seeds::LISTING,
            listing.seller.as_ref(),
            listing.mint.as_ref(),
            &[ctx.bumps.listing],
        ];
        let signer = &[&seeds[..]];
        anchor_spl::token::transfer(CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.escrow_token_account.to_account_info(),
                to: ctx.accounts.seller_token_account.to_account_info(),
                authority: ctx.accounts.listing.to_account_info(),
            },
            signer,
        ), listing.listed_amount)?;

        listing.status = 2;

        Ok(())
    }

    pub fn create_activity(
        ctx: Context<context::CreateActivity>,
        start_time: i64,
        end_time: i64,
        entry_fee: u64,
        reward_pcts: [u16; 3],
    ) -> Result<()> {
        require!(start_time < end_time, ActivityError::InvalidTimeRange);

        let activity = &mut ctx.accounts.activity;
        activity.authority = ctx.accounts.authority.key();
        activity.start_time = start_time;
        activity.end_time = end_time;
        activity.entry_fee = entry_fee;
        activity.reward_pcts = reward_pcts;
        activity.total_pool = 0;
        activity.participant_count = 0;

        Ok(())
    }

    pub fn participate_activity(
        ctx: Context<context::ParticipateActivity>,
    ) -> Result<()> {
        let activity = &mut ctx.accounts.activity;
        let clock = Clock::get()?;

        require!(clock.unix_timestamp >= activity.start_time, ActivityError::ActivityNotStarted);
        require!(clock.unix_timestamp <= activity.end_time, ActivityError::ActivityEnded);

        anchor_spl::token::transfer(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.participant_token_account.to_account_info(),
                to: ctx.accounts.pool_token_account.to_account_info(),
                authority: ctx.accounts.participant.to_account_info(),
            },
        ), activity.entry_fee)?;

        activity.total_pool = activity.total_pool.checked_add(activity.entry_fee).unwrap();
        activity.participant_count = activity.participant_count.checked_add(1).unwrap();

        Ok(())
    }

    pub fn set_merkle_root(
        ctx: Context<context::ClaimReward>,
        root: [u8; 32],
    ) -> Result<()> {
        let activity = &mut ctx.accounts.activity;
        activity.merkle_root = root;
        Ok(())
    }

    pub fn claim_reward(
        ctx: Context<context::ClaimReward>,
        rank: u8,
        proof: Vec<[u8; 32]>,
    ) -> Result<()> {
        let activity = &mut ctx.accounts.activity;

        let leaf = hashv(&[ctx.accounts.winner.key().as_ref(), &[rank]]).to_bytes();
        let mut computed_hash = leaf;

        for proof_element in &proof {
            let mut combined = [0u8; 64];
            if computed_hash <= *proof_element {
                combined[..32].copy_from_slice(&computed_hash);
                combined[32..].copy_from_slice(proof_element);
            } else {
                combined[..32].copy_from_slice(proof_element);
                combined[32..].copy_from_slice(&computed_hash);
            }
            computed_hash = hashv(&[&combined]).to_bytes();
        }

        require!(computed_hash == activity.merkle_root, ActivityError::InvalidProof);

        let reward_pct = if rank == 1 { activity.reward_pcts[0] }
            else if rank == 2 { activity.reward_pcts[1] }
            else { activity.reward_pcts[2] };

        let reward_amount = (activity.total_pool * reward_pct as u64) / 10000;

        let seeds = &[
            seeds::ACTIVITY,
            activity.authority.as_ref(),
            &[ctx.bumps.activity],
        ];
        let signer = &[&seeds[..]];
        anchor_spl::token::transfer(CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.pool_token_account.to_account_info(),
                to: ctx.accounts.winner_token_account.to_account_info(),
                authority: ctx.accounts.activity.to_account_info(),
            },
            signer,
        ), reward_amount)?;

        Ok(())
    }

    pub fn set_upline(
        ctx: Context<context::SetUpline>,
    ) -> Result<()> {
        let graph = &mut ctx.accounts.commission_graph;
        require!(graph.upline == Pubkey::default(), CommissionError::UplineAlreadySet);
        require!(ctx.accounts.target.key() != ctx.accounts.upline.key(), CommissionError::SelfReferral);

        graph.upline = ctx.accounts.upline.key();
        graph.level = 1;

        Ok(())
    }

    pub fn distribute_commission(
        ctx: Context<context::DistributeCommission>,
        sale_amount: u64,
    ) -> Result<()> {
        let commission_graph = &ctx.accounts.commission_graph;
        require!(commission_graph.upline != Pubkey::default(), CommissionError::UplineNotSet);

        // 10% commission rate (1000 basis points)
        let commission_amount = sale_amount * 1000 / 10000;

        let upline_token_account = &ctx.remaining_accounts[0];

        anchor_spl::token::transfer(CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.seller_token_account.to_account_info(),
                to: upline_token_account.to_account_info(),
                authority: ctx.accounts.seller.to_account_info(),
            },
        ), commission_amount)?;

        Ok(())
    }

    pub fn create_coupon(
        ctx: Context<context::CreateCoupon>,
        discount_amount: u64,
        max_uses: u64,
        merkle_root: [u8; 32],
        expiry: i64,
    ) -> Result<()> {
        let coupon = &mut ctx.accounts.coupon;
        coupon.authority = ctx.accounts.authority.key();
        coupon.mint = ctx.accounts.mint.key();
        coupon.discount_amount = discount_amount;
        coupon.max_uses = max_uses;
        coupon.total_redeemed = 0;
        coupon.merkle_root = merkle_root;
        coupon.expiry = expiry;
        Ok(())
    }

    pub fn redeem_coupon(
        ctx: Context<context::RedeemCoupon>,
        proof: Vec<[u8; 32]>,
    ) -> Result<()> {
        let coupon = &mut ctx.accounts.coupon;
        let clock = Clock::get()?;

        require!(clock.unix_timestamp <= coupon.expiry, CouponError::CouponExpired);
        require!(coupon.total_redeemed < coupon.max_uses, CouponError::MaxUsesReached);

        let leaf = solana_program::keccak::hashv(&[ctx.accounts.user.key().as_ref()]).to_bytes();
        let mut computed_hash = leaf;

        for proof_element in &proof {
            let mut combined = [0u8; 64];
            if computed_hash <= *proof_element {
                combined[..32].copy_from_slice(&computed_hash);
                combined[32..].copy_from_slice(proof_element);
            } else {
                combined[..32].copy_from_slice(proof_element);
                combined[32..].copy_from_slice(&computed_hash);
            }
            computed_hash = solana_program::keccak::hashv(&[&combined]).to_bytes();
        }

        require!(computed_hash == coupon.merkle_root, CouponError::InvalidCouponProof);

        let seeds = &[
            seeds::COUPON,
            ctx.accounts.authority.key().as_ref(),
            &[ctx.bumps.coupon],
        ];
        let signer = &[&seeds[..]];

        anchor_spl::token::transfer(CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.pool_token_account.to_account_info(),
                to: ctx.accounts.user_token_account.to_account_info(),
                authority: ctx.accounts.coupon.to_account_info(),
            },
            signer,
        ), coupon.discount_amount)?;

        coupon.total_redeemed = coupon.total_redeemed.checked_add(1).unwrap();

        Ok(())
    }
}

#[error_code]
pub enum ErrorCode {
    #[msg("Listing is not active")]
    ListingNotActive,
}

#[error_code]
pub enum ActivityError {
    #[msg("Invalid time range")]
    InvalidTimeRange,
    #[msg("Activity not started")]
    ActivityNotStarted,
    #[msg("Activity ended")]
    ActivityEnded,
    #[msg("Invalid merkle proof")]
    InvalidProof,
}

#[error_code]
pub enum CommissionError {
    #[msg("Upline already set")]
    UplineAlreadySet,
    #[msg("Cannot refer self")]
    SelfReferral,
    #[msg("Upline not set")]
    UplineNotSet,
}

#[error_code]
pub enum CouponError {
    #[msg("Coupon expired")]
    CouponExpired,
    #[msg("Maximum uses reached")]
    MaxUsesReached,
    #[msg("Invalid coupon proof")]
    InvalidCouponProof,
}
