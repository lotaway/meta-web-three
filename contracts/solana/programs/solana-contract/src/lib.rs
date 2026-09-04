#![allow(unexpected_cfgs)]

use anchor_lang::prelude::*;
use anchor_lang::solana_program::instruction::{AccountMeta, Instruction};
use anchor_lang::solana_program::program::invoke;
use anchor_spl::token::{Transfer, transfer, MintTo, Burn, burn, Mint, Token, TokenAccount};
use anchor_spl::associated_token::AssociatedToken;
use sha3::{Digest, Keccak256};
pub mod seeds;
pub mod context;
use context::*;

declare_id!("5Dk7wrWReetiNNcFbicN8ZFwXkrQeQu4LoMkhLJimGcP");

fn keccak_hashv(vals: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    for val in vals {
        hasher.update(val);
    }
    hasher.finalize().into()
}

fn create_metadata_v3_ix(
    program_id: Pubkey,
    metadata: Pubkey,
    mint: Pubkey,
    mint_authority: Pubkey,
    payer: Pubkey,
    update_authority: Pubkey,
    rent: Pubkey,
    name: String,
    symbol: String,
    uri: String,
    seller_fee_basis_points: u16,
    is_mutable: bool,
) -> Instruction {
    let accounts = vec![
        AccountMeta::new(metadata, false),
        AccountMeta::new_readonly(mint, false),
        AccountMeta::new_readonly(mint_authority, true),
        AccountMeta::new(payer, true),
        AccountMeta::new_readonly(update_authority, true),
        AccountMeta::new_readonly(Pubkey::default(), false),
        AccountMeta::new_readonly(rent, false),
    ];
    let mut data = vec![33u8];
    data.extend_from_slice(&(name.len() as u32).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&(symbol.len() as u32).to_le_bytes());
    data.extend_from_slice(symbol.as_bytes());
    data.extend_from_slice(&(uri.len() as u32).to_le_bytes());
    data.extend_from_slice(uri.as_bytes());
    data.extend_from_slice(&seller_fee_basis_points.to_le_bytes());
    data.push(0);
    data.push(0);
    data.push(0);
    data.push(is_mutable as u8);
    data.push(0);
    Instruction { program_id, accounts, data }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init_if_needed,
        payer = signer,
        seeds=[seeds::TOKEN_MANAGER],
        bump,
        space = 8,
    )]
    /// CHECK: PDA used as token account authority
    token_manager: UncheckedAccount<'info>,

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
    /// CHECK: token metadata PDA for mint
    pub metadata: UncheckedAccount<'info>,

    /// CHECK: token metadata program (mpl-token-metadata)
    pub token_metadata_program: UncheckedAccount<'info>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
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
    pub associated_token_program: Program<'info, AssociatedToken>,
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
    pub listing: Box<Account<'info, Listing>>,

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
    pub buyer_receive_token_account: Box<Account<'info, TokenAccount>>,

    #[account(
        mut,
        seeds = [seeds::LISTING_ESCROW, mint.key().as_ref()],
        bump,
        token::mint = mint,
        token::authority = listing,
    )]
    pub escrow_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
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
    /// CHECK: token metadata PDA for mint
    pub metadata: UncheckedAccount<'info>,

    /// CHECK: token metadata program (mpl-token-metadata)
    pub token_metadata_program: UncheckedAccount<'info>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
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
    /// CHECK: token metadata PDA for mint
    pub metadata: UncheckedAccount<'info>,

    /// CHECK: token metadata program (mpl-token-metadata)
    pub token_metadata_program: UncheckedAccount<'info>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
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
    pub associated_token_program: Program<'info, AssociatedToken>,
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
    /// CHECK: PDA used as token account authority
    pub token_manager: UncheckedAccount<'info>,

    #[account(
        mut,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        token::mint = token_mint_account,
        token::authority = token_manager,
    )]
    pub program_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    /// CHECK: validated via token program CPI
    pub sender_token_account: UncheckedAccount<'info>,

    /// CHECK: used as seed for PDA derivation
    pub token_mint_account: UncheckedAccount<'info>,

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
    /// CHECK: PDA used as token account authority
    pub token_manager: UncheckedAccount<'info>,

    #[account(
        mut,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        token::mint = token_mint_account,
        token::authority = token_manager,
    )]
    pub program_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    /// CHECK: validated via token program CPI
    pub receiver_token_account: UncheckedAccount<'info>,

    /// CHECK: used as seed for PDA derivation
    pub token_mint_account: UncheckedAccount<'info>,

    #[account(mut)]
    pub signer: Signer<'info>,

    pub system_program: Program<'info, System>,
    pub token_program: Program<'info, Token>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct CreateActivity<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        init,
        payer = authority,
        seeds = [seeds::ACTIVITY, authority.key().as_ref()],
        bump,
        space = Activity::LEN,
    )]
    pub activity: Account<'info, Activity>,

    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct ParticipateActivity<'info> {
    #[account(mut)]
    pub participant: Signer<'info>,

    #[account(
        mut,
        seeds = [seeds::ACTIVITY, activity.authority.key().as_ref()],
        bump,
    )]
    pub activity: Account<'info, Activity>,

    #[account(
        init_if_needed,
        payer = participant,
        seeds = [seeds::ACTIVITY, activity.authority.key().as_ref(), b"pool"],
        bump,
        token::mint = mint,
        token::authority = activity,
    )]
    pub pool_token_account: Account<'info, TokenAccount>,

    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = participant,
    )]
    pub participant_token_account: Account<'info, TokenAccount>,

    pub mint: Account<'info, Mint>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct ClaimReward<'info> {
    #[account(mut)]
    pub winner: Signer<'info>,

    #[account(
        mut,
        seeds = [seeds::ACTIVITY, activity.authority.key().as_ref()],
        bump,
    )]
    pub activity: Account<'info, Activity>,

    #[account(
        mut,
        seeds = [seeds::ACTIVITY, activity.authority.key().as_ref(), b"pool"],
        bump,
        token::mint = mint,
        token::authority = activity,
    )]
    pub pool_token_account: Account<'info, TokenAccount>,

    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = winner,
    )]
    pub winner_token_account: Account<'info, TokenAccount>,

    pub mint: Account<'info, Mint>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
}

#[derive(Accounts)]
pub struct DelistGood<'info> {
    #[account(mut)]
    pub seller: Signer<'info>,

    #[account(
        mut,
        seeds = [seeds::LISTING, listing.seller.key().as_ref(), listing.mint.key().as_ref()],
        bump,
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
        mut,
        seeds = [seeds::LISTING_ESCROW, mint.key().as_ref()],
        bump,
        token::mint = mint,
        token::authority = listing,
    )]
    pub escrow_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct SetUpline<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    /// CHECK: upline wallet address
    pub upline: UncheckedAccount<'info>,
    #[account(
        init,
        payer = signer,
        seeds = [seeds::COMMISSION, target.key().as_ref()],
        bump,
        space = CommissionGraph::LEN,
    )]
    pub commission_graph: Account<'info, CommissionGraph>,
    /// CHECK: target wallet address
    pub target: UncheckedAccount<'info>,
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
    /// CHECK: verified as upline's token account via commission_graph
    #[account(mut)]
    pub upline_token_account: UncheckedAccount<'info>,
    pub payment_mint: Account<'info, Mint>,
    pub token_program: Program<'info, Token>,
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
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct RedeemCoupon<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    /// CHECK: coupon authority (used for PDA seed derivation)
    pub authority: UncheckedAccount<'info>,
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
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}

#[program]
pub mod solana_contract {

    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        msg!("solana_contract initialized from: {:?}", ctx.program_id);
        Ok(())
    }

    pub fn create_token_and_nft(
        ctx: Context<CreateTokenAndNFT>,
        name: String,
        symbol: String,
        uri: String,
    ) -> Result<()> {
        anchor_spl::token::mint_to(CpiContext::new(
            ctx.accounts.token_program.key(),
            MintTo {
                mint: ctx.accounts.mint.to_account_info(),
                to: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), 1)?;

        let ix = create_metadata_v3_ix(
            ctx.accounts.token_metadata_program.key(),
            ctx.accounts.metadata.key(),
            ctx.accounts.mint.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.rent.key(),
            name,
            symbol,
            uri,
            1,
            false,
        );
        invoke(&ix, &[
            ctx.accounts.metadata.to_account_info(),
            ctx.accounts.mint.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.token_metadata_program.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
            ctx.accounts.rent.to_account_info(),
        ])?;

        Ok(())
    }

    pub fn create_token(
        ctx: Context<CreateToken>,
        name: String,
        symbol: String,
        uri: String,
        supply: u64,
    ) -> Result<()> {
        anchor_spl::token::mint_to(CpiContext::new(
            ctx.accounts.token_program.key(),
            MintTo {
                mint: ctx.accounts.mint.to_account_info(),
                to: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), supply)?;

        let ix = create_metadata_v3_ix(
            ctx.accounts.token_metadata_program.key(),
            ctx.accounts.metadata.key(),
            ctx.accounts.mint.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.rent.key(),
            name,
            symbol,
            uri,
            0,
            false,
        );
        invoke(&ix, &[
            ctx.accounts.metadata.to_account_info(),
            ctx.accounts.mint.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.token_metadata_program.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
            ctx.accounts.rent.to_account_info(),
        ])?;

        Ok(())
    }

    pub fn create_sft(
        ctx: Context<CreateSFT>,
        name: String,
        symbol: String,
        uri: String,
        supply: u64,
    ) -> Result<()> {
        anchor_spl::token::mint_to(CpiContext::new(
            ctx.accounts.token_program.key(),
            MintTo {
                mint: ctx.accounts.mint.to_account_info(),
                to: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), supply)?;

        let ix = create_metadata_v3_ix(
            ctx.accounts.token_metadata_program.key(),
            ctx.accounts.metadata.key(),
            ctx.accounts.mint.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.authority.key(),
            ctx.accounts.rent.key(),
            name,
            symbol,
            uri,
            0,
            true,
        );
        invoke(&ix, &[
            ctx.accounts.metadata.to_account_info(),
            ctx.accounts.mint.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.authority.to_account_info(),
            ctx.accounts.token_metadata_program.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
            ctx.accounts.rent.to_account_info(),
        ])?;

        Ok(())
    }

    pub fn mint_to(
        ctx: Context<MintTokens>,
        amount: u64,
    ) -> Result<()> {
        anchor_spl::token::mint_to(CpiContext::new(
            ctx.accounts.token_program.key(),
            MintTo {
                mint: ctx.accounts.mint.to_account_info(),
                to: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), amount)?;
        Ok(())
    }

    pub fn burn_tokens(
        ctx: Context<BurnTokens>,
        amount: u64,
    ) -> Result<()> {
        burn(CpiContext::new(
            ctx.accounts.token_program.key(),
            Burn {
                mint: ctx.accounts.mint.to_account_info(),
                from: ctx.accounts.token_account.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        ), amount)?;
        Ok(())
    }

    pub fn deposit(ctx: Context<DepositAccounts>, amount: u64) -> Result<()> {
        let transaction = Transfer {
            from: ctx.accounts.sender_token_account.to_account_info(),
            to: ctx.accounts.program_token_account.to_account_info(),
            authority: ctx.accounts.signer.to_account_info(),
        };
        let cpi_ctx = CpiContext::new(
            ctx.accounts.token_program.key(),
            transaction,
        );
        transfer(cpi_ctx, amount)?;
        Ok(())
    }

    pub fn withdraw(ctx: Context<WithdrawAccounts>, amount: u64) -> Result<()> {
        let transaction = Transfer {
            from: ctx.accounts.program_token_account.to_account_info(),
            to: ctx.accounts.receiver_token_account.to_account_info(),
            authority: ctx.accounts.token_manager.to_account_info(),
        };
        let bump: u8 = ctx.bumps.token_manager;
        let seeds = &[seeds::TOKEN_MANAGER, &[bump]];
        let signer = &[&seeds[..]];
        let cpi_ctx = CpiContext::new_with_signer(
            ctx.accounts.token_program.key(),
            transaction,
            signer,
        );
        transfer(cpi_ctx, amount)?;
        Ok(())
    }

    pub fn list_good(
        ctx: Context<ListGood>,
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
            ctx.accounts.token_program.key(),
            Transfer {
                from: ctx.accounts.seller_token_account.to_account_info(),
                to: ctx.accounts.escrow_token_account.to_account_info(),
                authority: ctx.accounts.seller.to_account_info(),
            },
        ), listed_amount)?;

        Ok(())
    }

    pub fn buy_good(
        ctx: Context<BuyGood>,
    ) -> Result<()> {
        let listing = &ctx.accounts.listing;
        require!(listing.status == 0, ErrorCode::ListingNotActive);
        let price = listing.price;
        let listed_amount = listing.listed_amount;
        let seller = listing.seller;
        let mint_key = listing.mint;
        let listing_info = ctx.accounts.listing.to_account_info();

        anchor_spl::token::transfer(CpiContext::new(
            ctx.accounts.token_program.key(),
            Transfer {
                from: ctx.accounts.buyer_payment_token_account.to_account_info(),
                to: ctx.accounts.seller_payment_token_account.to_account_info(),
                authority: ctx.accounts.buyer.to_account_info(),
            },
        ), price)?;

        let seeds = &[
            seeds::LISTING,
            seller.as_ref(),
            mint_key.as_ref(),
            &[ctx.bumps.listing],
        ];
        let signer = &[&seeds[..]];
        anchor_spl::token::transfer(CpiContext::new_with_signer(
            ctx.accounts.token_program.key(),
            Transfer {
                from: ctx.accounts.escrow_token_account.to_account_info(),
                to: ctx.accounts.buyer_receive_token_account.to_account_info(),
                authority: listing_info,
            },
            signer,
        ), listed_amount)?;

        ctx.accounts.listing.status = 1;

        Ok(())
    }

    pub fn delist_good(
        ctx: Context<DelistGood>,
    ) -> Result<()> {
        let listing = &ctx.accounts.listing;
        require!(listing.status == 0, ErrorCode::ListingNotActive);
        let listed_amount = listing.listed_amount;
        let seller = listing.seller;
        let mint_key = listing.mint;
        let listing_info = ctx.accounts.listing.to_account_info();

        let seeds = &[
            seeds::LISTING,
            seller.as_ref(),
            mint_key.as_ref(),
            &[ctx.bumps.listing],
        ];
        let signer = &[&seeds[..]];
        anchor_spl::token::transfer(CpiContext::new_with_signer(
            ctx.accounts.token_program.key(),
            Transfer {
                from: ctx.accounts.escrow_token_account.to_account_info(),
                to: ctx.accounts.seller_token_account.to_account_info(),
                authority: listing_info,
            },
            signer,
        ), listed_amount)?;

        ctx.accounts.listing.status = 2;

        Ok(())
    }

    pub fn create_activity(
        ctx: Context<CreateActivity>,
        start_time: i64,
        end_time: i64,
        entry_fee: u64,
        reward_pcts: [u16; 3],
    ) -> Result<()> {
        require!(start_time < end_time, ErrorCode::InvalidTimeRange);

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
        ctx: Context<ParticipateActivity>,
    ) -> Result<()> {
        let activity = &mut ctx.accounts.activity;
        let clock = Clock::get()?;

        require!(clock.unix_timestamp >= activity.start_time, ErrorCode::ActivityNotStarted);
        require!(clock.unix_timestamp <= activity.end_time, ErrorCode::ActivityEnded);

        anchor_spl::token::transfer(CpiContext::new(
            ctx.accounts.token_program.key(),
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
        ctx: Context<ClaimReward>,
        root: [u8; 32],
    ) -> Result<()> {
        let activity = &mut ctx.accounts.activity;
        activity.merkle_root = root;
        Ok(())
    }

    pub fn claim_reward(
        ctx: Context<ClaimReward>,
        rank: u8,
        proof: Vec<[u8; 32]>,
    ) -> Result<()> {
        let activity = &ctx.accounts.activity;
        let merkle_root = activity.merkle_root;
        let reward_pcts = activity.reward_pcts;
        let total_pool = activity.total_pool;
        let authority = activity.authority;
        let activity_info = ctx.accounts.activity.to_account_info();

        let leaf = keccak_hashv(&[ctx.accounts.winner.key().as_ref(), &[rank]]);
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
            computed_hash = keccak_hashv(&[&combined]);
        }

        require!(computed_hash == merkle_root, ErrorCode::InvalidProof);

        let reward_pct = if rank == 1 { reward_pcts[0] }
            else if rank == 2 { reward_pcts[1] }
            else { reward_pcts[2] };

        let reward_amount = (total_pool * reward_pct as u64) / 10000;

        let seeds = &[
            seeds::ACTIVITY,
            authority.as_ref(),
            &[ctx.bumps.activity],
        ];
        let signer = &[&seeds[..]];
        anchor_spl::token::transfer(CpiContext::new_with_signer(
            ctx.accounts.token_program.key(),
            Transfer {
                from: ctx.accounts.pool_token_account.to_account_info(),
                to: ctx.accounts.winner_token_account.to_account_info(),
                authority: activity_info,
            },
            signer,
        ), reward_amount)?;

        Ok(())
    }

    pub fn set_upline(
        ctx: Context<SetUpline>,
    ) -> Result<()> {
        let graph = &mut ctx.accounts.commission_graph;
        require!(graph.upline == Pubkey::default(), ErrorCode::UplineAlreadySet);
        require!(ctx.accounts.target.key() != ctx.accounts.upline.key(), ErrorCode::SelfReferral);

        graph.upline = ctx.accounts.upline.key();
        graph.level = 1;

        Ok(())
    }

    pub fn distribute_commission(
        ctx: Context<DistributeCommission>,
        sale_amount: u64,
    ) -> Result<()> {
        let commission_graph = &ctx.accounts.commission_graph;
        require!(commission_graph.upline != Pubkey::default(), ErrorCode::UplineNotSet);

        let commission_amount = sale_amount * 1000 / 10000;

        anchor_spl::token::transfer(CpiContext::new(
            ctx.accounts.token_program.key(),
            Transfer {
                from: ctx.accounts.seller_token_account.to_account_info(),
                to: ctx.accounts.upline_token_account.to_account_info(),
                authority: ctx.accounts.seller.to_account_info(),
            },
        ), commission_amount)?;

        Ok(())
    }

    pub fn create_coupon(
        ctx: Context<CreateCoupon>,
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
        ctx: Context<RedeemCoupon>,
        proof: Vec<[u8; 32]>,
    ) -> Result<()> {
        let coupon = &ctx.accounts.coupon;
        let clock = Clock::get()?;

        require!(clock.unix_timestamp <= coupon.expiry, ErrorCode::CouponExpired);
        require!(coupon.total_redeemed < coupon.max_uses, ErrorCode::MaxUsesReached);

        let discount_amount = coupon.discount_amount;
        let merkle_root = coupon.merkle_root;
        let coupon_info = ctx.accounts.coupon.to_account_info();

        let leaf = keccak_hashv(&[ctx.accounts.user.key().as_ref()]);
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
            computed_hash = keccak_hashv(&[&combined]);
        }

        require!(computed_hash == merkle_root, ErrorCode::InvalidCouponProof);

        let authority_key = ctx.accounts.authority.key();
        let seeds = &[
            seeds::COUPON,
            authority_key.as_ref(),
            &[ctx.bumps.coupon],
        ];
        let signer = &[&seeds[..]];

        anchor_spl::token::transfer(CpiContext::new_with_signer(
            ctx.accounts.token_program.key(),
            Transfer {
                from: ctx.accounts.pool_token_account.to_account_info(),
                to: ctx.accounts.user_token_account.to_account_info(),
                authority: coupon_info,
            },
            signer,
        ), discount_amount)?;

        ctx.accounts.coupon.total_redeemed = ctx.accounts.coupon.total_redeemed.checked_add(1).unwrap();

        Ok(())
    }
}

#[error_code]
pub enum ErrorCode {
    #[msg("Listing is not active")]
    ListingNotActive,
    #[msg("Invalid time range")]
    InvalidTimeRange,
    #[msg("Activity not started")]
    ActivityNotStarted,
    #[msg("Activity ended")]
    ActivityEnded,
    #[msg("Invalid merkle proof")]
    InvalidProof,
    #[msg("Upline already set")]
    UplineAlreadySet,
    #[msg("Cannot refer self")]
    SelfReferral,
    #[msg("Upline not set")]
    UplineNotSet,
    #[msg("Coupon expired")]
    CouponExpired,
    #[msg("Maximum uses reached")]
    MaxUsesReached,
    #[msg("Invalid coupon proof")]
    InvalidCouponProof,
}
