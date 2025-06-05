use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount, Transfer, transfer};
pub mod seeds;

declare_id!("FJBs8eFtXpdnGWcnihfqRJpJTx8Loak4f1pUUAS6nFxn");

#[program]
pub mod solana_contract {

    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        msg!("solana_contract initialized from: {:?}", ctx.program_id);
        Ok(())
    }

    pub fn deposit(ctx: Context<DepositAccounts>, amount: u64) -> Result<()> {
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

    pub fn withdraw(ctx: Context<WithdrawAccounts>, amount: u64) -> Result<()> {
        let transaction = Transfer {
            from: ctx.accounts.program_token_account.to_account_info(),
            to: ctx.accounts.recevier_token_account.to_account_info(),
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
    token_manager: AccountInfo<'info>,

    #[account(
        init_if_needed,
        payer = signer,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        space = TokenAccount::LEN,
    )]
    program_token_account: Account<'info, TokenAccount>,

    token_mint_account: AccountInfo<'info>,

    #[account(mut)]
    signer: Signer<'info>,

    system_program: Program<'info, System>,

    token_program: Program<'info, Token>,
    
    rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct DepositAccounts<'info> {

    #[account(
        mut, 
        seeds=[seeds::TOKEN_MANAGER], 
        bump,
    )]
    token_manager: AccountInfo<'info>,

    #[account(
        mut,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        token::mint = token_mint_account,
        token::authority = token_manager,
    )]
    program_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    sender_token_account: AccountInfo<'info>,

    token_mint_account: AccountInfo<'info>,

    #[account(mut)]
    signer: Signer<'info>,

    system_program: Program<'info, System>,

    token_program: Program<'info, Token>,
    
    rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct WithdrawAccounts<'info> {

    #[account(
        mut, 
        seeds=[seeds::TOKEN_MANAGER], 
        bump,
    )]
    token_manager: AccountInfo<'info>,

    #[account(
        mut,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        token::mint = token_mint_account,
        token::authority = token_manager,
    )]
    program_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    recevier_token_account: AccountInfo<'info>,

    token_mint_account: AccountInfo<'info>,

    #[account(mut)]
    signer: Signer<'info>,

    system_program: Program<'info, System>,

    token_program: Program<'info, Token>,
    
    rent: Sysvar<'info, Rent>,
}