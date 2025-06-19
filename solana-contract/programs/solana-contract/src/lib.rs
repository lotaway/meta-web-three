use anchor_lang::prelude::{borsh::de, *};
use anchor_spl::token::{Token, TokenAccount, Transfer, transfer};
pub mod seeds;

declare_id!("EUDxXt8kG9o76MWGwyZCGUL1oPPnoNvmAprdZskjyBTh");

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
    /// The PDA account that manages token operations.
    /// CHECK: This is a PDA account that serves as the authority for program token operations.
    /// Safety is ensured through PDA derivation using TOKEN_MANAGER seeds and bump.
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

    /// CHECK: This is the mint account for the token being managed.
    /// It is safe because it's only used as a reference for PDA derivation
    /// and is validated by the Token Program in program_token_account constraints.
    #[account(
        init_if_needed,
        payer = signer,
        mint::decimals = 9,
        mint::authority = signer,
    )]
    token_mint_account: Account<'info, anchor_spl::token::Mint>,

    #[account(mut)]
    signer: Signer<'info>,

    system_program: Program<'info, System>,

    token_program: Program<'info, Token>,
    
    rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct DepositAccounts<'info> {
    /// The PDA account that manages token operations.
    /// CHECK: This is a PDA account verified by seeds and bump constraint.
    /// It is safe because it can only be derived from the program's TOKEN_MANAGER seed.
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

    /// CHECK: This is the source token account owned by the signer.
    /// It is safe because:
    /// 1. The signer must sign the transaction (enforced by Anchor)
    /// 2. The Token Program validates ownership and balance during transfer
    #[account(mut)]
    sender_token_account: AccountInfo<'info>,

    /// CHECK: This is the mint account that represents the token type being transferred.
    /// It is safe because it's validated through the token constraints on program_token_account.
    token_mint_account: AccountInfo<'info>,

    #[account(mut)]
    signer: Signer<'info>,

    system_program: Program<'info, System>,

    token_program: Program<'info, Token>,
    
    rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct WithdrawAccounts<'info> {
    /// The PDA account that manages token operations.
    /// CHECK: This is a PDA account validated by seeds and bump constraint.
    /// Safety is guaranteed as it can only be derived from TOKEN_MANAGER seed and is used as a signing PDA.
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

    /// CHECK: This is the destination token account for withdrawal.
    /// It is safe because the Token Program validates during transfer:
    /// 1. The account is a valid token account
    /// 2. It has the correct mint
    /// 3. It can receive tokens
    #[account(mut)]
    recevier_token_account: AccountInfo<'info>,

    /// CHECK: This is the mint account for the token being withdrawn.
    /// It is safe because it's validated through the token constraints on program_token_account.
    token_mint_account: AccountInfo<'info>,

    #[account(mut)]
    signer: Signer<'info>,

    system_program: Program<'info, System>,

    token_program: Program<'info, Token>,
    
    rent: Sysvar<'info, Rent>,
}