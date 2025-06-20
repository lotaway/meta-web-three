use anchor_lang::prelude::*;
use anchor_spl::token::{Transfer, transfer};
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

    pub fn withdraw(ctx: Context<Wcontext::ithdrawAccounts>, amount: u64) -> Result<()> {
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