use anchor_lang::prelude::*;

declare_id!("FJBs8eFtXpdnGWcnihfqRJpJTx8Loak4f1pUUAS6nFxn");

#[program]
pub mod solana_contract {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        msg!("solana_contract initialized from: {:?}", ctx.program_id);
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(init, payer = admin, space = 8 + 8)]
    token_manager: AccountInfo<'info>,
    #[account(mut)]
    admin: Signer<'info>,
    system_program: Program<'info, System>,
}

