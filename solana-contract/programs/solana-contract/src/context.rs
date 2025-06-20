use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount};

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

    #[account(
        init_if_needed,
        payer = signer,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        token::mint = token_mint_account,
        token::authority = token_manager,
        // space = TokenAccount::LEN,
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

    /// CHECK: Metaplex metadata PDA
    #[account(mut)]
    pub metadata: UncheckedAccount<'info>,

    /// CHECK: metaplex program
    pub token_metadata_program: UncheckedAccount<'info>,

    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, anchor_spl::associated_token::AssociatedToken>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
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
    pub token_manager: AccountInfo<'info>,

    #[account(
        mut,
        seeds=[seeds::PROGRAM_TOKEN_ACCOUNT, token_mint_account.key().as_ref()],
        bump,
        token::mint = token_mint_account,
        token::authority = token_manager,
    )]
    pub program_token_account: Account<'info, TokenAccount>,

    /// CHECK: This is the source token account owned by the signer.
    /// It is safe because:
    /// 1. The signer must sign the transaction (enforced by Anchor)
    /// 2. The Token Program validates ownership and balance during transfer
    #[account(mut)]
    pub sender_token_account: AccountInfo<'info>,

    /// CHECK: This is the mint account that represents the token type being transferred.
    /// It is safe because it's validated through the token constraints on program_token_account.
    pub token_mint_account: AccountInfo<'info>,

    #[account(mut)]
    pub signer: Signer<'info>,

    pub system_program: Program<'info, System>,

    pub token_program: Program<'info, Token>,
    
    pub rent: Sysvar<'info, Rent>,
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
    pub program_token_account: Account<'info, TokenAccount>,

    /// CHECK: This is the destination token account for withdrawal.
    /// It is safe because the Token Program validates during transfer:
    /// 1. The account is a valid token account
    /// 2. It has the correct mint
    /// 3. It can receive tokens
    #[account(mut)]
    pub recevier_token_account: AccountInfo<'info>,

    /// CHECK: This is the mint account for the token being withdrawn.
    /// It is safe because it's validated through the token constraints on program_token_account.
    pub token_mint_account: AccountInfo<'info>,

    #[account(mut)]
    pub signer: Signer<'info>,

    pub system_program: Program<'info, System>,

    pub token_program: Program<'info, Token>,
    
    pub rent: Sysvar<'info, Rent>,
}