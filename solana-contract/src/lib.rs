use std::borrow::BorrowMut;

use solana_program::{account_info::{ next_account_info, AccountInfo}, entrypoint::{self, ProgramResult}, msg, pubkey::Pubkey};

entrypoint!(process_instruction);
fn process_instruction(program_id: &Pubkey, accounts: &[AccountInfo], instruction_data: &[u8]) -> ProgramResult {

    let account = next_account_info(accounts.iter().borrow_mut()).expect("No account found");

    if account.owner != program_id {
        msg!("Account owner is not the program");
        return Err(solana_program::program_error::ProgramError::IncorrectProgramId);
    }

    msg!("Hello, world!, {}, {:?}, {:?}", program_id, accounts, instruction_data);
    Ok(())
}
