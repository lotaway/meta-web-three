use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use crate::hashable::Hashable;
use crate::lib::{BlockHash, difficulty_bytes_as_u128, now, u128_bytes, u32_bytes, u64_bytes};
use crate::Transaction;
use crate::transaction::Output;

#[derive(Clone)]
pub struct Block {
    pub index: usize,
    pub timestamp: u128,
    pub prev_block_hash: BlockHash,
    pub hash: BlockHash,
    pub nonce: u64,
    pub transactions: Vec<Transaction>,
    pub difficulty: u128,
}

impl Hashable for Block {
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::from_iter(u32_bytes(&(self.index as u32)));
        bytes.extend(&u128_bytes(&self.timestamp));
        bytes.extend(&self.prev_block_hash);
        bytes.extend(&u64_bytes(&self.nonce));
        bytes.extend(self.transactions.bytes());
        bytes.extend(&u128_bytes(&self.difficulty));
        bytes
    }
}

impl Debug for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block[{}]: {} at: {} with: {} nonce: {}", &self.index, &hex::encode(&self.hash), &self.timestamp, &self.transactions.len(), &self.nonce)
    }
}

impl Block {
    pub fn new(index: usize, timestamp: u128, prev_block_hash: BlockHash, nonce: u64, transactions: Vec<Transaction>, difficulty: u128) -> Self {
        let mut block = Self {
            index,
            timestamp,
            prev_block_hash,
            hash: vec![0; 0],
            nonce,
            transactions,
            difficulty,
        };
        block.hash = block.hash();
        block
    }

    pub fn mine(&mut self) {
        for n in 0..u64::MAX {
            self.nonce = n;
            let hash = self.hash();
            if Block::check_difficulty(&hash, self.difficulty) {
                self.hash = hash;
                break;
            }
        }
    }

    pub fn check_difficulty(hash: &BlockHash, difficulty: u128) -> bool {
        difficulty < difficulty_bytes_as_u128(&hash)
    }
}

pub enum BlockValidationErr {
    MismatchedIndex,
    InvalidHash,
    AChronologicalTimestamp,
    MismatchedPreviousHash,
    InvalidGenesisBlockFormat,
    InvalidInput,
    InsufficientInputVal,
    InvalidCoinbaseTransaction,
}

#[derive(Debug)]
pub struct BlockChain {
    pub blocks: Vec<Block>,
    pub unspent_outputs: HashSet<BlockHash>,
}

impl BlockChain {
    pub fn new() -> Self {
        Self { blocks: Vec::new(), unspent_outputs: HashSet::new() }
    }

    pub fn origin_hash() -> BlockHash {
        vec![0; 32]
    }

    pub fn init(&mut self) -> Result<(), BlockValidationErr> {
        let prev_hash = BlockChain::origin_hash();
        let nonce = 0;
        let transactions = vec![
            Transaction {
                inputs: Vec::new(),
                outputs: vec![
                    Output {
                        to: String::from("creator"),
                        value: 50,
                    },
                    Output {
                        to: String::from("genesis"),
                        value: 25,
                    },
                ],
            }
        ];
        let difficulty = 0x000FFFFFFFFFFFFFFFFFFFFFFFFFFFF;
        let genesis_block = Block::new(0, now(), prev_hash, nonce, transactions, difficulty);
        self.update_with(genesis_block)
    }

    pub fn update_with(&mut self, block: Block) -> Result<(), BlockValidationErr> {
        let i = self.len();
        if let Err(error) = self.verify_block(&block, i) {
            return Err(error);
        }
        if let Some((coinbase, transactions)) = block.transactions.split_first() {
            if !coinbase.is_coinbase() {
                return Err(BlockValidationErr::InvalidCoinbaseTransaction);
            }
            let mut block_spent = HashSet::<BlockHash>::new();
            let mut block_created = HashSet::<BlockHash>::new();
            let mut total_fee = 0;
            for transaction in transactions {
                let input_hashes = transaction.input_hashes();
                if !(&input_hashes - &self.unspent_outputs).is_empty() || !(&input_hashes & &block_spent).is_empty() {
                    return Err(BlockValidationErr::InvalidInput);
                }
                let input_val = transaction.input_val();
                let output_val = transaction.output_val();
                if output_val > input_val {
                    return Err(BlockValidationErr::InsufficientInputVal);
                }
                total_fee += input_val - output_val;
                block_spent.extend(input_hashes);
                block_created.extend(transaction.output_hashes());
            }
            if coinbase.output_val() < total_fee {
                return Err(BlockValidationErr::InvalidCoinbaseTransaction);
            } else {
                block_created.extend(coinbase.output_hashes());
            }
            self.unspent_outputs.retain(|u_output| !block_spent.contains(u_output));
            self.unspent_outputs.extend(block_created);
        }
        self.add_block(block);
        Ok(())
    }

    pub fn verify_block(&self, block: &Block, index: usize) -> Result<(), BlockValidationErr> {
        if block.index != index {
            return Err(BlockValidationErr::MismatchedIndex);
        } else if !Block::check_difficulty(&block.hash, block.difficulty) {
            return Err(BlockValidationErr::InvalidHash);
        } else if index != 0 {
            let prev_block = &self.blocks[index - 1];
            if block.timestamp <= prev_block.timestamp {
                return Err(BlockValidationErr::AChronologicalTimestamp);
            } else if block.prev_block_hash != prev_block.hash {
                return Err(BlockValidationErr::MismatchedPreviousHash);
            }
        } else if block.prev_block_hash != vec![0; 32] {
            return Err(BlockValidationErr::InvalidGenesisBlockFormat);
        }
        Ok(())
    }

    pub fn verify_all_block(&self) -> Result<(), BlockValidationErr> {
        for (i, block) in self.blocks.iter().enumerate() {
            return self.verify_block(block, i);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    fn add_block(&mut self, block: Block) {
        self.blocks.push(block);
    }

    pub fn last_block(&self) -> Option<&Block> {
        self.blocks.last()
    }

    pub fn last_hash(&self) -> BlockHash {
        match self.last_block() {
            Some(block) => block.hash.clone(),
            None => vec![0; 32]
        }
    }
}

#[cfg(test)]
pub mod block_tests {
    use crate::block::{Block, BlockChain, Hashable};
    use crate::lib::now;
    use crate::Transaction;
    use crate::transaction::Output;

    #[test]
    pub fn test_create_block() {
        let index = 0;
        let _now = now();
        let prev_hash = BlockChain::origin_hash();
        let nonce = 0;
        let transactions = Vec::<Transaction>::new();
        let difficulty = 0x0088FFFFFFFFFFFFFFFFFFFFFFFFFFF;
        let mut block = Block::new(index, _now, prev_hash.clone(), nonce.clone(), transactions.clone(), difficulty);
        assert_eq!(block.nonce, nonce.clone(), "Block nonce not equal initial");
        block.mine();
        assert_eq!(block.index, index, "Block index not equal");
        assert_eq!(block.timestamp, _now, "Block timestamp not equal");
        assert_eq!(block.prev_block_hash, prev_hash, "Block prev block hash not equal");
        assert_eq!(block.hash, block.hash(), "Block hash not equal");
        assert_ne!(block.nonce, nonce, "Block nonce not change");
        assert_eq!(block.transactions, transactions, "Block transactions not equal");
        assert_eq!(block.difficulty, difficulty, "Block difficulty not equal");
    }

    #[test]
    pub fn test_create_block_chain() {
        let mut block_chain = BlockChain::new();
        assert_eq!(block_chain.init().is_ok(), true, "Block chain init fail!");
        let prev_block_hash = block_chain.last_hash();
        let origin_len = block_chain.blocks.len();
        let nonce = 0;
        let transactions = vec![
            Transaction {
                inputs: vec![
                    block_chain.blocks[0].transactions[0].outputs[0].clone(),
                ],
                outputs: vec![
                    Output {
                        to: String::from("Batman"),
                        value: 200,
                    },
                    Output {
                        to: String::from("Superman"),
                        value: 100,
                    },
                ],
            }
        ];
        let difficulty = 0x0088FFFFFFFFFFFFFFFFFFFFFFFFFFF;
        let mut block = Block::new(origin_len.clone(), now(), prev_block_hash, nonce, transactions, difficulty);
        block.mine();
        block_chain.add_block(block.clone());
        let last_block = block_chain.last_block().unwrap();
        assert_eq!(origin_len + 1, block_chain.len(), "Block chain length not increase");
        assert_eq!(block_chain.verify_all_block().is_ok(), true, "Block chain verify fail");
        assert_eq!(last_block.index, block.index, "Last block index not equal");
        assert_eq!(last_block.timestamp, block.timestamp, "Last block timestamp not equal");
        assert_eq!(last_block.hash, block.hash, "Last block hash not equal");
        assert_eq!(last_block.transactions, block.transactions, "Last block transactions not equal");
        assert_eq!(last_block.difficulty, block.difficulty, "Last block difficulty not equal");
    }
}