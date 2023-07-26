use std::fmt::{Debug, Formatter};
use crate::{BlockHash, difficulty_bytes_as_u128, now, u128_bytes, u32_bytes, u64_bytes};

pub trait Hashable {
    fn bytes(&self) -> Vec<u8>;
    fn hash(&self) -> Vec<u8> {
        crypto_hash::digest(crypto_hash::Algorithm::SHA256, &self.bytes())
    }
}

#[derive(Clone)]
pub struct Block {
    index: usize,
    timestamp: u128,
    prev_block_hash: BlockHash,
    hash: BlockHash,
    nonce: u64,
    payload: String,
    difficulty: u128,
}

impl Hashable for Block {
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::from_iter(u32_bytes(&(self.index as u32)));
        bytes.extend(&u128_bytes(&self.timestamp));
        bytes.extend(&self.prev_block_hash);
        bytes.extend(&u64_bytes(&self.nonce));
        bytes.extend(self.payload.as_bytes());
        bytes.extend(&u128_bytes(&self.difficulty));
        bytes
    }
}

impl Debug for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block[{}]: {} at: {} with: {} nonce: {}", &self.index, &hex::encode(&self.hash), &self.timestamp, &self.payload, &self.nonce)
    }
}

impl Block {
    pub fn new(index: usize, timestamp: u128, prev_block_hash: BlockHash, nonce: u64, payload: String, difficulty: u128) -> Self {
        let mut block = Block {
            index,
            timestamp,
            prev_block_hash,
            hash: vec![0; 0],
            nonce,
            payload,
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

struct BlockChain {
    blocks: Vec<Block>,
}

impl BlockChain {
    fn new() -> Self {
        let prev_hash = vec![0; 32];
        let nonce = 0;
        let payload = String::from("Genesis block");
        let difficulty = 0x0000FFFFFFFFFFFFFFFFFFFFFFFFFFF;
        let block = Block::new(0, now(), prev_hash, nonce, payload, difficulty);
        let block_chain = BlockChain {
            blocks: vec![block]
        };
        block_chain
    }

    fn verify(&self) -> bool {
        for (i, block) in self.blocks.iter().enumerate() {
            if block.index != i {
                println!("Index mismatch {} != {}", &block.index, &i);
                return false;
            } else if !Block::check_difficulty(&block.hash, block.difficulty) {
                println!("Difficulty check fail");
                return false;
            } else if i != 0 {
                let prev_block = &self.blocks[i - 1];
                if block.timestamp <= prev_block.timestamp {
                    println!("Time did not increase");
                    return false;
                } else if block.prev_block_hash != prev_block.hash {
                    println!("Prev block hash not match");
                    return false;
                }
            } else if block.prev_block_hash != vec![0; 32] {
                println!("Genesis block prev block hash invalid!");
                return false;
            }
        }
        true
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }

    fn add_block(&mut self, block: Block) {
        self.blocks.push(block);
    }

    fn get_last_block(&self) -> Option<&Block> {
        self.blocks.last()
    }

    fn get_last_hash(&self) -> BlockHash {
        match self.get_last_block() {
            Some(block) => block.hash.clone(),
            None => vec![0; 32]
        }
    }
}

#[cfg(test)]
pub mod block_test {
    use crate::{Block, now};
    use crate::block::{BlockChain, Hashable};

    #[test]
    pub fn test_create_block() {
        let index = 0;
        let _now = now();
        let prev_hash = vec![0; 32];
        let nonce = 0;
        let payload = String::from("test block");
        let difficulty = 0x0088FFFFFFFFFFFFFFFFFFFFFFFFFFF;
        let mut block = Block::new(index, _now, prev_hash.clone(), nonce.clone(), payload.clone(), difficulty);
        assert_eq!(block.nonce, nonce.clone());
        block.mine();
        assert_eq!(block.index, index);
        assert_eq!(block.timestamp, _now);
        assert_eq!(block.prev_block_hash, prev_hash);
        assert_eq!(block.hash, block.hash());
        assert_ne!(block.nonce, nonce);
        assert_eq!(block.payload, payload);
        assert_eq!(block.difficulty, difficulty);
    }

    #[test]
    pub fn test_create_block_chain() {
        let nonce = 0;
        let payload = String::from("test block chain");
        let difficulty = 0x0088FFFFFFFFFFFFFFFFFFFFFFFFFFF;
        let mut block_chain = BlockChain::new();
        let prev_block_hash = block_chain.get_last_hash();
        let origin_len = block_chain.blocks.len();
        let mut block = Block::new(origin_len.clone(), now(), prev_block_hash, nonce, payload, difficulty);
        block.mine();
        block_chain.add_block(block.clone());
        let last_block = block_chain.get_last_block().unwrap();
        assert_eq!(origin_len + 1, block_chain.len());
        assert_eq!(block_chain.verify(), true);
        assert_eq!(last_block.index, block.index);
        assert_eq!(last_block.timestamp, block.timestamp);
        assert_eq!(last_block.hash, block.hash);
        assert_eq!(last_block.payload, block.payload);
        assert_eq!(last_block.difficulty, block.difficulty);
    }
}