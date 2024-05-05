use bitcoin::consensus::encode::serialize;
use bitcoin::util::script::Builder;
use num_bigint::BigUint;
use num_traits::identities::Zero;
use uint128::Uint128;
use bitcoin::secp256k1::{PublicKey, Secp256k1};
use bitcoin::util::hash::BitcoinHash;

// https://cloud.tencent.com/developer/article/2413900
pub struct Etching {
    divisibility: Option<u8>, // 符文的可分割性，相当于ERC20中的decimal字段，小数位数
    premine: Option<u64>,     // 预挖矿的数量。不设就表示不预挖
    rune: Option<String>,     // 符文的名称，以修改后的基数-26整数编码
    spacers: Option<String>,  // 表示在符文名称字符之间显示的间隔符
    symbol: Option<String>,   // 符文的货币符号，一个UTF8字符
    terms: Option<Terms>,     // 包含铸造条款，如数量、上限、开始和结束的区块高度
}

impl Etching {
    pub fn build_etching_script(payload: &[u8], magic_number: u8) -> Vec<u8> {
        let mut builder = Builder::new();
        builder.push_opcode(bitcoin::blockdata::opcodes::all::OP_RETURN);
        builder.push_opcode(magic_number.into());
        let mut remaining_payload = payload;
        while !remaining_payload.is_empty() {
            let chunk_size = remaining_payload
                .len()
                .min(bitcoin::util::MAX_SCRIPT_ELEMENT_SIZE);
            let chunk = &remaining_payload[..chunk_size];
            builder.push_slice(chunk);
            remaining_payload = &remaining_payload[chunk_size..];
        }
        builder.into_script().to_bytes().unwrap()
    }
}

pub struct Terms {
    amount: Option<Uint128>,  // Mint一次能够铸造的数量
    cap: Option<Uint128>,     // 能够Mint多少次
    height: [Option<u64>; 2], // 允许Mint的开始高度和结束高度（绝对值）
    offset: [Option<u64>; 2], // 允许Mint的开始高度和结束高度（相当于发行符文的高度而言的相对值）
}

struct Edict {
    id: String,  // 涉及的符文ID
    amount: u64, // 转移的符文数量
    output: u32, // 指定的输出索引
}

struct Runestone {
    edicts: Vec<Edict>,       // 一个Edict（法令）的集合，用于转移Rune
    etching: Option<Etching>, // 一个可选的Etching（蚀刻），用于创建Rune
    mint: Option<String>,     // 一个可选的RuneId，表示要铸造的Rune的ID
    pointer: Option<u32>,     // 一个可选的u32，指向未被Edict分配的Rune应转移至的输出
}

impl Runestone {
    fn new(etching: Etching, mint: Option<String>, pointer: Option<String>) -> Self {
        Runestone {
            edicts: vec![],
            etching,
            mint,
            pointer,
        }
    }

    pub fn commitment(r: &Runestone) -> Vec<u8> {
        let mut bytes = r.value.to_bytes_le(); // Get bytes in little-endian representation
                                               // Reverse bytes to get big-endian representation
        bytes.reverse();
        // Trim leading zero bytes
        let mut end = bytes.len();
        while end > 0 && bytes[end - 1] == 0 {
            end -= 1;
        }
        bytes.truncate(end);
        bytes
    }

    pub fn create_tap_script(pk: &PublicKey, commitment: &[u8]) -> Vec<u8> {
        let secp = Secp256k1::new();
        let mut builder = Builder::new();
        // Push pubkey
        let pk_ser = pk.serialize();
        builder.push_slice(&pk_ser);
        builder.push_opcode(bitcoin::blockdata::opcodes::all::OP_CHECKSIG);
        // Commitment script
        builder.push_opcode(bitcoin::blockdata::opcodes::all::OP_FALSE);
        builder.push_opcode(bitcoin::blockdata::opcodes::all::OP_IF);
        builder.push_slice(commitment);
        builder.push_opcode(bitcoin::blockdata::opcodes::all::OP_ENDIF);
        builder.into_script().to_bytes().unwrap()
    }

    pub fn commitment(&mut self) -> Vec<u8> {
        Runestone.commitment(self)
    }
}
