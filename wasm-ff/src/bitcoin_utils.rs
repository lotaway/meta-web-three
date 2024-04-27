use std::str::FromStr;
use bitcoin::{consensus::encode::serialize, network::constants::Network, util::key::PublicKey, blockdata::script::Builder, util::address::Address, util::amount::Amount, Transaction, SigHashType, PrivateKey};
use bitcoin::hashes::hex::{FromHex, ToHex};
use bitcoin::secp256k1::{Message, Secp256k1, SecretKey};
use rand::{Rng, thread_rng};

pub struct P2trTransaction {
    secp: Secp256k1<bitcoin::secp256k1::All>,
    s_private_key: SecretKey,
    to_address: Option<Address>,
    transactions: Vec<Transaction>,
}

pub enum BitcoinNetwork {
    LiveNet,
    TestNet,
    RegNet,
    SigNet,
}

pub struct BitcoinNetworkHelper {}

impl BitcoinNetwork {
    pub fn from_str(network: &str) -> Option<BitcoinNetwork> {
        match network {
            "Livenet" => Option::Some(BitcoinNetwork::LiveNet),
            "Testnet" => Option::Some(BitcoinNetwork::TestNet),
            "Regnet" => Option::Some(BitcoinNetwork::RegNet),
            "Signet" => Option::Some(BitcoinNetwork::SigNet),
            _ => Option::None
        }
    }

    pub fn as_network(&self) -> Network {
        match self {
            BitcoinNetwork::LiveNet => Network::Bitcoin,
            BitcoinNetwork::TestNet => Network::Testnet,
            BitcoinNetwork::RegNet => Network::Regtest,
            BitcoinNetwork::SigNet => Network::Signet,
        }
    }
}

impl P2trTransaction {
    pub fn new() -> Self {
        let secp = Secp256k1::new();
        let mut rng = thread_rng();
        let mut private_key_bytes = [0u8; 32];
        rng.fill(private_key_bytes.as_mut());
        let s_private_key = SecretKey::from_slice(&private_key_bytes).unwrap();
        P2trTransaction {
            secp,
            s_private_key,
            to_address: None,
            transactions: Vec::new(),
        }
    }

    pub fn create_taproot_transaction(
        xpriv: &str, // 扩展私钥，用于签名
        recipient_address: &str, // 接收者地址
        amount: u64, // 发送金额，单位聪
        utxo_txid: &str, // UTXO的txid
        utxo_vout: u32, // UTXO的输出索引
        utxo_amount: u64, // UTXO中的金额
    ) -> Result<String, bitcoin::Error> {
        let secp = Secp256k1::new();
        let mut rng = thread_rng();
        // 解析扩展私钥
        let xpriv_secret_key = SecretKey::from_str(xpriv).map_err(|e| e.to_string())?;
        let mut rng = thread_rng();
        let mut priv_key = [0u8; 32];
        rng.fill(priv_key.as_mut());

        // 构建接收者的输出脚本
        let recipient_script_pubkey = Address::from_str(recipient_address)
            .map_err(|e| e.to_string())?.script_pubkey();

        // 创建交易输入
        let txin = bitcoin::TxIn {
            previous_output: bitcoin::OutPoint::new(bitcoin::Txid::from_hex(utxo_txid).map_err(|e| e.to_string())?, utxo_vout),
            script_sig: Builder::new().into_script(),
            sequence: 0xFFFFFFFF,
            witness: Vec::new(),
        };

        // 创建交易输出
        let txout = bitcoin::TxOut {
            value: amount,
            script_pubkey: recipient_script_pubkey,
        };
        let transaction = Transaction {
            version: 2,
            lock_time: 0,
            input: vec![txin],
            output: vec![txout],
        };
        let mut psbt = PartiallySignedTransaction::from_unsigned_tx(transaction)
            .map_err(|e| e.to_string())?;

        let raw_transaction = serialize(&psbt.extract_tx());

        Ok(bitcoin::hashes::encode(raw_transaction))
    }

    pub fn generate_address(&mut self, network: Network) -> &mut Self {
        let private_key = PrivateKey::new(self.s_private_key, network);
        let public_key = PublicKey::from_private_key(&self.secp, &private_key);
        self.to_address = Some(Address::p2wpkh(&public_key, Network::Bitcoin).unwrap());
        println!("P2TR Address: {:?}", self.to_address.clone());
        self
    }

    pub fn build_transaction(&mut self, utxo_tx_id: &str) {
        let mut transaction = Transaction {
            version: 2,
            lock_time: 0,
            input: vec![],
            output: vec![],
        };
        let utxo_txid = bitcoin::Txid::from_hex(utxo_tx_id).unwrap();
        let utxo_output_index = 0;
        let utxo_value = Amount::from_btc(1.0).unwrap();
        let input = bitcoin::TxIn {
            previous_output: bitcoin::OutPoint {
                txid: utxo_txid,
                vout: utxo_output_index,
            },
            script_sig: Builder::new().into_script(),
            sequence: 0xFFFFFFFF,
            witness: vec![],
        };
        transaction.input.push(input);
        // 添加输出
        let output = bitcoin::TxOut {
            value: utxo_value.as_sat(),
            script_pubkey: self.to_address.clone().unwrap().script_pubkey(),
        };
        transaction.output.push(output.clone());
        println!("Signed Transaction: {}", serialize(&transaction).to_hex());
        self.transactions.push(transaction)
    }

    pub fn sign(&mut self) {
        // 签名交易
        self.transactions.iter().map(|transaction| {
            let hash_type = SigHashType::All;
            let sig = self.secp.sign(
                &Message::from_slice(&transaction.signature_hash(0, &self.to_address.clone().unwrap().script_pubkey(), hash_type.as_u32()).as_ref()).unwrap(),
                &self.s_private_key,
            );
            transaction.input[0].witness = vec![sig.serialize_der().to_vec(), vec![]];
        });
    }

    pub fn broad_cast(&mut self) {
        todo!()
    }
}

pub fn get_utxos(bitcoin_address: &str) {
    todo!()
}

#[cfg(test)]
mod tests {}