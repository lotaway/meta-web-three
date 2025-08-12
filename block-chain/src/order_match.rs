use dubbo::{
    codegen::async_trait,
    config::{protocol::ProtocolConfig, registry::RegistryConfig, service::ServiceConfig},
    status::DubboError,
};
use serde::{Deserialize, Serialize};

use dubbo::{
    codegen::async_trait,
    config::{protocol::ProtocolConfig, registry::RegistryConfig, service::ServiceConfig},
    status::DubboError,
};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OrderType {
    Buy,
    Sell,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OrderPriceType {
    Fixed,
    Market,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OrderRequest {
    pub order_id: String,
    pub order_type: OrderType,
    pub price_type: OrderPriceType,
    pub price: Option<f64>,
    pub quantity: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Trade {
    pub taker_order_id: String,
    pub maker_order_id: String,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u128,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OrderResponse {
    pub success: bool,
    pub message: String,
    pub trades: Vec<Trade>,
    pub remaining: f64,
}

#[derive(Clone)]
struct OrderEntry {
    order_id: String,
    price: f64,
    remaining: f64,
    timestamp: u128,
    side: OrderType,
}

struct PriceLevel {
    orders: VecDeque<usize>,
}

impl PriceLevel {
    fn new() -> Self {
        Self {
            orders: VecDeque::new(),
        }
    }
}

struct OrderBook {
    asks: BTreeMap<OrderedFloat<f64>, PriceLevel>,
    bids: BTreeMap<OrderedFloat<f64>, PriceLevel>,
}

impl OrderBook {
    fn new() -> Self {
        Self {
            asks: BTreeMap::new(),
            bids: BTreeMap::new(),
        }
    }

    fn best_ask_price(&self) -> Option<f64> {
        self.asks.keys().next().map(|k| k.into_inner())
    }

    fn best_bid_price(&self) -> Option<f64> {
        self.bids.keys().rev().next().map(|k| k.into_inner())
    }

    fn insert_order(&mut self, ord: OrderEntry) {
        let key = OrderedFloat(ord.price);
        match ord.side {
            OrderType::Buy => {
                let level = self.bids.entry(key).or_insert_with(PriceLevel::new);
                level.orders.push_back(ord)
            }
            OrderType::Sell => {
                let level = self.asks.entry(key).or_insert_with(PriceLevel::new);
                level.orders.push_back(ord)
            }
        }
    }

    fn remove_empty_levels(&mut self) {
        let empty_keys: Vec<_> = self
            .asks
            .iter()
            .filter(|(_, lvl)| lvl.orders.is_empty())
            .map(|(k, _)| *k)
            .collect();
        for k in empty_keys {
            self.asks.remove(&k);
        }
        let empty_keys: Vec<_> = self
            .bids
            .iter()
            .filter(|(_, lvl)| lvl.orders.is_empty())
            .map(|(k, _)| *k)
            .collect();
        for k in empty_keys {
            self.bids.remove(&k);
        }
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

#[dubbo::service]
#[async_trait]
pub trait MatchingService {
    async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError>;
}

#[derive(Default)]
pub struct MatchingServiceImpl {
    book: Arc<Mutex<OrderBook>>,
    orders: Arc<Mutex<Slab<OrderEntry>>>,
    wal: Arc<Mutex<File>>,
    batch_queue: Arc<Mutex<Vec<Trade>>>,
    use_protocol_account: bool,
}

impl MatchingServiceImpl {
    fn new_with_wal(wal_path: &str, use_protocol_account: bool) -> Self {
        let file = OpenOptions::new().create(true).append(true).open(wal_path).unwrap();
        let svc = Self {
            book: Arc::new(Mutex::new(OrderBook::new())),
            orders: Arc::new(Mutex::new(Slab::with_capacity(1024))),
            wal: Arc::new(Mutex::new(file)),
            batch_queue: Arc::new(Mutex::new(Vec::new())),
            use_protocol_account,
        };
        {
            let batch_q = svc.batch_queue.clone();
            let wal_clone = svc.wal.clone();
            let use_protocol = svc.use_protocol_account;
            tokio::spawn(async move {
                loop {
                    sleep(Duration::from_secs(5)).await;
                    let mut q = batch_q.lock().unwrap();
                    if q.is_empty() { continue }
                    let batch: Vec<Trade> = q.drain(..).collect();
                    let _ = Self::create_and_settle_batch_internal(batch, wal_clone.clone(), use_protocol).await;
                }
            });
        }
        svc
    }

    fn insert_order_entry(&self, entry: OrderEntry) -> usize {
        let mut slab = self.orders.lock().unwrap();
        slab.insert(entry)
    }

    fn pop_order_entry(&self, idx: usize) -> Option<OrderEntry> {
        let mut slab = self.orders.lock().unwrap();
        slab.remove(idx)
    }

    fn append_wal_trade(&self, trade: &Trade) {
        let mut f = self.wal.lock().unwrap();
        if let Ok(line) = to_string(trade) {
            let _ = f.write_all(line.as_bytes());
            let _ = f.write_all(b"\n");
            let _ = f.flush();
        }
    }

    async fn create_and_settle_batch_internal(batch: Vec<Trade>, wal: Arc<Mutex<File>>, use_protocol_account: bool) -> Result<String, String> {
        for t in &batch {
            if let Ok(line) = to_string(t) {
                let mut f = wal.lock().unwrap();
                let _ = f.write_all(line.as_bytes());
                let _ = f.write_all(b"\n");
            }
        }
        let txid = Self::settle_batch_on_chain_simulated(&batch, use_protocol_account).await;
        Ok(txid)
    }

    async fn settle_batch_on_chain_simulated(batch: &[Trade], use_protocol_account: bool) -> String {
        if use_protocol_account {
            format!("protocol-tx-{}", now_millis())
        } else {
            format!("user-batch-tx-{}", now_millis())
        }
    }

    async fn create_and_settle_batch(&self, trades: Vec<Trade>) -> Result<String, String> {
        Self::create_and_settle_batch_internal(trades, self.wal.clone(), self.use_protocol_account).await
    }

    fn push_trade_to_batch(&self, trade: Trade) {
        self.append_wal_trade(&trade);
        let mut q = self.batch_queue.lock().unwrap();
        q.push(trade);
    }

    fn match_limit_buy_with_slab(book: &mut OrderBook, orders_slab: &Arc<Mutex<Slab<OrderEntry>>>, mut taker_idx: usize) -> (Vec<Trade>, f64) {
        let mut trades = Vec::new();
        loop {
            if {
                let slab = orders_slab.lock().unwrap();
                slab.get(taker_idx).map(|o| o.remaining > 0.0).unwrap_or(false)
            } == false { break }
            let best_ask_key = match book.asks.keys().next().cloned() { Some(k) => k, None => break };
            let best_price = best_ask_key.into_inner();
            let level = book.asks.get_mut(&OrderedFloat(best_price)).unwrap();
            while let Some(&maker_idx) = level.orders.front() {
                let mut slab = orders_slab.lock().unwrap();
                let taker_remaining = slab.get_mut(taker_idx).unwrap().remaining;
                let maker_remaining = slab.get_mut(maker_idx).unwrap().remaining;
                let qty = taker_remaining.min(maker_remaining);
                let trade = Trade {
                    taker_order_id: slab.get(taker_idx).unwrap().order_id.clone(),
                    maker_order_id: slab.get(maker_idx).unwrap().order_id.clone(),
                    price: slab.get(maker_idx).unwrap().price,
                    quantity: qty,
                    timestamp: now_millis(),
                };
                slab.get_mut(taker_idx).unwrap().remaining -= qty;
                slab.get_mut(maker_idx).unwrap().remaining -= qty;
                trades.push(trade);
                if slab.get(maker_idx).unwrap().remaining <= 0.0 {
                    level.orders.pop_front();
                    slab.remove(maker_idx);
                } else {
                    break
                }
                if slab.get(taker_idx).unwrap().remaining <= 0.0 { break }
            }
            if level.orders.is_empty() { book.asks.remove(&OrderedFloat(best_price)); }
        }
        let rem = {
            let slab = orders_slab.lock().unwrap();
            slab.get(taker_idx).map(|o| o.remaining).unwrap_or(0.0)
        };
        (trades, rem)
    }
}

#[async_trait]
impl MatchingService for MatchingServiceImpl {
    async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError> {
        let timestamp = now_millis();
        let mut book = self.book.lock().unwrap();
        let mut trades = Vec::new();
        let mut remaining = req.quantity;
        let side = req.order_type.clone();
        match (req.price_type.clone(), side) {
            (OrderPriceType::Fixed, OrderType::Buy) => {
                let price = req.price.unwrap_or(0.0);
                let taker = OrderEntry {
                    order_id: req.order_id.clone(),
                    price,
                    remaining: req.quantity,
                    timestamp,
                    side: OrderType::Buy,
                };
                let (t, rem) = Self::match_limit_buy(&mut book, taker);
                trades.extend(t);
                remaining = rem;
                if remaining > 0.0 {
                    let rest = OrderEntry {
                        order_id: req.order_id.clone(),
                        price,
                        remaining,
                        timestamp,
                        side: OrderType::Buy,
                    };
                    book.insert_order(rest);
                    book.remove_empty_levels();
                } else {
                    book.remove_empty_levels();
                }
            }
            (OrderPriceType::Fixed, OrderType::Sell) => {
                let price = req.price.unwrap_or(0.0);
                let taker = OrderEntry {
                    order_id: req.order_id.clone(),
                    price,
                    remaining: req.quantity,
                    timestamp,
                    side: OrderType::Sell,
                };
                let (t, rem) = Self::match_limit_sell(&mut book, taker);
                trades.extend(t);
                remaining = rem;
                if remaining > 0.0 {
                    let rest = OrderEntry {
                        order_id: req.order_id.clone(),
                        price,
                        remaining,
                        timestamp,
                        side: OrderType::Sell,
                    };
                    book.insert_order(rest);
                    book.remove_empty_levels();
                } else {
                    book.remove_empty_levels();
                }
            }
            (OrderPriceType::Market, OrderType::Buy) => {
                let taker = OrderEntry {
                    order_id: req.order_id.clone(),
                    price: 0.0,
                    remaining: req.quantity,
                    timestamp,
                    side: OrderType::Buy,
                };
                let (t, rem) = Self::match_market_buy(&mut book, taker);
                trades.extend(t);
                remaining = rem;
                book.remove_empty_levels();
            }
            (OrderPriceType::Market, OrderType::Sell) => {
                let taker = OrderEntry {
                    order_id: req.order_id.clone(),
                    price: 0.0,
                    remaining: req.quantity,
                    timestamp,
                    side: OrderType::Sell,
                };
                let (t, rem) = Self::match_market_sell(&mut book, taker);
                trades.extend(t);
                remaining = rem;
                book.remove_empty_levels();
            }
        }
        let success = !trades.is_empty();
        let msg = if success {
            "matched".to_string()
        } else {
            "no match, order placed or empty book".to_string()
        };
        Ok(OrderResponse {
            success,
            message: msg,
            trades,
            remaining,
        })
    }
}

pub fn start_rpc() {
    let registry_config = RegistryConfig {
        protocol: "zookeeper".to_string(),
        address: "127.0.0.1:2181".to_string(),
        ..Default::default()
    };
    let protocal = Protocal {
        name: "dubbo".to_string(),
        port: 20086,
        ..Default::default()
    };
    let protocol_config = ProtocolConfig::default();
    protocol_config.insert("dubbo", protocal);
    let svc_impl = MatchingServiceImpl::new();
    let service_config = ServiceConfig::new(
        "com.metawebthree.common.rpc.interface.MatchingService",
        svc_impl,
    )
    .with_group("/dev/metawebthree")
    .with_protocol("dubbo");
    dubbo::init()
        .with_registry(registry_config)
        .with_protocol(protocol_config)
        .with_service(service_config)
        .serve();
}
