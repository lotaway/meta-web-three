use dubbo::{
    codegen::async_trait,
    config::{protocol::ProtocolConfig, registry::RegistryConfig},
    status::DubboError,
};
use serde::{Deserialize, Serialize};
use serde_json::to_string;
use ordered_float::OrderedFloat;
use slab::Slab;
use std::{
    collections::{BTreeMap, VecDeque},
    fs::{File, OpenOptions},
    io::Write,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH, Duration},
};
use tokio::time::sleep;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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

#[derive(Default)]
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

    fn insert_order(&mut self, ord: OrderEntry, orders_slab: &Arc<Mutex<Slab<OrderEntry>>>) -> usize {
        let key = OrderedFloat(ord.price);
        let side = ord.side.clone();
        let idx = {
            let mut slab = orders_slab.lock().unwrap();
            slab.insert(ord)
        };
        match side {
            OrderType::Buy => {
                let level = self.bids.entry(key).or_insert_with(PriceLevel::new);
                level.orders.push_back(idx);
                idx
            }
            OrderType::Sell => {
                let level = self.asks.entry(key).or_insert_with(PriceLevel::new);
                level.orders.push_back(idx);
                idx
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

#[async_trait]
pub trait MatchingService {
    async fn match_order(&mut self, req: OrderRequest) -> Result<OrderResponse, DubboError>;
}

#[derive(Default)]
pub struct MatchingServiceImpl {
    book: Arc<Mutex<OrderBook>>,
    orders: Arc<Mutex<Slab<OrderEntry>>>,
    wal: Arc<Mutex<Option<File>>>,
    batch_queue: Arc<Mutex<Vec<Trade>>>,
    use_protocol_account: bool,
}

impl MatchingServiceImpl {
    fn new_with_wal(wal_path: &str, use_protocol_account: bool) -> Self {
        let file = OpenOptions::new().create(true).append(true).open(wal_path).ok();
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
                    let batch: Vec<Trade> = {
                        let mut q = batch_q.lock().unwrap();
                        if q.is_empty() {
                            Vec::new()
                        } else {
                            q.drain(..).collect()
                        }
                    };
                    if !batch.is_empty() {
                        let _ = Self::create_and_settle_batch_internal(batch, wal_clone.clone(), use_protocol).await;
                    }
                }
            });
        }
        svc
    }

    fn new() -> Self {
        Self::new_with_wal("order_match.wal", false)
    }

    fn insert_order_entry(&self, entry: OrderEntry) -> usize {
        let mut slab = self.orders.lock().unwrap();
        slab.insert(entry)
    }

    fn pop_order_entry(&self, idx: usize) -> Option<OrderEntry> {
        let mut slab = self.orders.lock().unwrap();
        Some(slab.remove(idx))
    }

    fn append_wal_trade(&self, trade: &Trade) {
        if let Ok(line) = to_string(trade) {
            if let Ok(mut f) = self.wal.lock() {
                if let Some(ref mut file) = *f {
                    let _ = file.write_all(line.as_bytes());
                    let _ = file.write_all(b"\n");
                    let _ = file.flush();
                }
            }
        }
    }

    async fn create_and_settle_batch_internal(batch: Vec<Trade>, wal: Arc<Mutex<Option<File>>>, use_protocol_account: bool) -> Result<String, String> {
        for t in &batch {
            if let Ok(line) = to_string(t) {
                if let Ok(mut f) = wal.lock() {
                    if let Some(ref mut file) = *f {
                        let _ = file.write_all(line.as_bytes());
                        let _ = file.write_all(b"\n");
                    }
                }
            }
        }
        let txid = Self::settle_batch_on_chain_simulated(&batch, use_protocol_account).await;
        Ok(txid)
    }

    async fn settle_batch_on_chain_simulated(_batch: &[Trade], use_protocol_account: bool) -> String {
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

    fn match_limit_buy_with_slab(book: &mut OrderBook, orders_slab: &Arc<Mutex<Slab<OrderEntry>>>, taker_idx: usize) -> (Vec<Trade>, f64) {
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

    fn match_limit_buy(&mut self, taker: OrderEntry) -> (Vec<Trade>, f64) {
        let orders_slab = self.orders.clone();
        let taker_idx = self.insert_order_entry(taker);
        let mut book = self.book.lock().unwrap();
        Self::match_limit_buy_with_slab(&mut book, &orders_slab, taker_idx)
    }

    fn match_limit_sell(&mut self, taker: OrderEntry) -> (Vec<Trade>, f64) {
        let orders_slab = self.orders.clone();
        let taker_idx = self.insert_order_entry(taker);
        let mut book = self.book.lock().unwrap();
        Self::match_limit_sell_with_slab(&mut book, &orders_slab, taker_idx)
    }

    fn match_market_buy(&mut self, taker: OrderEntry) -> (Vec<Trade>, f64) {
        let orders_slab = self.orders.clone();
        let taker_idx = self.insert_order_entry(taker);
        let mut book = self.book.lock().unwrap();
        Self::match_market_buy_with_slab(&mut book, &orders_slab, taker_idx)
    }

    fn match_market_sell(&mut self, taker: OrderEntry) -> (Vec<Trade>, f64) {
        let orders_slab = self.orders.clone();
        let taker_idx = self.insert_order_entry(taker);
        let mut book = self.book.lock().unwrap();
        Self::match_market_sell_with_slab(&mut book, &orders_slab, taker_idx)
    }

    fn match_limit_sell_with_slab(book: &mut OrderBook, orders_slab: &Arc<Mutex<Slab<OrderEntry>>>, taker_idx: usize) -> (Vec<Trade>, f64) {
        let mut trades = Vec::new();
        loop {
            if {
                let slab = orders_slab.lock().unwrap();
                slab.get(taker_idx).map(|o| o.remaining > 0.0).unwrap_or(false)
            } == false { break }
            let best_bid_key = match book.bids.keys().rev().next().cloned() { Some(k) => k, None => break };
            let best_price = best_bid_key.into_inner();
            let level = book.bids.get_mut(&OrderedFloat(best_price)).unwrap();
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
            if level.orders.is_empty() { book.bids.remove(&OrderedFloat(best_price)); }
        }
        let rem = {
            let slab = orders_slab.lock().unwrap();
            slab.get(taker_idx).map(|o| o.remaining).unwrap_or(0.0)
        };
        (trades, rem)
    }

    fn match_market_buy_with_slab(book: &mut OrderBook, orders_slab: &Arc<Mutex<Slab<OrderEntry>>>, taker_idx: usize) -> (Vec<Trade>, f64) {
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

    fn match_market_sell_with_slab(book: &mut OrderBook, orders_slab: &Arc<Mutex<Slab<OrderEntry>>>, taker_idx: usize) -> (Vec<Trade>, f64) {
        let mut trades = Vec::new();
        loop {
            if {
                let slab = orders_slab.lock().unwrap();
                slab.get(taker_idx).map(|o| o.remaining > 0.0).unwrap_or(false)
            } == false { break }
            let best_bid_key = match book.bids.keys().rev().next().cloned() { Some(k) => k, None => break };
            let best_price = best_bid_key.into_inner();
            let level = book.bids.get_mut(&OrderedFloat(best_price)).unwrap();
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
            if level.orders.is_empty() { book.bids.remove(&OrderedFloat(best_price)); }
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
    async fn match_order(&mut self, req: OrderRequest) -> Result<OrderResponse, DubboError> {
        let timestamp = now_millis();
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
                let (t, rem) = self.match_limit_buy(taker);
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
                    let mut book = self.book.lock().unwrap();
                    book.insert_order(rest, &self.orders);
                    book.remove_empty_levels();
                } else {
                    let mut book = self.book.lock().unwrap();
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
                let (t, rem) = self.match_limit_sell(taker);
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
                    let mut book = self.book.lock().unwrap();
                    book.insert_order(rest, &self.orders);
                    book.remove_empty_levels();
                } else {
                    let mut book = self.book.lock().unwrap();
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
                let (t, rem) = self.match_market_buy(taker);
                trades.extend(t);
                remaining = rem;
                let mut book = self.book.lock().unwrap();
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
                let (t, rem) = self.match_market_sell(taker);
                trades.extend(t);
                remaining = rem;
                let mut book = self.book.lock().unwrap();
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
    let protocol = ProtocolConfig::default();
    let svc_impl = MatchingServiceImpl::new();
    // Note: This is a placeholder - you'll need to implement the actual RPC server
    // based on your dubbo version and requirements
    println!("RPC server would start here");
}
