use dubbo::{codegen::async_trait, config::{protocol::ProtocolConfig, registry::RegistryConfig, service::ServiceConfig}, status::DubboError};
use serde::{Deserialize, Serialize};


use dubbo::{
    codegen::async_trait,
    config::{protocol::ProtocolConfig, registry::RegistryConfig, service::ServiceConfig},
    status::DubboError,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use ordered_float::OrderedFloat;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OrderKind {
    Limit,
    Market,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OrderRequest {
    pub order_id: String,
    pub side: Side,
    pub kind: OrderKind,
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
struct Order {
    order_id: String,
    price: f64,
    remaining: f64,
    timestamp: u128,
    side: Side,
}

struct PriceLevel {
    orders: VecDeque<Order>,
}

impl PriceLevel {
    fn new() -> Self {
        Self { orders: VecDeque::new() }
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

    fn insert_order(&mut self, ord: Order) {
        let key = OrderedFloat(ord.price);
        match ord.side {
            Side::Buy => {
                let level = self.bids.entry(key).or_insert_with(PriceLevel::new);
                level.orders.push_back(ord)
            }
            Side::Sell => {
                let level = self.asks.entry(key).or_insert_with(PriceLevel::new);
                level.orders.push_back(ord)
            }
        }
    }

    fn remove_empty_levels(&mut self) {
        let empty_keys: Vec<_> = self.asks.iter()
            .filter(|(_, lvl)| lvl.orders.is_empty())
            .map(|(k, _)| *k)
            .collect();
        for k in empty_keys { self.asks.remove(&k); }
        let empty_keys: Vec<_> = self.bids.iter()
            .filter(|(_, lvl)| lvl.orders.is_empty())
            .map(|(k, _)| *k)
            .collect();
        for k in empty_keys { self.bids.remove(&k); }
    }
}

fn now_millis() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()
}

#[dubbo::service]
#[async_trait]
pub trait MatchingService {
    async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError>;
}

#[derive(Default)]
pub struct MatchingServiceImpl {
    book: Arc<Mutex<OrderBook>>,
}

impl MatchingServiceImpl {
    fn new() -> Self {
        Self { book: Arc::new(Mutex::new(OrderBook::new())) }
    }

    fn match_limit_buy(book: &mut OrderBook, mut order: Order) -> (Vec<Trade>, f64) {
        let mut trades = Vec::new();
        while order.remaining > 0.0 {
            let best_ask_key = match book.asks.keys().next().cloned() { Some(k) => k, None => break };
            let best_price = best_ask_key.into_inner();
            if order.price < best_price { break }
            let level = book.asks.get_mut(&OrderedFloat(best_price)).unwrap();
            while let Some(mut maker) = level.orders.front_mut() {
                let qty = order.remaining.min(maker.remaining);
                let trade = Trade {
                    taker_order_id: order.order_id.clone(),
                    maker_order_id: maker.order_id.clone(),
                    price: maker.price,
                    quantity: qty,
                    timestamp: now_millis(),
                };
                trades.push(trade);
                order.remaining -= qty;
                maker.remaining -= qty;
                if maker.remaining <= 0.0 {
                    level.orders.pop_front();
                } else {
                    break;
                }
                if order.remaining <= 0.0 { break }
            }
            if level.orders.is_empty() { book.asks.remove(&OrderedFloat(best_price)); }
        }
        (trades, order.remaining)
    }

    fn match_limit_sell(book: &mut OrderBook, mut order: Order) -> (Vec<Trade>, f64) {
        let mut trades = Vec::new();
        while order.remaining > 0.0 {
            let best_bid_key = match book.bids.keys().rev().next().cloned() { Some(k) => k, None => break };
            let best_price = best_bid_key.into_inner();
            if order.price > best_price { break }
            let level = book.bids.get_mut(&OrderedFloat(best_price)).unwrap();
            while let Some(mut maker) = level.orders.front_mut() {
                let qty = order.remaining.min(maker.remaining);
                let trade = Trade {
                    taker_order_id: order.order_id.clone(),
                    maker_order_id: maker.order_id.clone(),
                    price: maker.price,
                    quantity: qty,
                    timestamp: now_millis(),
                };
                trades.push(trade);
                order.remaining -= qty;
                maker.remaining -= qty;
                if maker.remaining <= 0.0 {
                    level.orders.pop_front();
                } else {
                    break;
                }
                if order.remaining <= 0.0 { break }
            }
            if level.orders.is_empty() { book.bids.remove(&OrderedFloat(best_price)); }
        }
        (trades, order.remaining)
    }

    fn match_market_buy(book: &mut OrderBook, mut order: Order) -> (Vec<Trade>, f64) {
        let mut trades = Vec::new();
        while order.remaining > 0.0 {
            let best_ask_key = match book.asks.keys().next().cloned() { Some(k) => k, None => break };
            let best_price = best_ask_key.into_inner();
            let level = book.asks.get_mut(&OrderedFloat(best_price)).unwrap();
            while let Some(mut maker) = level.orders.front_mut() {
                let qty = order.remaining.min(maker.remaining);
                let trade = Trade {
                    taker_order_id: order.order_id.clone(),
                    maker_order_id: maker.order_id.clone(),
                    price: maker.price,
                    quantity: qty,
                    timestamp: now_millis(),
                };
                trades.push(trade);
                order.remaining -= qty;
                maker.remaining -= qty;
                if maker.remaining <= 0.0 {
                    level.orders.pop_front();
                } else {
                    break;
                }
                if order.remaining <= 0.0 { break }
            }
            if level.orders.is_empty() { book.asks.remove(&OrderedFloat(best_price)); }
        }
        (trades, order.remaining)
    }

    fn match_market_sell(book: &mut OrderBook, mut order: Order) -> (Vec<Trade>, f64) {
        let mut trades = Vec::new();
        while order.remaining > 0.0 {
            let best_bid_key = match book.bids.keys().rev().next().cloned() { Some(k) => k, None => break };
            let best_price = best_bid_key.into_inner();
            let level = book.bids.get_mut(&OrderedFloat(best_price)).unwrap();
            while let Some(mut maker) = level.orders.front_mut() {
                let qty = order.remaining.min(maker.remaining);
                let trade = Trade {
                    taker_order_id: order.order_id.clone(),
                    maker_order_id: maker.order_id.clone(),
                    price: maker.price,
                    quantity: qty,
                    timestamp: now_millis(),
                };
                trades.push(trade);
                order.remaining -= qty;
                maker.remaining -= qty;
                if maker.remaining <= 0.0 {
                    level.orders.pop_front();
                } else {
                    break;
                }
                if order.remaining <= 0.0 { break }
            }
            if level.orders.is_empty() { book.bids.remove(&OrderedFloat(best_price)); }
        }
        (trades, order.remaining)
    }
}

#[async_trait]
impl MatchingService for MatchingServiceImpl {
    async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError> {
        let timestamp = now_millis();
        let mut book = self.book.lock().unwrap();
        let mut trades = Vec::new();
        let mut remaining = req.quantity;
        let side = req.side.clone();
        match (req.kind.clone(), side) {
            (OrderKind::Limit, Side::Buy) => {
                let price = req.price.unwrap_or(0.0);
                let taker = Order { order_id: req.order_id.clone(), price, remaining: req.quantity, timestamp, side: Side::Buy };
                let (t, rem) = Self::match_limit_buy(&mut book, taker);
                trades.extend(t);
                remaining = rem;
                if remaining > 0.0 {
                    let rest = Order { order_id: req.order_id.clone(), price, remaining, timestamp, side: Side::Buy };
                    book.insert_order(rest);
                    book.remove_empty_levels();
                } else {
                    book.remove_empty_levels();
                }
            }
            (OrderKind::Limit, Side::Sell) => {
                let price = req.price.unwrap_or(0.0);
                let taker = Order { order_id: req.order_id.clone(), price, remaining: req.quantity, timestamp, side: Side::Sell };
                let (t, rem) = Self::match_limit_sell(&mut book, taker);
                trades.extend(t);
                remaining = rem;
                if remaining > 0.0 {
                    let rest = Order { order_id: req.order_id.clone(), price, remaining, timestamp, side: Side::Sell };
                    book.insert_order(rest);
                    book.remove_empty_levels();
                } else {
                    book.remove_empty_levels();
                }
            }
            (OrderKind::Market, Side::Buy) => {
                let taker = Order { order_id: req.order_id.clone(), price: 0.0, remaining: req.quantity, timestamp, side: Side::Buy };
                let (t, rem) = Self::match_market_buy(&mut book, taker);
                trades.extend(t);
                remaining = rem;
                book.remove_empty_levels();
            }
            (OrderKind::Market, Side::Sell) => {
                let taker = Order { order_id: req.order_id.clone(), price: 0.0, remaining: req.quantity, timestamp, side: Side::Sell };
                let (t, rem) = Self::match_market_sell(&mut book, taker);
                trades.extend(t);
                remaining = rem;
                book.remove_empty_levels();
            }
        }
        let success = !trades.is_empty();
        let msg = if success { "matched".to_string() } else { "no match, order placed or empty book".to_string() };
        Ok(OrderResponse { success, message: msg, trades, remaining })
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
    let service_config = ServiceConfig::new("com.metawebthree.common.rpc.interface.MatchingService", svc_impl)
        .with_group("/dev/metawebthree")
        .with_protocol("dubbo");
    dubbo::init()
        .with_registry(registry_config)
        .with_protocol(protocol_config)
        .with_service(service_config)
        .serve();
}