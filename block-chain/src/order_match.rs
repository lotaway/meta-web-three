use anyhow::Result;
use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use crossbeam_skiplist::SkipMap;
use dubbo::{
    codegen::async_trait,
    config::{protocol::ProtocolConfig, registry::RegistryConfig, service::ServiceConfig},
    status::DubboError,
};
use ordered_float::OrderedFloat;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rocksdb::{Options, DB};
use serde::{Deserialize, Serialize};
use serde_json::to_string;
use slab::Slab;
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Seek, SeekFrom, Write},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum OrderKind {
    Limit,
    Market,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct OrderRequest {
    pub market: String,
    pub chain: String,
    pub order_id: String,
    pub side: Side,
    pub kind: OrderKind,
    pub price: Option<f64>,
    pub quantity: f64,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Trade {
    pub market: String,
    pub chain: String,
    pub taker_order_id: String,
    pub maker_order_id: String,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u128,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
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
    side: Side,
    market: String,
    chain: String,
}

struct PriceLevel {
    orders: Vec<usize>,
}
impl PriceLevel {
    fn new() -> Self {
        Self { orders: Vec::new() }
    }
}

struct OrderBook {
    buys: SkipMap<OrderedFloat<f64>, Arc<Mutex<PriceLevel>>>,
    sells: SkipMap<OrderedFloat<f64>, Arc<Mutex<PriceLevel>>>,
    slab: Slab<OrderEntry>,
}

impl OrderBook {
    fn new() -> Self {
        Self {
            buys: SkipMap::new(),
            sells: SkipMap::new(),
            slab: Slab::with_capacity(1024),
        }
    }
}

enum EngineCommand {
    NewOrder(OrderRequest),
    CancelOrder { market: String, order_id: String },
}

struct Shard {
    id: usize,
    price_min: f64,
    price_max: f64,
    book: Arc<Mutex<OrderBook>>,
    cmd_rx: Receiver<EngineCommand>,
    trade_tx: Sender<Trade>,
    wal: Arc<Mutex<File>>,
    producer: FutureProducer,
    kafka_topic: String,
}

impl Shard {
    fn run(self) {
        loop {
            match self.cmd_rx.try_recv() {
                Ok(cmd) => match cmd {
                    EngineCommand::NewOrder(req) => {
                        let (trades, _) = Self::process_new_order(&self, req);
                        for t in trades {
                            let _ = self.trade_tx.send(t.clone());
                            let _ = Self::append_wal_trade(&self.wal, &t);
                            let _ = Self::produce_kafka(&self.producer, &self.kafka_topic, &t);
                        }
                    }
                    EngineCommand::CancelOrder { order_id, .. } => {
                        let _ = Self::process_cancel(&self, &order_id);
                    }
                },
                Err(TryRecvError::Empty) => {
                    thread::sleep(Duration::from_micros(50));
                }
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    fn append_wal_trade(wal: &Arc<Mutex<File>>, trade: &Trade) -> bool {
        if let Ok(line) = to_string(trade) {
            let mut f = wal.lock().unwrap();
            let _ = f.write_all(line.as_bytes());
            let _ = f.write_all(b"\n");
            let _ = f.flush();
            return true;
        }
        false
    }

    fn produce_kafka(producer: &FutureProducer, topic: &str, trade: &Trade) -> bool {
        if let Ok(payload) = to_string(trade) {
            let rec: FutureRecord<String, String> =
                FutureRecord::to(topic).payload(&payload).key(&trade.market);
            let _ = producer.send(rec, Duration::from_secs(0));
            return true;
        }
        false
    }

    fn process_new_order(s: &Shard, req: OrderRequest) -> (Vec<Trade>, f64) {
        let timestamp = now_millis();
        let mut trades = Vec::new();
        let mut remaining = req.quantity;
        let mut book = s.book.lock().unwrap();
        let slab_idx = book.slab.insert(OrderEntry {
            order_id: req.order_id.clone(),
            price: req.price.unwrap_or(0.0),
            remaining: req.quantity,
            timestamp,
            side: req.side.clone(),
            market: req.market.clone(),
            chain: req.chain.clone(),
        });
        match (req.kind.clone(), req.side.clone()) {
            (OrderKind::Limit, Side::Buy) => {
                while remaining > 0.0 {
                    let opt = book
                        .sells
                        .front()
                        .and_then(|e| Some((e.key().clone(), e.value().clone())));
                    let (best_price_k, best_price_pl) = match opt {
                        Some(x) => x,
                        None => break,
                    };
                    let best_price = best_price_k.into_inner();
                    if req.price.unwrap_or(0.0) < best_price {
                        break;
                    }
                    let mut pl = best_price_pl.lock().unwrap();
                    while remaining > 0.0 && !pl.orders.is_empty() {
                        let maker_idx = pl.orders[0];
                        {
                            let maker = book.slab.get_mut(maker_idx).unwrap();
                            let qty = remaining.min(maker.remaining);
                            let trade = Trade {
                                market: req.market.clone(),
                                chain: req.chain.clone(),
                                taker_order_id: req.order_id.clone(),
                                maker_order_id: maker.order_id.clone(),
                                price: maker.price,
                                quantity: qty,
                                timestamp: now_millis(),
                            };
                            trades.push(trade);
                            remaining -= qty;
                            maker.remaining -= qty;
                        }
                        if book
                            .slab
                            .get(maker_idx)
                            .map(|m| m.remaining <= 0.0)
                            .unwrap_or(true)
                        {
                            pl.orders.remove(0);
                            let _ = book.slab.remove(maker_idx);
                        } else {
                            break;
                        }
                    }
                    if pl.orders.is_empty() {
                        book.sells.remove(&best_price_k);
                    }
                }
                if remaining > 0.0 {
                    if let Some(mut this_entry) = book.slab.get_mut(slab_idx) {
                        this_entry.remaining = remaining;
                    }
                    let key = OrderedFloat(req.price.unwrap_or(0.0));
                    if let Some(g) = book.buys.get(&key) {
                        let mut pl = g.value().lock().unwrap();
                        pl.orders.push(slab_idx);
                    } else {
                        let new_pl = Arc::new(Mutex::new(PriceLevel::new()));
                        new_pl.lock().unwrap().orders.push(slab_idx);
                        book.buys.insert(key, new_pl.clone());
                    }
                } else {
                    let _ = book.slab.remove(slab_idx);
                }
            }
            (OrderKind::Limit, Side::Sell) => {
                while remaining > 0.0 {
                    let opt = book
                        .buys
                        .back()
                        .and_then(|e| Some((e.key().clone(), e.value().clone())));
                    let (best_price_k, best_price_pl) = match opt {
                        Some(x) => x,
                        None => break,
                    };
                    let best_price = best_price_k.into_inner();
                    if req.price.unwrap_or(0.0) > best_price {
                        break;
                    }
                    let mut pl = best_price_pl.lock().unwrap();
                    while remaining > 0.0 && !pl.orders.is_empty() {
                        let maker_idx = pl.orders[0];
                        {
                            let maker = book.slab.get_mut(maker_idx).unwrap();
                            let qty = remaining.min(maker.remaining);
                            let trade = Trade {
                                market: req.market.clone(),
                                chain: req.chain.clone(),
                                taker_order_id: req.order_id.clone(),
                                maker_order_id: maker.order_id.clone(),
                                price: maker.price,
                                quantity: qty,
                                timestamp: now_millis(),
                            };
                            trades.push(trade);
                            remaining -= qty;
                            maker.remaining -= qty;
                        }
                        if book
                            .slab
                            .get(maker_idx)
                            .map(|m| m.remaining <= 0.0)
                            .unwrap_or(true)
                        {
                            pl.orders.remove(0);
                            let _ = book.slab.remove(maker_idx);
                        } else {
                            break;
                        }
                    }
                    if pl.orders.is_empty() {
                        book.buys.remove(&best_price_k);
                    }
                }
                if remaining > 0.0 {
                    if let Some(mut this_entry) = book.slab.get_mut(slab_idx) {
                        this_entry.remaining = remaining;
                    }
                    let key = OrderedFloat(req.price.unwrap_or(0.0));
                    if let Some(g) = book.sells.get(&key) {
                        let mut pl = g.value().lock().unwrap();
                        pl.orders.push(slab_idx);
                    } else {
                        let new_pl = Arc::new(Mutex::new(PriceLevel::new()));
                        new_pl.lock().unwrap().orders.push(slab_idx);
                        book.sells.insert(key, new_pl.clone());
                    }
                } else {
                    let _ = book.slab.remove(slab_idx);
                }
            }
            (OrderKind::Market, Side::Buy) | (OrderKind::Market, Side::Sell) => {
                // 简化：市场单吃对方直到耗尽或对手空
                let is_buy = matches!(req.side, Side::Buy);
                while remaining > 0.0 {
                    let opt = if is_buy {
                        book.sells
                            .front()
                            .and_then(|e| Some((e.key().clone(), e.value().clone())))
                    } else {
                        book.buys
                            .back()
                            .and_then(|e| Some((e.key().clone(), e.value().clone())))
                    };
                    let (best_k, best_v) = match opt {
                        Some(x) => x,
                        None => break,
                    };
                    let mut pl = best_v.lock().unwrap();
                    while remaining > 0.0 && !pl.orders.is_empty() {
                        let maker_idx = pl.orders[0];
                        {
                            let maker = book.slab.get_mut(maker_idx).unwrap();
                            let qty = remaining.min(maker.remaining);
                            let trade = Trade {
                                market: req.market.clone(),
                                chain: req.chain.clone(),
                                taker_order_id: req.order_id.clone(),
                                maker_order_id: maker.order_id.clone(),
                                price: maker.price,
                                quantity: qty,
                                timestamp: now_millis(),
                            };
                            trades.push(trade);
                            remaining -= qty;
                            maker.remaining -= qty;
                        }
                        if book
                            .slab
                            .get(maker_idx)
                            .map(|m| m.remaining <= 0.0)
                            .unwrap_or(true)
                        {
                            pl.orders.remove(0);
                            let _ = book.slab.remove(maker_idx);
                        } else {
                            break;
                        }
                    }
                    if pl.orders.is_empty() {
                        if is_buy {
                            book.sells.remove(&best_k);
                        } else {
                            book.buys.remove(&best_k);
                        }
                    }
                }
                if remaining > 0.0 {
                    let _ = book.slab.remove(slab_idx);
                }
            }
        }
        (trades, remaining)
    }

    fn process_cancel(_s: &Shard, order_id: &str) -> bool {
        let mut found = false;
        // 简化：扫描 slab（在高 QPS 下应用 map order_id->idx 索引，这里为示意）
        true
    }
}

struct ShardManager {
    market: String,
    shards: Arc<Mutex<Vec<(usize, Sender<EngineCommand>)>>>,
    producer: FutureProducer,
    kafka_topic: String,
    wal_dir: String,
    db: Arc<Mutex<DB>>,
}

impl ShardManager {
    fn new(
        market: &str,
        producer: FutureProducer,
        kafka_topic: &str,
        wal_dir: &str,
        db: Arc<Mutex<DB>>,
    ) -> Self {
        Self {
            market: market.to_string(),
            shards: Arc::new(Mutex::new(Vec::new())),
            producer,
            kafka_topic: kafka_topic.to_string(),
            wal_dir: wal_dir.to_string(),
            db,
        }
    }

    fn start_with_shards(&self, initial: usize) {
        for i in 0..initial {
            self.spawn_shard(i);
        }
        self.spawn_monitor();
    }

    fn spawn_shard(&self, id: usize) {
        let (cmd_tx, cmd_rx) = bounded::<EngineCommand>(16384);
        let (trade_tx, _trade_rx) = bounded::<Trade>(65536);
        let wal_path = format!(
            "{}/wal_{}_shard_{}.log",
            self.wal_dir,
            self.market.replace("/", "_"),
            id
        );
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)
            .unwrap();
        let shard = Shard {
            id,
            price_min: f64::MIN,
            price_max: f64::MAX,
            book: Arc::new(Mutex::new(OrderBook::new())),
            cmd_rx,
            trade_tx,
            wal: Arc::new(Mutex::new(file)),
            producer: self.producer.clone(),
            kafka_topic: self.kafka_topic.clone(),
        };
        thread::spawn(move || {
            shard.run();
        });
        self.shards.lock().unwrap().push((id, cmd_tx));
    }

    fn spawn_monitor(&self) {
        let shards = self.shards.clone();
        let producer = self.producer.clone();
        let topic = self.kafka_topic.clone();
        let market = self.market.clone();
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(1));
            let s = shards.lock().unwrap();
            let total_q: usize = s.iter().map(|(_, tx)| tx.len()).sum();
            if total_q > 10000 {
                let new_id = s.len();
                drop(s);
                let (cmd_tx, cmd_rx) = bounded::<EngineCommand>(16384);
                let (trade_tx, _trade_rx) = bounded::<Trade>(65536);
                let wal_path =
                    format!("/tmp/wal_{}_shard_{}.log", market.replace("/", "_"), new_id);
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&wal_path)
                    .unwrap();
                let shard = Shard {
                    id: new_id,
                    price_min: f64::MIN,
                    price_max: f64::MAX,
                    book: Arc::new(Mutex::new(OrderBook::new())),
                    cmd_rx,
                    trade_tx,
                    wal: Arc::new(Mutex::new(file)),
                    producer: producer.clone(),
                    kafka_topic: topic.clone(),
                };
                thread::spawn(move || {
                    shard.run();
                });
                let mut mm = shards.lock().unwrap();
                mm.push((new_id, cmd_tx));
            }
        });
    }

    fn route(&self, req: OrderRequest) -> Result<()> {
        let shards = self.shards.lock().unwrap();
        if shards.is_empty() {
            return Err(anyhow::anyhow!("no shards"));
        }
        let idx =
            (hash_market_price(&req.market, req.price.unwrap_or(0.0)) as usize) % shards.len();
        let cmd = EngineCommand::NewOrder(req);
        let tx = &shards[idx].1;
        tx.send(cmd).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(())
    }
}

fn hash_market_price(market: &str, price: f64) -> u64 {
    let mut h = fxhash::FxHasher::default();
    use std::hash::{Hash, Hasher};
    market.hash(&mut h);
    ((price as u64).wrapping_mul(31)).hash(&mut h);
    h.finish()
}

#[dubbo::service]
#[async_trait]
pub trait MatchingService {
    async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError>;
}

pub struct MatchingServiceImpl {
    managers: HashMap<String, Arc<ShardManager>>,
    producer: FutureProducer,
    kafka_topic: String,
}

impl MatchingServiceImpl {
    pub fn new(
        markets: Vec<String>,
        kafka_brokers: &str,
        kafka_topic: &str,
        wal_dir: &str,
        db_path: &str,
    ) -> Self {
        env_logger::init();
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", kafka_brokers)
            .create()
            .expect("producer");
        let mut managers = HashMap::new();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = Arc::new(Mutex::new(DB::open(&opts, db_path).unwrap()));
        for m in markets {
            let mgr = Arc::new(ShardManager::new(
                &m,
                producer.clone(),
                kafka_topic,
                wal_dir,
                db.clone(),
            ));
            mgr.start_with_shards(1);
            managers.insert(m.clone(), mgr);
        }
        MatchingServiceImpl {
            managers,
            producer,
            kafka_topic: kafka_topic.to_string(),
        }
    }
}

#[async_trait]
impl MatchingService for MatchingServiceImpl {
    async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError> {
        if let Some(mgr) = self.managers.get(&req.market) {
            if let Err(e) = mgr.route(req.clone()) {
                return Ok(OrderResponse {
                    success: false,
                    message: format!("route err: {}", e),
                    trades: vec![],
                    remaining: req.quantity,
                });
            }
            Ok(OrderResponse {
                success: true,
                message: "accepted".to_string(),
                trades: vec![],
                remaining: req.quantity,
            })
        } else {
            Ok(OrderResponse {
                success: false,
                message: "unknown market".to_string(),
                trades: vec![],
                remaining: req.quantity,
            })
        }
    }
}

pub fn start_rpc(
    markets: Vec<String>,
    kafka_brokers: &str,
    kafka_topic: &str,
    wal_dir: &str,
    db_path: &str,
) {
    let registry_config = RegistryConfig {
        protocol: "zookeeper".to_string(),
        address: "127.0.0.1:2181".to_string(),
        ..Default::default()
    };
    let protocol_config = ProtocolConfig {
        name: "dubbo".to_string(),
        port: 20086,
        ..Default::default()
    };
    let svc = MatchingServiceImpl::new(markets, kafka_brokers, kafka_topic, wal_dir, db_path);
    let service_config = ServiceConfig::new("org.dex.matching.MatchingService", svc)
        .with_group("/dev/metawebthree")
        .with_protocol("dubbo");
    dubbo::init()
        .with_registry(registry_config)
        .with_protocol(protocol_config)
        .with_service(service_config)
        .serve();
}

// #[tokio::main]
async fn start() {
    let markets = vec![
        "BTC/USDT".to_string(),
        "ETH/USDT".to_string(),
        // "DOGE/USDT".to_string(),
    ];
    start_rpc(
        markets,
        "localhost:9092",
        "dex-trades",
        "/tmp",
        "/tmp/rocksdb",
    );
}
