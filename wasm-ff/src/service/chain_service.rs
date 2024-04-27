pub struct ChainService {
    pub pre_fix: &str,
}

impl ChainService {
    pub fn new(_pre_fix: Option<&str>) -> Self {
        ChainService { pre_fix: _pre_fix.or_else("").unwrap() }
    }

    pub fn start(&self) {

    }
}

