pub struct ChainService {
    pub pre_fix: String,
}

impl ChainService {
    pub fn new(_pre_fix: Option<&str>) -> Self {
        ChainService { pre_fix: _pre_fix.unwrap_or("").to_string() }
    }

    pub fn start(&self) {

    }
}

