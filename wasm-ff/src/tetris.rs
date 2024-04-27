use std::collections::HashSet;

pub struct Shape {
    pub positions: HashSet<Vec<usize>>,
}

pub struct Tetris {
    width: usize,
    height: usize,
}

impl Tetris {
    pub fn new(vec: Vec<usize>) -> Self {
        Tetris {
            width: *vec.get(0).unwrap(),
            height: *vec.get(1).unwrap(),
        }
    }
}