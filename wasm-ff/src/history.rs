use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

#[derive(Clone, Serialize, Deserialize)]
pub enum ExcelAction {
    MergeCells {
        start_row: usize,
        start_col: usize,
        end_row: usize,
        end_col: usize,
    },
    UnmergeCells {
        row: usize,
        col: usize,
        original_span: CellSpan,
    },
    SetValue {
        row: usize,
        col: usize,
        old_value: String,
        new_value: String,
    },
}

pub struct History {
    undo_stack: VecDeque<ExcelAction>,
    redo_stack: VecDeque<ExcelAction>,
    max_history: usize,
}

impl History {
    pub fn new(max_history: usize) -> Self {
        History {
            undo_stack: VecDeque::with_capacity(max_history),
            redo_stack: VecDeque::new(),
            max_history,
        }
    }

    pub fn push(&mut self, action: ExcelAction) {
        if self.undo_stack.len() >= self.max_history {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(action);
        self.redo_stack.clear(); // 清除重做栈
    }

    pub fn undo(&mut self) -> Option<ExcelAction> {
        if let Some(action) = self.undo_stack.pop_back() {
            self.redo_stack.push_back(action.clone());
            Some(action)
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<ExcelAction> {
        if let Some(action) = self.redo_stack.pop_back() {
            self.undo_stack.push_back(action.clone());
            Some(action)
        } else {
            None
        }
    }
} 