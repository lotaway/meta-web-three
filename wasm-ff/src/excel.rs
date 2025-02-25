use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::excel::history::{History, ExcelAction};

#[derive(Serialize, Deserialize, Clone)]
pub struct CellSpan {
    row_span: usize,
    col_span: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Cell {
    value: String,
    span: Option<CellSpan>,
    is_merged_cell: bool,
    merged_parent: Option<(usize, usize)>, // 存储合并单元格的父单元格位置
}

#[wasm_bindgen]
pub struct ExcelSheet {
    rows: usize,
    cols: usize,
    data: Vec<Vec<Cell>>,
    merged_regions: HashMap<(usize, usize), CellSpan>, // 存储合并区域信息
    history: History,
}

#[wasm_bindgen]
impl ExcelSheet {
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize) -> ExcelSheet {
        let mut data = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                row.push(Cell {
                    value: String::new(),
                    span: None,
                    is_merged_cell: false,
                    merged_parent: None,
                });
            }
            data.push(row);
        }
        
        ExcelSheet { 
            rows, 
            cols, 
            data,
            merged_regions: HashMap::new(),
            history: History::new(100),
        }
    }

    pub fn merge_cells(&mut self, start_row: usize, start_col: usize, 
                      end_row: usize, end_col: usize) -> Result<(), JsValue> {
        // 验证输入范围
        if start_row >= self.rows || end_row >= self.rows || 
           start_col >= self.cols || end_col >= self.cols ||
           start_row > end_row || start_col > end_col {
            return Err(JsValue::from_str("Invalid merge range"));
        }

        // 检查是否与现有合并区域重叠
        for row in start_row..=end_row {
            for col in start_col..=end_col {
                if self.data[row][col].is_merged_cell {
                    return Err(JsValue::from_str("Cannot merge cells that are already part of a merged region"));
                }
            }
        }

        let span = CellSpan {
            row_span: end_row - start_row + 1,
            col_span: end_col - start_col + 1,
        };

        // 设置合并区域的父单元格
        self.data[start_row][start_col].span = Some(span.clone());
        self.merged_regions.insert((start_row, start_col), span);

        // 标记合并区域内的其他单元格
        for row in start_row..=end_row {
            for col in start_col..=end_col {
                if row != start_row || col != start_col {
                    self.data[row][col].is_merged_cell = true;
                    self.data[row][col].merged_parent = Some((start_row, start_col));
                }
            }
        }

        // 记录合并操作
        self.history.push(ExcelAction::MergeCells {
            start_row,
            start_col,
            end_row,
            end_col,
        });

        Ok(())
    }

    pub fn unmerge_cells(&mut self, row: usize, col: usize) -> Result<(), JsValue> {
        let cell = &self.data[row][col];
        
        // 检查是否是合并区域的父单元格
        if let Some(span) = &cell.span {
            let row_span = span.row_span;
            let col_span = span.col_span;
            
            // 清除合并区域内所有单元格的合并状态
            for r in row..row + row_span {
                for c in col..col + col_span {
                    self.data[r][c].is_merged_cell = false;
                    self.data[r][c].merged_parent = None;
                }
            }
            
            // 清除父单元格的span信息
            self.data[row][col].span = None;
            self.merged_regions.remove(&(row, col));
            
            // 记录取消合并操作
            self.history.push(ExcelAction::UnmergeCells {
                row,
                col,
                original_span: span.clone(),
            });
            
            Ok(())
        } else {
            Err(JsValue::from_str("Selected cell is not a merge parent"))
        }
    }

    pub fn get_cell_value(&self, row: usize, col: usize) -> Result<String, JsValue> {
        if row >= self.rows || col >= self.cols {
            return Err(JsValue::from_str("Cell index out of bounds"));
        }

        let cell = &self.data[row][col];
        if cell.is_merged_cell {
            if let Some((parent_row, parent_col)) = cell.merged_parent {
                Ok(self.data[parent_row][parent_col].value.clone())
            } else {
                Ok(String::new())
            }
        } else {
            Ok(cell.value.clone())
        }
    }

    pub fn set_cell_value(&mut self, row: usize, col: usize, value: String) -> Result<(), JsValue> {
        if row >= self.rows || col >= self.cols {
            return Err(JsValue::from_str("Cell index out of bounds"));
        }

        let cell = &mut self.data[row][col];
        if cell.is_merged_cell {
            if let Some((parent_row, parent_col)) = cell.merged_parent {
                self.data[parent_row][parent_col].value = value;
            }
        } else {
            cell.value = value;
        }
        Ok(())
    }

    pub fn undo(&mut self) -> Result<(), JsValue> {
        if let Some(action) = self.history.undo() {
            match action {
                ExcelAction::MergeCells { start_row, start_col, end_row, end_col } => {
                    self.unmerge_cells(start_row, start_col)?;
                },
                ExcelAction::UnmergeCells { row, col, original_span } => {
                    self.merge_cells(row, col, 
                        row + original_span.row_span - 1,
                        col + original_span.col_span - 1)?;
                },
                ExcelAction::SetValue { row, col, old_value, .. } => {
                    self.set_cell_value(row, col, old_value)?;
                }
            }
            Ok(())
        } else {
            Err(JsValue::from_str("No action to undo"))
        }
    }

    pub fn redo(&mut self) -> Result<(), JsValue> {
        if let Some(action) = self.history.redo() {
            match action {
                ExcelAction::MergeCells { start_row, start_col, end_row, end_col } => {
                    self.merge_cells(start_row, start_col, end_row, end_col)?;
                },
                ExcelAction::UnmergeCells { row, col, .. } => {
                    self.unmerge_cells(row, col)?;
                },
                ExcelAction::SetValue { row, col, new_value, .. } => {
                    self.set_cell_value(row, col, new_value)?;
                }
            }
            Ok(())
        } else {
            Err(JsValue::from_str("No action to redo"))
        }
    }
} 