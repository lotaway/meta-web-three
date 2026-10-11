package com.metawebthree.warehouse.interfaces.controller;

import com.metawebthree.warehouse.application.StocktakeOrderService;
import com.metawebthree.warehouse.application.dto.StocktakeOrderDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/stocktake")
public class StocktakeOrderController {

    private final StocktakeOrderService stocktakeOrderService;

    public StocktakeOrderController(StocktakeOrderService stocktakeOrderService) {
        this.stocktakeOrderService = stocktakeOrderService;
    }

    @PostMapping
    public ResponseEntity<StocktakeOrderDTO> create(@RequestBody StocktakeOrderDTO dto) {
        return ResponseEntity.ok(stocktakeOrderService.createStocktakeOrder(dto));
    }

    @GetMapping("/{orderNo}")
    public ResponseEntity<StocktakeOrderDTO> get(@PathVariable String orderNo) {
        return ResponseEntity.ok(stocktakeOrderService.queryStocktakeOrder(orderNo));
    }

    @GetMapping
    public ResponseEntity<List<StocktakeOrderDTO>> list(
            @RequestParam(required = false) Long warehouseId,
            @RequestParam(required = false) String status) {
        return ResponseEntity.ok(stocktakeOrderService.listStocktakeOrders(warehouseId, status));
    }

    @PostMapping("/{orderNo}/submit")
    public ResponseEntity<StocktakeOrderDTO> submit(@PathVariable String orderNo) {
        return ResponseEntity.ok(stocktakeOrderService.submitStocktakeOrder(orderNo));
    }

    @PostMapping("/{orderNo}/start")
    public ResponseEntity<StocktakeOrderDTO> start(@PathVariable String orderNo) {
        return ResponseEntity.ok(stocktakeOrderService.startStocktake(orderNo));
    }

    @PostMapping("/{orderNo}/complete-counting")
    public ResponseEntity<StocktakeOrderDTO> completeCounting(@PathVariable String orderNo) {
        return ResponseEntity.ok(stocktakeOrderService.completeCounting(orderNo));
    }

    @PostMapping("/{orderNo}/report-discrepancy")
    public ResponseEntity<StocktakeOrderDTO> reportDiscrepancy(@PathVariable String orderNo) {
        return ResponseEntity.ok(stocktakeOrderService.reportDiscrepancy(orderNo));
    }

    @PostMapping("/{orderNo}/adjust")
    public ResponseEntity<StocktakeOrderDTO> adjust(@PathVariable String orderNo) {
        return ResponseEntity.ok(stocktakeOrderService.adjustInventory(orderNo));
    }

    @PostMapping("/{orderNo}/complete")
    public ResponseEntity<StocktakeOrderDTO> complete(@PathVariable String orderNo) {
        return ResponseEntity.ok(stocktakeOrderService.completeStocktake(orderNo));
    }

    @PostMapping("/{orderNo}/cancel")
    public ResponseEntity<StocktakeOrderDTO> cancel(@PathVariable String orderNo) {
        return ResponseEntity.ok(stocktakeOrderService.cancelStocktake(orderNo));
    }
}
