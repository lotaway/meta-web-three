package com.metawebthree.aftersale.interfaces.controller;

import com.metawebthree.aftersale.application.dto.AfterSaleApplyDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleProcessDTO;
import com.metawebthree.aftersale.application.service.AfterSaleApplicationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/after-sale")
public class AfterSaleController {

    private final AfterSaleApplicationService afterSaleService;

    public AfterSaleController(AfterSaleApplicationService afterSaleService) {
        this.afterSaleService = afterSaleService;
    }

    /**
     * Apply for after-sale
     */
    @PostMapping("/apply")
    public ResponseEntity<AfterSaleDTO> apply(
            @RequestBody AfterSaleApplyDTO applyDTO,
            @RequestHeader("X-User-Id") Long userId) {
        AfterSaleDTO result = afterSaleService.apply(applyDTO, userId);
        return ResponseEntity.ok(result);
    }

    /**
     * Process after-sale (admin)
     */
    @PostMapping("/process")
    public ResponseEntity<AfterSaleDTO> process(@RequestBody AfterSaleProcessDTO processDTO) {
        AfterSaleDTO result = afterSaleService.process(processDTO);
        return ResponseEntity.ok(result);
    }

    /**
     * Get after-sale by ID
     */
    @GetMapping("/{id}")
    public ResponseEntity<AfterSaleDTO> getById(@PathVariable Long id) {
        AfterSaleDTO result = afterSaleService.getById(id);
        return ResponseEntity.ok(result);
    }

    /**
     * Get after-sale list by user ID
     */
    @GetMapping("/user/{userId}")
    public ResponseEntity<List<AfterSaleDTO>> getByUserId(@PathVariable Long userId) {
        List<AfterSaleDTO> result = afterSaleService.getByUserId(userId);
        return ResponseEntity.ok(result);
    }

    /**
     * Get after-sale list by order ID
     */
    @GetMapping("/order/{orderId}")
    public ResponseEntity<List<AfterSaleDTO>> getByOrderId(@PathVariable Long orderId) {
        List<AfterSaleDTO> result = afterSaleService.getByOrderId(orderId);
        return ResponseEntity.ok(result);
    }

    /**
     * Get all after-sale records (admin)
     */
    @GetMapping("/list")
    public ResponseEntity<List<AfterSaleDTO>> getAll() {
        List<AfterSaleDTO> result = afterSaleService.getAll();
        return ResponseEntity.ok(result);
    }
}