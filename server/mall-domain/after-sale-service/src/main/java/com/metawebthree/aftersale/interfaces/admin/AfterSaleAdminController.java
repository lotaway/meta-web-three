package com.metawebthree.aftersale.interfaces.admin;

import com.metawebthree.aftersale.application.dto.AfterSaleDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleProcessDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleQueryDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleStatisticDTO;
import com.metawebthree.aftersale.application.service.AfterSaleApplicationService;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/admin/after-sale")
public class AfterSaleAdminController {

    @Autowired
    private AfterSaleApplicationService afterSaleService;

    /**
     * Get after-sale list with pagination
     */
    @GetMapping("/list")
    public ApiResponse<Map<String, Object>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String orderNo,
            @RequestParam(required = false) String userId,
            @RequestParam(required = false) Integer status,
            @RequestParam(required = false) Integer type,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate) {
        
        AfterSaleQueryDTO queryDTO = new AfterSaleQueryDTO();
        queryDTO.setPageNum(pageNum);
        queryDTO.setPageSize(pageSize);
        queryDTO.setOrderNo(orderNo);
        queryDTO.setUserId(userId);
        queryDTO.setStatus(status);
        queryDTO.setType(type);
        queryDTO.setStartDate(startDate);
        queryDTO.setEndDate(endDate);
        
        Map<String, Object> result = afterSaleService.getAllPaged(queryDTO);
        return ApiResponse.success(result);
    }

    /**
     * Get after-sale by ID
     */
    @GetMapping("/{id}")
    public ApiResponse<AfterSaleDTO> getById(@PathVariable Long id) {
        AfterSaleDTO result = afterSaleService.getById(id);
        if (result != null) {
            return ApiResponse.success(result);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "After-sale record not found");
    }

    /**
     * Process after-sale (approve/reject)
     */
    @PostMapping("/process")
    public ApiResponse<AfterSaleDTO> process(@RequestBody AfterSaleProcessDTO processDTO) {
        AfterSaleDTO result = afterSaleService.process(processDTO);
        return ApiResponse.success(result);
    }

    /**
     * Batch approve after-sale records
     */
    @PostMapping("/batch-approve")
    public ApiResponse<Void> batchApprove(@RequestBody List<Long> ids) {
        int count = afterSaleService.batchApprove(ids);
        return ApiResponse.success();
    }

    /**
     * Batch reject after-sale records
     */
    @PostMapping("/batch-reject")
    public ApiResponse<Void> batchReject(@RequestBody Map<String, Object> request) {
        @SuppressWarnings("unchecked")
        List<Long> ids = (List<Long>) request.get("ids");
        String reason = (String) request.get("reason");
        int count = afterSaleService.batchReject(ids, reason);
        return ApiResponse.success();
    }

    /**
     * Get after-sale statistics
     */
    @GetMapping("/statistics")
    public ApiResponse<AfterSaleStatisticDTO> getStatistics() {
        AfterSaleStatisticDTO result = afterSaleService.getStatistics();
        return ApiResponse.success(result);
    }

    /**
     * Export after-sale records
     */
    @GetMapping("/export")
    public ApiResponse<List<AfterSaleDTO>> export(
            @RequestParam(required = false) String orderNo,
            @RequestParam(required = false) String userId,
            @RequestParam(required = false) Integer status,
            @RequestParam(required = false) Integer type) {
        
        AfterSaleQueryDTO queryDTO = new AfterSaleQueryDTO();
        queryDTO.setOrderNo(orderNo);
        queryDTO.setUserId(userId);
        queryDTO.setStatus(status);
        queryDTO.setType(type);
        queryDTO.setPageNum(1);
        queryDTO.setPageSize(10000);
        
        Map<String, Object> result = afterSaleService.getAllPaged(queryDTO);
        @SuppressWarnings("unchecked")
        List<AfterSaleDTO> list = (List<AfterSaleDTO>) result.get("list");
        return ApiResponse.success(list);
    }
}